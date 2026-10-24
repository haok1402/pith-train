"""
BF16SwiGLUExpertsFunc (operators.bf16_experts: the fused CuTe stage 3 from the received tokens) against
the unfused path (padded_index_gather -> scatter_for_grouped_gemm -> GroupedLinear gate/up
-> silu_mul -> GroupedLinear down -> padded_index_gather) and an fp32 reference, on EP-dedup
routings: the output, the received tokens' gradient and the three weight gradients accumulated
into fp32 main_grads. Tolerance: relative error vs the fp32 reference at most 1.5x the unfused path's.
The weight gradients must wait for WeightGradStore's W phase, and the node must be deterministic.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.contexts import training
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels.bf16_expert_support import QUACK_VERSION
from pithtrain.kernels.bf16_expert_support import quack_version as _quack_version
from pithtrain.operators.grouped_linear import GroupedLinear
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9
    or _quack_version() != QUACK_VERSION,
    reason="needs SM90 and the validated quack-kernels",
)


def ep_routing(G, ep, tokens_per_sender, k, seed=0):
    """Slots sender-major, expert-minor; each token on up to k local experts (dedup'd)."""
    g = torch.Generator().manual_seed(seed)
    eids, exps, base = [], [], 0
    for _ in range(ep):
        choice = []
        for t in range(tokens_per_sender):
            n = int(torch.randint(1, k + 1, (1,), generator=g))
            for e in torch.randperm(G, generator=g)[:n].tolist():
                choice.append((e, t))
        choice.sort()
        eids += [e for e, _ in choice]
        exps += [base + t for _, t in choice]
        base += tokens_per_sender
    return torch.tensor(eids).cuda(), torch.tensor(exps).cuda(), base


def make_modules(G, H, inter, seed=0):
    torch.manual_seed(seed)
    mods = [GroupedLinear(G, H, inter), GroupedLinear(G, H, inter), GroupedLinear(G, inter, H)]
    for m in mods:
        torch.nn.init.normal_(m.weight, std=m.in_features**-0.5)
        m.cuda().to(torch.bfloat16)
        m.main_grad = torch.zeros(m.weight.shape, dtype=torch.float32, device="cuda")
    return mods


def run_r5base(mods, gathered, expert_idxs, expand_idx, dout):
    G = mods[0].num_groups
    src = gathered.detach().requires_grad_()
    x = padded_index_gather(src, expand_idx)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    xs, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(x, expert_idxs, G, group_counts=counts)
    kw = dict(grouped_mm_offs=offs, ks=ks, ks_tensor=ks_t)
    y = mods[2](silu_mul(mods[0](xs, **kw), mods[1](xs, **kw)), **kw)
    out = padded_index_gather(y, rsi)
    out.backward(dout)
    return out.detach(), src.grad, [m.main_grad.clone() for m in mods]


def run_fused(mods, gathered, expert_idxs, expand_idx, dout, k, deferred=False):
    from pithtrain.operators.bf16_experts import bf16_swiglu_experts

    src = gathered.detach().requires_grad_()
    out = bf16_swiglu_experts(*mods, src, expert_idxs, expand_idx, k)
    WeightGradStore.enabled = deferred
    try:
        out.backward(dout)
    finally:
        WeightGradStore.enabled = False
    if deferred:
        assert all(torch.count_nonzero(m.main_grad) == 0 for m in mods), "wgrad ran before W"
        WeightGradStore.flush()
        WeightGradStore.pop()
    grads = [m.main_grad.clone() if m.main_grad is not None else m.weight.grad for m in mods]
    return out.detach(), src.grad, grads


def run_fp32(mods, gathered, expert_idxs, expand_idx, dout):
    ws = [m.weight.detach().float().requires_grad_() for m in mods]
    src = gathered.detach().float().requires_grad_()
    x = src[expand_idx]
    out = torch.empty(x.shape[0], ws[2].shape[1], device="cuda")
    for e in range(ws[0].shape[0]):
        sel = (expert_idxs == e).nonzero().squeeze(1)
        xe = x[sel]
        out = out.index_put((sel,), (F.silu(xe @ ws[0][e].T) * (xe @ ws[1][e].T)) @ ws[2][e].T)
    out.backward(dout.float())
    return out.detach(), src.grad, [w.grad for w in ws]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


@pytest.fixture(autouse=True)
def bf16(monkeypatch):
    monkeypatch.setattr(training, "fp8", False, raising=False)


@pytest.mark.parametrize(
    "G,ep,tps,k,H,inter",
    [(32, 4, 1024, 8, 2048, 768), (16, 4, 700, 6, 2048, 1408), (8, 2, 300, 4, 1024, 512)],
)
def test_fused_vs_r5base_and_fp32(G, ep, tps, k, H, inter):
    expert_idxs, expand_idx, T = ep_routing(G, ep, tps, k)
    torch.manual_seed(5)
    gathered = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(expert_idxs.shape[0], H, device="cuda", dtype=torch.bfloat16)
    ref = run_fp32(make_modules(G, H, inter), gathered, expert_idxs, expand_idx, dout)
    r5 = run_r5base(make_modules(G, H, inter), gathered, expert_idxs, expand_idx, dout)
    new = run_fused(make_modules(G, H, inter), gathered, expert_idxs, expand_idx, dout, k)
    names = ("out", "dgathered", "main_grad gate", "main_grad up", "main_grad down")
    for name, a, b, r in zip(names, (new[0], new[1], *new[2]), (r5[0], r5[1], *r5[2]), (ref[0], ref[1], *ref[2])):  # fmt: skip
        assert rel(a, r) <= 1.5 * rel(b, r), f"{name}: {rel(a, r):.3e} vs unfused {rel(b, r):.3e}"


def test_deferred_wgrad_and_determinism():
    G, H, inter, k = 32, 2048, 768, 8
    expert_idxs, expand_idx, T = ep_routing(G, 4, 512, k, seed=1)
    torch.manual_seed(6)
    gathered = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(expert_idxs.shape[0], H, device="cuda", dtype=torch.bfloat16)
    a = run_fused(
        make_modules(G, H, inter), gathered, expert_idxs, expand_idx, dout, k, deferred=True
    )
    b = run_fused(make_modules(G, H, inter), gathered, expert_idxs, expand_idx, dout, k)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for x, y in zip(a[2], b[2]):
        assert torch.equal(x, y)


def test_without_main_grad_adds_to_grad():
    """Modules without an fp32 main_grad (no wgrad_accumulation) get their bf16 .grad."""
    G, H, inter, k = 8, 1024, 512, 4
    expert_idxs, expand_idx, T = ep_routing(G, 2, 200, k, seed=2)
    gathered = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(expert_idxs.shape[0], H, device="cuda", dtype=torch.bfloat16)
    with_mg = run_fused(make_modules(G, H, inter), gathered, expert_idxs, expand_idx, dout, k)[2]
    mods = make_modules(G, H, inter)
    for m in mods:
        m.main_grad = None
    run_fused(mods, gathered, expert_idxs, expand_idx, dout, k)
    for m, mg in zip(mods, with_mg):
        assert torch.equal(m.weight.grad, mg.to(torch.bfloat16))


def test_fused_gate_up_layout_vs_r5base_and_fp32():
    """qwen3.5's experts: one gate_up_proj whose rows are [gate; up] (up_proj=None), read in place
    as the pair; its main_grad gets both halves."""
    from pithtrain.operators.bf16_experts import bf16_swiglu_experts

    G, H, inter, k = 64, 2048, 512, 8
    expert_idxs, expand_idx, T = ep_routing(G, 4, 400, k, seed=3)
    torch.manual_seed(7)
    gathered = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(expert_idxs.shape[0], H, device="cuda", dtype=torch.bfloat16)

    def modules():
        sep = make_modules(G, H, inter)
        gate_up = GroupedLinear(G, H, 2 * inter).cuda().to(torch.bfloat16)
        with torch.no_grad():
            gate_up.weight.copy_(torch.cat([sep[0].weight, sep[1].weight], 1))
        gate_up.main_grad = torch.zeros(gate_up.weight.shape, dtype=torch.float32, device="cuda")
        return gate_up, sep[2], sep

    # unfused qwen3.5 path
    gate_up, down, sep = modules()
    src = gathered.detach().requires_grad_()
    x = padded_index_gather(src, expand_idx)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    xs, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(x, expert_idxs, G, group_counts=counts)
    kw = dict(grouped_mm_offs=offs, ks=ks, ks_tensor=ks_t)
    gu = gate_up(xs, **kw)
    y = down(silu_mul(gu[:, :inter].contiguous(), gu[:, inter:].contiguous()), **kw)
    out = padded_index_gather(y, rsi)
    out.backward(dout)
    r5 = (out.detach(), src.grad, gate_up.main_grad.clone(), down.main_grad.clone())

    ref = run_fp32(sep, gathered, expert_idxs, expand_idx, dout)
    ref = (ref[0], ref[1], torch.cat([ref[2][0], ref[2][1]], 1), ref[2][2])

    gate_up, down, _ = modules()
    src = gathered.detach().requires_grad_()
    out = bf16_swiglu_experts(gate_up, None, down, src, expert_idxs, expand_idx, k)
    out.backward(dout)
    new = (out.detach(), src.grad, gate_up.main_grad, down.main_grad)
    for name, a, b, r in zip(("out", "dgathered", "main_grad gate_up", "main_grad down"), new, r5, ref):  # fmt: skip
        assert rel(a, r) <= 1.5 * rel(b, r), f"{name}: {rel(a, r):.3e} vs unfused {rel(b, r):.3e}"


def gptoss_experts(G, H, inter, limit, seed=0):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from pithtrain.models.gpt_oss import GptOssExperts

    cfg = GptOssConfig(hidden_size=H, intermediate_size=inter, num_local_experts=G, num_experts_per_tok=4, swiglu_limit=limit)  # fmt: skip
    torch.manual_seed(seed)
    m = GptOssExperts(cfg, G)
    with torch.no_grad():
        m.gate_up_proj.normal_(std=H**-0.5)
        m.down_proj.normal_(std=inter**-0.5)
        m.gate_up_proj_bias.normal_(std=0.5)
        m.down_proj_bias.normal_(std=0.5)
    return m.cuda().to(torch.bfloat16)


@pytest.mark.parametrize("limit", [7.0, 2.5])
def test_gptoss_vs_r5base_and_fp32(limit):
    """gpt-oss experts: interleaved gate_up rows, biases in the epilogues, the clamped SwiGLU of
    the configured swiglu_limit (a non-default limit catches a hard-coded clamp)."""
    from pithtrain.operators.bf16_experts import bf16_clamped_swiglu_experts

    G, H, inter, k = 8, 2880, 2880, 4
    expert_idxs, expand_idx, T = ep_routing(G, 4, 600, k, seed=4)
    torch.manual_seed(8)
    gathered = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(expert_idxs.shape[0], H, device="cuda", dtype=torch.bfloat16)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    names = ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias")

    # unfused: scatter + GptOssExperts.forward (torch BF16 path) + gather
    m = gptoss_experts(G, H, inter, limit)
    src = gathered.detach().requires_grad_()
    x = padded_index_gather(src, expand_idx)
    xs, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(x, expert_idxs, G, group_counts=counts)
    out = padded_index_gather(m(xs, offs, ks=ks, ks_tensor=ks_t), rsi)
    out.backward(dout)
    r5 = [out.detach(), src.grad] + [getattr(m, n).grad.float() for n in names]

    # fp32 reference
    m = gptoss_experts(G, H, inter, limit)
    ps = {n: getattr(m, n).detach().float().requires_grad_() for n in names}
    src = gathered.detach().float().requires_grad_()
    xr = src[expand_idx]
    ref_out = torch.empty(xr.shape[0], H, device="cuda")
    for e in range(G):
        sel = (expert_idxs == e).nonzero().squeeze(1)
        gu = xr[sel] @ ps["gate_up_proj"][e].T + ps["gate_up_proj_bias"][e]
        g, u = gu[:, 0::2].clamp(max=limit), gu[:, 1::2].clamp(-limit, limit)
        a = (u + 1) * g * torch.sigmoid(1.702 * g)
        ref_out = ref_out.index_put((sel,), a @ ps["down_proj"][e].T + ps["down_proj_bias"][e])
    ref_out.backward(dout.float())
    ref = [ref_out.detach(), src.grad] + [ps[n].grad for n in names]

    # fused
    m = gptoss_experts(G, H, inter, limit)
    src = gathered.detach().requires_grad_()
    out = bf16_clamped_swiglu_experts(m, src, expert_idxs, expand_idx, counts, k, 1.702, limit)
    out.backward(dout)
    new = [out.detach(), src.grad] + [getattr(m, n).grad.float() for n in names]
    for name, a, b, r in zip(("out", "dgathered") + names, new, r5, ref):
        assert rel(a, r) <= 1.5 * rel(b, r), f"{name}: {rel(a, r):.3e} vs unfused {rel(b, r):.3e}"
