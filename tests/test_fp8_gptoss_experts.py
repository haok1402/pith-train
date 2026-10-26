"""
gpt-oss FP8 experts on Hopper: FP8ClampedSwiGLUExpertsFunc (bias + clamped SwiGLU fused into the
FP8 quantization of its neighbours) against the unfused FP8GroupedLinearFunc -> indexed_bias_add ->
clamped_swiglu -> FP8GroupedLinearFunc path of GptOssExperts. Forward output, input gradient,
both weight gradients and the down bias gradient must be bit-identical; the gate_up bias gradient
sums the same bf16 values in another fp32 order. Also checks forward_from_tokens (fused scatter +
quantization) per token against scatter_for_grouped_gemm + experts.
"""

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

import pithtrain.models.gpt_oss as gpt_oss
from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import fp8_fused_experts_supported
from pithtrain.models.gpt_oss import GptOssExperts
from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(), reason="needs the fused FP8 expert ops"
)


def make(G: int, H: int, inter: int, seed: int) -> GptOssExperts:
    cfg = GptOssConfig(hidden_size=H, intermediate_size=inter, swiglu_limit=7.0)
    torch.manual_seed(seed)
    m = GptOssExperts(cfg, G)
    for p, std in ((m.gate_up_proj, 0.05), (m.down_proj, 0.05), (m.gate_up_proj_bias, 0.5), (m.down_proj_bias, 0.5)):  # fmt: skip
        torch.nn.init.normal_(p, std=std)
    return m.cuda().to(torch.bfloat16)


def run(m: GptOssExperts, x, dy, offs, ks, ks_t, fused: bool, defer: bool, monkeypatch):
    monkeypatch.setattr(gpt_oss, "fp8_fused_experts_supported", lambda: fused)
    monkeypatch.setattr(WeightGradStore, "enabled", defer)
    WeightGradStore.clear()
    FP8WeightCacheControl.step()
    x = x.detach().requires_grad_()
    y = m(x, offs, ks=ks, ks_tensor=ks_t)
    y.backward(dy)
    WeightGradStore.clear()  # deferred weight gradients are dropped
    grads = [p.grad for p in (m.gate_up_proj, m.down_proj, m.gate_up_proj_bias, m.down_proj_bias)]
    return y.detach(), x.grad, grads


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


@pytest.mark.parametrize(
    "counts,H,inter",
    [
        ([2048] * 4, 2880, 2880),
        ([0, 128, 256, 0, 640, 128, 1024, 0], 2880, 2880),
        ([512] * 3, 1024, 768),
    ],
)
def test_fused_matches_unfused(counts, H, inter, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    At gpt-oss's 2880 widths the unfused path's k-grouped wgrad deadlocks in DeepGEMM (see
    fp8_gemm.k_grouped_safe_size), so there the reference defers and drops its weight gradients and
    the fused ones are checked against a bf16 autograd product instead.
    """
    monkeypatch.setattr(training, "fp8", True, raising=False)
    G = len(counts)
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks, ks_t = counts, torch.tensor(counts, device="cuda", dtype=torch.int32)
    M = sum(counts)
    torch.manual_seed(1)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 4
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 1e-2
    safe = H % 128 == 0 and inter % 128 == 0
    y0, dx0, g0 = run(make(G, H, inter, seed=0), x, dy, offs, ks, ks_t, False, not safe, monkeypatch)  # fmt: skip
    m1 = make(G, H, inter, seed=0)
    y1, dx1, g1 = run(m1, x, dy, offs, ks, ks_t, True, False, monkeypatch)
    assert torch.equal(y0, y1)
    assert torch.equal(dx0, dx1)
    assert torch.equal(g0[3], g1[3]), "down bias"
    torch.testing.assert_close(g1[2], g0[2], rtol=2e-2, atol=1e-2)
    if safe:
        assert torch.equal(g0[0], g1[0]), "gate_up"
        assert torch.equal(g0[1], g1[1]), "down"
        return
    # Weight gradients against bf16 autograd on the same inputs. Their FP8 operands are the
    # unfused path's bytes (dx above is bit-identical and reads the same quantized d(gate_up));
    # the GEMM layouts themselves are checked tightly in test_wgrad_layouts. Here FP8 noise
    # flips clamp decisions of the x4-scaled inputs, so the bound is loose.
    ref = make(G, H, inter, seed=0)
    monkeypatch.setattr(training, "fp8", False, raising=False)
    xr = x.detach().requires_grad_()
    ref(xr, offs, ks=ks, ks_tensor=ks_t).backward(dy)
    for name, a, b in (("gate_up", g1[0], ref.gate_up_proj.grad), ("down", g1[1], ref.down_proj.grad)):  # fmt: skip
        d = calc_diff(a, b)
        assert d < 5e-2, (name, d)


@pytest.mark.parametrize("ntok,G", [(4096, 4), (1000, 8)])
def test_forward_from_tokens(ntok: int, G: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    H = inter = 2880
    m = make(G, H, inter, seed=2)
    torch.manual_seed(3)
    tokens = torch.randn(ntok, H, device="cuda", dtype=torch.bfloat16)
    eidx = torch.randint(0, G, (ntok,), device="cuda")
    FP8WeightCacheControl.step()
    out_tok, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(tokens, eidx, G)
    ref = padded_index_gather(m(out_tok, offs, ks=ks, ks_tensor=ks_t), rsi)[:ntok]
    outs, rsi2 = m.forward_from_tokens(tokens, eidx, None, None)
    got = padded_index_gather(outs, rsi2)[:ntok]
    assert torch.equal(ref, got)


@pytest.mark.parametrize("N,K", [(2880, 1024), (1024, 2880)])
def test_wgrad_layouts(N: int, K: int) -> None:
    """fp8_padded_wgrad (operand order / padding chosen to dodge DeepGEMM's odd-tile deadlock) against
    an fp32 product: the error is FP8 quantization only (~3.8% relative for Gaussian data)."""
    from pithtrain.kernels import quantize_grouped_input
    from pithtrain.operators.fp8_experts import fp8_padded_wgrad

    torch.manual_seed(0)
    ks = [512, 1024, 256]
    G, M = len(ks), sum(ks)
    offs = torch.tensor(ks, device="cuda").cumsum(0).to(torch.int32)
    kt = torch.tensor(ks, device="cuda", dtype=torch.int32)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    a = quantize_grouped_input(dy, offs)[2:]
    b = quantize_grouped_input(x, offs)[2:]
    w = torch.nn.Parameter(torch.empty(G, N, K, device="cuda", dtype=torch.bfloat16))
    fp8_padded_wgrad(a, b, ks, kt, w)
    ref = torch.stack([dy[o - k : o].float().T @ x[o - k : o].float() for o, k in zip(offs.tolist(), ks)])  # fmt: skip
    rel = ((w.grad.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.045, rel


def test_recompute_gate_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """RECOMPUTE_GATE_UP gives the same forward output and gradients, bit for bit."""
    import pithtrain.operators.fp8_experts as fg

    monkeypatch.setattr(training, "fp8", True, raising=False)
    counts, H, inter = [0, 128, 256, 0, 640, 128, 1024, 0], 2880, 2880
    G = len(counts)
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks, ks_t = counts, torch.tensor(counts, device="cuda", dtype=torch.int32)
    torch.manual_seed(1)
    x = torch.randn(sum(counts), H, device="cuda", dtype=torch.bfloat16) * 4
    dy = torch.randn(sum(counts), H, device="cuda", dtype=torch.bfloat16) * 1e-2
    res = []
    for recompute in (False, True):
        monkeypatch.setattr(fg, "RECOMPUTE_GATE_UP", recompute)
        res.append(run(make(G, H, inter, seed=0), x, dy, offs, ks, ks_t, True, False, monkeypatch))
    (y0, dx0, g0), (y1, dx1, g1) = res
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
    for a, b in zip(g0, g1):
        assert torch.equal(a, b)
