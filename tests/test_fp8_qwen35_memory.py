"""
Round-4 memory changes of the Qwen3.5 path, against the code they replace.

- Gated DeltaNet with grouped value attention: gated_delta_rule on unrepeated q / k against
  repeat_interleave'd q / k. The autotuned custom op repeats them itself and the pinned path reads
  them per group with the repeat's backward order (tests/test_gated_delta_rule_pinned.py), so both
  match bit for bit. (FLA's native GVA backward rounds differently; round 4 kept it out.)
- The GDN conv and FP8 input projections recomputed in backward (selective checkpoint).
"""

import pytest
import torch

from pithtrain.operators.gated_delta_rule import gated_delta_rule


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize("S", [4096, 1000])
def test_gva_matches_repeat(S: int) -> None:
    torch.manual_seed(0)
    B, H, HV, K, V = 1, 16, 32, 128, 128
    dev, bf = "cuda", torch.bfloat16
    q0 = torch.nn.functional.normalize(torch.randn(B, S, H, K, device=dev), dim=-1).to(bf)
    k0 = torch.nn.functional.normalize(torch.randn(B, S, H, K, device=dev), dim=-1).to(bf)
    v0 = torch.randn(B, S, HV, V, device=dev, dtype=bf)
    g0 = -torch.rand(B, S, HV, device=dev) * 0.1
    b0 = torch.rand(B, S, HV, device=dev, dtype=bf)
    do = torch.randn(B, S, HV, V, device=dev, dtype=bf)

    def run(repeat: bool):
        q, k, v, g, b = (t.clone().requires_grad_() for t in (q0, k0, v0, g0, b0))
        qq, kk = (q.repeat_interleave(2, 2), k.repeat_interleave(2, 2)) if repeat else (q, k)
        o = gated_delta_rule(qq, kk, v, g, b)
        o.backward(do)
        return o.detach(), q.grad, k.grad, v.grad, g.grad, b.grad

    ref, new = run(True), run(False)
    for a, b in zip(new, ref):
        assert torch.equal(a, b)


@pytest.mark.parametrize("fp8", [False, True])
def test_gdn_selective_recompute(fp8: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The GDN block, compiled, with the conv (and under FP8 the in_proj_qkv / in_proj_z GEMMs)
    recomputed in backward by selective checkpointing, against saving them: output bit-identical,
    gradients to bf16 rounding, and less memory held between forward and backward.
    """
    from transformers import AutoConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models import qwen35_moe
    from pithtrain.operators.linear import BF16Linear, FP8Linear

    monkeypatch.setattr(training, "Linear", FP8Linear if fp8 else BF16Linear, raising=False)
    monkeypatch.setattr(training, "fp8", fp8, raising=False)
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    cfg = AutoConfig.from_pretrained("examples/pretrain_lm/qwen3.5-35b-a3b")
    torch.manual_seed(0)
    gdn = qwen35_moe.Qwen35MoeGatedDeltaNet(cfg, 0).cuda().to(torch.bfloat16)
    for p in gdn.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    x0 = torch.randn(1, 4096, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    def run(ckpt):
        if ckpt is not None:
            monkeypatch.setattr(qwen35_moe, "checkpoint", ckpt)
        torch._dynamo.reset()
        f = torch.compile(gdn, fullgraph=True)
        gdn.zero_grad(set_to_none=True)
        f(x0.clone().requires_grad_()).float().sum().backward()  # warm up / compile
        gdn.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_()
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        y = f(x)
        held = torch.cuda.memory_allocated() - m0
        y.float().square().mean().backward()
        return y.detach(), x.grad, {n: p.grad for n, p in gdn.named_parameters()}, held

    nockpt = lambda fn, *args, **kwargs: fn(*args)  # noqa: E731
    (y0, dx0, g0, h0), (y1, dx1, g1, h1) = run(nockpt), run(torch.utils.checkpoint.checkpoint)
    print("held MiB", h0 / 2**20, h1 / 2**20)
    assert torch.equal(y0, y1)
    assert rel(dx1, dx0) < 1e-2
    for n in g0:
        assert rel(g1[n], g0[n]) < 1e-2, n
    assert h1 < h0


def test_attention_selective_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gated attention under FP8, compiled: q/k/v projections recomputed in backward."""
    from transformers import AutoConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models import qwen35_moe
    from pithtrain.operators.linear import FP8Linear

    monkeypatch.setattr(training, "Linear", FP8Linear, raising=False)
    monkeypatch.setattr(training, "fp8", True, raising=False)
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    cfg = AutoConfig.from_pretrained("examples/pretrain_lm/qwen3.5-35b-a3b")
    torch.manual_seed(0)
    attn = qwen35_moe.Qwen35MoeAttention(cfg).cuda().to(torch.bfloat16)
    for p in attn.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    S = 4096
    rope = qwen35_moe.Qwen35MoeRotaryEmbedding(cfg).cuda()
    cos, sin = rope(S)
    posemb = (cos.unsqueeze(0), sin.unsqueeze(0))
    x0 = torch.randn(1, S, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    def run(ckpt):
        monkeypatch.setattr(qwen35_moe, "checkpoint", ckpt)
        torch._dynamo.reset()
        f = torch.compile(attn, fullgraph=True)
        attn.zero_grad(set_to_none=True)
        f(x0.clone().requires_grad_(), posemb).float().sum().backward()
        attn.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_()
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        y = f(x, posemb)
        held = torch.cuda.memory_allocated() - m0
        y.float().square().mean().backward()
        return y.detach(), x.grad, {n: p.grad for n, p in attn.named_parameters()}, held

    nockpt = lambda fn, *args, **kwargs: fn(*args)  # noqa: E731
    (y0, dx0, g0, h0), (y1, dx1, g1, h1) = run(nockpt), run(torch.utils.checkpoint.checkpoint)
    print("attn held MiB", h0 / 2**20, h1 / 2**20)
    assert torch.equal(y0, y1)
    assert rel(dx1, dx0) < 1e-2
    for n in g0:
        assert rel(g1[n], g0[n]) < 1e-2, n
    assert h1 < h0
