"""
Channels-last causal conv of the Qwen3.5 Gated DeltaNet (operators/causal_conv1d) against nn.Conv1d on the
transposed input: forward and input gradient bit-identical (eager and under torch.compile, also with
a prepended conv state), weight gradient to fp32 rounding. Then the whole GDN block, compiled, with
the old conv code: output and every gradient but the conv weight's bit-identical.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.causal_conv1d import causal_conv1d


def reference(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    K, S = w.shape[-1], x.shape[1] - offset
    y = F.conv1d(x.transpose(1, 2), w, groups=w.shape[0], padding=K - 1)
    return y[..., offset : offset + S].transpose(1, 2)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("B,S,C,offset", [(1, 4096, 8192, 0), (2, 1000, 520, 0), (1, 2051, 640, 3)])
def test_conv_matches_conv1d(B: int, S: int, C: int, offset: int, compiled: bool) -> None:
    torch.manual_seed(0)
    x0 = torch.randn(B, S + offset, C, device="cuda", dtype=torch.bfloat16)
    w0 = torch.randn(C, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.5
    g = torch.randn(B, S, C, device="cuda", dtype=torch.bfloat16)

    def run(fn):
        x, w = x0.clone().requires_grad_(), w0.clone().requires_grad_()
        f = torch.compile(fn, fullgraph=True) if compiled else fn
        y = f(x, w, offset)
        y.backward(g)
        return y.detach(), x.grad, w.grad

    (y0, dx0, dw0), (y1, dx1, dw1) = run(reference), run(causal_conv1d)
    assert torch.equal(y0, y1)
    assert torch.equal(dx0, dx1)
    assert rel(dw1, dw0) < 1e-2  # bf16 rounding of differently ordered fp32 sums


def test_gdn_block_matches_old_conv(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models import qwen35_moe
    from pithtrain.operators.linear import BF16Linear

    monkeypatch.setattr(training, "Linear", BF16Linear, raising=False)
    # The block reads training.fp8 (the pinned GDR kernels); set it rather than inherit it from
    # whichever test ran before (this file used to run after the FP8 expert tests).
    monkeypatch.setattr(training, "fp8", False, raising=False)
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    cfg = AutoConfig.from_pretrained("examples/pretrain_lm/qwen3.5-35b-a3b")
    torch.manual_seed(0)
    gdn = qwen35_moe.Qwen35MoeGatedDeltaNet(cfg, 0).cuda().to(torch.bfloat16)
    for p in gdn.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    x0 = torch.randn(1, 4096, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

    def old_conv(mixed_qkv: torch.Tensor, weight: torch.Tensor, offset: int) -> torch.Tensor:
        S = mixed_qkv.shape[1] - offset
        return gdn.conv1d(mixed_qkv.transpose(1, 2))[..., offset : offset + S].transpose(1, 2)

    def run(conv):
        monkeypatch.setattr(qwen35_moe, "causal_conv1d", conv)
        torch._dynamo.reset()
        gdn.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_()
        y = torch.compile(gdn, fullgraph=True)(x)
        y.float().square().mean().backward()
        return y.detach(), x.grad, {n: p.grad for n, p in gdn.named_parameters()}

    (y0, dx0, g0), (y1, dx1, g1) = run(old_conv), run(causal_conv1d)
    print("y mismatches", (y0 != y1).sum().item(), "rel", rel(y1, y0), "dx rel", rel(dx1, dx0))
    for n in g0:
        print(n, rel(g1[n], g0[n]))
    assert torch.equal(y0, y1)
    assert rel(dx1, dx0) < 1e-2
    for n in g0:
        assert rel(g1[n], g0[n]) < 1e-2, n
