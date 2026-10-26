"""
Channels-last causal conv of the Qwen3.5 Gated DeltaNet (operators/causal_conv1d) against nn.Conv1d on the
transposed input: forward and input gradient bit-identical (eager and under torch.compile, also with
a prepended conv state), weight gradient to fp32 rounding.
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
