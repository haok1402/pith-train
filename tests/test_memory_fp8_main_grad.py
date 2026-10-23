"""
FP8 expert weight gradients accumulated into an fp32 main_grad (as wgrad_accumulation installs it)
against the bf16 .grad path: one micro-batch's main_grad rounded to bf16 equals the bf16 .grad bit
for bit (both are the same fp32 GEMM result), and two micro-batches sum in fp32. The forward and
input gradient do not depend on where the weight gradient goes. Hopper only.
"""

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.operators.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.operators.grouped_linear import FP8GroupedLinear, fp8_swiglu_experts
from pithtrain.operators.linear import ARCH_MAJOR
from pithtrain.operators.token_scatter import precompute_group_indices
from pithtrain.pipeline.execution import WeightGradStore

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper-only fused path")


def make(G: int, H: int, inter: int, main_grad: bool) -> tuple[FP8GroupedLinear, ...]:
    torch.manual_seed(0)
    mods = (
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, inter, H),
    )
    for m in mods:
        torch.nn.init.normal_(m.weight, std=0.02)
    mods = tuple(m.cuda().to(torch.bfloat16) for m in mods)
    if main_grad:
        for m in mods:
            m.main_grad = torch.zeros(m.weight.shape, device="cuda", dtype=torch.float32)
    return mods


def step(mods, x, dy, counts, deferred: bool):
    FP8WeightCacheControl.step()
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks_t = torch.tensor(counts, device="cuda", dtype=torch.int32)
    x = x.detach().requires_grad_()
    gi = precompute_group_indices(offs, x.shape[0])
    y = fp8_swiglu_experts(*mods, x, offs, counts, ks_t, gi)
    WeightGradStore.enabled = deferred
    y.backward(dy)
    WeightGradStore.enabled = False
    if deferred:
        WeightGradStore.flush()
        WeightGradStore.pop()
    return y.detach(), x.grad


@pytest.mark.parametrize("deferred", [False, True])
def test_main_grad_matches_bf16_grad(deferred: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    counts = [0, 128, 256, 0, 640, 128, 1024, 0]
    G, H, inter, M = len(counts), 2048, 768, sum(counts)
    torch.manual_seed(1)
    xs = [torch.randn(M, H, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    dys = [torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 1e-2 for _ in range(2)]

    ref, new = make(G, H, inter, False), make(G, H, inter, True)
    y0, dx0 = step(ref, xs[0], dys[0], counts, deferred)
    y1, dx1 = step(new, xs[0], dys[0], counts, deferred)
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
    for r, n in zip(ref, new):
        assert n.weight.grad is None
        assert torch.equal(n.main_grad.to(torch.bfloat16), r.weight.grad)

    # A second micro-batch: main_grad is the fp32 sum of the two GEMM results.
    first = [n.main_grad.clone() for n in new]
    ref2 = make(G, H, inter, True)
    step(ref2, xs[1], dys[1], counts, deferred)
    step(new, xs[1], dys[1], counts, deferred)
    for f, r2, n in zip(first, ref2, new):
        assert torch.equal(n.main_grad, f + r2.main_grad)
