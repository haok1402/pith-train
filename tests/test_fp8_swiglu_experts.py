"""
FP8SwiGLUExpertsFunc (SwiGLU fused into the FP8 quantization of its neighbours), where
fp8_fused_experts_supported():

- against the unfused FP8GroupedLinear -> silu_mul -> FP8GroupedLinear path: forward output, input
  gradient and all three weight gradients bit-identical;
- with recompute_h (down_proj's K-major FP8 input re-quantized from g and u in backward instead of
  saved) against the default: the same, bit for bit, and the saved activations smaller by that
  tensor;
- with the weight gradients accumulated into an fp32 main_grad (as wgrad_accumulation installs it)
  against the bf16 .grad path: one micro-batch's main_grad rounded to bf16 equals the bf16 .grad
  bit for bit (the same fp32 GEMM result), two micro-batches sum in fp32, and the forward and input
  gradient do not depend on where the weight gradient goes.
"""

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import fp8_fused_experts_supported
from pithtrain.operators.fp8_experts import FP8SwiGLUExpertsFunc, fp8_swiglu_experts
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import precompute_group_indices

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(), reason="needs the fused FP8 expert ops"
)

COUNTS = [[2048] * 16, [0, 128, 256, 0, 640, 128, 1024, 0]]
H, INTER = 2048, 768


@pytest.fixture(autouse=True)
def fp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)


def make(G: int, main_grad: bool = False) -> tuple[FP8GroupedLinear, ...]:
    torch.manual_seed(0)
    mods = (
        FP8GroupedLinear(G, H, INTER),
        FP8GroupedLinear(G, H, INTER),
        FP8GroupedLinear(G, INTER, H),
    )
    for m in mods:
        torch.nn.init.normal_(m.weight, std=0.02)
    mods = tuple(m.cuda().to(torch.bfloat16) for m in mods)
    if main_grad:
        for m in mods:
            m.main_grad = torch.zeros(m.weight.shape, device="cuda", dtype=torch.float32)
    return mods


def inputs(counts: list[int], n: int = 1) -> tuple:
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks_t = torch.tensor(counts, device="cuda", dtype=torch.int32)
    M = sum(counts)
    torch.manual_seed(1)
    xs = [torch.randn(M, H, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    dys = [torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 1e-2 for _ in range(n)]
    return offs, ks_t, xs, dys


def run(mods, x, dy, offs, counts, ks_t, fused: bool = True, deferred: bool = False):
    """One forward + backward; returns (output, input gradient, saved-activation bytes)."""
    FP8WeightCacheControl.step()
    x = x.detach().requires_grad_()
    gi = precompute_group_indices(offs, x.shape[0])
    saved = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t.untyped_storage().nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        if fused:
            y = fp8_swiglu_experts(*mods, x, offs, counts, ks_t, gi)
        else:
            kw = dict(grouped_mm_offs=offs, ks=counts, ks_tensor=ks_t, group_indices=gi)
            y = mods[2](silu_mul(mods[0](x, **kw), mods[1](x, **kw)), **kw)
    WeightGradStore.enabled = deferred
    y.backward(dy)
    WeightGradStore.enabled = False
    if deferred:
        WeightGradStore.flush()
        WeightGradStore.pop()
    return y.detach(), x.grad, sum(saved)


@pytest.mark.parametrize("counts", COUNTS)
def test_fused_matches_unfused(counts: list[int]) -> None:
    offs, ks_t, (x,), (dy,) = inputs(counts)
    ref, new = make(len(counts)), make(len(counts))
    y0, dx0, _ = run(ref, x, dy, offs, counts, ks_t, fused=False)
    y1, dx1, _ = run(new, x, dy, offs, counts, ks_t, fused=True)
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
    for r, n in zip(ref, new):
        assert torch.equal(r.weight.grad, n.weight.grad)


@pytest.mark.parametrize("counts", COUNTS)
def test_recompute_h_bit_identical(counts: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    offs, ks_t, (x,), (dy,) = inputs(counts)
    results = []
    for recompute in (False, True):
        monkeypatch.setattr(FP8SwiGLUExpertsFunc, "recompute_h", recompute)
        mods = make(len(counts))
        results.append((mods, *run(mods, x, dy, offs, counts, ks_t)))
    (m0, y0, dx0, s0), (m1, y1, dx1, s1) = results
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
    for a, b in zip(m0, m1):
        assert torch.equal(a.weight.grad, b.weight.grad)
    assert s0 - s1 >= sum(counts) * INTER  # the K-major FP8 h, one byte per element, plus scales


@pytest.mark.parametrize("deferred", [False, True])
def test_main_grad_matches_bf16_grad(deferred: bool) -> None:
    counts = COUNTS[1]
    offs, ks_t, xs, dys = inputs(counts, n=2)
    G = len(counts)
    ref, new = make(G), make(G, main_grad=True)
    y0, dx0, _ = run(ref, xs[0], dys[0], offs, counts, ks_t, deferred=deferred)
    y1, dx1, _ = run(new, xs[0], dys[0], offs, counts, ks_t, deferred=deferred)
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
    for r, n in zip(ref, new):
        assert n.weight.grad is None
        assert torch.equal(n.main_grad.to(torch.bfloat16), r.weight.grad)

    # A second micro-batch: main_grad is the fp32 sum of the two GEMM results.
    first = [n.main_grad.clone() for n in new]
    ref2 = make(G, main_grad=True)
    run(ref2, xs[1], dys[1], offs, counts, ks_t, deferred=deferred)
    run(new, xs[1], dys[1], offs, counts, ks_t, deferred=deferred)
    for f, r2, n in zip(first, ref2, new):
        assert torch.equal(n.main_grad, f + r2.main_grad)
