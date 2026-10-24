"""
FP8SwiGLUExpertsFunc with recompute_h (down_proj's K-major FP8 input re-quantized from g and u in
backward instead of saved) against the default: output, input gradient and the three weight
gradients bit-identical, and the saved activations smaller by that tensor. Hopper only.
"""

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.operators.fp8_experts import FP8SwiGLUExpertsFunc, fp8_swiglu_experts
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.token_scatter import precompute_group_indices

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper-only fused path")


def make(G: int, H: int, inter: int) -> tuple[FP8GroupedLinear, ...]:
    torch.manual_seed(0)
    mods = (
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, inter, H),
    )
    for m in mods:
        torch.nn.init.normal_(m.weight, std=0.02)
    return tuple(m.cuda().to(torch.bfloat16) for m in mods)


def run(x, dy, counts, recompute: bool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(FP8SwiGLUExpertsFunc, "recompute_h", recompute)
    mods = make(len(counts), x.shape[1], 768)
    FP8WeightCacheControl.step()
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks_t = torch.tensor(counts, device="cuda", dtype=torch.int32)
    x = x.detach().requires_grad_()
    gi = precompute_group_indices(offs, x.shape[0])
    saved = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t.untyped_storage().nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = fp8_swiglu_experts(*mods, x, offs, counts, ks_t, gi)
    y.backward(dy)
    return y.detach(), x.grad, [m.weight.grad for m in mods], sum(saved)


@pytest.mark.parametrize("counts", [[2048] * 16, [0, 128, 256, 0, 640, 128, 1024, 0]])
def test_recompute_h_bit_identical(counts: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    M, H = sum(counts), 2048
    torch.manual_seed(1)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 1e-2
    y0, dx0, g0, s0 = run(x, dy, counts, False, monkeypatch)
    y1, dx1, g1, s1 = run(x, dy, counts, True, monkeypatch)
    assert torch.equal(y0, y1) and torch.equal(dx0, dx1)
    for a, b in zip(g0, g1):
        assert torch.equal(a, b)
    assert s0 - s1 >= M * 768  # the K-major FP8 h, one byte per element, plus its scales
