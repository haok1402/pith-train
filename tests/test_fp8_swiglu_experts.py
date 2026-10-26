"""
FP8SwiGLUExpertsFunc (SwiGLU fused into the FP8 quantization of its neighbours) against the unfused
FP8GroupedLinear -> silu_mul -> FP8GroupedLinear path: forward output, input gradient and all three
weight gradients must be bit-identical. Hopper only.
"""

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels import fp8_fused_experts_supported
from pithtrain.operators.fp8_experts import fp8_swiglu_experts
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import precompute_group_indices

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(), reason="needs the fused FP8 expert ops"
)


def make(G: int, H: int, inter: int, seed: int) -> tuple[FP8GroupedLinear, ...]:
    torch.manual_seed(seed)
    mods = (
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, inter, H),
    )
    for m in mods:
        torch.nn.init.normal_(m.weight, std=0.02)
    return tuple(m.cuda().to(torch.bfloat16) for m in mods)


def run(mods, x, dy, offs, ks, ks_t, fused: bool):
    FP8WeightCacheControl.step()
    x = x.detach().requires_grad_()
    gi = precompute_group_indices(offs, x.shape[0])
    if fused:
        y = fp8_swiglu_experts(*mods, x, offs, ks, ks_t, gi)
    else:
        kw = dict(grouped_mm_offs=offs, ks=ks, ks_tensor=ks_t, group_indices=gi)
        y = mods[2](silu_mul(mods[0](x, **kw), mods[1](x, **kw)), **kw)
    y.backward(dy)
    grads = [m.weight.grad for m in mods]
    return y.detach(), x.grad, grads


@pytest.mark.parametrize("counts", [[2048] * 16, [0, 128, 256, 0, 640, 128, 1024, 0]])
def test_fused_matches_unfused(counts: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    G, H, inter = len(counts), 2048, 768
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks, ks_t = counts, torch.tensor(counts, device="cuda", dtype=torch.int32)
    M = sum(counts)
    torch.manual_seed(1)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 1e-2
    results = []
    for fused in (False, True):
        mods = make(G, H, inter, seed=0)
        results.append(run(mods, x, dy, offs, ks, ks_t, fused))
    (y0, dx0, g0), (y1, dx1, g1) = results
    assert torch.equal(y0, y1)
    assert torch.equal(dx0, dx1)
    for a, b in zip(g0, g1):
        assert torch.equal(a, b)
