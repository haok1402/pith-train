"""
Qwen3.5 FP8 routed experts (fused gate_up weight) through FP8SwiGLUGateUpExpertsFunc against the
unfused FP8GroupedLinear(gate_up) -> slice -> silu_mul -> FP8GroupedLinear(down) path: kernels,
forward output, input gradient and both weight gradients bit-identical. Plus the fused stage-3
entry (forward_from_tokens) against the three-step path. Hopper only.
"""

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.kernels.fp8_rowwise_kmajor import fp8cast_rowwise_kmajor
from pithtrain.kernels.fp8_silu_mul_quant import silu_mul_bwd_quant_gu, silu_mul_quant_gu_fwd
from pithtrain.kernels.silu_mul import silu_mul_bwd
from pithtrain.operators.fp8_experts import fp8_scatter_quant, fp8_swiglu_gate_up_experts
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper-only fused path")

COUNTS = [[1024] * 16, [0, 128, 256, 0, 640, 128, 1024, 0], [128]]


def _equal(a: tuple, b: tuple) -> None:
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.stride() == y.stride()
        if x.dtype == torch.float8_e4m3fn:
            x, y = x.view(torch.uint8), y.view(torch.uint8)
        assert torch.equal(x, y)


def _offs(counts: list[int]) -> torch.Tensor:
    return torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)


@pytest.mark.parametrize("counts", COUNTS)
@pytest.mark.parametrize("N", [512, 768, 520])
def test_kernels_match_unfused(counts: list[int], N: int) -> None:
    torch.manual_seed(0)
    M, offs = sum(counts), _offs(counts)
    gu = torch.randn(M, 2 * N, device="cuda", dtype=torch.bfloat16) * 3
    gu[:5, :7] = 0
    gu[7, 9] = 300.0
    dh = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    _equal(silu_mul_quant_gu_fwd(gu, offs), fp8cast_rowwise_kmajor(silu_mul(g, u), offs))
    dg, du = silu_mul_bwd(dh, g, u)
    _equal(
        silu_mul_bwd_quant_gu(dh, gu, offs), fp8cast_rowwise_kmajor(torch.cat([dg, du], 1), offs)
    )


def make(G: int, H: int, inter: int) -> tuple[FP8GroupedLinear, FP8GroupedLinear]:
    torch.manual_seed(0)
    mods = (FP8GroupedLinear(G, H, 2 * inter), FP8GroupedLinear(G, inter, H))
    for m in mods:
        torch.nn.init.normal_(m.weight, std=0.02)
    return tuple(m.cuda().to(torch.bfloat16) for m in mods)


def experts(mods, x, offs, ks, ks_t, fused: bool, xq=None) -> torch.Tensor:
    gi = precompute_group_indices(offs, x.shape[0])
    if fused:
        return fp8_swiglu_gate_up_experts(*mods, x, offs, ks, ks_t, gi, xq)
    kw = dict(grouped_mm_offs=offs, ks=ks, ks_tensor=ks_t, group_indices=gi)
    inter = mods[1].in_features
    gu = mods[0](x, **kw)
    return mods[1](silu_mul(gu[:, :inter].contiguous(), gu[:, inter:].contiguous()), **kw)


@pytest.mark.parametrize("counts", COUNTS)
def test_fused_matches_unfused(counts: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    G, H, inter = len(counts), 2048, 512
    offs, ks, ks_t = _offs(counts), counts, torch.tensor(counts, device="cuda", dtype=torch.int32)
    torch.manual_seed(1)
    x0 = torch.randn(sum(counts), H, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(sum(counts), H, device="cuda", dtype=torch.bfloat16) * 1e-2
    results = []
    for fused in (False, True):
        mods = make(G, H, inter)
        FP8WeightCacheControl.step()
        x = x0.detach().requires_grad_()
        y = experts(mods, x, offs, ks, ks_t, fused)
        y.backward(dy)
        results.append((y.detach(), x.grad, [m.weight.grad for m in mods]))
    (y0, dx0, g0), (y1, dx1, g1) = results
    assert torch.equal(y0, y1)
    assert torch.equal(dx0, dx1)
    for a, b in zip(g0, g1):
        assert torch.equal(a, b)


@pytest.mark.parametrize("expansion", [1, 4])
def test_fused_stage3_forward(expansion: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows of the fused stage 3 equal the three-step path's (gradients are covered by
    test_fp8_scatter_quant_experts: their 128-row blocks follow token order, not atomic order)."""
    monkeypatch.setattr(training, "fp8", True, raising=False)
    G, H, inter, m = 16, 2048, 512, 16384
    torch.manual_seed(1)
    gathered = torch.randn(m // expansion, H, device="cuda", dtype=torch.bfloat16)
    expand_idx = torch.randint(0, m // expansion, (m,), device="cuda")
    probs = torch.rand(G, device="cuda") ** 4
    probs[3] = 0
    expert_idxs = torch.multinomial(probs, m, replacement=True)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    outs = []
    for fused in (False, True):
        mods = make(G, H, inter)
        FP8WeightCacheControl.step()
        if fused:
            x, rsi, offs, ks, ks_t, xq = fp8_scatter_quant(gathered, expert_idxs, G, expand_idx, counts)  # fmt: skip
        else:
            expanded = padded_index_gather(gathered, expand_idx)
            x, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(expanded, expert_idxs, G, group_counts=counts)  # fmt: skip
            xq = None
        with torch.no_grad():
            outs.append(padded_index_gather(experts(mods, x, offs, ks, ks_t, True, xq), rsi))
    assert torch.equal(outs[0], outs[1])
