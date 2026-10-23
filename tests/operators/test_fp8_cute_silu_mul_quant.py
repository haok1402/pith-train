"""Bit-exactness of the fused CuTe-DSL SwiGLU + FP8 rowwise/K-major quantization kernels.

Reference: the unfused Triton sequence, ``fp8cast_rowwise_kmajor(silu_mul(g, u), offs)`` for the
forward and ``fp8cast_rowwise_kmajor`` of the BF16 ``silu_mul`` gradients for the backward.
"""

import pytest
import torch
import triton
import triton.language as tl

from pithtrain.operators import deepgemm_quantize
from pithtrain.operators.fp8_cute.scale_layout import is_mn_major
from pithtrain.operators.silu_mul import silu_mul

# Compare against the Triton kernel itself, not a dispatcher that may route to a CuTe port.
fp8cast_rowwise_kmajor = getattr(
    deepgemm_quantize, "_fp8cast_rowwise_kmajor_triton", deepgemm_quantize.fp8cast_rowwise_kmajor
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper (SM90) only",
)


def _offs(sizes: list[int]) -> torch.Tensor:
    return torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)


def _inputs(M: int, N: int, kind: str, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn(M, N, device="cuda", generator=gen) for _ in range(3)]
    if kind == "outliers":
        for x in xs:
            mask = torch.rand(M, N, device="cuda", generator=gen) < 1e-3
            x[mask] *= 1e4
            x[torch.rand(M, N, device="cuda", generator=gen) < 0.05] = 0.0
            x[:128, :64] = 0.0  # whole zero blocks: amax clamps to 1e-4
            x[128:256] *= 1e-30  # subnormal-ish: tiny products and flushed FP8
    elif kind == "wide":
        for x in xs:
            x.mul_(torch.exp2(torch.randint(-20, 20, (M, 1), device="cuda", generator=gen)))
    return [x.to(torch.bfloat16) for x in xs]


def _assert_quant_equal(got: tuple, ref: tuple) -> None:
    names = ("tok_fp8", "tok_scale", "kmajor_fp8", "kmajor_scale")
    for name, a, b in zip(names, got, ref):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype == torch.float8_e4m3fn:
            assert a.stride() == b.stride(), name
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        else:  # scales: same values, in the MN-major layout DeepGEMM reads as is
            assert is_mn_major(a), (name, a.stride())
        if not torch.equal(a, b):
            bad = (a != b).nonzero()
            raise AssertionError(f"{name}: {bad.shape[0]} mismatches, first at {bad[:4].tolist()}")


def _ref_bwd(dh, g, u):
    g = g.detach().clone().requires_grad_()
    u = u.detach().clone().requires_grad_()
    silu_mul(g, u).backward(dh)
    return g.grad, u.grad


GROUPS = {
    "balanced16x2048": [2048] * 16,
    "imbalanced": [128, 0, 384, 1024, 0, 0, 256, 128, 2048, 0, 640],
    "single": [256],
    "empty_first_last": [0, 512, 128, 0],
    "many_small": [128] * 40 + [0] * 8,
    "experts128": [0, 256] * 32 + [128] * 64,  # 128 groups: multi-word ballot search
    "over_cap": [0] * 1100 + [256],  # > 1024 groups: unfused fallback
}

CASES = [
    ("balanced16x2048", 768, "normal"),
    ("imbalanced", 768, "normal"),
    ("imbalanced", 768, "outliers"),
    ("imbalanced", 768, "wide"),
    ("single", 768, "normal"),
    ("empty_first_last", 2048, "normal"),
    ("many_small", 768, "outliers"),
    ("imbalanced", 200, "outliers"),  # ragged N, partial last column tile
    ("single", 72, "normal"),  # N < 128
    ("empty_first_last", 1416, "wide"),
    ("single", 100, "normal"),  # N % 8 != 0 -> unfused fallback
    ("experts128", 768, "normal"),
    ("over_cap", 768, "normal"),
]


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_silu_mul_quant_fwd(groups: str, N: int, kind: str) -> None:
    from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_quant_fwd

    offs = _offs(GROUPS[groups])
    M = int(sum(GROUPS[groups]))
    _, g, u = _inputs(M, N, kind, seed=N + M)
    ref = fp8cast_rowwise_kmajor(silu_mul(g, u), offs)
    got = silu_mul_quant_fwd(g, u, offs)
    torch.cuda.synchronize()
    _assert_quant_equal(got, ref)


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_silu_mul_bwd_quant(groups: str, N: int, kind: str) -> None:
    from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_bwd_quant

    offs = _offs(GROUPS[groups])
    M = int(sum(GROUPS[groups]))
    dh, g, u = _inputs(M, N, kind, seed=3 * N + M)
    dg, du = _ref_bwd(dh, g, u)
    ref_g = fp8cast_rowwise_kmajor(dg, offs)
    ref_u = fp8cast_rowwise_kmajor(du, offs)
    got_g, got_u = silu_mul_bwd_quant(dh, g, u, offs)
    torch.cuda.synchronize()
    _assert_quant_equal(got_g, ref_g)
    _assert_quant_equal(got_u, ref_u)


def test_exhaustive_bf16_gate() -> None:
    """Every finite BF16 gate value against several up / dh values, fwd and bwd."""
    from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_bwd_quant, silu_mul_quant_fwd

    bits = torch.arange(0, 1 << 16, device="cuda", dtype=torch.int32).to(torch.int16)
    g_all = bits.view(torch.bfloat16)
    g_all = g_all[torch.isfinite(g_all)]
    N = 256
    M = (g_all.numel() + N - 1) // N
    M = (M + 127) // 128 * 128
    g = torch.zeros(M * N, device="cuda", dtype=torch.bfloat16)
    g[: g_all.numel()] = g_all
    g = g.view(M, N)
    offs = _offs([M])
    gen = torch.Generator(device="cuda").manual_seed(0)
    for scale in (1.0, 1e-3, 1e3):
        u = (torch.randn(M, N, device="cuda", generator=gen) * scale).to(torch.bfloat16)
        dh = torch.randn(M, N, device="cuda", generator=gen).to(torch.bfloat16)
        _assert_quant_equal(
            silu_mul_quant_fwd(g, u, offs), fp8cast_rowwise_kmajor(silu_mul(g, u), offs)
        )
        dg, du = _ref_bwd(dh, g, u)
        got_g, got_u = silu_mul_bwd_quant(dh, g, u, offs)
        _assert_quant_equal(got_g, fp8cast_rowwise_kmajor(dg, offs))
        _assert_quant_equal(got_u, fp8cast_rowwise_kmajor(du, offs))


def test_empty_input() -> None:
    """M == 0 (no tokens routed to this rank) returns correctly shaped empty outputs."""
    from pithtrain.operators.fp8_cute.silu_mul_quant import silu_mul_bwd_quant, silu_mul_quant_fwd

    g = torch.empty(0, 768, device="cuda", dtype=torch.bfloat16)
    offs = torch.zeros(4, device="cuda", dtype=torch.int32)
    out = silu_mul_quant_fwd(g, g, offs)
    assert out[0].shape == (0, 768) and out[3].shape == (768, 0)
    dq, uq = silu_mul_bwd_quant(g, g, g, offs)
    assert dq[2].numel() == 0 and uq[1].shape == (0, 6)


@triton.jit
def _asm_kernel(x_ptr, o_ptr, n, ASM: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    y = tl.inline_asm_elementwise(ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(o_ptr + offs, y, mask=offs < n)


def test_sigmoid_sequence_exhaustive() -> None:
    """The kernels' shortened sigmoid PTX equals Triton's ``1 / (1 + exp(-x))`` lowering
    (sub, mul by log2e, ex2.approx, add, div.full) bit for bit on every BF16 input."""
    from pithtrain.operators.fp8_cute.silu_mul_quant import _SIGMOID_PTX

    ref_asm = (
        "{\n.reg .f32 t, one;\nsub.f32 t, 0f00000000, $1;\nmul.f32 t, t, 0f3FB8AA3B;\n"
        "ex2.approx.f32 t, t;\nadd.f32 t, t, 0f3F800000;\nmov.f32 one, 0f3F800000;\n"
        "div.full.f32 $0, one, t;\n}"
    )
    new_asm = "{\n.reg .f32 t, s;\n" + _SIGMOID_PTX.replace("$G", "$1") + "mov.b32 $0, s;\n}"
    x = torch.arange(0, 1 << 16, device="cuda", dtype=torch.int32).to(torch.int16)
    x = x.view(torch.bfloat16).float()
    outs = []
    for asm in (ref_asm, new_asm):
        o = torch.empty_like(x)
        _asm_kernel[(x.numel() // 1024,)](x, o, x.numel(), ASM=asm, BLOCK=1024)
        outs.append(o.view(torch.int32))
    finite = ~torch.isnan(x)
    assert torch.equal(outs[0][finite], outs[1][finite])
