"""
Bit-exactness tests for the CuTe-DSL ``fp8cast_rowwise_blockwise_transpose``.

The reference is the Triton kernel in ``pithtrain.operators.deepgemm_quantize``: FP8 outputs are
compared as raw bytes and scales with ``torch.equal``, along with shapes, dtypes and strides
(``tok_scale`` is MN-major, see ``fp8_cute/scale_layout.py``).
"""

import pytest
import torch

from pithtrain.operators.deepgemm_quantize import (
    _fp8cast_rowwise_blockwise_transpose_triton as triton_fn,
)
from pithtrain.operators.fp8_cute.rowwise_blockwise_transpose import (
    fp8cast_rowwise_blockwise_transpose as cute_fn,
)
from pithtrain.operators.fp8_cute.scale_layout import is_mn_major

SHAPES = [
    (4096, 2048),
    (4096, 4096),
    (4100, 576),
    (1000, 10944),
    (128, 128),
    (300, 384),
    (1, 128),
    (1, 1),
    (7, 300),
    (64, 64),
    (130, 129),
    (33, 4097),
    (4098, 2050),
    (256, 7168),
]


def _assert_identical(x: torch.Tensor) -> None:
    ref = triton_fn(x)
    out = cute_fn(x)
    for name, a, b in zip(["tok_fp8", "tok_scale", "blk_t_fp8", "blk_t_scale"], ref, out):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert a.dtype == b.dtype, (name, a.dtype, b.dtype)
        if name == "tok_scale":  # MN-major, the layout DeepGEMM's SM90 GEMMs read as is
            assert is_mn_major(b), (name, b.stride())
        else:
            assert a.stride() == b.stride(), (name, a.stride(), b.stride())
        if a.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b), f"{name}: {(a != b).sum().item()} mismatching elements"


def _with_outliers(x: torch.Tensor) -> torch.Tensor:
    flat = x.view(-1)
    n = flat.numel()
    flat[:: max(1, n // 97)] *= 1000.0
    flat[5 :: max(1, n // 61)] = 0.0
    flat[3 :: max(1, n // 13)] *= -1e-3
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M,K", SHAPES)
def test_random_with_outliers(M: int, K: int, dtype: torch.dtype) -> None:
    torch.manual_seed(M * 7919 + K)
    x = _with_outliers(torch.randn(M, K, device="cuda", dtype=dtype))
    _assert_identical(x)


@pytest.mark.parametrize("M,K", [(4096, 2048), (4100, 576), (7, 300)])
def test_special_values(M: int, K: int) -> None:
    _assert_identical(torch.zeros(M, K, device="cuda", dtype=torch.bfloat16))
    # Tiny values: amax clamps to 1e-4, products underflow fp8.
    _assert_identical(torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 1e-30)
    # Huge values: saturation and large-exponent scales.
    _assert_identical(torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 1e30)
    # Exact powers of two hit the "mantissa == 0" branch of the E8M0 ceil.
    p2 = torch.exp2(torch.randint(-20, 20, (M, K), device="cuda").float()).bfloat16()
    _assert_identical(p2 * 448.0)
    _assert_identical(-p2)


def test_strided_rows() -> None:
    """A row-strided view (e.g. a slice of a wider buffer) is handled in place."""
    base = torch.randn(1000, 2304, device="cuda", dtype=torch.bfloat16)
    _assert_identical(base[:, :2048])
    _assert_identical(base[:, 8:584])
    _assert_identical(base[:, 1:300])


def test_transposed_input() -> None:
    """A non-unit inner stride falls back to a contiguous copy, same result."""
    x = torch.randn(576, 1000, device="cuda", dtype=torch.bfloat16).T
    _assert_identical(x)


def test_degenerate_shapes() -> None:
    _assert_identical(torch.randn(1000, 1, device="cuda", dtype=torch.bfloat16))
    _assert_identical(torch.randn(1, 1000, device="cuda", dtype=torch.bfloat16).T)
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    out = cute_fn(x)
    assert [t.shape for t in out] == [(0, 256), (0, 2), (256, 0), (2, 0)]
