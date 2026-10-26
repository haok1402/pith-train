"""
Verifier of ``fp8cast_rowwise_blockwise_transpose`` against a pure-PyTorch implementation of the
E8M0 FP8 recipe.

``(x_fp8, x_scale, xt_fp8, xt_scale) = fp8cast_rowwise_blockwise_transpose(x)`` is the rowwise
1x128 quantization of x ((M, K) e4m3, scale (M, ceil(K/128))) plus the 128x128 blockwise
quantization of x.T ((K, M) e4m3, scale (ceil(K/128), ceil(M/128))). FP8 bytes and scale values
must be identical to the reference; the rowwise scale's strides are unspecified by contract, so
scales are compared by logical value.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import fp8cast_rowwise_blockwise_transpose

BLOCK = 128


# ---------------------------------------------------------------------------
# PyTorch reference of the recipe
# ---------------------------------------------------------------------------


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(max(amax, 1e-4) / 448)), exact (frexp, no float log2)."""
    v = amax.float().clamp(min=1e-4) / 448.0
    m, e = torch.frexp(v)  # v = m * 2**e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def to_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def ref_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x128 rowwise quantization of a 2D tensor: (q (M, K), scale (M, ceil(K/128)))."""
    M, K = x.shape
    kb = (K + BLOCK - 1) // BLOCK
    xb = F.pad(x.float(), (0, kb * BLOCK - K)).view(M, kb, BLOCK)
    s = e8m0_scale(xb.abs().amax(dim=-1)) if M * kb else torch.empty(M, kb, device=x.device)
    q = to_e4m3(xb / s[..., None]).view(M, kb * BLOCK)[:, :K]
    return q.contiguous(), s


def ref_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 blockwise quantization of a 2D tensor: (q (N, K), scale (ceil(N/128), ceil(K/128)))."""
    N, K = x.shape
    nb, kb = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
    xb = F.pad(x.float(), (0, kb * BLOCK - K, 0, nb * BLOCK - N)).view(nb, BLOCK, kb, BLOCK)
    if nb * kb:
        s = e8m0_scale(xb.abs().amax(dim=(1, 3)))
    else:
        s = torch.empty(nb, kb, device=x.device)
    q = to_e4m3(xb / s[:, None, :, None]).view(nb * BLOCK, kb * BLOCK)[:N, :K]
    return q.contiguous(), s


def assert_fp8_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float8_e4m3fn, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    a, b = out.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(a, b), f"{name}: {(a != b).sum().item()} mismatching bytes"


def assert_scale_equal(name: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.dtype == torch.float32, (name, out.dtype)
    assert out.shape == ref.shape, (name, out.shape, ref.shape)
    assert torch.equal(out, ref), f"{name}: {(out != ref).sum().item()} mismatching scales"


def check(x: torch.Tensor) -> None:
    x_fp8, x_scale, xt_fp8, xt_scale = fp8cast_rowwise_blockwise_transpose(x)
    q, s = ref_rowwise(x)
    qt, st = ref_blockwise(x.T)
    assert_fp8_equal("x_fp8", x_fp8, q)
    assert_scale_equal("x_scale", x_scale, s)
    assert_fp8_equal("xt_fp8", xt_fp8, qt)
    assert_scale_equal("xt_scale", xt_scale, st)
    assert xt_scale.is_contiguous(), xt_scale.stride()  # 128x128 scales are row-major


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# (M, K): the model hidden / intermediate / dense sizes plus tile edges and odd sizes.
SHAPES = [
    (4096, 2048),
    (4096, 4096),
    (4096, 1408),
    (4096, 768),
    (4096, 512),
    (4096, 2880),
    (1000, 10944),
    (4100, 576),
    (128, 128),
    (256, 128),
    (128, 256),
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


def with_outliers(x: torch.Tensor) -> torch.Tensor:
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
    check(with_outliers(torch.randn(M, K, device="cuda", dtype=dtype)))


@pytest.mark.parametrize("M,K", [(4096, 2048), (4100, 576), (7, 300)])
def test_special_values(M: int, K: int) -> None:
    torch.manual_seed(0)
    bf16 = torch.bfloat16
    check(torch.zeros(M, K, device="cuda", dtype=bf16))
    # Tiny values: amax clamps to 1e-4 and the products underflow e4m3.
    check(torch.randn(M, K, device="cuda", dtype=bf16) * 1e-30)
    # Huge values: large-exponent scales, |q| up to 448 (saturation boundary).
    check(torch.randn(M, K, device="cuda", dtype=bf16) * 1e30)
    # Exact powers of two: amax = 448 * 2**k hits the no-round-up branch of the E8M0 ceil.
    p2 = torch.exp2(torch.randint(-20, 20, (M, K), device="cuda").float()).to(bf16)
    check(p2 * 448.0)
    check(-p2)
    # All negative.
    check(-torch.rand(M, K, device="cuda", dtype=bf16) * 3.0)


def test_strided_rows() -> None:
    """A row-strided view (a slice of a wider buffer)."""
    torch.manual_seed(1)
    base = torch.randn(1000, 2304, device="cuda", dtype=torch.bfloat16)
    check(base[:, :2048])
    check(base[:, 8:584])
    check(base[:, 1:300])


def test_transposed_input() -> None:
    """A non-unit inner stride."""
    torch.manual_seed(2)
    check(torch.randn(576, 1000, device="cuda", dtype=torch.bfloat16).T)


def test_degenerate_shapes() -> None:
    torch.manual_seed(3)
    check(torch.randn(1000, 1, device="cuda", dtype=torch.bfloat16))
    check(torch.randn(1, 1000, device="cuda", dtype=torch.bfloat16).T)
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    out = fp8cast_rowwise_blockwise_transpose(x)
    assert [tuple(t.shape) for t in out] == [(0, 256), (0, 2), (256, 0), (2, 0)]
