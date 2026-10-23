"""Bit-exactness of the CuTe-DSL ``fp8cast_rowwise_kmajor`` against the Triton reference."""

import pytest
import torch

from pithtrain.operators.deepgemm_quantize import _fp8cast_rowwise_kmajor_triton as triton_impl
from pithtrain.operators.fp8_cute.rowwise_kmajor import fp8cast_rowwise_kmajor as cute_impl
from pithtrain.operators.fp8_cute.scale_layout import is_mn_major

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper (SM90) K-major path",
)


def _offs(sizes: list[int]) -> torch.Tensor:
    return torch.tensor(sizes, dtype=torch.int64).cumsum(0).to(torch.int32).cuda()


def _input(M: int, N: int, kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, N, device="cuda", generator=g)
    if kind == "scaled":
        # per-row and per-column dynamic range spread over many binades
        rs = torch.exp2(torch.randint(-20, 20, (M, 1), device="cuda", generator=g).float())
        cs = torch.exp2(torch.randint(-6, 6, (1, N), device="cuda", generator=g).float())
        x = x * rs * cs
    elif kind == "outliers":
        mask = torch.rand(M, N, device="cuda", generator=g) < 1e-3
        x = torch.where(mask, x * 1e4, x)
        x[:, : min(N, 3)] = 0.0  # all-zero columns
        x[5::97] = 0.0  # all-zero rows
        x[7::131] *= 1e-30  # tiny rows (amax clamp)
        x[11::127] *= 1e-39  # bf16 subnormals
    return x.to(torch.bfloat16)


def _check(x: torch.Tensor, offs: torch.Tensor) -> None:
    ref = triton_impl(x, offs)
    out = cute_impl(x, offs)
    names = ("tok_fp8", "tok_scale", "kmajor_fp8", "kmajor_scale")
    for name, a, b in zip(names, out, ref):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        if a.dtype == torch.float8_e4m3fn:
            assert a.stride() == b.stride(), name
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        else:  # scales: same values, MN-major layout DeepGEMM reads without a transpose
            assert is_mn_major(a), (name, a.stride())
        assert torch.equal(a, b), f"{name} mismatch: {(a != b).sum().item()} elements"


BALANCED = [2048] * 16
IMBALANCED = [0, 128, 384, 0, 0, 1024, 256, 128, 0, 2944, 128, 0, 640, 384, 128, 0]
SHAPES = [
    (BALANCED, 2048),
    (BALANCED, 768),
    (IMBALANCED, 2048),
    (IMBALANCED, 768),
    ([256, 0, 512, 256], 200),  # ragged N (not a multiple of 128)
    ([128, 128, 0, 768], 136),
    ([1024], 4096),
    ([128], 128),
    ([0, 0, 128], 8),
    ([512, 512], 2880),
]


@pytest.mark.parametrize("kind", ["randn", "scaled", "outliers"])
@pytest.mark.parametrize("sizes,N", SHAPES)
def test_bit_exact(sizes: list[int], N: int, kind: str) -> None:
    M = sum(sizes)
    _check(_input(M, N, kind), _offs(sizes))


def test_many_groups() -> None:
    g = torch.Generator().manual_seed(1)
    sizes = (torch.randint(0, 6, (100,), generator=g) * 128).tolist()
    M = sum(sizes)
    _check(_input(M, 512, "scaled"), _offs(sizes))


def test_fallbacks() -> None:
    # N % 8 != 0 and a non-contiguous input both go through the Triton kernel.
    _check(_input(384, 100, "randn"), _offs([128, 256]))
    x = _input(512, 1024, "randn")[:, :512]
    _check(x, _offs([256, 256]))


def test_empty() -> None:
    x = torch.empty(0, 2048, dtype=torch.bfloat16, device="cuda")
    out = cute_impl(x, torch.zeros(16, dtype=torch.int32, device="cuda"))
    assert [t.shape for t in out] == [(0, 2048), (0, 16), (0,), (2048, 0)]
