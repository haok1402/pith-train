"""Bitwise parity of the CuTe-DSL fp8cast_rowwise_transpose against the Triton reference."""

import pytest
import torch

from pithtrain.operators.deepgemm_quantize import _fp8cast_rowwise_transpose_triton as triton_ref
from pithtrain.operators.fp8_cute.rowwise_transpose import fp8cast_rowwise_transpose as cute_impl
from pithtrain.operators.fp8_cute.scale_layout import is_mn_major

SHAPES = [
    (4096, 4096),
    (4096, 512),
    (4096, 2048),
    (4100, 576),
    (1000, 10944),
    (128, 128),
    (1, 128),
    (7, 64),
    (16, 8),
    (300, 200),
    (130, 1000),
    (257, 129),
    (64, 4104),
    (333, 1001),
    (96, 1028),
    (258, 1030),
    (4101, 512),
    (136, 384),
    (2056, 2048),
    (8, 4096),
]


def _make_input(m: int, n: int, kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, generator=g)
    if kind == "outliers":
        mask = torch.rand(m, n, device="cuda", generator=g) < 0.001
        x = torch.where(mask, x * 1e4, x)
        x[:, ::7] *= 1e-3
        x[::5, :] *= 30.0
    elif kind == "zeros":
        x[:, : max(1, n // 3)] = 0.0
        x[: max(1, m // 4), :] = 0.0
    elif kind == "tiny":
        x = x * 1e-7
    elif kind == "pow2":
        # amax exactly 448 * 2^k exercises the exact-power-of-2 (no exponent bump) path.
        x = x.clamp(-1, 1)
        x[:, 0] = 448.0 * 2.0**-3
        x[0, :] = -448.0
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-40, 40, (m, n), device="cuda", generator=g).float())
    return x.to(torch.bfloat16)


def _assert_bitwise(x: torch.Tensor) -> None:
    ref = triton_ref(x)
    out = cute_impl(x)
    names = ("tok_fp8", "tok_scale", "t_fp8", "t_scale")
    for name, r, o in zip(names, ref, out):
        assert r.shape == o.shape and r.dtype == o.dtype, name
        # Scales come back MN-major (DeepGEMM's SM90 layout); the values must still match.
        assert is_mn_major(o) if name.endswith("scale") else r.stride() == o.stride(), name
        if r.dtype == torch.float8_e4m3fn:
            ok = torch.equal(r.view(torch.uint8), o.view(torch.uint8))
        else:
            ok = torch.equal(r.view(torch.int32), o.view(torch.int32))
        assert ok, f"{name} mismatch for shape {tuple(x.shape)}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", ["randn", "outliers", "zeros", "tiny", "pow2", "wide"])
def test_bitwise_vs_triton(shape: tuple[int, int], kind: str) -> None:
    _assert_bitwise(_make_input(*shape, kind))


def test_misaligned_base_pointer() -> None:
    base = _make_input(512, 1032, "outliers")
    for off in (3, 4, 8):
        _assert_bitwise(base.view(-1)[off : off + 512 * 1024].view(512, 1024))


def test_fallback_non_contiguous() -> None:
    x = _make_input(256, 512, "randn")[:, :384]
    _assert_bitwise(x)
