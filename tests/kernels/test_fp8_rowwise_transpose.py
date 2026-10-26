"""
Verifier of ``fp8cast_rowwise_transpose`` against a pure-PyTorch implementation of the E8M0 FP8
recipe.

``(x_fp8, x_scale, xt_fp8, xt_scale) = fp8cast_rowwise_transpose(x)`` is the rowwise 1x128
quantization of x ((M, N) e4m3, scale (M, ceil(N/128))) and of x.T ((N, M) e4m3, scale
(N, ceil(M/128))). FP8 bytes and scale values must be identical to the reference; scale strides are
unspecified by contract, so scales are compared by logical value.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import fp8cast_rowwise_transpose

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


def ref_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x128 rowwise quantization of a 2D tensor: (q (M, K), scale (M, ceil(K/128)))."""
    M, K = x.shape
    kb = (K + BLOCK - 1) // BLOCK
    xb = F.pad(x.float(), (0, kb * BLOCK - K)).view(M, kb, BLOCK)
    s = e8m0_scale(xb.abs().amax(dim=-1)) if M * kb else torch.empty(M, kb, device=x.device)
    q = (xb / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(M, kb * BLOCK)[:, :K].contiguous(), s


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
    x_fp8, x_scale, xt_fp8, xt_scale = fp8cast_rowwise_transpose(x)
    q, s = ref_rowwise(x)
    qt, st = ref_rowwise(x.T)
    assert_fp8_equal("x_fp8", x_fp8, q)
    assert_scale_equal("x_scale", x_scale, s)
    assert_fp8_equal("xt_fp8", xt_fp8, qt)
    assert_scale_equal("xt_scale", xt_scale, st)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# (M, N): model hidden / intermediate / dense sizes, tile edges, unaligned sizes.
SHAPES = [
    (4096, 4096),
    (4096, 2048),
    (4096, 1408),
    (4096, 768),
    (4096, 512),
    (4096, 2880),
    (1000, 10944),
    (4100, 576),
    (128, 128),
    (128, 256),
    (256, 256),
    (64, 256),
    (1, 128),
    (7, 64),
    (7, 300),
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
    (256, 7168),
]


def make_input(m: int, n: int, kind: str, seed: int = 0) -> torch.Tensor:
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
        # amax exactly 448 * 2**k: the exact-power-of-two (no round-up) branch of the E8M0 ceil.
        x = x.clamp(-1, 1)
        x[:, 0] = 448.0 * 2.0**-3
        x[0, :] = -448.0
    elif kind == "wide":
        x = x * torch.exp2(torch.randint(-40, 40, (m, n), device="cuda", generator=g).float())
    elif kind == "huge":
        x = x * 1e30
    elif kind == "negative":
        x = -x.abs()
    return x.to(torch.bfloat16)


KINDS = ["randn", "outliers", "zeros", "tiny", "pow2", "wide", "huge", "negative"]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("kind", KINDS)
def test_matches_reference(shape: tuple[int, int], kind: str) -> None:
    check(make_input(*shape, kind))


def test_all_zeros() -> None:
    check(torch.zeros(64, 256, device="cuda", dtype=torch.bfloat16))


def test_misaligned_base_pointer() -> None:
    base = make_input(512, 1032, "outliers")
    for off in (3, 4, 8):
        check(base.view(-1)[off : off + 512 * 1024].view(512, 1024))


def test_non_contiguous() -> None:
    check(make_input(256, 512, "randn")[:, :384])
    check(make_input(512, 300, "outliers").T)


def test_empty() -> None:
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    out = fp8cast_rowwise_transpose(x)
    assert [tuple(t.shape) for t in out] == [(0, 256), (0, 2), (256, 0), (256, 0)]
