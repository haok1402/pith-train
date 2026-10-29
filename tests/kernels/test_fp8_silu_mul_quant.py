"""
Verifier of the ``fp8_silu_mul_quant`` feature (``silu_mul_quant_fwd``, ``silu_mul_bwd_quant``,
``silu_mul_quant_gu_fwd``, ``silu_mul_bwd_quant_gu``) against plain PyTorch.

Contract: the SwiGLU ``silu(g) * u`` (or its two gradients), rounded to bf16 as the plain kernels
``silu_mul_fwd`` / ``silu_mul_bwd`` round it, then quantized as ``quantize_grouped_input`` does on
SM90: rowwise 1x128 FP8 plus the per-channel FP8 laid out K-major per group.

Reference. The kernels' sigmoid is ``1 / (1 + ex2.approx(-x * log2e))``, which PyTorch cannot
reproduce bit for bit, so exact equality with a PyTorch activation is impossible. Instead:

- the FP8 bytes and scale values must equal, exactly, a PyTorch quantization (the recipe, below)
  of the exported plain kernels' bf16 output (``silu_mul_fwd`` / ``silu_mul_bwd``), and
- those bf16 activations must be within one bf16 rounding of a float64 PyTorch evaluation of the
  same formula (plus fp32-level noise).

Recipe: per 128-element block ``amax = max |x|`` (NaN ignored), ``scale = 2^ceil(log2(max(amax,
1e-4) / 448))`` computed as the kernels do (fp32 product with 1/448, then a power-of-two ceil by
bit arithmetic; the same value), ``q = e4m3_satfinite(x * (1 / scale))`` (round to nearest even).

    pytest tests/kernels/test_fp8_silu_mul_quant.py -v
"""

import pytest
import torch

from pithtrain.kernels import (
    fp8_fused_experts_supported,
    silu_mul_bwd,
    silu_mul_bwd_quant,
    silu_mul_bwd_quant_gu,
    silu_mul_fwd,
    silu_mul_quant_fwd,
    silu_mul_quant_gu_fwd,
)

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(),
    reason="needs the fused FP8 expert path (fp8_fused_experts_supported)",
)

BLOCK = 128

# ---------------------------------------------------------------- PyTorch quantization reference


def _e8m0(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The recipe's power-of-two scale of an fp32 amax, and its exact reciprocal."""
    a = torch.maximum(amax, torch.full_like(amax, 1e-4))
    s = a * torch.full_like(a, 1.0 / 448.0)
    bits = s.view(torch.int32)
    bits = (bits & 0x7F800000) + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    scale = torch.maximum(bits.view(torch.float32), torch.full_like(a, 2.0**-126))
    rcp = ((254 << 23) - scale.view(torch.int32)).view(torch.float32)
    return scale, rcp


def _absmax_input(x: torch.Tensor) -> torch.Tensor:
    a = x.abs()
    return torch.where(torch.isnan(a), torch.zeros_like(a), a)


def _q(x: torch.Tensor, rcp: torch.Tensor) -> torch.Tensor:
    return (x * rcp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_grouped_ref(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``quantize_grouped_input`` on SM90 in PyTorch: ``(x_fp8 (M, K) as uint8, x_scale (M,
    ceil(K/128)), x_ch flat (M*K,) uint8 K-major per group, x_ch_scale (K, M/128))``."""
    M, K = x.shape
    nb = -(-K // BLOCK)
    xf = torch.zeros(M, nb * BLOCK, dtype=torch.float32, device=x.device)
    xf[:, :K] = x.float()
    s_row, r_row = _e8m0(_absmax_input(xf).view(M, nb, BLOCK).amax(-1))
    q_row = _q(xf.view(M, nb, BLOCK), r_row[..., None]).view(M, nb * BLOCK)[:, :K]
    xc = xf[:, :K].reshape(M // BLOCK, BLOCK, K)
    s_ch, r_ch = _e8m0(_absmax_input(xc).amax(1))
    q_ch = _q(xc, r_ch[:, None, :]).view(M, K)
    flat = torch.zeros(M * K, dtype=torch.uint8, device=x.device)
    start = 0
    for end in offs.tolist():
        flat[start * K : end * K].view(K, end - start).copy_(q_ch[start:end].T)
        start = end
    return q_row.contiguous(), s_row, flat, s_ch.T.contiguous()


def _assert_bytes(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    """Same e4m3 bytes; a NaN may carry either sign."""
    got = got.view(torch.uint8)
    nan = (ref & 0x7F) == 0x7F
    ok = (got == ref) | (nan & ((got & 0x7F) == 0x7F))
    if not bool(ok.all()):
        bad = (~ok).nonzero()
        raise AssertionError(f"{name}: {bad.shape[0]} byte mismatches, first at {bad[:4].tolist()}")


def assert_quant_equal(got: tuple, ref: tuple) -> None:
    names = ("x_fp8", "x_scale", "x_ch_fp8", "x_ch_scale")
    assert len(got) == 4
    for name, a, b in zip(names, got, ref):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        if name.endswith("fp8"):
            assert a.dtype == torch.float8_e4m3fn, name
            _assert_bytes(a, b, name)
        else:
            assert a.dtype == torch.float32, name
            if not torch.equal(a, b):
                bad = (a != b).nonzero()
                raise AssertionError(f"{name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}")


# ------------------------------------------------------------------- activation accuracy check


def _assert_bf16_close(
    got: torch.Tensor, ref64: torch.Tensor, mag: torch.Tensor, floor: torch.Tensor
) -> None:
    """``got`` (bf16) is the bf16 rounding of a value fp32-close to the float64 ``ref64``: within
    one bf16 rounding, fp32 noise relative to the terms' magnitude ``mag``, and ``floor`` times the
    fp32 sigmoid's absolute underflow (sigmoid(x) < ~1e-38 flushes to 0 for x < ~-88)."""
    g = got.double()
    special = ~torch.isfinite(ref64.to(torch.bfloat16).double())
    nan = torch.isnan(ref64)
    assert torch.equal(torch.isnan(g), nan)
    assert torch.equal(g[special & ~nan], ref64.to(torch.bfloat16).double()[special & ~nan])
    fin = ~special
    err = (g[fin] - ref64[fin]).abs()
    tol = 2.0**-8 * ref64[fin].abs() + 2.0**-20 * mag[fin] + 1e-37 * floor[fin] + 1e-37
    if not bool((err <= tol).all()):
        i = (err - tol).argmax()
        raise AssertionError(f"activation off: got {g[fin][i]}, want {ref64[fin][i]}")


def check_fwd_activation(g: torch.Tensor, u: torch.Tensor, h: torch.Tensor) -> None:
    g64, u64 = g.double(), u.double()
    silu = g64 * torch.sigmoid(g64)
    _assert_bf16_close(h, silu * u64, (silu * u64).abs(), (g64 * u64).abs())


def check_bwd_activation(dh, g, u, dg, du) -> None:
    d64, g64, u64 = dh.double(), g.double(), u.double()
    sig = torch.sigmoid(g64)
    silu = g64 * sig
    _assert_bf16_close(du, d64 * silu, (d64 * silu).abs(), (d64 * g64).abs())
    front = (d64 * u64 * sig).abs()
    floor = (d64 * u64).abs() * (1 + g64.abs())
    _assert_bf16_close(dg, d64 * u64 * sig * (1.0 + g64 - silu), front * (1 + g64.abs()), floor)


# --------------------------------------------------------------------------------- inputs


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
            x[128:256] *= 1e-30  # tiny products and flushed FP8
    elif kind == "wide":
        for x in xs:
            x.mul_(torch.exp2(torch.randint(-20, 20, (M, 1), device="cuda", generator=gen)))
    elif kind == "large":  # large finite values (fp32 gradients up to ~1e36)
        for x in xs:
            x.mul_(1e12)
    return [x.to(torch.bfloat16) for x in xs]


GROUPS = {
    "balanced16x2048": [2048] * 16,
    "imbalanced": [128, 0, 384, 1024, 0, 0, 256, 128, 2048, 0, 640],
    "single": [256],
    "empty_first_last": [0, 512, 128, 0],
    "many_small": [128] * 40 + [0] * 8,
    "experts128": [0, 256] * 32 + [128] * 64,
    "over_cap": [0] * 1100 + [256],  # > 1024 groups
    "all_empty_but_one": [0] * 7 + [128],
}

# (groups, N, kind). N: the expert intermediates of qwen3-30b-a3b (768), deepseek-v2-lite (1408),
# qwen3.5-35b-a3b (512), deepseek-v2-lite's dense MLP (10944), plus ragged / small ones.
CASES = [
    ("balanced16x2048", 768, "normal"),
    ("imbalanced", 768, "normal"),
    ("imbalanced", 768, "outliers"),
    ("imbalanced", 768, "wide"),
    ("imbalanced", 1408, "normal"),
    ("imbalanced", 512, "normal"),
    ("single", 10944, "normal"),
    ("single", 768, "normal"),
    ("single", 768, "large"),
    ("empty_first_last", 2048, "normal"),
    ("many_small", 768, "outliers"),
    ("imbalanced", 200, "outliers"),  # ragged N, partial last column block
    ("single", 72, "normal"),  # N < 128
    ("empty_first_last", 1416, "wide"),
    ("single", 100, "normal"),  # N % 8 != 0
    ("experts128", 768, "normal"),
    ("over_cap", 768, "normal"),
    ("all_empty_but_one", 512, "negative"),
]


def _case(groups: str, N: int, kind: str, seed: int):
    sizes = GROUPS[groups]
    M = int(sum(sizes))
    dh, g, u = _inputs(M, N, kind, seed=seed + N + M)
    if kind == "negative":
        g, u = -g.abs(), -u.abs()
    return _offs(sizes), dh, g, u


# ----------------------------------------------------------------------------------- tests


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_silu_mul_quant_fwd(groups: str, N: int, kind: str) -> None:
    offs, _, g, u = _case(groups, N, kind, seed=0)
    h = silu_mul_fwd(g, u)
    check_fwd_activation(g, u, h)
    assert_quant_equal(silu_mul_quant_fwd(g, u, offs), quantize_grouped_ref(h, offs))


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_silu_mul_quant_gu_fwd(groups: str, N: int, kind: str) -> None:
    offs, _, g, u = _case(groups, N, kind, seed=1)
    gu = torch.cat([g, u], 1).contiguous()
    h = silu_mul_fwd(g, u)
    assert_quant_equal(silu_mul_quant_gu_fwd(gu, offs), quantize_grouped_ref(h, offs))


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_silu_mul_bwd_quant(groups: str, N: int, kind: str) -> None:
    offs, dh, g, u = _case(groups, N, kind, seed=2)
    dg, du = silu_mul_bwd(dh, g, u)
    check_bwd_activation(dh, g, u, dg, du)
    got_g, got_u = silu_mul_bwd_quant(dh, g, u, offs)
    assert_quant_equal(got_g, quantize_grouped_ref(dg, offs))
    assert_quant_equal(got_u, quantize_grouped_ref(du, offs))


@pytest.mark.parametrize("groups,N,kind", CASES)
def test_silu_mul_bwd_quant_gu(groups: str, N: int, kind: str) -> None:
    offs, dh, g, u = _case(groups, N, kind, seed=3)
    gu = torch.cat([g, u], 1).contiguous()
    dg, du = silu_mul_bwd(dh, g, u)
    got = silu_mul_bwd_quant_gu(dh, gu, offs)
    assert_quant_equal(got, quantize_grouped_ref(torch.cat([dg, du], 1), offs))


def test_exhaustive_bf16_gate() -> None:
    """Every finite BF16 gate value against several up / dh scales, all four entry points."""
    bits = torch.arange(0, 1 << 16, device="cuda", dtype=torch.int32).to(torch.int16)
    g_all = bits.view(torch.bfloat16)
    g_all = g_all[torch.isfinite(g_all)]
    N = 256
    M = (g_all.numel() + N - 1) // N
    M = (M + 127) // 128 * 128
    g = torch.zeros(M * N, device="cuda", dtype=torch.bfloat16)
    g[: g_all.numel()] = g_all
    g = g.view(M, N)
    offs = _offs([M // 2, 0, M // 2])
    gen = torch.Generator(device="cuda").manual_seed(0)
    for scale in (1.0, 1e-3, 1e3):
        u = (torch.randn(M, N, device="cuda", generator=gen) * scale).to(torch.bfloat16)
        dh = torch.randn(M, N, device="cuda", generator=gen).to(torch.bfloat16)
        gu = torch.cat([g, u], 1).contiguous()
        h = silu_mul_fwd(g, u)
        check_fwd_activation(g, u, h)
        ref_h = quantize_grouped_ref(h, offs)
        assert_quant_equal(silu_mul_quant_fwd(g, u, offs), ref_h)
        assert_quant_equal(silu_mul_quant_gu_fwd(gu, offs), ref_h)
        dg, du = silu_mul_bwd(dh, g, u)
        check_bwd_activation(dh, g, u, dg, du)
        got_g, got_u = silu_mul_bwd_quant(dh, g, u, offs)
        assert_quant_equal(got_g, quantize_grouped_ref(dg, offs))
        assert_quant_equal(got_u, quantize_grouped_ref(du, offs))
        got = silu_mul_bwd_quant_gu(dh, gu, offs)
        assert_quant_equal(got, quantize_grouped_ref(torch.cat([dg, du], 1), offs))


def test_zeros() -> None:
    """All-zero inputs: zero bytes, every scale the floor 2^ceil(log2(1e-4 / 448))."""
    M, N = 384, 768
    z = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    offs = _offs([128, 256])
    floor = _e8m0(torch.zeros(1, device="cuda"))[0].item()
    for out in (
        silu_mul_quant_fwd(z, z, offs),
        silu_mul_quant_gu_fwd(torch.cat([z, z], 1), offs),
        *silu_mul_bwd_quant(z, z, z, offs),
        silu_mul_bwd_quant_gu(z, torch.cat([z, z], 1), offs),
    ):
        assert not bool(out[0].view(torch.uint8).any()) and not bool(out[2].view(torch.uint8).any())
        assert bool((out[1] == floor).all()) and bool((out[3] == floor).all())


def test_empty_input() -> None:
    """M == 0 (no tokens routed to this rank) returns correctly shaped empty outputs."""
    N = 768
    g = torch.empty(0, N, device="cuda", dtype=torch.bfloat16)
    gu = torch.empty(0, 2 * N, device="cuda", dtype=torch.bfloat16)
    offs = torch.zeros(4, device="cuda", dtype=torch.int32)

    def shapes(out, K):
        assert out[0].shape == (0, K) and out[0].dtype == torch.float8_e4m3fn
        assert out[1].shape == (0, -(-K // BLOCK)) and out[1].dtype == torch.float32
        assert out[2].numel() == 0 and out[2].dtype == torch.float8_e4m3fn
        assert out[3].shape == (K, 0) and out[3].dtype == torch.float32

    shapes(silu_mul_quant_fwd(g, g, offs), N)
    shapes(silu_mul_quant_gu_fwd(gu, offs), N)
    dq, uq = silu_mul_bwd_quant(g, g, g, offs)
    shapes(dq, N)
    shapes(uq, N)
    shapes(silu_mul_bwd_quant_gu(g, gu, offs), 2 * N)


QWEN35_COUNTS = [[1024] * 16, [0, 128, 256, 0, 640, 128, 1024, 0], [128]]


@pytest.mark.parametrize("counts", QWEN35_COUNTS)
@pytest.mark.parametrize("N", [512, 768, 520])
def test_gu_qwen35_cases(counts: list[int], N: int) -> None:
    """
    The fused gate_up entry points on qwen3.5-like groups.
    """
    gen = torch.Generator(device="cuda").manual_seed(0)
    M, offs = sum(counts), _offs(counts)
    gu = torch.randn(M, 2 * N, device="cuda", generator=gen).to(torch.bfloat16) * 3
    gu[:5, :7] = 0
    gu[7, 9] = 300.0
    dh = torch.randn(M, N, device="cuda", generator=gen).to(torch.bfloat16)
    g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    h = silu_mul_fwd(g, u)
    check_fwd_activation(g, u, h)
    assert_quant_equal(silu_mul_quant_gu_fwd(gu, offs), quantize_grouped_ref(h, offs))
    dg, du = silu_mul_bwd(dh, g, u)
    check_bwd_activation(dh, g, u, dg, du)
    got = silu_mul_bwd_quant_gu(dh, gu, offs)
    assert_quant_equal(got, quantize_grouped_ref(torch.cat([dg, du], 1), offs))
