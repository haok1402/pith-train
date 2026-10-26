"""
Verifier of the ``fp8_clamped_swiglu_quant`` feature (``clamped_swiglu_quant_rowwise``,
``clamped_swiglu_quant_kmajor``, ``clamped_swiglu_bwd_quant``) against plain PyTorch.

Contract: gpt-oss's clamped SwiGLU of ``gate_up + bias[group]`` (the bias added in fp32 and rounded
to bf16; gate_up interleaved, gate at even / up at odd columns), rounded to bf16 as the plain
kernels ``clamped_swiglu_fwd`` / ``clamped_swiglu_bwd`` round it, then quantized as
``quantize_grouped_input`` does on SM90 (rowwise 1x128 plus the per-channel FP8 K-major per
group). ``padded=True`` lays the K-major pair out with ``ceil(C / 256) * 256`` channels (C = N in
forward, 2N in backward) where the fused kernel runs, C where the input takes the unfused fallback;
the extra channels are unwritten, and ``k_grouped_fp8_gemm`` gives the same product with either
layout (it drops the outputs of extra channels). The bias is bf16. The backward also
returns the per-group bias gradient: fp32 sums of each 128-row block of the bf16 gradient, added in
block order, rounded once to the bias dtype.

Reference. The kernels' sigmoid (Triton's ``1 / (1 + ex2.approx(...))`` lowering) cannot be
reproduced bit for bit in PyTorch, so exact equality with a PyTorch activation is impossible.
Instead the FP8 bytes and scale values must equal, exactly, a PyTorch quantization (the recipe,
below) of the exported plain kernels' bf16 output on the PyTorch-biased input, and those bf16
activations must be within one bf16 rounding of a float64 PyTorch evaluation of the formula (plus
fp32-level noise). The bias add itself is compared exactly (it is PyTorch's). The bias gradient is
compared to the fp32 block sums within the bf16 rounding plus fp32 reassociation inside a block
(the in-block summation order is the kernel's).

Recipe: per 128-element block ``amax = max |x|`` (NaN ignored), ``scale = 2^ceil(log2(max(amax,
1e-4) / 448))`` computed as the kernels do (fp32 product with 1/448, then a power-of-two ceil by
bit arithmetic; the same value), ``q = e4m3_satfinite(x * (1 / scale))`` (round to nearest even).

    pytest tests/kernels/test_fp8_clamped_swiglu_quant.py -v
"""

import pytest
import torch

from pithtrain.kernels import (
    clamped_swiglu_bwd,
    clamped_swiglu_bwd_quant,
    clamped_swiglu_fwd,
    clamped_swiglu_quant_kmajor,
    clamped_swiglu_quant_rowwise,
    fp8_fused_experts_supported,
    k_grouped_fp8_gemm,
    quantize_grouped_input,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8_fused_experts_supported(),
    reason="the fused FP8 expert ops run only where fp8_fused_experts_supported()",
)

BLOCK = 128
ALPHA, LIMIT = 1.702, 7.0

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


def assert_quant_equal(got: tuple, ref: tuple, names: tuple[str, ...]) -> None:
    assert len(got) == len(ref) == len(names)
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


ROW = ("x_fp8", "x_scale")
KMAJOR = ("x_ch_fp8", "x_ch_scale")
ALL = ROW + KMAJOR


def _padded(c: int) -> int:
    return -(-c // 256) * 256


def assert_kmajor(got: tuple, ref: tuple, offs: torch.Tensor, K: int, padded: bool = False) -> None:
    """The K-major pair ``got``: laid out with ``K`` channels, or with ``padded`` with ``K`` or
    ``_padded(K)`` (the returned scale's row count says which); its first ``K`` channels equal the
    unpadded reference ``ref`` exactly. Extra channels are unspecified."""
    fp8, scale = got
    KP = scale.shape[0] if scale.ndim == 2 else K
    assert KP in ((K, _padded(K)) if padded else (K,)), (scale.shape, K, padded)
    M = scale.shape[1] * BLOCK if scale.ndim == 2 else 0
    assert fp8.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert fp8.numel() == M * KP, (fp8.shape, scale.shape, KP)
    raw = fp8.view(torch.uint8)
    real = torch.empty(M * K, dtype=torch.uint8, device=raw.device)
    start = 0
    for end in offs.tolist():
        blk = raw[start * KP : end * KP].view(KP, end - start)
        real[start * K : end * K].view(K, end - start).copy_(blk[:K])
        start = end
    assert_quant_equal((real.view(torch.float8_e4m3fn), scale[:K]), ref, KMAJOR)


# ---------------------------------------------------------------------- activation references


def _group_ids(offs: torch.Tensor, M: int) -> torch.Tensor:
    ids = torch.arange(M, device=offs.device)
    return torch.searchsorted(offs.long(), ids, right=True).clamp_(max=offs.shape[0] - 1)


def biased_ref(gate_up: torch.Tensor, bias: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """gate_up + bias[group of each row], one fp32 add rounded to bf16."""
    return (gate_up.float() + bias.float()[_group_ids(offs, gate_up.shape[0])]).to(torch.bfloat16)


def _assert_bf16_close(got: torch.Tensor, ref64: torch.Tensor, mag: torch.Tensor) -> None:
    """``got`` (bf16) is the bf16 rounding of a value fp32-close to the float64 ``ref64``."""
    g = got.double()
    assert bool(torch.isfinite(ref64).all()) and bool(torch.isfinite(g).all())
    err = (g - ref64).abs()
    tol = 2.0**-8 * ref64.abs() + 2.0**-20 * mag + 1e-30
    if not bool((err <= tol).all()):
        i = (err - tol).argmax()
        raise AssertionError(f"activation off: got {g.flatten()[i]}, want {ref64.flatten()[i]}")


def _f32(v: float) -> float:
    return torch.tensor(v, dtype=torch.float32).item()


def check_fwd_activation(x: torch.Tensor, h: torch.Tensor) -> None:
    """Plain ``clamped_swiglu_fwd`` output ``h`` of the biased ``x`` against float64."""
    a, lim = _f32(ALPHA), _f32(LIMIT)
    g, u = x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = g.clamp(max=lim), u.clamp(-lim, lim)
    glu = gc * torch.sigmoid(a * gc)
    _assert_bf16_close(h, (uc + 1) * glu, (uc.abs() + 1) * glu.abs())


def check_bwd_activation(dh: torch.Tensor, x: torch.Tensor, dx: torch.Tensor) -> None:
    """Plain ``clamped_swiglu_bwd`` output ``dx`` (interleaved) against float64."""
    a, lim = _f32(ALPHA), _f32(LIMIT)
    d, g, u = dh.double(), x[:, 0::2].double(), x[:, 1::2].double()
    gc, uc = g.clamp(max=lim), u.clamp(-lim, lim)
    sig = torch.sigmoid(a * gc)
    glu = gc * sig
    front = d * (uc + 1) * sig
    dg = torch.where(g <= lim, front * (1 + a * gc * (1 - sig)), 0.0)
    du = torch.where((u >= -lim) & (u <= lim), d * glu, 0.0)
    _assert_bf16_close(dx[:, 0::2], dg, (d * (uc.abs() + 1) * sig).abs() * (1 + abs(a) * gc.abs()))
    _assert_bf16_close(dx[:, 1::2], du, (d * glu).abs())


def bias_grad_blocks(dgu: torch.Tensor, offs: torch.Tensor, G: int) -> tuple:
    """fp32 per-group sums of the 128-row block sums, in block order, and the per-group sums of
    |dgu| (the reassociation bound)."""
    M, C = dgu.shape
    blocks = dgu.float().view(M // BLOCK, BLOCK, C).sum(1)
    ablocks = dgu.float().abs().view(M // BLOCK, BLOCK, C).sum(1)
    exp = torch.zeros(G, C, device=dgu.device)
    absum = torch.zeros(G, C, device=dgu.device)
    start = 0
    for g, end in enumerate((offs // BLOCK).tolist()):
        for b in range(start, end):
            exp[g] += blocks[b]
        absum[g] = ablocks[start:end].sum(0)
        start = end
    return exp, absum


def check_bias_grad(got: torch.Tensor, dgu: torch.Tensor, offs: torch.Tensor, bias) -> None:
    """bf16 bias gradient: within one bf16 rounding of the fp32 block-order sums, plus fp32
    reassociation inside a block; exactly zero for empty groups."""
    assert got.dtype == bias.dtype and got.shape == bias.shape
    exp, absum = bias_grad_blocks(dgu, offs, bias.shape[0])
    err = (got.double() - exp.double()).abs()
    tol = 2.0**-8 * exp.double().abs() + 2.0**-16 * absum.double()
    assert bool((err <= tol).all()), f"bias grad off by {(err - tol).max().item()}"
    empty = absum == 0
    assert bool((got[empty] == 0).all())


# --------------------------------------------------------------------------------- inputs


def _offs(M: int, G: int, gen: torch.Generator) -> torch.Tensor:
    """Random 128-aligned cumulative group ends (some groups empty), last entry M."""
    blocks = M // BLOCK
    cuts = torch.sort(torch.randint(0, blocks + 1, (G - 1,), generator=gen)).values
    ends = torch.cat([cuts, torch.tensor([blocks])]) * BLOCK
    return ends.to(torch.int32).cuda()


def _inputs(M: int, N: int, G: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    gu = torch.randn(M, 2 * N, generator=gen) * 4
    gu[torch.rand(M, 2 * N, generator=gen) < 0.01] *= 100  # outliers past the clamp
    gu[torch.rand(M, 2 * N, generator=gen) < 0.01] = 0
    if M >= 256:
        gu[128:256] = 0  # an all-zero row block
    gu[:, 0:16:2] = -gu[:, 0:16:2].abs()  # negative gates
    bias = torch.randn(G, 2 * N, generator=gen)
    dh = torch.randn(M, N, generator=gen)
    dh[torch.rand(M, N, generator=gen) < 0.01] *= 1000
    bf = dict(dtype=torch.bfloat16, device="cuda")
    return gu.to(**bf), bias.to(**bf), dh.to(**bf), _offs(M, G, gen)


# (M, N, G): gpt-oss-20b's 2880 (hidden = intermediate), small / ragged N, 32 experts.
SHAPES = [
    (128, 2880, 1),
    (1024, 2880, 4),
    (4096, 2880, 32),
    (2048, 2880, 16),
    (512, 64, 3),
    (640, 136, 5),  # N and 2N not multiples of 128
    (256, 8, 2),
    (384, 100, 3),  # N % 8 != 0: unfused fallback
]


# ----------------------------------------------------------------------------------- tests


@pytest.mark.parametrize("M,N,G", SHAPES)
def test_quant_rowwise(M: int, N: int, G: int) -> None:
    gu, bias, _, offs = _inputs(M, N, G, seed=M + N + G)
    x = biased_ref(gu, bias, offs)
    h = clamped_swiglu_fwd(x, ALPHA, LIMIT)
    check_fwd_activation(x, h)
    ref = quantize_grouped_ref(h, offs)
    got = clamped_swiglu_quant_rowwise(gu, bias, offs, ALPHA, LIMIT)
    assert_quant_equal(got, ref[:2], ROW)


KMAJOR_CASES = [pytest.param(*shape, padded) for shape in SHAPES for padded in (False, True)]


@pytest.mark.parametrize("M,N,G,padded", KMAJOR_CASES)
def test_quant_kmajor(M: int, N: int, G: int, padded: bool) -> None:
    gu, bias, _, offs = _inputs(M, N, G, seed=3 * M + N + G)
    h = clamped_swiglu_fwd(biased_ref(gu, bias, offs), ALPHA, LIMIT)
    ref = quantize_grouped_ref(h, offs)
    got = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=padded)
    assert_kmajor(got, ref[2:], offs, N, padded)


@pytest.mark.parametrize("M,N,G,padded", KMAJOR_CASES)
def test_bwd_quant(M: int, N: int, G: int, padded: bool) -> None:
    gu, bias, dh, offs = _inputs(M, N, G, seed=7 * M + N + G)
    x = biased_ref(gu, bias, offs)
    dgu = clamped_swiglu_bwd(dh, x, ALPHA, LIMIT)
    check_bwd_activation(dh, x, dgu)
    ref = quantize_grouped_ref(dgu, offs)
    got, d_bias = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT, padded=padded)
    assert_quant_equal(got[:2], ref[:2], ROW)
    assert_kmajor(got[2:], ref[2:], offs, 2 * N, padded)
    check_bias_grad(d_bias, dgu, offs, bias)


@pytest.mark.parametrize("backward", [False, True])
def test_padded_layout_feeds_k_grouped_gemm(backward: bool) -> None:
    """With ``padded=True`` the fused kernels lay out ``_padded(C)`` channels, the extra ones
    unwritten (the allocator is dirtied first); ``k_grouped_fp8_gemm`` drops them, so the weight
    gradient equals the one from the unpadded layout."""
    M, N, G = 1024, 2880, 4
    gu, bias, dh, offs = _inputs(M, N, G, seed=17)
    torch.cuda.empty_cache()
    torch.empty(64 << 20, dtype=torch.uint8, device="cuda").fill_(0x7F)  # FP8 NaN bytes
    if backward:
        C = 2 * N
        pad = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT, padded=True)[0][2:]
        flat = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT)[0][2:]
    else:
        C = N
        pad = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=True)
        flat = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT)
    assert pad[1].shape[0] == _padded(C) and flat[1].shape[0] == C
    other = torch.randn(M, 2048, device="cuda").to(torch.bfloat16)
    b = quantize_grouped_input(other, offs)[2:]
    ks = [int(v) for v in torch.diff(offs, prepend=offs.new_zeros(1)).tolist()]
    kt = torch.tensor(ks, dtype=torch.int32, device="cuda")
    got = torch.zeros(G, C, 2048, device="cuda")
    want = torch.zeros(G, C, 2048, device="cuda")
    k_grouped_fp8_gemm(pad, b, got, ks, kt)
    k_grouped_fp8_gemm(flat, b, want, ks, kt)
    assert bool(torch.isfinite(got).all())
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backward", [False, True])
def test_every_bf16_gate_and_up(backward: bool) -> None:
    """Every BF16 bit pattern (inf and NaN included) as gate and, shuffled, as up, with a random
    bias; NaN-ignoring block amax."""
    M, N, G = 512, 128, 2
    gen = torch.Generator().manual_seed(3)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16)
    up = bits[torch.randperm(65536, generator=gen)].view(torch.bfloat16)
    gu = torch.stack([gate, up], dim=-1).view(M, 2 * N).cuda()
    bias = (torch.randn(G, 2 * N, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    x = biased_ref(gu, bias, offs)
    if not backward:
        ref = quantize_grouped_ref(clamped_swiglu_fwd(x, ALPHA, LIMIT), offs)
        assert_quant_equal(clamped_swiglu_quant_rowwise(gu, bias, offs, ALPHA, LIMIT), ref[:2], ROW)
        for padded in (False, True):
            got = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT, padded=padded)
            assert_kmajor(got, ref[2:], offs, N, padded)
        return
    dh = torch.randn(M, N, generator=gen).to(torch.bfloat16).cuda()
    dgu = clamped_swiglu_bwd(dh, x, ALPHA, LIMIT)
    out, _ = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT)
    assert_quant_equal(out, quantize_grouped_ref(dgu, offs), ALL)


def test_zeros() -> None:
    """Zero inputs and a zero bias: zero bytes, the scale floor 2^ceil(log2(1e-4 / 448)) and a
    zero bias gradient, from every entry point."""
    M, N, G = 384, 2880, 3
    gu = torch.zeros(M, 2 * N, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(G, 2 * N, dtype=torch.bfloat16, device="cuda")
    dh = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([128, 128, 384], dtype=torch.int32, device="cuda")
    floor = _e8m0(torch.zeros(1, device="cuda"))[0].item()
    row = clamped_swiglu_quant_rowwise(gu, bias, offs, ALPHA, LIMIT)
    km = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT)
    (q, d_bias) = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT)
    for fp8, scale in (row, km, q[:2], q[2:]):
        assert not bool(fp8.view(torch.uint8).any())
        assert bool((scale == floor).all())
    assert not bool(d_bias.any())


def test_empty_input() -> None:
    """M == 0 (no tokens routed to this rank): correctly shaped empty outputs, zero bias grad."""
    N, G = 2880, 4
    gu = torch.empty(0, 2 * N, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(G, 2 * N, device="cuda").to(torch.bfloat16)
    dh = torch.empty(0, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.zeros(G, dtype=torch.int32, device="cuda")
    q, s = clamped_swiglu_quant_rowwise(gu, bias, offs, ALPHA, LIMIT)
    assert q.shape == (0, N) and s.shape == (0, -(-N // BLOCK))
    km, kms = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT)
    assert km.numel() == 0 and kms.shape == (N, 0)
    out, d_bias = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT)
    assert out[0].shape == (0, 2 * N) and out[2].numel() == 0 and out[3].shape == (2 * N, 0)
    assert d_bias.shape == bias.shape and d_bias.dtype == bias.dtype and not bool(d_bias.any())
