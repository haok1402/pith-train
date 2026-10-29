"""
Verifier of the ``fp8_gemm`` feature of ``pithtrain.kernels`` against plain PyTorch.

- ``quantize_grouped_input``: exact FP8 bytes and E8M0 scale values of both outputs (rowwise 1x128
  per row; per-channel 128x1 within each group in the K-major per-group layout of SM90).
- ``m_grouped_group_indices``: exact ids against ``searchsorted``, for offsets the one-launch
  kernel covers and offsets it does not.
- ``fp8_m_grouped_gemm`` (with and without the hint), ``k_grouped_fp8_gemm`` (in place, row-limit
  split, the SM90 deadlock guard's direct / transposed / padded branches, padded operands, the
  ``fp8_grouped_wgrad_buffer`` layouts), and the dense ``fp8_gemm_nt`` / ``fp8_act_weight_gemm``
  / ``fp8_dgrad_wgrad``: against fp32 PyTorch matmuls of the dequantized FP8 operands.
- ``fp8_grouped_wgrad_buffer`` layout and ``fp8_fused_experts_supported``.

The deadlock-guard shapes of k_grouped_fp8_gemm run in a subprocess under a timeout, so a hang
fails the test instead of the session.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from pithtrain.kernels import (
    fp8_act_weight_gemm,
    fp8_dgrad_wgrad,
    fp8_fused_experts_supported,
    fp8_gemm_nt,
    fp8_grouped_wgrad_buffer,
    fp8_m_grouped_gemm,
    fp8cast_blockwise_transpose,
    fp8cast_blockwise_transpose_batched,
    fp8cast_rowwise_blockwise_transpose,
    k_grouped_fp8_gemm,
    m_grouped_group_indices,
    quantize_grouped_input,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8_fused_experts_supported(),
    reason="needs the fused FP8 expert path (fp8_fused_experts_supported)",
)

DEV = "cuda"
BLOCK = 128


# ---------------------------------------------------------------------------
# PyTorch reference of the FP8 recipe
# ---------------------------------------------------------------------------


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Normalized squared error ``1 - 2 sum(xy) / sum(x^2 + y^2)`` in float64."""
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    if den == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / den).item()


def e8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(max(amax, 1e-4) / 448))`` as an exact fp32 power of two. ``/ 448`` is the
    kernels' multiply by fp32(1 / 448); the power-of-two ceil is done on the bits."""
    a = amax.float().clamp(min=1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    bits = a.view(torch.int32)
    up = torch.where((bits & 0x007FFFFF) != 0, 0x00800000, 0).to(torch.int32)
    return ((bits & 0x7F800000) + up).view(torch.float32)


def cast(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def ref_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1x128 along each row: fp8 (M, K), scale (M, ceil(K / 128))."""
    M, K = x.shape
    kb = -(-K // BLOCK)
    xp = torch.zeros(M, kb * BLOCK, dtype=torch.float32, device=x.device)
    xp[:, :K] = x.float()
    xb = xp.view(M, kb, BLOCK)
    s = e8m0_scale(xb.abs().amax(-1))
    q = cast(xb, s[..., None]).view(M, kb * BLOCK)[:, :K].contiguous()
    return q, s


def ref_channelwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x1 along the rows (M a multiple of 128): fp8 (M, K), scale (K, M / 128)."""
    M, K = x.shape
    xb = x.float().view(M // BLOCK, BLOCK, K)
    s = e8m0_scale(xb.abs().amax(1))
    q = cast(xb, s[:, None, :]).view(M, K)
    return q, s.T


def ref_blockwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """128x128 tiles of a 2D x: fp8 (R, C), scale (ceil(R / 128), ceil(C / 128))."""
    R, C = x.shape
    rb, cb = -(-R // BLOCK), -(-C // BLOCK)
    xp = torch.zeros(rb * BLOCK, cb * BLOCK, dtype=torch.float32, device=x.device)
    xp[:R, :C] = x.float()
    xb = xp.view(rb, BLOCK, cb, BLOCK)
    s = e8m0_scale(xb.abs().amax(dim=(1, 3)))
    q = cast(xb, s[:, None, :, None]).view(rb * BLOCK, cb * BLOCK)[:R, :C].contiguous()
    return q, s


def to_kmajor(q: torch.Tensor, ks: list[int]) -> torch.Tensor:
    """(M, K) per-channel FP8 -> flat K-major per group: group g's (K, ks[g]) block row-major."""
    parts, row = [], 0
    for k in ks:
        parts.append(q[row : row + k].T.contiguous().view(-1))
        row += k
    return torch.cat(parts) if parts else q.new_empty(0)


def dq_rowwise(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * s.repeat_interleave(BLOCK, dim=1)[:, : q.shape[1]]


def dq_blockwise(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    R, C = q.shape[-2:]
    full = s.repeat_interleave(BLOCK, dim=-2).repeat_interleave(BLOCK, dim=-1)
    return q.float() * full[..., :R, :C]


def dq_kmajor(data: torch.Tensor, scale: torch.Tensor, ks: list[int], ch: int) -> list:
    """Per group (ch, ks[g]) fp32 from the K-major flat data and (ch, M / 128) scales."""
    out, row = [], 0
    for k in ks:
        block = data[row * ch : (row + k) * ch].view(ch, k).float()
        s = scale[:, row // BLOCK : (row + k) // BLOCK].repeat_interleave(BLOCK, dim=1)
        out.append(block * s)
        row += k
    return out


def offsets(ks: list[int]) -> torch.Tensor:
    return torch.tensor(ks, dtype=torch.int64).cumsum(0).to(torch.int32).to(DEV)


def make_input(M: int, K: int, kind: str = "randn", seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, K, device=DEV, generator=g)
    if kind == "scaled":  # per-row and per-column dynamic range over many binades
        rs = torch.exp2(torch.randint(-20, 20, (M, 1), device=DEV, generator=g).float())
        cs = torch.exp2(torch.randint(-6, 6, (1, K), device=DEV, generator=g).float())
        x = x * rs * cs
    elif kind == "outliers":
        mask = torch.rand(M, K, device=DEV, generator=g) < 1e-3
        x = torch.where(mask, x * 1e4, x)
        x[:, : min(K, 3)] = 0.0  # all-zero columns
        x[5::97] = 0.0  # all-zero rows
        x[7::131] *= 1e-30  # tiny rows (amax clamp)
        x[11::127] *= 1e-39  # bf16 subnormals
        x[13::89, ::17] = 448.0 * 2.0**-3  # amax exactly 448 * 2^k: a scale boundary
    elif kind == "large":  # near the bf16 range, sign mixed
        x = x * 1e36
    elif kind == "negative":
        x = -x.abs()
    elif kind == "zeros":
        x = torch.zeros_like(x)
    return x.to(torch.bfloat16)


def assert_fp8_equal(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    assert got.dtype == torch.float8_e4m3fn and got.shape == ref.shape, (name, got.shape)
    a, b = got.view(torch.uint8), ref.view(torch.uint8)
    assert torch.equal(a, b), f"{name}: {(a != b).sum().item()} bytes differ"


def assert_scale_equal(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    assert got.dtype == torch.float32 and got.shape == ref.shape, (name, got.shape, ref.shape)
    assert torch.equal(got, ref), f"{name}: {(got != ref).sum().item()} scales differ"


# ---------------------------------------------------------------------------
# fp8_fused_experts_supported / fp8_grouped_wgrad_buffer
# ---------------------------------------------------------------------------


def test_fused_experts_supported() -> None:
    assert fp8_fused_experts_supported() is True


def blocks(n: int) -> int:
    return -(-n // BLOCK)


@pytest.mark.parametrize(
    "size",
    [
        (4, 2880, 1024),  # odd N-blocks, even K-blocks: the guard runs it transposed
        (2, 384, 256),
        (3, 1408, 2048),
        (4, 1024, 2880),
        (4, 2880, 2880),
        (2, 5760, 2880),
        (2, 2880, 5760),
        (8, 2816, 2048),
        (8, 2048, 1408),
        (8, 1536, 2048),
        (8, 2048, 768),
        (8, 1024, 2048),
        (8, 2048, 512),
        (2048, 1408),  # not a grouped weight
    ],
)
def test_wgrad_buffer_layout(size: tuple) -> None:
    buf = fp8_grouped_wgrad_buffer(torch.Size(size), torch.device(DEV))
    assert buf.dtype == torch.float32 and buf.device.type == "cuda"
    assert buf.shape == size and buf.contiguous().shape == size
    transposed = len(size) == 3 and blocks(size[1]) % 2 == 1 and blocks(size[2]) % 2 == 0
    if transposed:
        g, n, k = size
        assert not buf.is_contiguous() and buf.stride() == (n * k, 1, n)
        assert buf.transpose(1, 2).is_contiguous()
    else:
        assert buf.is_contiguous()


# ---------------------------------------------------------------------------
# quantize_grouped_input
# ---------------------------------------------------------------------------


def check_grouped_input(x: torch.Tensor, ks: list[int]) -> None:
    M, K = x.shape
    x_fp8, x_scale, ch_fp8, ch_scale = quantize_grouped_input(x, offsets(ks))
    rq, rs = ref_rowwise(x)
    assert_fp8_equal(x_fp8, rq, "x_fp8")
    assert_scale_equal(x_scale, rs, "x_scale")
    cq, cs = ref_channelwise(x)
    assert ch_fp8.shape == (M * K,)
    assert_fp8_equal(ch_fp8, to_kmajor(cq, ks), "x_ch_fp8")
    assert_scale_equal(ch_scale, cs, "x_ch_scale")


BALANCED = [2048] * 16
IMBALANCED = [0, 128, 384, 0, 0, 1024, 256, 128, 0, 2944, 128, 0, 640, 384, 128, 0]
GROUPED_SHAPES = [
    (BALANCED, 2048),
    (BALANCED, 768),
    (IMBALANCED, 2048),
    (IMBALANCED, 768),
    (IMBALANCED, 1408),
    (IMBALANCED, 512),
    ([256, 0, 512, 256], 200),  # K not a multiple of 128
    ([128, 128, 0, 768], 136),
    ([1024], 4096),
    ([128], 128),
    ([0, 0, 128], 8),
    ([512, 512], 2880),
    ([384, 0, 640], 5760),
    ([128, 256], 100),  # K % 8 != 0
]


@pytest.mark.parametrize("kind", ["randn", "scaled", "outliers", "large", "negative"])
@pytest.mark.parametrize("ks,K", GROUPED_SHAPES)
def test_quantize_grouped_input(ks: list[int], K: int, kind: str) -> None:
    check_grouped_input(make_input(sum(ks), K, kind), ks)


@pytest.mark.parametrize(
    "M,K", [(128, 128), (128, 256), (256, 256), (256, 512), (384, 384), (128, 300), (256, 7168)]
)
def test_quantize_grouped_input_one_group(M: int, K: int) -> None:
    check_grouped_input(make_input(M, K), [M])
    check_grouped_input(make_input(M, K), [128] * (M // 128))


def test_quantize_grouped_input_zeros() -> None:
    x = torch.zeros(128, 256, dtype=torch.bfloat16, device=DEV)
    check_grouped_input(x, [128])
    x_fp8, x_scale, ch_fp8, ch_scale = quantize_grouped_input(x, offsets([128]))
    assert (x_fp8.float() == 0).all() and (ch_fp8.float() == 0).all()
    assert torch.isfinite(x_scale).all() and torch.isfinite(ch_scale).all()


def test_quantize_grouped_input_many_groups() -> None:
    g = torch.Generator().manual_seed(1)
    ks = (torch.randint(0, 6, (100,), generator=g) * 128).tolist()
    check_grouped_input(make_input(sum(ks), 512, "scaled"), ks)


def test_quantize_grouped_input_empty() -> None:
    x = torch.empty(0, 2048, dtype=torch.bfloat16, device=DEV)
    out = quantize_grouped_input(x, torch.zeros(16, dtype=torch.int32, device=DEV))
    assert [tuple(t.shape) for t in out] == [(0, 2048), (0, 16), (0,), (2048, 0)]


# ---------------------------------------------------------------------------
# m_grouped_group_indices
# ---------------------------------------------------------------------------


def ref_group_indices(offs: torch.Tensor, M: int) -> torch.Tensor:
    rows = torch.arange(M, device=offs.device)
    gi = torch.searchsorted(offs.contiguous(), rows, right=True).to(torch.int32)
    return gi.clamp_(max=offs.shape[0] - 1)


def check_group_indices(offs: torch.Tensor, M: int) -> None:
    gi = m_grouped_group_indices(offs, M)
    assert gi is not None and gi.dtype == torch.int32 and gi.shape == (M,)
    assert torch.equal(gi, ref_group_indices(offs, M))


@pytest.mark.parametrize("G", [1, 3, 8, 16, 32, 64, 128, 1024, 1025, 1500])
@pytest.mark.parametrize("seed", [0, 1])
def test_group_indices(G: int, seed: int) -> None:
    # G <= 1024 int32 contiguous runs the one-launch kernel, G > 1024 the searchsorted fallback.
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 3000, (G,), generator=g)
    counts[torch.rand(G, generator=g) < 0.3] = 0  # empty groups
    ks = (counts + 127) // 128 * 128
    offs = ks.cumsum(0).to(torch.int32).to(DEV)
    M = int(offs[-1])
    for rows in {M, max(M - 5, 0), M + 300}:  # exact, short, and past the last end (clamped)
        check_group_indices(offs, rows)


def test_group_indices_uncovered_offsets() -> None:
    ks = [128, 0, 384, 256, 0, 0, 640]
    offs = offsets(ks)
    M = sum(ks)
    check_group_indices(offs.to(torch.int64), M)  # not int32
    wide = torch.stack([offs, torch.zeros_like(offs)], 1).view(-1)
    check_group_indices(wide[::2], M)  # non-contiguous


def test_group_indices_empty() -> None:
    offs = torch.tensor([0, 0, 0], dtype=torch.int32, device=DEV)
    assert m_grouped_group_indices(offs, 0).shape == (0,)
    check_group_indices(offs, 7)


def test_group_indices_example() -> None:
    offs = torch.tensor([128, 256, 384, 512], dtype=torch.int32, device=DEV)
    ref = torch.arange(4, dtype=torch.int32, device=DEV).repeat_interleave(128)
    assert torch.equal(m_grouped_group_indices(offs, 512), ref)


# ---------------------------------------------------------------------------
# fp8_m_grouped_gemm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ks,n,K",
    [
        ([256, 0, 640, 128, 0, 384], 2816, 2048),  # deepseek-v2-lite gate_up
        ([256, 0, 640, 128, 0, 384], 2048, 1408),  # deepseek-v2-lite down
        ([512, 128, 0, 256], 1536, 2048),  # qwen3-30b-a3b gate_up
        ([512, 128, 0, 256], 2048, 768),  # qwen3-30b-a3b down
        ([0, 384, 128, 128], 1024, 2048),  # qwen3.5-35b-a3b gate_up
        ([0, 384, 128, 128], 2048, 512),  # qwen3.5-35b-a3b down
        ([256, 0, 384], 5760, 2880),  # gpt-oss-20b gate_up
        ([256, 0, 384], 2880, 2880),  # gpt-oss-20b down
        ([128], 256, 128),
        ([0, 0, 128, 0], 256, 256),
    ],
)
def test_m_grouped_gemm(ks: list[int], n: int, K: int) -> None:
    G, M = len(ks), sum(ks)
    offs = offsets(ks)
    x = make_input(M, K, seed=1)
    w = (make_input(G * n, K, seed=2).float() * 0.05).to(torch.bfloat16).view(G, n, K)
    x_fp8, x_scale, _, _ = quantize_grouped_input(x, offs)
    w_fp8, w_scale, _, _ = fp8cast_blockwise_transpose_batched(w)
    out = fp8_m_grouped_gemm((x_fp8, x_scale), (w_fp8, w_scale), n, offs)
    gi = m_grouped_group_indices(offs, M)
    hinted = fp8_m_grouped_gemm((x_fp8, x_scale), (w_fp8, w_scale), n, offs, group_indices=gi)
    assert out.shape == (M, n) and out.dtype == torch.bfloat16
    assert torch.equal(out, hinted)
    xd, wd = dq_rowwise(x_fp8, x_scale), dq_blockwise(w_fp8, w_scale)
    row = 0
    for g, k in enumerate(ks):
        if k:
            ref = xd[row : row + k] @ wd[g].T
            assert calc_diff(out[row : row + k], ref) < 1e-3, g
        row += k


# ---------------------------------------------------------------------------
# k_grouped_fp8_gemm
# ---------------------------------------------------------------------------


def kgrouped_case(ks: list[int], N: int, K: int, NA: int | None = None, KB: int | None = None):
    """grad_output (M, N) and input (M, K) per-channel FP8 from quantize_grouped_input, optionally
    zero-padded to NA / KB channels, plus their dequantized per-group (N, k) / (K, k) fp32."""
    M = sum(ks)
    offs = offsets(ks)
    ops, refs = [], []
    for ch, padded, seed in ((N, NA or N, 3), (K, KB or K, 4)):
        x = torch.zeros(M, padded, dtype=torch.bfloat16, device=DEV)
        x[:, :ch] = make_input(M, ch, seed=seed)
        _, _, data, scale = quantize_grouped_input(x, offs)
        ops.append((data, scale))
        refs.append([t[:ch] for t in dq_kmajor(data, scale, ks, padded)])
    ks_t = torch.tensor(ks, dtype=torch.int32, device=DEV)
    return ops[0], ops[1], ks_t, refs[0], refs[1]


def check_kgrouped(d: torch.Tensor, ks: list[int], N: int, K: int, **pad) -> None:
    a, b, ks_t, ra, rb = kgrouped_case(ks, N, K, **pad)
    # large enough that overwriting d instead of accumulating into it fails the check below
    d.copy_(100 * torch.randn(d.shape, device=DEV, generator=torch.Generator(DEV).manual_seed(5)))
    d0 = d.clone()
    ptr, stride = d.data_ptr(), d.stride()
    k_grouped_fp8_gemm(a, b, d, ks, ks_t)
    assert d.data_ptr() == ptr and d.stride() == stride
    for g, k in enumerate(ks):
        if k == 0:  # d0 has no subnormals, so even a reduce-add of zeros leaves it bit-identical
            assert torch.equal(d[g], d0[g]), f"empty group {g} changed"
            continue
        prod = ra[g] @ rb[g].T
        # tolerance, not bit equality: DeepGEMM's reduce-add into fp32 d accumulates at its own
        # precision (the FP8 GEMM alone is ~1e-4 relative)
        assert calc_diff(d[g] - d0[g], prod) < 1e-3, g


def run_kgrouped_case(ks: list[int], N: int, K: int, NA: int | None, KB: int | None) -> None:
    """Both main_grad layouts: the fp8_grouped_wgrad_buffer one and a plain contiguous tensor."""
    size = torch.Size((len(ks), N, K))
    for d in (
        fp8_grouped_wgrad_buffer(size, torch.device(DEV)),
        torch.empty(size, dtype=torch.float32, device=DEV),
    ):
        check_kgrouped(d, ks, N, K, NA=NA, KB=KB)
    torch.cuda.synchronize()


# A case whose first operand has an odd 128-block count would deadlock DeepGEMM's SM90 2-CTA
# k-grouped GEMM if made directly; k_grouped_fp8_gemm must guard it. Those cases run in a
# subprocess under a timeout, so a regression fails instead of hanging the session (a hung kernel
# cannot be interrupted from Python).
SUBPROCESS = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("fp8_gemm_verifier", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.run_kgrouped_case(*json.loads(sys.argv[2]))
print("OK")
"""

RAGGED = [256, 0, 640, 128, 0, 384]
# (ks, N, K, NA, KB): d is (G, N, K); NA / KB the channels a / b are laid out with (padded) or None
KGROUPED_CASES = [
    (RAGGED, 2816, 2048, None, None),  # deepseek-v2-lite gate_up
    (RAGGED, 2048, 1408, None, None),  # deepseek-v2-lite down
    (RAGGED, 1536, 2048, None, None),  # qwen3-30b-a3b gate_up
    (RAGGED, 2048, 768, None, None),  # qwen3-30b-a3b down
    (RAGGED, 1024, 2048, None, None),  # qwen3.5-35b-a3b gate_up
    (RAGGED, 2048, 512, None, None),  # qwen3.5-35b-a3b down
    ([1024, 0, 512, 2048, 128, 0, 768, 256], 256, 384, None, None),
    ([128] * 40, 256, 384, None, None),
    ([0, 0, 0], 256, 384, None, None),  # no rows at all
    ([256, 0, 384], 1024, 2880, None, None),  # even M-blocks: direct
    # above DeepGEMM's 65536-row limit: split at group boundaries / inside a group
    ([40960, 0, 30720, 128], 256, 128, None, None),
    ([70016], 256, 128, None, None),
    ([128, 66048, 0, 256], 256, 128, None, None),
    # deadlock guard (odd first-operand block count)
    ([512, 256, 384, 128], 2880, 2880, None, None),  # gpt-oss down: both odd, pads a
    ([256, 128, 640], 2880, 1024, None, None),  # odd M, even N: transposed product
    ([384, 128], 5760, 2880, None, None),  # gpt-oss gate_up: pads a
    ([1024, 0, 512, 256], 1408, 2048, None, None),  # deepseek-v2-lite (11 blocks), empty group
    ([256, 384], 2048, 1408, None, None),
    ([640, 128, 0, 256], 768, 2048, None, None),
    ([256, 0, 384], 384, 256, None, None),  # transposed, small
    # padded operands (more channels than d), as the fused ops lay them out
    ([256, 0, 384], 2880, 2880, 3072, None),  # a padded to a multiple of 256
    ([256, 0, 384], 2880, 2880, None, 3072),  # only b padded: transposed temporary
    ([256, 0, 384], 2880, 2880, 3072, 3072),
    ([256, 0, 384], 5760, 2880, 5888, 3072),
    ([256, 0, 384], 2880, 1024, 2944, None),  # a padded to an odd block count
    ([256, 0, 384], 2048, 768, None, 1024),
    ([256, 0, 384], 1408, 2048, 1536, None),
]


def needs_guard(N: int, NA: int | None) -> bool:
    return blocks(NA or N) % 2 == 1


@pytest.mark.parametrize("ks,N,K,NA,KB", KGROUPED_CASES)
def test_k_grouped_gemm(ks: list[int], N: int, K: int, NA: int | None, KB: int | None) -> None:
    if not needs_guard(N, NA):
        run_kgrouped_case(ks, N, K, NA, KB)
        return
    args = json.dumps([ks, N, K, NA, KB])
    try:
        r = subprocess.run(
            [sys.executable, "-c", SUBPROCESS, os.path.abspath(__file__), args],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"k_grouped_fp8_gemm hung for (N, K, NA, KB) = ({N}, {K}, {NA}, {KB}), {ks}")
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


# ---------------------------------------------------------------------------
# Dense: fp8_gemm_nt, fp8_act_weight_gemm, fp8_dgrad_wgrad
# ---------------------------------------------------------------------------

# (M, K, N): the dense projections of the models plus M not a multiple of 128 (all multiples of 16)
DENSE_SHAPES = [
    (4096, 2048, 10944),  # deepseek-v2-lite dense MLP up
    (1008, 10944, 2048),  # deepseek-v2-lite dense MLP down
    (4096, 2048, 4096),  # qwen3 q_proj
    (4096, 2048, 512),  # qwen3 k/v_proj
    (1008, 2880, 4096),  # gpt-oss q_proj
    (144, 4096, 2880),  # gpt-oss o_proj
    (1008, 2000, 1040),
    (144, 400, 272),
    (16, 128, 16),
]


def dense_case(M: int, K: int, N: int):
    x = make_input(M, K, seed=M + K + N)
    w = (make_input(N, K, seed=N).float() * 0.02).to(torch.bfloat16)
    dy = make_input(M, N, seed=K)
    return x, w, dy


@pytest.mark.parametrize("M,K,N", DENSE_SHAPES)
def test_fp8_gemm_nt(M: int, K: int, N: int) -> None:
    x, w, _ = dense_case(M, K, N)
    a_fp8, a_scale, _, _ = fp8cast_rowwise_blockwise_transpose(x)
    w_fp8, w_scale, _, _ = fp8cast_blockwise_transpose(w)
    # bf16 output only: DeepGEMM's SM90 kernel asserts d is bf16 (every caller passes bf16)
    out = fp8_gemm_nt(a_fp8, a_scale, w_fp8, w_scale, M, N, torch.device(DEV), torch.bfloat16)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    ref = dq_rowwise(a_fp8, a_scale) @ dq_blockwise(w_fp8, w_scale).T
    assert calc_diff(out, ref) < 1e-3


@pytest.mark.parametrize("M,K,N", DENSE_SHAPES)
def test_act_weight_gemm_and_dgrad_wgrad(M: int, K: int, N: int) -> None:
    x, w, dy = dense_case(M, K, N)
    w_fp8, w_scale, wt_fp8, wt_scale = fp8cast_blockwise_transpose(w)

    out, xt_fp8, xt_scale = fp8_act_weight_gemm(x, w_fp8, w_scale)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    xq, xs = ref_rowwise(x)
    ref = dq_rowwise(xq, xs) @ dq_blockwise(w_fp8, w_scale).T
    assert calc_diff(out, ref) < 1e-3
    # the transposed activation is the 128x128 blockwise FP8 of x.T
    rq, rs = ref_blockwise(x.T)
    assert_fp8_equal(xt_fp8, rq, "input_t_fp8")
    assert_scale_equal(xt_scale, rs, "scale_input_t")

    grad_input, weight_grad = fp8_dgrad_wgrad(dy, wt_fp8, wt_scale, xt_fp8, xt_scale, K)
    assert grad_input.shape == (M, K) and grad_input.dtype == torch.bfloat16
    assert weight_grad.shape == (N, K) and weight_grad.dtype == torch.bfloat16
    gq, gs = ref_rowwise(dy)
    ref_gi = dq_rowwise(gq, gs) @ dq_blockwise(wt_fp8, wt_scale).T
    assert calc_diff(grad_input, ref_gi) < 1e-3
    gtq, gts = ref_rowwise(dy.T.contiguous())
    ref_wg = dq_rowwise(gtq, gts) @ dq_blockwise(xt_fp8, xt_scale).T
    assert calc_diff(weight_grad, ref_wg) < 1e-3
    # and close to the unquantized products
    assert calc_diff(grad_input, dy.float() @ w.float()) < 1e-2
    assert calc_diff(weight_grad, dy.float().T @ x.float()) < 1e-2
