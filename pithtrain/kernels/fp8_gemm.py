"""
DeepGEMM FP8 GEMM launchers and the recipes built on them, shared by the FP8 linear layers, the
FP8 expert ops and the MLA pass-latent ring attention so each GEMM convention has one source.

- Dense: :func:`fp8_gemm_nt` (``out = a @ b.T``) and the linear recipes
  :func:`fp8_act_weight_gemm` (forward) / :func:`fp8_dgrad_wgrad` (backward).
- m-grouped (expert forward / dgrad): :func:`m_grouped_fp8_gemm_nt`, :func:`fp8_m_grouped_gemm`,
  with the per-row group ids SM90 needs from :func:`m_grouped_group_indices`.
- k-grouped (expert wgrad): :func:`k_grouped_fp8_gemm_nt`, which splits calls above
  ``K_GROUPED_MAX_ROWS`` and avoids DeepGEMM's SM90 2-CTA-cluster deadlock (see
  :func:`k_grouped_runs_transposed`, :func:`k_grouped_safe_size`).
- Grouped-GEMM input prep: :func:`quantize_grouped_input`.

DeepGEMM picks its SM90 or SM100 kernels itself; the m- / k-grouped entry points differ by
architecture and are chosen here with ``ARCH_MAJOR``.
"""

from typing import Tuple

import deep_gemm
import torch

from pithtrain.kernels.arch import ARCH_MAJOR, is_sm90
from pithtrain.kernels.fp8_rowwise_blockwise_transpose import fp8cast_rowwise_blockwise_transpose
from pithtrain.kernels.fp8_rowwise_colwise import fp8cast_rowwise_colwise
from pithtrain.kernels.fp8_rowwise_kmajor import fp8cast_rowwise_kmajor
from pithtrain.kernels.fp8_rowwise_transpose import fp8cast_rowwise_transpose

__all__ = [
    "K_GROUPED_MAX_ROWS",
    "fp8_act_weight_gemm",
    "fp8_dgrad_wgrad",
    "fp8_gemm_nt",
    "fp8_m_grouped_gemm",
    "k_grouped_fp8_gemm_nt",
    "k_grouped_runs_transposed",
    "k_grouped_safe_size",
    "m_grouped_fp8_gemm_nt",
    "m_grouped_group_indices",
    "pad_k_major_channels",
    "quantize_grouped_input",
]


def fp8_gemm_nt(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    m: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``out[m, n] = a @ b.T`` via the DeepGEMM fp8 NT GEMM. The single place that owns the
    allocate-then-``fp8_fp4_gemm_nt`` convention shared by the FP8 linear layer and the MLA
    pass-latent ring decompress. ``a_scale`` (1x128) is used as is when MN-major, as the SM90
    quantization kernels return it, and transposed by DeepGEMM otherwise; ``b_scale`` (128x128)
    is always used as is."""
    out = torch.empty((m, n), device=device, dtype=dtype)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
    return out


def fp8_act_weight_gemm(
    input_2d: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 forward GEMM ``y = x @ W.T``: rowwise-quantize the activation, run the NT GEMM, and
    return ``(output, input_t_fp8, scale_input_t)`` -- the transposed activation is handed back
    so a later wgrad can reuse it. Shared by ``_fp8_linear_fwd`` and the MLA ring so the recipe
    has one source of truth."""
    m, n = input_2d.shape[0], weight_fp8.shape[0]
    input_fp8, scale_input, input_t_fp8, scale_input_t = fp8cast_rowwise_blockwise_transpose(
        input_2d
    )
    output = fp8_gemm_nt(
        input_fp8, scale_input, weight_fp8, scale_weight, m, n, input_2d.device, input_2d.dtype
    )
    return output, input_t_fp8, scale_input_t


def fp8_dgrad_wgrad(
    grad_2d: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP8 backward GEMMs: ``grad_input = grad @ W`` (via the transposed weight) and
    ``weight_grad = grad.T @ x`` (via the transposed activation), both returned in ``grad_2d``'s
    dtype. Shared by ``_fp8_linear_bwd`` and the MLA ring so dgrad/wgrad have one source of
    truth; the caller supplies ``input_t_fp8`` (saved from the forward, or recomputed)."""
    m, n = grad_2d.shape
    grad_fp8, scale_grad, grad_t_fp8, scale_grad_t = fp8cast_rowwise_transpose(grad_2d)
    grad_input = fp8_gemm_nt(
        grad_fp8, scale_grad, weight_t_fp8, scale_weight_t, m, k, grad_2d.device, grad_2d.dtype
    )
    weight_grad = fp8_gemm_nt(
        grad_t_fp8, scale_grad_t, input_t_fp8, scale_input_t, n, k, grad_2d.device, grad_2d.dtype
    )
    return grad_input, weight_grad


def m_grouped_fp8_gemm_nt(a, b, d, grouped_mm_offs, M, group_indices=None):
    """Dispatch m_grouped FP8 GEMM NT to the right API for the current GPU arch."""
    if ARCH_MAJOR >= 10:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            b,
            d,
            grouped_mm_offs,
            use_psum_layout=True,
            expected_m_for_psum_layout=M,
        )
    else:
        assert group_indices is not None
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, group_indices)


K_GROUPED_MAX_ROWS = 65536
"""
Rows per call of DeepGEMM's SM90 k-grouped FP8 GEMM. The kernel fails to launch
(CUDA_ERROR_INVALID_VALUE) once the rows of all groups reach 116224 whatever N and K are, which
imbalanced routing reaches (an EP rank receiving ~3.5x its share), so larger calls are split.
"""


def pad_k_major_channels(
    x: tuple[torch.Tensor, torch.Tensor], ks: list, channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A K-major grouped operand (flat data of (n, k_g) row-major blocks, scales (n, rows/128) with
    any strides) re-laid out with ``channels`` >= n channels; the extra channels are zero data
    with unit scales, so they contribute exact zeros to the product.
    """
    data, scale = x
    n, blocks = scale.shape
    rows = sum(ks)
    out = data.new_zeros(channels * rows)
    row = 0
    for k in ks:
        if k:
            dst = out[row * channels : (row + k) * channels].view(channels, k)
            dst[:n].copy_(data[row * n : (row + k) * n].view(n, k))
        row += k
    # MN-major (column-major) like the quantization kernels write them.
    out_scale = torch.ones((blocks, channels), device=scale.device, dtype=scale.dtype).T
    out_scale[:n].copy_(scale)
    return out, out_scale


def k_grouped_runs_transposed(m: int, n: int) -> bool:
    """
    Whether the SM90 k-grouped FP8 GEMM of an (m, n) output (m the first operand's channels) runs as
    the transposed product in :func:`_k_grouped_call`: an odd 128-block count on m (which would
    deadlock) and an even one on n. ``install_wgrad_accumulation`` lays out main_grad for it.
    """
    return -(-m // 128) % 2 == 1 and -(-n // 128) % 2 == 0


def _k_grouped_call(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
) -> None:
    """
    One deep_gemm.k_grouped_fp8_gemm_nt_contiguous call accumulating into d (c is d), made safe
    from a DeepGEMM SM90 deadlock: the kernel runs 2-CTA clusters along M (block_m = 128,
    cluster_m = 2) and never finishes when ceil(M / 128) * ceil(N / block_n) is odd, M being the
    first operand's channels (found by the gptoss agent: e.g. (M, N) = (2880, 2880), (2880, 1024),
    (5760, 2880) hang; workspace/kda-tasks/round3/gptoss/docs/orchestrator_diffs.md). An even
    M-block count is always safe, so when a's is odd the call computes the transposed product
    b a^T if b's is even, and otherwise pads a's channels to a multiple of 256 with zeros. The
    result is the same sum of the same FP8 products; padded rows of the output are discarded.
    """
    m, n = a[1].shape[0], b[1].shape[0]
    if -(-m // 128) % 2 == 0:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c=d)
    elif k_grouped_runs_transposed(m, n):
        # d is usually a main_grad that install_wgrad_accumulation stores transposed for exactly
        # this case, so its transpose is contiguous and accumulates in place, copy-free.
        dt = d.transpose(1, 2)
        if dt.is_contiguous():
            deep_gemm.k_grouped_fp8_gemm_nt_contiguous(b, a, dt, ks, ks_tensor, c=dt)
            return
        dt = dt.contiguous()
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(b, a, dt, ks, ks_tensor, c=dt)
        d.copy_(dt.transpose(1, 2))
    else:
        mp = -(-m // 256) * 256
        dp = d.new_zeros((d.shape[0], mp, d.shape[2]))
        dp[:, :m].copy_(d)
        ap = pad_k_major_channels(a, ks, mp)
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(ap, b, dp, ks, ks_tensor, c=dp)
        d.copy_(dp[:, :m])


def k_grouped_fp8_gemm_nt(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    c: torch.Tensor,
) -> None:
    """
    deep_gemm.k_grouped_fp8_gemm_nt_contiguous with c accumulated into d (d is c), split into calls
    of at most K_GROUPED_MAX_ROWS rows, each made through _k_grouped_call (deadlock guard). Each group's operand is a (channels, k) row-major block of
    the flat K-major buffer and its scales a column range of (channels, rows/128), so runs of whole
    groups are contiguous slices; a group longer than the limit is cut along k and accumulates
    chunk by chunk into its slice of d. Scale slices keep the parent's strides, so MN-major scales
    (kernels/fp8_scale_layout.py) stay in the layout DeepGEMM reads without a transpose.
    """
    assert d is c
    if sum(ks) <= K_GROUPED_MAX_ROWS:
        _k_grouped_call(a, b, d, ks, ks_tensor)
        return
    channels = (a[1].shape[0], b[1].shape[0])

    def call(g0: int, g1: int, row0: int, row1: int, chunk: tuple[int, int] | None) -> None:
        ops = []
        for (data, scale), n in zip((a, b), channels):
            if chunk is None:
                data = data[row0 * n : row1 * n]
                scale = scale[:, row0 // 128 : row1 // 128]
            else:
                k, (r0, r1) = row1 - row0, chunk
                data = data[row0 * n : row1 * n].view(n, k)[:, r0:r1].contiguous().view(-1)
                scale = scale[:, (row0 + r0) // 128 : (row0 + r1) // 128]
            ops.append((data, scale))
        sub_ks = ks[g0:g1] if chunk is None else [chunk[1] - chunk[0]]
        sub_t = ks_tensor[g0:g1] if chunk is None else torch.tensor(sub_ks, dtype=ks_tensor.dtype, device=ks_tensor.device)  # fmt: skip
        out = d[g0:g1]
        _k_grouped_call(ops[0], ops[1], out, sub_ks, sub_t)

    g0, row0, row = 0, 0, 0
    for g, k in enumerate(ks):
        if k > K_GROUPED_MAX_ROWS:
            if g > g0:
                call(g0, g, row0, row, None)
            for r0 in range(0, k, K_GROUPED_MAX_ROWS):
                call(g, g + 1, row, row + k, (r0, min(r0 + K_GROUPED_MAX_ROWS, k)))
            g0, row0 = g + 1, row + k
        elif row + k - row0 > K_GROUPED_MAX_ROWS:
            call(g0, g, row0, row, None)
            g0, row0 = g, row
        row += k
    if len(ks) > g0:
        call(g0, len(ks), row0, row, None)


def quantize_grouped_input(
    input: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a grouped-GEMM input for FP8GroupedLinear: per-token FP8 for the m-grouped forward and
    per-channel FP8 for the k-grouped wgrad, from one read. On Hopper the per-channel output is
    written directly in K-major layout.

    Projections that share an input (gate_proj and up_proj) can quantize it once and pass the
    result to each as quantized_input; the bytes are the same either way.
    """
    if ARCH_MAJOR >= 10:
        return fp8cast_rowwise_colwise(input)
    return fp8cast_rowwise_kmajor(input, grouped_mm_offs)


def fp8_m_grouped_gemm(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    n: int,
    grouped_mm_offs: torch.Tensor,
    group_indices: torch.Tensor | None,
) -> torch.Tensor:
    """m-grouped FP8 GEMM ``a @ b.T`` per group into a fresh bf16 (M, n) tensor."""
    M = a[0].shape[0]
    out = torch.empty((M, n), device=a[0].device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt(a, b, out, grouped_mm_offs, M, group_indices=group_indices)
    return out


def k_grouped_safe_size(n: int) -> int:
    """
    Channel count to lay out a k-grouped wgrad operand with so DeepGEMM can take it as the first
    (M-side) operand: a multiple of 256.

    DeepGEMM's SM90 k-grouped FP8 GEMM deadlocks (the kernel never finishes, so the next launch
    or sync hangs) whenever ceil(M / 128) * ceil(N / block_n) is odd: it runs 2-CTA clusters
    (cluster_m = 2) and an odd tile count leaves one CTA without its partner. block_n depends on
    the problem (136 or 152 for gpt-oss), but block_m is 128, so an even M-block count is always
    safe. gpt-oss hits it with its 2880-wide hidden and intermediate (23 M-blocks; 19 or 7
    N-blocks at block_n = 152), measured on H200 with workspace/kda-tasks/round3/gptoss/scripts/
    kg_case.py: e.g. (M, N) = (2880, 2880), (2880, 1024), (5760, 2880) hang, (2880, 2944),
    (2944, 2944), (1024, 2880), (2880, 5760) complete.
    """
    return -(-n // 256) * 256


def m_grouped_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor:
    """
    Per-row group ids of an m-grouped GEMM input (what DeepGEMM's SM90 m-grouped GEMM takes), e.g.
    ``grouped_mm_offs = [128, 256, 384, 512]`` -> ``[0,0,...,1,1,...,2,2,...,3,3,...]``. On SM90 the
    one-launch CuTe kernel (``group_indices_sm90``) computes it for the offsets it supports;
    otherwise ``arange`` + ``searchsorted`` + ``clamp`` give the same values.
    """
    if is_sm90():
        from pithtrain.kernels import group_indices_sm90 as cgi

        if cgi.supports_group_indices(grouped_mm_offs, M):
            return cgi.group_indices(grouped_mm_offs, M)  # same values, one launch
    row_indices = torch.arange(M, device=grouped_mm_offs.device)
    gi = torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)
    gi.clamp_(max=grouped_mm_offs.shape[0] - 1)
    return gi
