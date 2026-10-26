"""
DeepGEMM FP8 GEMMs and the recipes built on them: dense (fp8_gemm_nt, fp8_act_weight_gemm,
fp8_dgrad_wgrad), m-grouped (fp8_m_grouped_gemm, m_grouped_group_indices) and k-grouped
(k_grouped_fp8_gemm, fp8_grouped_wgrad_buffer) GEMMs, plus the grouped-input quantization
(quantize_grouped_input). The architecture-specific entry points are chosen here.

Every 1x128 activation scale is fp32 of logical shape (rows, ceil(K / 128)) with unspecified
strides (MN-major on SM90, row-major on SM100); the GEMMs take either.
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
    "K_GROUPED_MAX_ROWS", "fp8_act_weight_gemm", "fp8_dgrad_wgrad", "fp8_fused_experts_supported", "fp8_gemm_nt", "fp8_grouped_wgrad_buffer", "fp8_m_grouped_gemm", "k_grouped_fp8_gemm",
    "k_grouped_fp8_gemm_nt", "k_grouped_runs_transposed", "k_grouped_safe_size", "m_grouped_fp8_gemm_nt", "m_grouped_group_indices", "pad_k_major_channels", "quantize_grouped_input"
]


def fp8_gemm_nt(a_fp8: torch.Tensor, a_scale: torch.Tensor, b_fp8: torch.Tensor, b_scale: torch.Tensor, m: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    out = a @ b.T into a fresh (m, n) tensor of dtype on device, fp32 accumulation. dtype must be
    bf16 (DeepGEMM's SM90 kernel rejects others).

    - a_fp8: e4m3 (m, K) row-major; a_scale: its 1x128 scale, fp32 (m, ceil(K / 128)), any strides.
    - b_fp8: e4m3 (n, K) row-major; b_scale: its 128x128 scale, fp32 (ceil(n / 128), ceil(K / 128))
      row-major, as fp8cast_blockwise_transpose returns it.
    """
    out = torch.empty((m, n), device=device, dtype=dtype)
    deep_gemm.fp8_fp4_gemm_nt((a_fp8, a_scale), (b_fp8, b_scale), out)
    return out


def fp8_act_weight_gemm(input_2d: torch.Tensor, weight_fp8: torch.Tensor, scale_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward FP8 linear y = x @ W.T, returning (output, input_t_fp8, scale_input_t).

    - input_2d: x, bf16 (m, K), quantized rowwise 1x128.
    - weight_fp8, scale_weight: W, e4m3 (n, K) with its 128x128 scale as fp8cast_blockwise_transpose
      returns it.
    - output: fresh (m, n) in input_2d's dtype.
    - input_t_fp8, scale_input_t: x.T (K, m) e4m3 quantized per 128x128 tile, scale fp32
      (ceil(K / 128), ceil(m / 128)) row-major; the activation operand of fp8_dgrad_wgrad.
    """
    m, n = input_2d.shape[0], weight_fp8.shape[0]
    input_fp8, scale_input, input_t_fp8, scale_input_t = fp8cast_rowwise_blockwise_transpose(input_2d)
    output = fp8_gemm_nt(input_fp8, scale_input, weight_fp8, scale_weight, m, n, input_2d.device, input_2d.dtype)
    return output, input_t_fp8, scale_input_t


def fp8_dgrad_wgrad(grad_2d: torch.Tensor, weight_t_fp8: torch.Tensor, scale_weight_t: torch.Tensor, input_t_fp8: torch.Tensor, scale_input_t: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward FP8 linear of y = x @ W.T, returning (grad_input, weight_grad).

    - grad_2d: the gradient of y, bf16 (m, n).
    - weight_t_fp8, scale_weight_t: W.T, e4m3 (k, n) with its 128x128 scale (the transpose pair of
      fp8cast_blockwise_transpose).
    - input_t_fp8, scale_input_t: x.T (k, m) blockwise, as fp8_act_weight_gemm returns them.
    - k: the input features.
    - grad_input = grad @ W, fresh (m, k); weight_grad = grad.T @ x, fresh (n, k) (not accumulated);
      both in grad_2d's dtype.
    """
    m, n = grad_2d.shape
    grad_fp8, scale_grad, grad_t_fp8, scale_grad_t = fp8cast_rowwise_transpose(grad_2d)
    grad_input = fp8_gemm_nt(grad_fp8, scale_grad, weight_t_fp8, scale_weight_t, m, k, grad_2d.device, grad_2d.dtype)
    weight_grad = fp8_gemm_nt(grad_t_fp8, scale_grad_t, input_t_fp8, scale_input_t, n, k, grad_2d.device, grad_2d.dtype)
    return grad_input, weight_grad


def m_grouped_fp8_gemm_nt(a, b, d, grouped_mm_offs, M, group_indices=None):
    """m-grouped FP8 GEMM NT into d; on SM90 group_indices is computed when None."""
    if ARCH_MAJOR >= 10:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_mm_offs, use_psum_layout=True, expected_m_for_psum_layout=M)
    else:
        if group_indices is None:
            group_indices = m_grouped_group_indices(grouped_mm_offs, M)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, group_indices)


K_GROUPED_MAX_ROWS = 65536
"""
Rows per call of DeepGEMM's SM90 k-grouped FP8 GEMM: it fails to launch (CUDA_ERROR_INVALID_VALUE)
once all groups' rows reach 116224, which imbalanced routing reaches, so larger calls are split.
"""


def pad_k_major_channels(x: tuple[torch.Tensor, torch.Tensor], ks: list, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A K-major grouped operand (flat data of (n, k_g) row-major blocks, scales (n, rows / 128))
    re-laid out with channels >= n; the extra channels are zero data with unit scales.
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
    out_scale = torch.ones((blocks, channels), device=scale.device, dtype=scale.dtype).T
    out_scale[:n].copy_(scale)
    return out, out_scale


def k_grouped_runs_transposed(m: int, n: int) -> bool:
    """
    Whether the SM90 k-grouped FP8 GEMM of an (m, n) output runs as the transposed product (see
    _k_grouped_call): an odd 128-block count on m and an even one on n.
    """
    return -(-m // 128) % 2 == 1 and -(-n // 128) % 2 == 0


def _k_grouped_call(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], d: torch.Tensor, ks: list, ks_tensor: torch.Tensor) -> None:
    """
    One deep_gemm.k_grouped_fp8_gemm_nt_contiguous call accumulating into d, made safe from a
    DeepGEMM SM90 deadlock: the kernel runs 2-CTA clusters along M (block_m = 128, cluster_m = 2)
    and never finishes when ceil(M / 128) * ceil(N / block_n) is odd, M being the first operand's
    channels (e.g. (M, N) = (2880, 2880), (2880, 1024), (5760, 2880) hang). An even M-block count
    is always safe, so when a's is odd the call computes the transposed product b a^T if b's is
    even, and otherwise pads a's channels to a multiple of 256 with zeros.
    """
    m, n = a[1].shape[0], b[1].shape[0]
    if -(-m // 128) % 2 == 0:
        deep_gemm.k_grouped_fp8_gemm_nt_contiguous(a, b, d, ks, ks_tensor, c=d)
    elif k_grouped_runs_transposed(m, n):
        # fp8_grouped_wgrad_buffer stores d transposed for this case, so it accumulates in place.
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


def k_grouped_fp8_gemm_nt(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], d: torch.Tensor, ks: list, ks_tensor: torch.Tensor, c: torch.Tensor) -> None:
    """
    SM90 k-grouped FP8 GEMM NT accumulating into d (d is c), split into calls of at most
    K_GROUPED_MAX_ROWS rows, each through _k_grouped_call. Runs of whole groups are contiguous
    slices of the K-major operands; a group longer than the limit is cut along k and accumulates
    chunk by chunk. Scale slices keep the parent's MN-major strides, so DeepGEMM reads them as is.
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
        sub_t = ks_tensor[g0:g1] if chunk is None else torch.tensor(sub_ks, dtype=ks_tensor.dtype, device=ks_tensor.device)
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


def k_grouped_fp8_gemm(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], d: torch.Tensor, ks: list, ks_tensor: torch.Tensor) -> None:
    """
    FP8 grouped weight gradient, accumulated in place: d[g] += a_g^T b_g in fp32 for each group g
    of ks rows.

    - d: fp32 (G, N, K), contiguous or as fp8_grouped_wgrad_buffer returns it.
    - a, b: the per-channel FP8 data and scales (x_ch_fp8, x_ch_scale) of quantize_grouped_input
      (or of the fused quantization ops) of the grad_output (N channels) and of the input (K
      channels), over the same grouped rows.
    - ks (host list) and ks_tensor (int32, device): per-group row counts, multiples of 128.
    - Either operand may have more channels than d (zero-padded, as the fused ops' padded=True
      writes it); the extra channels are dropped. Only the SM90 fused ops produce such operands.
    """
    if ARCH_MAJOR >= 10:
        deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c=d)
        return
    G, N, K = d.shape
    NA, KB = a[1].shape[0], b[1].shape[0]
    if (NA, KB) == (N, K):
        k_grouped_fp8_gemm_nt(a, b, d, ks, ks_tensor, c=d)
        return
    if NA % 256 != 0 and KB % 256 == 0:
        c = torch.zeros((G, KB, NA), device=d.device, dtype=torch.float32)
        k_grouped_fp8_gemm_nt(b, a, c, ks, ks_tensor, c=c)
        d.add_(c[:, :K, :N].transpose(1, 2))
    else:
        c = torch.zeros((G, NA, KB), device=d.device, dtype=torch.float32)
        k_grouped_fp8_gemm_nt(a, b, c, ks, ks_tensor, c=c)
        d.add_(c[:, :N, :K])


def fp8_grouped_wgrad_buffer(size: torch.Size, device: torch.device) -> torch.Tensor:
    """
    Uninitialized fp32 main_grad of shape size for an FP8 grouped weight (G, N, K), laid out so
    k_grouped_fp8_gemm accumulates into it in place: a transposed view of (G, K, N) storage where
    SM90 computes the product transposed, otherwise contiguous.
    """
    if len(size) == 3 and is_sm90():
        g, n, k = size
        if k_grouped_runs_transposed(n, k):
            return torch.empty((g, k, n), dtype=torch.float32, device=device).transpose(1, 2)
    return torch.empty(size, dtype=torch.float32, device=device)


def fp8_fused_experts_supported() -> bool:
    """
    Whether the fused FP8 expert ops run on this device (the activation-fused quantizations,
    scatter_quant, the fused dense SwiGLU MLP): True on SM90 only.
    """
    return is_sm90()


def quantize_grouped_input(input: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The FP8 operands of a grouped-GEMM input from one read of it: (x_fp8, x_scale, x_ch_fp8,
    x_ch_scale).

    - input: bf16 (M, K) contiguous, rows grouped by expert, each group padded to a multiple of 128
      rows; grouped_mm_offs: int32 cumulative group ends (last = M).
    - x_fp8 e4m3 (M, K), x_scale fp32 (M, ceil(K / 128)): rowwise 1x128, the a operand of
      fp8_m_grouped_gemm.
    - x_ch_fp8, x_ch_scale: per-channel (128 rows x 1 within each group), the operand of
      k_grouped_fp8_gemm. SM90: K-major per group (group g's (K, ks[g]) row-major block flat at
      offs[g-1] * K), scale (K, M / 128). SM100: (M, K), scale (M / 128, K).

    Per 128-element block, scale = the power of two at or above max(amax, 1e-4) / 448 (at least
    2^-126) and fp8 = e4m3_satfinite(x / scale).
    """
    if ARCH_MAJOR >= 10:
        return fp8cast_rowwise_colwise(input)
    return fp8cast_rowwise_kmajor(input, grouped_mm_offs)


def fp8_m_grouped_gemm(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor], n: int, grouped_mm_offs: torch.Tensor, group_indices: torch.Tensor | None = None) -> torch.Tensor:
    """
    m-grouped FP8 GEMM into a fresh bf16 (M, n) tensor: rows [offs[g-1], offs[g]) are a_g @ b[g].T,
    fp32 accumulation.

    - a: (fp8 (M, K), 1x128 scale (M, ceil(K / 128))) as quantize_grouped_input returns them.
    - b: (fp8 (G, n, K), 128x128 scale (G, ceil(n / 128), ceil(K / 128))) as
      fp8cast_blockwise_transpose_batched returns them.
    - grouped_mm_offs: int32 (G,) cumulative group ends, each group a multiple of 128 rows.
    - group_indices: optional hint from m_grouped_group_indices, computed when None.
    """
    M = a[0].shape[0]
    out = torch.empty((M, n), device=a[0].device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt(a, b, out, grouped_mm_offs, M, group_indices=group_indices)
    return out


def k_grouped_safe_size(n: int) -> int:
    """
    Channel count (a multiple of 256) to lay out a k-grouped wgrad operand with so DeepGEMM's SM90
    k-grouped GEMM can take it as the first operand without the 2-CTA deadlock (see
    _k_grouped_call).
    """
    return -(-n // 256) * 256


def m_grouped_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor | None:
    """
    The group-id hint of fp8_m_grouped_gemm, reusable by every m-grouped GEMM over the same rows;
    None on SM100, which takes none.

    On SM90: int32 (M,) per-row group ids, e.g. grouped_mm_offs = [128, 256, 384, 512] gives
    [0, ..., 1, ..., 2, ..., 3, ...]; rows at or past the last offset get the last group.
    """
    if not is_sm90():
        return None
    from pithtrain.kernels import group_indices_sm90 as cgi

    if cgi.supports_group_indices(grouped_mm_offs, M):
        return cgi.group_indices(grouped_mm_offs, M)
    row_indices = torch.arange(M, device=grouped_mm_offs.device)
    gi = torch.searchsorted(grouped_mm_offs, row_indices, right=True).to(torch.int32)
    gi.clamp_(max=grouped_mm_offs.shape[0] - 1)
    return gi
