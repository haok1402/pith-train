"""
Fused rowwise + colwise FP8 quantization from one read of the input (Triton, run on both
architectures). fp8_rowwise_kmajor_sm100 launches the same kernel with the K-major colwise output.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_rowwise_colwise", "fp8cast_rowwise_colwise_kernel"]


@triton.jit
def fp8cast_rowwise_colwise_kernel(x_ptr, out_tok_ptr, scale_tok_ptr, out_ch_ptr, scale_ch_ptr, M, N, stride_x_row, stride_x_col, stride_out_tok_row, scale_tok_cols, stride_out_ch_row, scale_ch_cols, block_to_group_ptr, cumsum_ptr, BLOCK_N: tl.constexpr, NATIVE_E8M0: tl.constexpr, WRITE_KMAJOR: tl.constexpr = False):
    """
    One 128 x BLOCK_N tile per program: 1x128 rowwise and 128x1 colwise quantization. M % 128 == 0.

    With WRITE_KMAJOR the colwise output is written K-major (each group's (K_g, N) block stored as
    (N, K_g), concatenated) with scales (N, M / 128); this needs block_to_group_ptr and cumsum_ptr
    (grouped_mm_offs). Otherwise colwise is (M, N) with scales (M / 128, N).
    """
    BLOCK_ROWS: tl.constexpr = 128
    CHUNKS: tl.constexpr = BLOCK_N // BLOCK_ROWS

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK_ROWS
    col_start = pid_col * BLOCK_N

    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # amax runs on the input dtype; only the multiply is widened to fp32
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    abs_reshaped = tl.reshape(abs_bf16, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_amax = tl.max(abs_reshaped, axis=1)  # (128 * CHUNKS,)
    tok_scale, tok_rcp = compute_fp8_scale(tok_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    x_reshaped = tl.reshape(x_f32, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_fp8_flat = (x_reshaped * tok_rcp[:, None]).to(tl.float8e4nv)
    tok_fp8 = tl.reshape(tok_fp8_flat, BLOCK_ROWS, BLOCK_N)

    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    tok_scale_2d = tl.reshape(tok_scale, BLOCK_ROWS, CHUNKS)
    tok_col_offs = pid_col * CHUNKS + tl.arange(0, CHUNKS)
    tok_scale_ptrs = scale_tok_ptr + row_offs[:, None] * scale_tok_cols + tok_col_offs[None, :]
    tok_scale_mask = (row_offs[:, None] < M) & (tok_col_offs[None, :] < scale_tok_cols)
    tl.store(tok_scale_ptrs, tok_scale_2d, mask=tok_scale_mask)

    ch_amax = tl.max(abs_bf16, axis=0)  # (BLOCK_N,)
    ch_scale, ch_rcp = compute_fp8_scale(ch_amax, NATIVE_E8M0)

    ch_fp8 = (x_f32 * ch_rcp[None, :]).to(tl.float8e4nv)

    ch_scale_mask = col_offs < N

    if WRITE_KMAJOR:
        # Groups are 128-aligned, so each pid_row belongs to exactly one group.
        group_id = tl.load(block_to_group_ptr + pid_row).to(tl.int64)
        group_end = tl.load(cumsum_ptr + group_id).to(tl.int64)
        group_start = tl.load(cumsum_ptr + group_id - 1, mask=group_id > 0, other=0).to(tl.int64)
        k_g = group_end - group_start
        k_offset = row_start - group_start

        flat_base = group_start * N
        row_local = tl.arange(0, BLOCK_ROWS).to(tl.int64)
        out_offs = (flat_base + col_offs[None, :].to(tl.int64) * k_g + (k_offset + row_local)[:, None])
        tl.store(out_ch_ptr + out_offs, ch_fp8, mask=mask)

        scale_ch_ptrs = scale_ch_ptr + col_offs * scale_ch_cols + pid_row
        tl.store(scale_ch_ptrs, ch_scale, mask=ch_scale_mask)
    else:
        out_ch_ptrs = out_ch_ptr + row_offs[:, None] * stride_out_ch_row + col_offs[None, :]
        tl.store(out_ch_ptrs, ch_fp8, mask=mask)

        scale_ch_ptrs = scale_ch_ptr + pid_row * scale_ch_cols + col_offs
        tl.store(scale_ch_ptrs, ch_scale, mask=ch_scale_mask)


def fp8cast_rowwise_colwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise and colwise FP8 quantization of x from one read. x is bf16 or fp32 (M, N) with M %
    128 == 0.

    Returns (tok_fp8, tok_scale, ch_fp8, ch_scale): tok_fp8 float8_e4m3fn (M, N) with 1x128 row
    blocks and tok_scale float32 (M, ceil(N / 128)); ch_fp8 float8_e4m3fn (M, N) with 128x1 column
    blocks and ch_scale float32 (M / 128, N). Scales are row-major.
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (f"Expected 2D tensor with M%128==0, got shape {x.shape}")
    M, N = x.shape
    BLOCK = 128

    num_row_groups = M // BLOCK
    scale_tok_cols = (N + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_ch = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_ch = torch.empty((num_row_groups, N), dtype=torch.float32, device=x.device)

    _BLOCK_N = 128
    grid = (num_row_groups, (N + _BLOCK_N - 1) // _BLOCK_N)

    fp8cast_rowwise_colwise_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_ch_ptr=out_ch, scale_ch_ptr=scale_ch, M=M, N=N, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=N, scale_tok_cols=scale_tok_cols, stride_out_ch_row=N, scale_ch_cols=N, block_to_group_ptr=out_tok, cumsum_ptr=out_tok, BLOCK_N=_BLOCK_N, NATIVE_E8M0=NATIVE_E8M0, WRITE_KMAJOR=False, num_warps=4, num_stages=2)

    return out_tok, scale_tok, out_ch, scale_ch
