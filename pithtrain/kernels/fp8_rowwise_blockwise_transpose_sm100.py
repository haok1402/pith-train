"""
Triton rowwise FP8 quantization of x and 128x128 blockwise FP8 quantization of x.T from one read of
x (the SM100 path and the SM90 kernel's reference).
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_rowwise_blockwise_transpose_sm100"]


@triton.jit
def _fp8cast_rowwise_blockwise_transpose_kernel(x_ptr, out_tok_ptr, scale_tok_ptr, out_blk_t_ptr, scale_blk_t_ptr, M, K, stride_x_row, stride_x_col, stride_out_tok_row, stride_scale_tok_row, stride_out_blk_t_row, scale_blk_t_cols, NATIVE_E8M0: tl.constexpr):
    """One 128x128 tile of x per program."""
    BLOCK: tl.constexpr = 128

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK
    col_start = pid_col * BLOCK

    row_offs = row_start + tl.arange(0, BLOCK)
    col_offs = col_start + tl.arange(0, BLOCK)

    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    tok_amax = tl.max(abs_bf16, axis=1)
    tok_scale, tok_rcp = compute_fp8_scale(tok_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    tok_fp8 = (x_f32 * tok_rcp[:, None]).to(tl.float8e4nv)

    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    scale_tok_ptrs = scale_tok_ptr + row_offs * stride_scale_tok_row + pid_col
    scale_tok_mask = row_offs < M
    tl.store(scale_tok_ptrs, tok_scale, mask=scale_tok_mask)

    block_amax = tl.max(tok_amax, axis=0)
    blk_scale, blk_rcp = compute_fp8_scale(block_amax, NATIVE_E8M0)

    blk_fp8 = (x_f32 * blk_rcp).to(tl.float8e4nv)

    blk_fp8_t = tl.trans(blk_fp8)

    t_row_offs = col_start + tl.arange(0, BLOCK)
    t_col_offs = row_start + tl.arange(0, BLOCK)
    out_blk_t_ptrs = (out_blk_t_ptr + t_row_offs[:, None] * stride_out_blk_t_row + t_col_offs[None, :])
    t_mask = (t_row_offs[:, None] < K) & (t_col_offs[None, :] < M)
    tl.store(out_blk_t_ptrs, blk_fp8_t, mask=t_mask)

    scale_blk_t_offset = pid_col * scale_blk_t_cols + pid_row
    tl.store(scale_blk_t_ptr + scale_blk_t_offset, blk_scale)


def fp8cast_rowwise_blockwise_transpose_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_blockwise_transpose with the rowwise scale row-major."""
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128

    scale_tok_cols = (K + BLOCK - 1) // BLOCK
    scale_blk_t_rows = (K + BLOCK - 1) // BLOCK
    scale_blk_t_cols = (M + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_blk_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_blk_t = torch.empty((scale_blk_t_rows, scale_blk_t_cols), dtype=torch.float32, device=x.device)

    grid = ((M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK)

    _fp8cast_rowwise_blockwise_transpose_kernel[grid](x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_blk_t_ptr=out_blk_t, scale_blk_t_ptr=scale_blk_t, M=M, K=K, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_tok_row=K, stride_scale_tok_row=scale_tok_cols, stride_out_blk_t_row=M, scale_blk_t_cols=scale_blk_t_cols, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out_tok, scale_tok, out_blk_t, scale_blk_t
