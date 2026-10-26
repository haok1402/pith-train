"""
Rowwise FP8 quantization of x and blockwise (128x128) FP8 quantization of x.T from one read of x
(Triton), the ``fp8_rowwise_blockwise_transpose`` feature's SM100 implementation and the reference
its SM90 port (``fp8_rowwise_blockwise_transpose_sm90``) is bit-identical to. Scales come back
row-major.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_rowwise_blockwise_transpose_sm100"]


@triton.jit
def _fp8cast_rowwise_blockwise_transpose_kernel(
    x_ptr, out_tok_ptr, scale_tok_ptr, out_blk_t_ptr, scale_blk_t_ptr, M, K, stride_x_row, stride_x_col, stride_out_tok_row, stride_scale_tok_row, stride_out_blk_t_row, scale_blk_t_cols,
    NATIVE_E8M0: tl.constexpr
):
    """Fused rowwise quantization of x and blockwise quantization of x.T.

    Each program processes a single 128x128 tile of x.

    Rowwise path: row-wise amax over 128 columns -> (128,) scales ->
        FP8 output to ``out_tok (M, K)`` + scales to ``scale_tok (M, ceil(K/128))``.
    Blockwise transpose path: tile-wide amax (reuses row amax) -> scalar scale ->
        transposed FP8 output to ``out_blk_t (K, M)`` + scale to
        ``scale_blk_t (ceil(K/128), ceil(M/128))``.
    """
    BLOCK: tl.constexpr = 128

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK
    col_start = pid_col * BLOCK

    row_offs = row_start + tl.arange(0, BLOCK)
    col_offs = col_start + tl.arange(0, BLOCK)

    # Load 128x128 tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    # --- Rowwise path (one 128-element column block per tile) ---
    tok_amax = tl.max(abs_bf16, axis=1)  # (128,)
    tok_scale, tok_rcp = compute_fp8_scale(tok_amax, NATIVE_E8M0)  # (128,)

    x_f32 = x_bf16.to(tl.float32)
    tok_fp8 = (x_f32 * tok_rcp[:, None]).to(tl.float8e4nv)

    # Store rowwise FP8 data: out_tok (M, K)
    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    # Store rowwise scale: (M, ceil(K/128)), one value per row at column pid_col
    scale_tok_ptrs = scale_tok_ptr + row_offs * stride_scale_tok_row + pid_col
    scale_tok_mask = row_offs < M
    tl.store(scale_tok_ptrs, tok_scale, mask=scale_tok_mask)

    # --- Blockwise transpose path (reuse tok_amax) ---
    block_amax = tl.max(tok_amax, axis=0)  # scalar
    blk_scale, blk_rcp = compute_fp8_scale(block_amax, NATIVE_E8M0)  # scalar

    blk_fp8 = (x_f32 * blk_rcp).to(tl.float8e4nv)

    # Transpose to (128, 128) for coalesced store to out_blk_t (K, M)
    blk_fp8_t = tl.trans(blk_fp8)

    # Store transposed FP8 data: rows [col_start..+128), cols [row_start..+128)
    t_row_offs = col_start + tl.arange(0, BLOCK)
    t_col_offs = row_start + tl.arange(0, BLOCK)
    out_blk_t_ptrs = (out_blk_t_ptr + t_row_offs[:, None] * stride_out_blk_t_row + t_col_offs[None, :])
    t_mask = (t_row_offs[:, None] < K) & (t_col_offs[None, :] < M)
    tl.store(out_blk_t_ptrs, blk_fp8_t, mask=t_mask)

    # Store block scale: scale_blk_t (ceil(K/128), ceil(M/128))
    # Block (pid_col, pid_row) of x.T maps to tile (pid_row, pid_col) of x
    scale_blk_t_offset = pid_col * scale_blk_t_cols + pid_row
    tl.store(scale_blk_t_ptr + scale_blk_t_offset, blk_scale)


def fp8cast_rowwise_blockwise_transpose_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise FP8 quant of *x* and blockwise FP8 quant of *x*.T.

    Args:
        x: 2D tensor (M, K) in BF16/FP32.

    Returns:
        (tok_fp8, tok_scale, blk_t_fp8, blk_t_scale):
            tok_fp8     -- float8_e4m3fn (M, K),
            tok_scale   -- float32 (M, ceil(K/128)),
            blk_t_fp8   -- float8_e4m3fn (K, M),
            blk_t_scale -- float32 (ceil(K/128), ceil(M/128)).
    """
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

    _fp8cast_rowwise_blockwise_transpose_kernel[grid](
        x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_blk_t_ptr=out_blk_t, scale_blk_t_ptr=scale_blk_t, M=M, K=K, stride_x_row=x.stride(0), stride_x_col=x.stride(1),
        stride_out_tok_row=K, stride_scale_tok_row=scale_tok_cols, stride_out_blk_t_row=M, scale_blk_t_cols=scale_blk_t_cols, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2
    )

    return out_tok, scale_tok, out_blk_t, scale_blk_t
