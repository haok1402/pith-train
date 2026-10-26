"""
Rowwise FP8 quantization of x and of x.T from one read of x (Triton), the
``fp8_rowwise_transpose`` feature's SM100 implementation and the reference its SM90 port
(``fp8_rowwise_transpose_sm90``) is bit-identical to. Scales come back row-major.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_rowwise_transpose_sm100"]


@triton.jit
def _fp8cast_rowwise_transpose_kernel(x_ptr, out_tok_ptr, scale_tok_ptr, out_t_ptr, scale_t_ptr, M, N, stride_x_row, stride_x_col, stride_out_tok_row, scale_tok_cols, stride_out_t_row, stride_scale_t_row, BLOCK_N: tl.constexpr, NATIVE_E8M0: tl.constexpr):
    """Fused rowwise quantization of x (M, N) and x.T (N, M).

    Each program processes a 128 x BLOCK_N tile of x.

    Original rowwise: axis=1 reduction -> (M, N) FP8 + float32 scales.
    Transposed rowwise: axis=0 reduction -> (N, M) FP8 + float32 scales.
    """
    BLOCK_ROWS: tl.constexpr = 128
    CHUNKS: tl.constexpr = BLOCK_N // BLOCK_ROWS

    pid_row = tl.program_id(0)  # which 128-row group of x
    pid_col = tl.program_id(1)  # which BLOCK_N column block of x

    row_start = pid_row * BLOCK_ROWS
    col_start = pid_col * BLOCK_N

    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Load 128 x BLOCK_N tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    # --- Rowwise on x (axis=1 reduction) ---
    abs_reshaped = tl.reshape(abs_bf16, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_amax = tl.max(abs_reshaped, axis=1)  # (128 * CHUNKS,)
    tok_scale, tok_rcp = compute_fp8_scale(tok_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    x_reshaped = tl.reshape(x_f32, BLOCK_ROWS * CHUNKS, BLOCK_ROWS)
    tok_fp8_flat = (x_reshaped * tok_rcp[:, None]).to(tl.float8e4nv)
    tok_fp8 = tl.reshape(tok_fp8_flat, BLOCK_ROWS, BLOCK_N)

    # Store rowwise FP8 data: out_tok (M, N)
    out_tok_ptrs = out_tok_ptr + row_offs[:, None] * stride_out_tok_row + col_offs[None, :]
    tl.store(out_tok_ptrs, tok_fp8, mask=mask)

    # Store rowwise float32 scales: (M, scale_tok_cols)
    tok_scale_2d = tl.reshape(tok_scale, BLOCK_ROWS, CHUNKS)
    tok_col_offs = pid_col * CHUNKS + tl.arange(0, CHUNKS)
    tok_scale_ptrs = scale_tok_ptr + row_offs[:, None] * scale_tok_cols + tok_col_offs[None, :]
    tok_scale_mask = (row_offs[:, None] < M) & (tok_col_offs[None, :] < scale_tok_cols)
    tl.store(tok_scale_ptrs, tok_scale_2d, mask=tok_scale_mask)

    # --- Transposed rowwise on x.T (axis=0 reduction of x tile) ---
    t_amax = tl.max(abs_bf16, axis=0)  # (BLOCK_N,)
    t_scale, t_rcp = compute_fp8_scale(t_amax, NATIVE_E8M0)

    # Scale each column of x by its transpose-token scale, cast to FP8
    t_fp8 = (x_f32 * t_rcp[None, :]).to(tl.float8e4nv)

    # Transpose to (BLOCK_N, 128) for coalesced store to out_t (N, M)
    t_fp8_t = tl.trans(t_fp8)  # (BLOCK_N, 128)

    t_row_offs = col_start + tl.arange(0, BLOCK_N)
    t_col_offs = row_start + tl.arange(0, BLOCK_ROWS)
    out_t_ptrs = out_t_ptr + t_row_offs[:, None] * stride_out_t_row + t_col_offs[None, :]
    t_mask = (t_row_offs[:, None] < N) & (t_col_offs[None, :] < M)
    tl.store(out_t_ptrs, t_fp8_t, mask=t_mask)

    # Store transposed float32 scales: scale_t (N, ceil(M/128))
    # Each program contributes one column (pid_row) per transposed row.
    scale_t_ptrs = scale_t_ptr + col_offs * stride_scale_t_row + pid_row
    scale_t_mask = col_offs < N
    tl.store(scale_t_ptrs, t_scale, mask=scale_t_mask)


def fp8cast_rowwise_transpose_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise FP8 quantization of x *and* x.T (single read of *x*).

    Args:
        x: 2D tensor (M, N) in BF16/FP32.

    Returns:
        (tok_fp8, tok_scale, t_fp8, t_scale):
            tok_fp8   -- float8_e4m3fn (M, N),
            tok_scale -- float32 (M, ceil(N/128)),
            t_fp8     -- float8_e4m3fn (N, M),
            t_scale   -- float32 (N, ceil(M/128)).
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    BLOCK = 128

    scale_tok_cols = (N + BLOCK - 1) // BLOCK
    scale_t_cols = (M + BLOCK - 1) // BLOCK

    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_tok = torch.empty((M, scale_tok_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((N, scale_t_cols), dtype=torch.float32, device=x.device)

    _BLOCK_N = 128
    grid = ((M + BLOCK - 1) // BLOCK, (N + _BLOCK_N - 1) // _BLOCK_N)

    _fp8cast_rowwise_transpose_kernel[grid](
        x_ptr=x, out_tok_ptr=out_tok, scale_tok_ptr=scale_tok, out_t_ptr=out_t, scale_t_ptr=scale_t, M=M, N=N, stride_x_row=x.stride(0), stride_x_col=x.stride(1),
        stride_out_tok_row=N, scale_tok_cols=scale_tok_cols, stride_out_t_row=M, stride_scale_t_row=scale_t_cols, BLOCK_N=_BLOCK_N, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2
    )

    return out_tok, scale_tok, out_t, scale_t
