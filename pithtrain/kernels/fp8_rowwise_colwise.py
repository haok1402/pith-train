"""
Fused rowwise + colwise FP8 quantization from one read of the input (Triton).

Rowwise: 1x128 blocks along each row. Colwise: 128x1 blocks down each column, written either
row-major ``(M, N)`` or, for the grouped wgrad on SM90, in DeepGEMM's K-major per-group layout (see
``fp8_rowwise_kmajor_sm100``, which launches the same kernel). No SM90-specific version exists, so
both architectures run it.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8cast_rowwise_colwise", "fp8cast_rowwise_colwise_kernel"]


@triton.jit
def fp8cast_rowwise_colwise_kernel(
    x_ptr,
    out_tok_ptr,
    scale_tok_ptr,
    out_ch_ptr,
    scale_ch_ptr,
    M,
    N,
    stride_x_row,
    stride_x_col,
    stride_out_tok_row,
    scale_tok_cols,
    stride_out_ch_row,
    scale_ch_cols,
    block_to_group_ptr,
    cumsum_ptr,
    BLOCK_N: tl.constexpr,
    NATIVE_E8M0: tl.constexpr,
    WRITE_KMAJOR: tl.constexpr = False,
):
    """Fused rowwise + colwise FP8 quantization from a single tile load.

    Each program processes a 128 x BLOCK_N tile.  M must be divisible by 128.

    Rowwise path: 128-element column-block scaling (axis=1 reduction).
    Colwise path: 128-row group, per-column scaling (axis=0 reduction).

    When WRITE_KMAJOR=True (Hopper grouped wgrad), the colwise FP8 output is
    written directly in K-major flat layout - per-group (K_g, N) blocks are
    transposed to (N, K_g) and concatenated - eliminating a separate transpose
    kernel.  Colwise scales are written transposed: (N, M//128) instead of
    (M//128, N).  Requires block_to_group_ptr and cumsum_ptr (cumulative
    row counts without a leading zero, i.e. grouped_mm_offs directly).
    """
    BLOCK_ROWS: tl.constexpr = 128
    CHUNKS: tl.constexpr = BLOCK_N // BLOCK_ROWS

    pid_row = tl.program_id(0)  # which 128-row group
    pid_col = tl.program_id(1)  # which BLOCK_N column block

    row_start = pid_row * BLOCK_ROWS
    col_start = pid_col * BLOCK_N

    row_offs = row_start + tl.arange(0, BLOCK_ROWS)
    col_offs = col_start + tl.arange(0, BLOCK_N)

    # Load 128 x BLOCK_N tile (single global-memory read) -- keep bf16
    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < N)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)

    # --- Rowwise path (axis=1 reduction, per-row, per-128-col-block) ---
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

    # --- Colwise path (axis=0 reduction, per-column within 128-row group) ---
    ch_amax = tl.max(abs_bf16, axis=0)  # (BLOCK_N,)
    ch_scale, ch_rcp = compute_fp8_scale(ch_amax, NATIVE_E8M0)

    ch_fp8 = (x_f32 * ch_rcp[None, :]).to(tl.float8e4nv)

    ch_scale_mask = col_offs < N

    if WRITE_KMAJOR:
        # K-major: write per-group (K_g, N) blocks transposed to (N, K_g),
        # concatenated into a 1D flat buffer.  Groups are 128-aligned so each
        # pid_row belongs to exactly one group.
        group_id = tl.load(block_to_group_ptr + pid_row).to(tl.int64)
        group_end = tl.load(cumsum_ptr + group_id).to(tl.int64)
        group_start = tl.load(cumsum_ptr + group_id - 1, mask=group_id > 0, other=0).to(tl.int64)
        k_g = group_end - group_start
        k_offset = row_start - group_start

        flat_base = group_start * N
        row_local = tl.arange(0, BLOCK_ROWS).to(tl.int64)
        out_offs = (
            flat_base + col_offs[None, :].to(tl.int64) * k_g + (k_offset + row_local)[:, None]
        )
        tl.store(out_ch_ptr + out_offs, ch_fp8, mask=mask)

        # Scales: (N, num_row_groups) transposed layout.
        # scale_ch_cols = num_row_groups when WRITE_KMAJOR=True.
        scale_ch_ptrs = scale_ch_ptr + col_offs * scale_ch_cols + pid_row
        tl.store(scale_ch_ptrs, ch_scale, mask=ch_scale_mask)
    else:
        # MN-major: out_ch (M, N), scale_ch (M//128, N)
        out_ch_ptrs = out_ch_ptr + row_offs[:, None] * stride_out_ch_row + col_offs[None, :]
        tl.store(out_ch_ptrs, ch_fp8, mask=mask)

        scale_ch_ptrs = scale_ch_ptr + pid_row * scale_ch_cols + col_offs
        tl.store(scale_ch_ptrs, ch_scale, mask=ch_scale_mask)


def fp8cast_rowwise_colwise(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise + colwise FP8 quantization (single read of *x*).

    Args:
        x: 2D tensor (M, N) in BF16/FP32.  M must be divisible by 128.

    Returns:
        (tok_fp8, tok_scale, ch_fp8, ch_scale):
            tok_fp8 -- float8_e4m3fn (M, N),
            tok_scale -- float32 (M, ceil(N/128)),
            ch_fp8  -- float8_e4m3fn (M, N),
            ch_scale -- float32 (M//128, N).
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (
        f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    )
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

    fp8cast_rowwise_colwise_kernel[grid](
        x_ptr=x,
        out_tok_ptr=out_tok,
        scale_tok_ptr=scale_tok,
        out_ch_ptr=out_ch,
        scale_ch_ptr=scale_ch,
        M=M,
        N=N,
        stride_x_row=x.stride(0),
        stride_x_col=x.stride(1),
        stride_out_tok_row=N,
        scale_tok_cols=scale_tok_cols,
        stride_out_ch_row=N,
        scale_ch_cols=N,
        block_to_group_ptr=out_tok,
        cumsum_ptr=out_tok,
        BLOCK_N=_BLOCK_N,
        NATIVE_E8M0=NATIVE_E8M0,
        WRITE_KMAJOR=False,
        num_warps=4,
        num_stages=2,
    )

    return out_tok, scale_tok, out_ch, scale_ch
