"""
Triton kernels of the fp8_weight_quant feature: the 128x128 blockwise FP8 of a weight and its
transpose (2-D, and batched 3-D for grouped experts), and the FP8 parameter all-gather kernels.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8_pack_transpose_sm100", "fp8_scales_sm100", "fp8_shard_amax_sm100", "fp8_shard_cast_sm100", "fp8cast_blockwise_transpose_batched_sm100", "fp8cast_blockwise_transpose_sm100"]


@triton.jit
def fp8cast_blockwise_transpose_kernel(x_ptr, out_ptr, scale_ptr, out_t_ptr, scale_t_ptr, M, K, stride_x_row, stride_x_col, stride_out_row, scale_cols, stride_out_t_row, scale_t_cols, NATIVE_E8M0: tl.constexpr):
    """
    One 128x128 tile of x: its FP8 to out (M, K) and out_t (K, M), its scale to scale and scale_t.
    """
    BLOCK: tl.constexpr = 128

    pid_row, pid_col = tl.program_id(0), tl.program_id(1)
    row_start, col_start = pid_row * BLOCK, pid_col * BLOCK
    row_offs, col_offs = row_start + tl.arange(0, BLOCK), col_start + tl.arange(0, BLOCK)

    ptrs = x_ptr + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < M) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)
    row_max = tl.max(abs_bf16, axis=1)
    block_amax = tl.max(row_max, axis=0)

    scale_val, rcp_val = compute_fp8_scale(block_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp_val).to(tl.float8e4nv)

    out_ptrs = out_ptr + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=mask)

    # tl.trans miscompiles fp8 tiles: store the same tile to swapped-index addresses instead.
    out_t_ptrs = out_t_ptr + col_offs[None, :] * stride_out_t_row + row_offs[:, None]
    tl.store(out_t_ptrs, fp8_tile, mask=mask)

    scale_offset = pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale_val)

    scale_t_offset = pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale_val)


def fp8cast_blockwise_transpose_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton fp8cast_blockwise_transpose: x 2-D (M, K) bf16 / fp32.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128
    scale_rows, scale_cols = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK

    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    grid = (scale_rows, scale_cols)

    fp8cast_blockwise_transpose_kernel[grid](x_ptr=x, out_ptr=out, scale_ptr=scale, out_t_ptr=out_t, scale_t_ptr=scale_t, M=M, K=K, stride_x_row=x.stride(0), stride_x_col=x.stride(1), stride_out_row=K, scale_cols=scale_cols, stride_out_t_row=M, scale_t_cols=scale_rows, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out, scale, out_t, scale_t


@triton.jit
def fp8cast_blockwise_transpose_batched_kernel(x_ptr, out_ptr, scale_ptr, out_t_ptr, scale_t_ptr, G, N, K, stride_x_g, stride_x_row, stride_x_col, stride_out_g, stride_out_row, scale_cols, stride_scale_g, stride_out_t_g, stride_out_t_row, scale_t_cols, stride_scale_t_g, NATIVE_E8M0: tl.constexpr):
    """
    One 128x128 tile of expert pid(2): grid (ceil(N / 128), ceil(K / 128), G).
    """
    BLOCK: tl.constexpr = 128

    pid_row, pid_col, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row_start, col_start = pid_row * BLOCK, pid_col * BLOCK
    row_offs, col_offs = row_start + tl.arange(0, BLOCK), col_start + tl.arange(0, BLOCK)

    base = x_ptr + pid_g * stride_x_g
    ptrs = base + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < N) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    abs_bf16 = tl.abs(x_bf16)
    row_max = tl.max(abs_bf16, axis=1)
    block_amax = tl.max(row_max, axis=0)

    scale_val, rcp_val = compute_fp8_scale(block_amax, NATIVE_E8M0)

    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp_val).to(tl.float8e4nv)

    out_base = out_ptr + pid_g * stride_out_g
    out_ptrs = out_base + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=mask)

    # tl.trans miscompiles fp8 tiles: store the same tile to swapped-index addresses instead.
    out_t_base = out_t_ptr + pid_g * stride_out_t_g
    out_t_ptrs = out_t_base + col_offs[None, :] * stride_out_t_row + row_offs[:, None]
    tl.store(out_t_ptrs, fp8_tile, mask=mask)

    scale_offset = pid_g * stride_scale_g + pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale_val)

    scale_t_offset = pid_g * stride_scale_t_g + pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale_val)


def fp8cast_blockwise_transpose_batched_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton fp8cast_blockwise_transpose_batched: x 3-D (G, N, K) bf16 / fp32.
    """
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    G, N, K = x.shape
    BLOCK = 128
    scale_rows, scale_cols = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK

    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((G, scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((G, scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    grid = (scale_rows, scale_cols, G)

    fp8cast_blockwise_transpose_batched_kernel[grid](x_ptr=x, out_ptr=out, scale_ptr=scale, out_t_ptr=out_t, scale_t_ptr=scale_t, G=G, N=N, K=K, stride_x_g=x.stride(0), stride_x_row=x.stride(1), stride_x_col=x.stride(2), stride_out_g=N * K, stride_out_row=K, scale_cols=scale_cols, stride_scale_g=scale_rows * scale_cols, stride_out_t_g=K * N, stride_out_t_row=N, scale_t_cols=scale_rows, stride_scale_t_g=scale_cols * scale_rows, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2)

    return out, scale, out_t, scale_t


BLOCK = 128


@triton.jit
def fp8_shard_tile_kernel(x_ptr, amax_ptr, out_ptr, row_offset, local_rows, K, tile_row_start, amax_g_offset, amax_stride_g, amax_cols, CAST: tl.constexpr, NATIVE_E8M0: tl.constexpr):
    """
    One 128x128 tile (global rows [tr * 128, tr * 128 + 128) of local expert pid(2)) of a row shard
    holding global rows [row_offset, row_offset + local_rows). CAST False writes the amax of the
    shard's rows in the tile to the grid; CAST True casts them with the reduced grid.
    """
    BLOCK: tl.constexpr = 128
    pid_row, pid_col, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    tr = tile_row_start + pid_row
    rows = tr * BLOCK + tl.arange(0, BLOCK) - row_offset
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = (rows[:, None] >= 0) & (rows[:, None] < local_rows) & (cols[None, :] < K)
    offs = pid_g.to(tl.int64) * local_rows * K + rows[:, None] * K + cols[None, :]
    # Round to bf16 first, as FSDP's mixed-precision cast does.
    x_bf16 = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.bfloat16)

    amax_offs = (amax_g_offset + pid_g) * amax_stride_g + tr * amax_cols + pid_col
    if CAST:
        amax = tl.load(amax_ptr + amax_offs)
        _, rcp = compute_fp8_scale(amax, NATIVE_E8M0)
        fp8 = (x_bf16.to(tl.float32) * rcp).to(tl.float8e4nv)
        tl.store(out_ptr + offs, fp8, mask=mask)
    else:
        amax = tl.max(tl.max(tl.abs(x_bf16), axis=1), axis=0)
        tl.store(amax_ptr + amax_offs, amax.to(tl.float32))


@triton.jit
def fp8_scale_kernel(amax_ptr, scale_ptr, scale_t_ptr, R, C, NATIVE_E8M0: tl.constexpr):
    """
    Scales of the (R, C) amax grid of group pid(1), written as (R, C) and transposed as (C, R).
    """
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    g = tl.program_id(1)
    mask = offs < R * C
    amax = tl.load(amax_ptr + g * R * C + offs, mask=mask, other=0.0)
    scale, _ = compute_fp8_scale(amax, NATIVE_E8M0)
    tl.store(scale_ptr + g * R * C + offs, scale, mask=mask)
    r, c = offs // C, offs % C
    tl.store(scale_t_ptr + g * R * C + c * R + r, scale, mask=mask)


@triton.jit
def fp8_pack_transpose_kernel(x_ptr, out_ptr, out_t_ptr, N, K):
    """
    One 128x128 byte tile of x (G, N, K) to out and, transposed, to out_t (G, K, N).
    """
    BLOCK: tl.constexpr = 128
    pid_row, pid_col, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows, cols = pid_row * BLOCK + tl.arange(0, BLOCK), pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = (rows[:, None] < N) & (cols[None, :] < K)
    base = pid_g.to(tl.int64) * N * K
    tile = tl.load(x_ptr + base + rows[:, None] * K + cols[None, :], mask=mask)
    tl.store(out_ptr + base + rows[:, None] * K + cols[None, :], tile, mask=mask)
    # Swapped-index store, since tl.trans miscompiles fp8 tiles.
    tl.store(out_t_ptr + base + cols[None, :] * N + rows[:, None], tile, mask=mask)


def shard_tiles(shard: torch.Tensor, amax: torch.Tensor, out: Optional[torch.Tensor], row_offset: int, group_offset: int) -> None:
    g_local, local_rows, K = shard.shape
    if shard.numel() == 0:
        return
    tile_row_start, tile_row_end = row_offset // BLOCK, (row_offset + local_rows + BLOCK - 1) // BLOCK
    grid = (tile_row_end - tile_row_start, triton.cdiv(K, BLOCK), g_local)
    fp8_shard_tile_kernel[grid](shard, amax, out if out is not None else shard, row_offset, local_rows, K, tile_row_start, group_offset, amax.stride(0), amax.shape[2], CAST=out is not None, NATIVE_E8M0=NATIVE_E8M0, num_warps=8)


def fp8_shard_amax_sm100(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """
    Triton fp8_shard_amax.
    """
    shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm100(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """
    Triton fp8_shard_cast.
    """
    shard_tiles(shard, amax, out, row_offset, group_offset)


def fp8_scales_sm100(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """
    Triton fp8_scales.
    """
    G, R, C = amax.shape
    grid = (triton.cdiv(R * C, 1024), G)
    fp8_scale_kernel[grid](amax, scale, scale_t, R, C, NATIVE_E8M0=NATIVE_E8M0)


def fp8_pack_transpose_sm100(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    Triton fp8_pack_transpose.
    """
    G, N, K = x.shape
    x = x.view(torch.uint8)
    out = out.view(torch.uint8)
    grid = (triton.cdiv(N, BLOCK), triton.cdiv(K, BLOCK), G)
    fp8_pack_transpose_kernel[grid](x, out, out[x.numel() :], N, K, num_warps=4)
