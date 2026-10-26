"""
Blockwise (128x128) FP8 quantization of the weights (Triton), the ``fp8_weight_quant`` feature's
SM100 implementation: a weight and its transpose (2D, and batched 3D for grouped experts) in one
read; and the FP8 parameter all-gather kernels (per-tile amax of a weight shard, its FP8 cast, the
scale grids, the FP8 pack + transpose; ``modules/fp8_param_gather.py``). The SM90 ports in
``fp8_weight_quant_sm90`` are bit-identical.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import NATIVE_E8M0
from pithtrain.kernels.fp8_scale import compute_fp8_scale

__all__ = ["fp8_pack_transpose_sm100", "fp8_scales_sm100", "fp8_shard_amax_sm100", "fp8_shard_cast_sm100", "fp8cast_blockwise_transpose_batched_sm100", "fp8cast_blockwise_transpose_sm100"]


# ---------------------------------------------------------------------------
# fp8cast_blockwise_transpose: blockwise(x) + blockwise(x.T), 2D
# ---------------------------------------------------------------------------


@triton.jit
def _fp8cast_blockwise_transpose_kernel(x_ptr, out_ptr, scale_ptr, out_t_ptr, scale_t_ptr, M, K, stride_x_row, stride_x_col, stride_out_row, scale_cols, stride_out_t_row, scale_t_cols, NATIVE_E8M0: tl.constexpr):
    """Fused blockwise (128x128) FP8 quantization of x and x.T.

    Each program processes one 128x128 tile of x, computing a single amax
    and FP8 scale for the tile.  The quantized tile is stored to both
    ``out (M, K)`` and, transposed via swapped-index stores, to ``out_t (K, M)``.
    The same scalar scale is written to both ``scale`` and ``scale_t`` at
    swapped indices.
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

    # Single scalar amax for the 128x128 block
    abs_bf16 = tl.abs(x_bf16)
    row_max = tl.max(abs_bf16, axis=1)  # (BLOCK,)
    block_amax = tl.max(row_max, axis=0)  # scalar

    # Compute scalar scale
    scale_val, rcp_val = compute_fp8_scale(block_amax, NATIVE_E8M0)

    # Scale and cast (once)
    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp_val).to(tl.float8e4nv)

    # Store FP8 data: out(M, K) at tile (pid_row, pid_col)
    out_ptrs = out_ptr + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=mask)

    # Store transposed FP8 data to out_t(K, M) without tl.trans (which
    # miscompiles fp8 tiles): write the same tile to swapped-index addresses,
    # so element (i, j) lands at out_t[col_start + j, row_start + i].
    out_t_ptrs = out_t_ptr + col_offs[None, :] * stride_out_t_row + row_offs[:, None]
    tl.store(out_t_ptrs, fp8_tile, mask=mask)

    # Store scale: one value at (pid_row, pid_col)
    scale_offset = pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale_val)

    # Store scale_t: same value at (pid_col, pid_row) -- swapped indices
    scale_t_offset = pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale_val)


def fp8cast_blockwise_transpose_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused blockwise FP8 quantization of x *and* x.T (single read of *x*).

    Args:
        x: 2D tensor (M, K) in BF16/FP32.

    Returns:
        (fp8, scale, fp8_t, scale_t):
            fp8     -- float8_e4m3fn (M, K),
            scale   -- float32 (ceil(M/128), ceil(K/128)),
            fp8_t   -- float8_e4m3fn (K, M),
            scale_t -- float32 (ceil(K/128), ceil(M/128)).
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    BLOCK = 128
    scale_rows = (M + BLOCK - 1) // BLOCK
    scale_cols = (K + BLOCK - 1) // BLOCK

    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    grid = (scale_rows, scale_cols)

    _fp8cast_blockwise_transpose_kernel[grid](
        x_ptr=x, out_ptr=out, scale_ptr=scale, out_t_ptr=out_t, scale_t_ptr=scale_t, M=M, K=K, stride_x_row=x.stride(0), stride_x_col=x.stride(1),
        stride_out_row=K, scale_cols=scale_cols, stride_out_t_row=M, scale_t_cols=scale_rows, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2
    )

    return out, scale, out_t, scale_t


# ---------------------------------------------------------------------------
# fp8cast_blockwise_transpose_batched: blockwise(x) + blockwise(x.T), 3D
# ---------------------------------------------------------------------------


@triton.jit
def _fp8cast_blockwise_transpose_batched_kernel(
    x_ptr, out_ptr, scale_ptr, out_t_ptr, scale_t_ptr, G, N, K, stride_x_g, stride_x_row, stride_x_col, stride_out_g, stride_out_row, scale_cols, stride_scale_g, stride_out_t_g, stride_out_t_row, scale_t_cols, stride_scale_t_g,
    NATIVE_E8M0: tl.constexpr
):
    """Fused batched blockwise (128x128) FP8 quantization of x and x.transpose(1,2).

    Grid: (ceil(N/128), ceil(K/128), G)

    Each program processes one 128x128 tile within one group, computing a
    single amax and FP8 scale.  The quantized tile is stored to both
    ``out (G, N, K)`` and, transposed, to ``out_t (G, K, N)``.  The same
    scalar scale is written to ``scale`` and ``scale_t`` at swapped indices.
    """
    BLOCK: tl.constexpr = 128

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    pid_g = tl.program_id(2)

    row_start = pid_row * BLOCK
    col_start = pid_col * BLOCK

    row_offs = row_start + tl.arange(0, BLOCK)
    col_offs = col_start + tl.arange(0, BLOCK)

    # Load 128x128 tile from group pid_g -- keep bf16
    base = x_ptr + pid_g * stride_x_g
    ptrs = base + row_offs[:, None] * stride_x_row + col_offs[None, :] * stride_x_col
    mask = (row_offs[:, None] < N) & (col_offs[None, :] < K)
    x_bf16 = tl.load(ptrs, mask=mask, other=0.0)

    # Single scalar amax for the block
    abs_bf16 = tl.abs(x_bf16)
    row_max = tl.max(abs_bf16, axis=1)  # (BLOCK,)
    block_amax = tl.max(row_max, axis=0)  # scalar

    scale_val, rcp_val = compute_fp8_scale(block_amax, NATIVE_E8M0)

    # Scale and cast (once)
    x_f32 = x_bf16.to(tl.float32)
    fp8_tile = (x_f32 * rcp_val).to(tl.float8e4nv)

    # Store FP8 data: out(G, N, K)
    out_base = out_ptr + pid_g * stride_out_g
    out_ptrs = out_base + row_offs[:, None] * stride_out_row + col_offs[None, :]
    tl.store(out_ptrs, fp8_tile, mask=mask)

    # Store transposed FP8 data to out_t(G, K, N) without tl.trans (which
    # miscompiles fp8 tiles): write the same tile to swapped-index addresses,
    # so element (i, j) lands at out_t[pid_g, col_start + j, row_start + i].
    out_t_base = out_t_ptr + pid_g * stride_out_t_g
    out_t_ptrs = out_t_base + col_offs[None, :] * stride_out_t_row + row_offs[:, None]
    tl.store(out_t_ptrs, fp8_tile, mask=mask)

    # Store scale: (G, ceil(N/128), ceil(K/128)) at (pid_g, pid_row, pid_col)
    scale_offset = pid_g * stride_scale_g + pid_row * scale_cols + pid_col
    tl.store(scale_ptr + scale_offset, scale_val)

    # Store scale_t: (G, ceil(K/128), ceil(N/128)) at (pid_g, pid_col, pid_row)
    scale_t_offset = pid_g * stride_scale_t_g + pid_col * scale_t_cols + pid_row
    tl.store(scale_t_ptr + scale_t_offset, scale_val)


def fp8cast_blockwise_transpose_batched_sm100(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused blockwise FP8 quantization of x *and* x.transpose(1,2) for batched (G, N, K).

    Args:
        x: 3D tensor (G, N, K) in BF16/FP32.

    Returns:
        (fp8, scale, fp8_t, scale_t):
            fp8     -- float8_e4m3fn (G, N, K),
            scale   -- float32 (G, ceil(N/128), ceil(K/128)),
            fp8_t   -- float8_e4m3fn (G, K, N),
            scale_t -- float32 (G, ceil(K/128), ceil(N/128)).
    """
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    G, N, K = x.shape
    BLOCK = 128
    scale_rows = (N + BLOCK - 1) // BLOCK
    scale_cols = (K + BLOCK - 1) // BLOCK

    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((G, scale_rows, scale_cols), dtype=torch.float32, device=x.device)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale_t = torch.empty((G, scale_cols, scale_rows), dtype=torch.float32, device=x.device)

    grid = (scale_rows, scale_cols, G)

    _fp8cast_blockwise_transpose_batched_kernel[grid](
        x_ptr=x, out_ptr=out, scale_ptr=scale, out_t_ptr=out_t, scale_t_ptr=scale_t, G=G, N=N, K=K, stride_x_g=x.stride(0), stride_x_row=x.stride(1), stride_x_col=x.stride(2), stride_out_g=N * K, stride_out_row=K,
        scale_cols=scale_cols, stride_scale_g=scale_rows * scale_cols, stride_out_t_g=K * N, stride_out_t_row=N, scale_t_cols=scale_rows, stride_scale_t_g=scale_cols * scale_rows, NATIVE_E8M0=NATIVE_E8M0, num_warps=4, num_stages=2
    )

    return out, scale, out_t, scale_t


# ---------------------------------------------------------------------------
# FP8 parameter all-gather kernels (modules/fp8_param_gather.py)
# ---------------------------------------------------------------------------

BLOCK = 128


@triton.jit
def _fp8_shard_tile_kernel(x_ptr, amax_ptr, out_ptr, row_offset, local_rows, K, tile_row_start, amax_g_offset, amax_stride_g, amax_cols, CAST: tl.constexpr, NATIVE_E8M0: tl.constexpr):
    """
    One 128x128 tile of a row shard of a (G, N, K) weight, local shard (g_local, local_rows, K).

    The tile covers global rows [tr * 128, tr * 128 + 128) of group g; the shard holds global rows
    [row_offset, row_offset + local_rows). With CAST False, the amax of the rows the shard holds is
    written to the grid; with CAST True, those rows are cast to FP8 with the reduced grid.
    """
    BLOCK: tl.constexpr = 128
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    pid_g = tl.program_id(2)

    tr = tile_row_start + pid_row
    rows = tr * BLOCK + tl.arange(0, BLOCK) - row_offset
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = (rows[:, None] >= 0) & (rows[:, None] < local_rows) & (cols[None, :] < K)
    offs = pid_g.to(tl.int64) * local_rows * K + rows[:, None] * K + cols[None, :]
    # Round to bf16 first, as FSDP's mixed-precision cast does with the flag off.
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
def _fp8_scale_kernel(amax_ptr, scale_ptr, scale_t_ptr, R, C, NATIVE_E8M0: tl.constexpr):
    """
    Scales of one (R, C) amax grid of group pid(1), written as (R, C) and transposed as (C, R).
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
def _fp8_pack_transpose_kernel(x_ptr, out_ptr, out_t_ptr, N, K):
    """
    Copy one 128x128 byte tile of x (G, N, K) to out and, transposed, to out_t (G, K, N).

    Bytes are moved as uint8 with swapped-index stores, since tl.trans miscompiles fp8 tiles.
    """
    BLOCK: tl.constexpr = 128
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    pid_g = tl.program_id(2)
    rows = pid_row * BLOCK + tl.arange(0, BLOCK)
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = (rows[:, None] < N) & (cols[None, :] < K)
    base = pid_g.to(tl.int64) * N * K
    tile = tl.load(x_ptr + base + rows[:, None] * K + cols[None, :], mask=mask)
    tl.store(out_ptr + base + rows[:, None] * K + cols[None, :], tile, mask=mask)
    tl.store(out_t_ptr + base + cols[None, :] * N + rows[:, None], tile, mask=mask)


def _shard_tiles(shard: torch.Tensor, amax: torch.Tensor, out: Optional[torch.Tensor], row_offset: int, group_offset: int) -> None:
    """
    Launch _fp8_shard_tile_kernel over every tile a shard touches.

    shard is (g_local, local_rows, K) fp32; amax is the full (G, ceil(N/128), ceil(K/128)) grid.
    """
    g_local, local_rows, K = shard.shape
    if shard.numel() == 0:
        return
    tile_row_start = row_offset // BLOCK
    tile_row_end = (row_offset + local_rows + BLOCK - 1) // BLOCK
    grid = (tile_row_end - tile_row_start, triton.cdiv(K, BLOCK), g_local)
    _fp8_shard_tile_kernel[grid](shard, amax, out if out is not None else shard, row_offset, local_rows, K, tile_row_start, group_offset, amax.stride(0), amax.shape[2], CAST=out is not None, NATIVE_E8M0=NATIVE_E8M0, num_warps=8)


def fp8_shard_amax_sm100(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:  # fmt: skip
    """
    Write the per-tile amax of the bf16-rounded shard into amax at the tiles it touches.

    amax must be zero where other shards contribute; a MAX across shards then gives the full grid.
    """
    _shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm100(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:  # fmt: skip
    """
    Cast the shard to FP8 into out (same shape) with the scales of the reduced amax grid.
    """
    _shard_tiles(shard, amax, out, row_offset, group_offset)


def fp8_scales_sm100(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """
    The (G, R, C) scale grid and its (G, C, R) transpose from a (G, R, C) amax grid.
    """
    G, R, C = amax.shape
    grid = (triton.cdiv(R * C, 1024), G)
    _fp8_scale_kernel[grid](amax, scale, scale_t, R, C, NATIVE_E8M0=NATIVE_E8M0)


def fp8_pack_transpose_sm100(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    Write x (G, N, K) and its (G, K, N) transpose back to back into the flat byte buffer out.
    """
    G, N, K = x.shape
    x = x.view(torch.uint8)
    out = out.view(torch.uint8)
    grid = (triton.cdiv(N, BLOCK), triton.cdiv(K, BLOCK), G)
    _fp8_pack_transpose_kernel[grid](x, out, out[x.numel() :], N, K, num_warps=4)
