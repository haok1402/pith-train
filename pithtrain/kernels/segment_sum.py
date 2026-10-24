"""
Per-group column sums over rows already sorted by group (the expert bias gradient), as Triton
kernels: :func:`segment_sum_along_rows`.

The reduction is fully parallel and atomic-free: one block per (group, h-tile), or per (row tile,
h-tile) followed by an ordered per-group pass when there are few groups, with the same bits either
way (tile sums added in row order). Both architectures run it; ``pithtrain.operators.
indexed_bias_add`` uses it for the backward of ``x + bias[group_ids]``.

:func:`block_segment_sum` adds per-128-row-block fp32 partials per expert in row order (the gate_up
bias gradient of the fused gpt-oss experts, whose kernels write the block partials).
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import sm_count

__all__ = ["block_segment_sum", "segment_sum_along_rows"]


@triton.jit
def _segment_sum_kernel(
    x_ptr,
    offs_padded_ptr,
    out_ptr,
    H: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    One block reduces rows [start, end) of group g for a HEAD_BLOCK-wide
    slab of the hidden dim.  Accumulation is in fp32 for numerical
    stability; the cast back to the source dtype happens at the call site.
    """
    g = tl.program_id(0)
    h_blk = tl.program_id(1)

    start = tl.load(offs_padded_ptr + g)
    end = tl.load(offs_padded_ptr + g + 1)

    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H

    acc = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for m_start in tl.range(start, end, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < end
        ptrs = x_ptr + m_offs[:, None] * H + h_offs[None, :]
        x = tl.load(ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)

    tl.store(out_ptr + g * H + h_offs, acc, mask=h_mask)


@triton.jit
def _segment_tile(offs_padded_ptr, E, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr):
    """
    Locate row tiles: groups are cut into BLOCK_M-row tiles from their own start, numbered
    consecutively across groups. Returns (number of tiles before each group, tiles per group,
    group starts, group ends) as BLOCK_E vectors (zero past E).
    """
    e = tl.arange(0, BLOCK_E)
    starts = tl.load(offs_padded_ptr + e, mask=e < E, other=0)
    ends = tl.load(offs_padded_ptr + e + 1, mask=e < E, other=0)
    ntiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
    first = tl.cumsum(ntiles, 0) - ntiles
    return first, ntiles, starts, ends


@triton.jit
def _segment_tile_sum_kernel(
    x_ptr,
    offs_padded_ptr,
    part_ptr,
    E,
    H: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """
    Column sums (fp32) of one BLOCK_M x HEAD_BLOCK tile of x, written to row t of part. The tile
    is a group's BLOCK_M-row tile as a sequential per-group loop would take it, so its sum has the
    same bits.
    """
    t = tl.program_id(0)
    h_blk = tl.program_id(1)
    first, ntiles, starts, ends = _segment_tile(offs_padded_ptr, E, BLOCK_M, BLOCK_E)
    e = tl.arange(0, BLOCK_E)
    g = tl.sum(((first + ntiles) <= t).to(tl.int32), axis=0)
    if g >= E:
        return
    sel = e == g
    m_start = tl.sum(tl.where(sel, starts, 0), axis=0) + (t - tl.sum(tl.where(sel, first, 0), axis=0)) * BLOCK_M  # fmt: skip
    end = tl.sum(tl.where(sel, ends, 0), axis=0)

    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < end
    ptrs = x_ptr + m_offs[:, None].to(tl.int64) * H + h_offs[None, :]
    x = tl.load(ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
    tl.store(part_ptr + t.to(tl.int64) * H + h_offs, tl.sum(x, axis=0), mask=h_mask)


@triton.jit
def _segment_tile_reduce_kernel(
    part_ptr,
    offs_padded_ptr,
    out_ptr,
    E,
    H: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """
    out[g] = the tile sums of group g added in row order, starting from zero (fp32).
    """
    g = tl.program_id(0)
    h_blk = tl.program_id(1)
    first, ntiles, _, _ = _segment_tile(offs_padded_ptr, E, BLOCK_M, BLOCK_E)
    sel = tl.arange(0, BLOCK_E) == g
    t0 = tl.sum(tl.where(sel, first, 0), axis=0)
    n = tl.sum(tl.where(sel, ntiles, 0), axis=0)

    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    acc = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for i in tl.range(0, n):
        acc += tl.load(part_ptr + (t0 + i).to(tl.int64) * H + h_offs, mask=h_mask, other=0.0)
    tl.store(out_ptr + g * H + h_offs, acc, mask=h_mask)


def segment_sum_along_rows(
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Per-group sum over rows.  Tokens are assumed already sorted by group.

    Each group is summed tile by tile (128 rows from the group's start), adding the tiles' fp32
    column sums in row order: one block per (group, 128 columns) does exactly that. With too few
    groups to fill the GPU (a few experts per rank), the tile sums run instead as one block per
    tile and 128 columns, and a second pass adds them up per group in the same order, so both
    give the same bits.

    Parameters:
        x: (M, H) contiguous tensor.
        grouped_mm_offs: (E,) cumulative end-offsets per group (group g
            spans rows [offs[g-1], offs[g]) with offs[-1] := 0).
        out_dtype: dtype of the returned bias gradient (matches the bias
            parameter's dtype).

    Returns:
        (E, H) tensor where row g holds the sum of x rows in group g.
    """
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    E = grouped_mm_offs.shape[0]

    # Prepend zero so each block can read both [start, end) without a branch.
    offs_padded = torch.empty(E + 1, dtype=torch.int32, device=x.device)
    offs_padded[0] = 0
    # .clamp keeps padding-rows out of the reduction if grouped_mm_offs
    # exceeds M; copy into int32 once.
    offs_padded[1:].copy_(grouped_mm_offs.to(torch.int32).clamp(max=M))

    HEAD_BLOCK = 128
    BLOCK_M = 128
    out = torch.empty((E, H), dtype=torch.float32, device=x.device)
    h_blocks = triton.cdiv(H, HEAD_BLOCK)
    if E * h_blocks >= 2 * sm_count(x.device):
        # Enough groups to fill the GPU with one block per (group, h-tile).
        meta = dict(H=H, HEAD_BLOCK=HEAD_BLOCK, BLOCK_M=BLOCK_M)
        _segment_sum_kernel[(E, h_blocks)](x, offs_padded, out, **meta)
    else:
        BLOCK_E = max(16, triton.next_power_of_2(E))
        # Each group has at most one partial tile, so this bounds the tile count without a sync.
        max_tiles = triton.cdiv(M, BLOCK_M) + E
        part = torch.empty((max_tiles, H), dtype=torch.float32, device=x.device)
        meta = dict(H=H, HEAD_BLOCK=HEAD_BLOCK, BLOCK_M=BLOCK_M, BLOCK_E=BLOCK_E)
        _segment_tile_sum_kernel[(max_tiles, h_blocks)](x, offs_padded, part, E, **meta)
        _segment_tile_reduce_kernel[(E, h_blocks)](part, offs_padded, out, E, **meta)
    return out.to(out_dtype) if out_dtype is not torch.float32 else out


@triton.jit
def _block_segment_sum_kernel(part_ptr, offs_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
    """out[g] = sum of the 128-row-block partials of expert g, in row order (fp32)."""
    g = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h < H
    end = tl.load(offs_ptr + g) // 128
    start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0) // 128
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for b in tl.range(start, end):
        acc += tl.load(part_ptr + b.to(tl.int64) * H + h, mask=mask, other=0.0)
    tl.store(out_ptr + g * H + h, acc.to(out_ptr.dtype.element_ty), mask=mask)


def block_segment_sum(
    partials: torch.Tensor, grouped_mm_offs: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """(G, H) per-expert sums of the (M / 128, H) fp32 partials of 128-row blocks, in row order,
    in dtype. Experts are 128-aligned row ranges given by grouped_mm_offs."""
    G, H = grouped_mm_offs.shape[0], partials.shape[1]
    out = torch.empty((G, H), dtype=dtype, device=partials.device)
    _block_segment_sum_kernel[(G, triton.cdiv(H, 512))](
        partials, grouped_mm_offs, out, H, BLOCK_H=512
    )
    return out
