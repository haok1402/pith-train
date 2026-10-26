"""
Triton per-group column sums over rows sorted by group: segment_sum_along_rows (the expert bias
gradient) and block_segment_sum (per-expert sums of 128-row-block partials). Deterministic and
atomic-free; both architectures run them.
"""

import torch
import triton
import triton.language as tl

from pithtrain.kernels.arch import SM_COUNT

__all__ = ["block_segment_sum", "segment_sum_along_rows"]


@triton.jit
def segment_sum_kernel(x_ptr, offs_padded_ptr, out_ptr, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr):
    """One block reduces rows [start, end) of group g over HEAD_BLOCK columns, in fp32."""
    g, h_blk = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(offs_padded_ptr + g), tl.load(offs_padded_ptr + g + 1)

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
def segment_tile(offs_padded_ptr, E, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr):
    """
    Groups are cut into BLOCK_M-row tiles from their own start, numbered consecutively across
    groups. Returns (tiles before each group, tiles per group, group starts, group ends) as BLOCK_E
    vectors (zero past E).
    """
    e = tl.arange(0, BLOCK_E)
    starts, ends = tl.load(offs_padded_ptr + e, mask=e < E, other=0), tl.load(offs_padded_ptr + e + 1, mask=e < E, other=0)
    ntiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
    first = tl.cumsum(ntiles, 0) - ntiles
    return first, ntiles, starts, ends


@triton.jit
def segment_tile_sum_kernel(x_ptr, offs_padded_ptr, part_ptr, E, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr):
    """
    fp32 column sums of row tile t of x, written to row t of part. The tile is the one the per-group
    loop of segment_sum_kernel would take, so its sum has the same bits.
    """
    t, h_blk = tl.program_id(0), tl.program_id(1)
    first, ntiles, starts, ends = segment_tile(offs_padded_ptr, E, BLOCK_M, BLOCK_E)
    e = tl.arange(0, BLOCK_E)
    g = tl.sum(((first + ntiles) <= t).to(tl.int32), axis=0)
    if g >= E:
        return
    sel = e == g
    m_start = tl.sum(tl.where(sel, starts, 0), axis=0) + (t - tl.sum(tl.where(sel, first, 0), axis=0)) * BLOCK_M
    end = tl.sum(tl.where(sel, ends, 0), axis=0)

    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < end
    ptrs = x_ptr + m_offs[:, None].to(tl.int64) * H + h_offs[None, :]
    x = tl.load(ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
    tl.store(part_ptr + t.to(tl.int64) * H + h_offs, tl.sum(x, axis=0), mask=h_mask)


@triton.jit
def segment_tile_reduce_kernel(part_ptr, offs_padded_ptr, out_ptr, E, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr):
    """out[g] = the tile sums of group g added in row order, starting from zero (fp32)."""
    g, h_blk = tl.program_id(0), tl.program_id(1)
    first, ntiles, _, _ = segment_tile(offs_padded_ptr, E, BLOCK_M, BLOCK_E)
    sel = tl.arange(0, BLOCK_E) == g
    t0, n = tl.sum(tl.where(sel, first, 0), axis=0), tl.sum(tl.where(sel, ntiles, 0), axis=0)

    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    acc = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for i in tl.range(0, n):
        acc += tl.load(part_ptr + (t0 + i).to(tl.int64) * H + h_offs, mask=h_mask, other=0.0)
    tl.store(out_ptr + g * H + h_offs, acc, mask=h_mask)


def segment_sum_along_rows(x: torch.Tensor, grouped_mm_offs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """
    Per-group column sums of rows already sorted by group (the bias gradient of x +
    bias[group_ids]): a new (G, H) tensor of out_dtype, row g the sum of rows [offs[g - 1], offs[g])
    of x (offs[-1] = 0).

    - x: (M, H) contiguous, any float dtype; not modified.
    - grouped_mm_offs: integer (G,) non-decreasing cumulative group ends; ends past M are clamped to
      M and rows past the last end are ignored. Groups need not be multiples of 128; an empty group
      sums to zero.
    - out_dtype: the result dtype; the sums are fp32, rounded once.

    Deterministic summation order: each group is cut into 128-row tiles from its own start, and the
    tiles' fp32 column sums are added in row order starting from zero.
    """
    assert x.is_contiguous(), "x must be contiguous"
    M, H = x.shape
    E = grouped_mm_offs.shape[0]

    # a leading zero lets every group read [start, end) without a branch
    offs_padded = torch.empty(E + 1, dtype=torch.int32, device=x.device)
    offs_padded[0] = 0
    offs_padded[1:].copy_(grouped_mm_offs.to(torch.int32).clamp(max=M))

    HEAD_BLOCK = 128
    BLOCK_M = 128
    out = torch.empty((E, H), dtype=torch.float32, device=x.device)
    h_blocks = triton.cdiv(H, HEAD_BLOCK)
    if E * h_blocks >= 2 * SM_COUNT:
        # enough groups to fill the GPU with one block per (group, h-tile)
        meta = dict(H=H, HEAD_BLOCK=HEAD_BLOCK, BLOCK_M=BLOCK_M)
        segment_sum_kernel[(E, h_blocks)](x, offs_padded, out, **meta)
    else:
        BLOCK_E = max(16, triton.next_power_of_2(E))
        # each group has at most one partial tile: a bound on the tile count without a sync
        max_tiles = triton.cdiv(M, BLOCK_M) + E
        part = torch.empty((max_tiles, H), dtype=torch.float32, device=x.device)
        meta = dict(H=H, HEAD_BLOCK=HEAD_BLOCK, BLOCK_M=BLOCK_M, BLOCK_E=BLOCK_E)
        segment_tile_sum_kernel[(max_tiles, h_blocks)](x, offs_padded, part, E, **meta)
        segment_tile_reduce_kernel[(E, h_blocks)](part, offs_padded, out, E, **meta)
    return out.to(out_dtype) if out_dtype is not torch.float32 else out


@triton.jit
def block_segment_sum_kernel(part_ptr, offs_ptr, out_ptr, H, BLOCK_H: tl.constexpr):
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


def block_segment_sum(partials: torch.Tensor, grouped_mm_offs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    (G, H) per-expert sums, in dtype, of the (M / 128, H) fp32 partials of 128-row blocks, added in
    row order. Experts are the 128-aligned row ranges given by grouped_mm_offs (int32 (G,)).
    """
    G, H = grouped_mm_offs.shape[0], partials.shape[1]
    out = torch.empty((G, H), dtype=dtype, device=partials.device)
    block_segment_sum_kernel[(G, triton.cdiv(H, 512))](partials, grouped_mm_offs, out, H, BLOCK_H=512)
    return out
