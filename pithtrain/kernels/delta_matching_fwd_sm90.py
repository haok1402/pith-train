"""
Causal E4M3 QK and PV with online FP32 softmax and row-wise probability group scales.
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32
from cutlass.utils import SmemAllocator
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.delta_matching_sm90 import BLOCK_QUERY, LOG2_E, allocate_operand, copy_operand, finish_operand_copies, launch_delta_matching_kernel, make_mma, multiply_tile, quantize_delta_matching, reduce_mma_row, share_mma_operand

__all__ = ["delta_matching_fwd_sm90"]


@cute.jit
def scale_causal_scores(scores, coords, mKeyScale, head_idx, query_start, key_start, query_scale, GROUP_KEYS: cutlass.Constexpr):
    for elem_idx in cutlass.range_constexpr(cute.size(scores)):
        row_idx, col_idx = coords[elem_idx]
        key_pos = key_start + col_idx
        scale = query_scale * (mKeyScale[head_idx, key_pos // GROUP_KEYS] * LOG2_E)
        scores[elem_idx] = scores[elem_idx] * scale if key_pos <= query_start + row_idx else Float32(float("-inf"))


@cute.jit
def update_softmax(scores, row_max, row_sum, rescale):
    rows = reshape_acc_to_mn(scores)  # (query rows, key columns)
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        tile_max = reduce_mma_row(scores, row_idx, Float32(float("-inf")), cute.arch.fmax)
        next_max = cute.arch.fmax(row_max[row_idx], tile_max)
        rescale[row_idx] = cute.math.exp2(row_max[row_idx] - next_max, fastmath=True)
        row_max[row_idx] = next_max
        tile_sum = Float32(0.0)
        for col_idx in cutlass.range_constexpr(cute.size(rows, mode=[1])):
            tile_sum += cute.math.exp2(rows[row_idx, col_idx] - next_max, fastmath=True)
        row_sum[row_idx] = row_sum[row_idx] * rescale[row_idx] + warp_reduce(tile_sum, operator.add, threads_in_group=4)


@cute.jit
def encode_probability_group(scores, coords, s_probability, row_max, weight, value_scale, group_idx: cutlass.Constexpr, GROUP_KEYS: cutlass.Constexpr):
    rows, positions = reshape_acc_to_mn(scores), reshape_acc_to_mn(coords)  # (query rows, key columns)
    GROUP_COLUMNS = GROUP_KEYS // 4
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        group_max = Float32(float("-inf"))
        for col_idx in cutlass.range_constexpr(GROUP_COLUMNS):
            group_max = cute.arch.fmax(group_max, rows[row_idx, group_idx * GROUP_COLUMNS + col_idx])
        group_max = warp_reduce(group_max, cute.arch.fmax, threads_in_group=4)
        basis = cute.arch.fmax(group_max, row_max[row_idx] - 12.0)
        weight[row_idx] = cute.math.exp2(basis - row_max[row_idx], fastmath=True) * (value_scale / 448.0)
        for col_idx in cutlass.range_constexpr(GROUP_COLUMNS):
            elem_col = group_idx * GROUP_COLUMNS + col_idx
            query_row, key_col = positions[row_idx, elem_col]
            code = 448.0 * cute.math.exp2(rows[row_idx, elem_col] - basis, fastmath=True)
            s_probability[query_row, key_col % GROUP_KEYS] = code.to(Float8E4M3FN)
    share_mma_operand()


@cute.jit
def accumulate_value_tile(scores, coords, s_probability, s_value, s_output, mma_value, thread_value, mValueT, mValueScale, head_idx, tile_idx, tidx, row_max, rescale, GROUP_KEYS: cutlass.Constexpr, GROUPS: cutlass.Constexpr):
    D = cute.size(s_value, mode=[0])
    acc_value = cute.make_rmem_tensor(thread_value.partition_shape_C((BLOCK_QUERY, D)), Float32)
    value_rows = reshape_acc_to_mn(acc_value)  # (query rows, value columns)
    shared_rows = reshape_acc_to_mn(thread_value.partition_C(s_output))  # (query rows, value columns)
    weight = cute.make_rmem_tensor(cute.size(value_rows, mode=[0]), Float32)
    previous = cute.make_rmem_tensor(cute.size(weight), Float32)
    for group_idx in cutlass.range_constexpr(GROUPS):
        key_group_idx = tile_idx * GROUPS + group_idx
        copy_operand(mValueT, s_value, head_idx, 0, key_group_idx, tidx)
        value_scale = mValueScale[head_idx, key_group_idx]
        encode_probability_group(scores, coords, s_probability, row_max, weight, value_scale, group_idx, GROUP_KEYS)
        if cutlass.const_expr(group_idx > 0):
            for row_idx in cutlass.range_constexpr(cute.size(weight)):
                ratio = cute.arch.fmin(cute.arch.fmax(previous[row_idx] / weight[row_idx], 1e-30), 1e30)
                for col_idx in cutlass.range_constexpr(cute.size(value_rows, mode=[1])):
                    value_rows[row_idx, col_idx] *= ratio
        previous.store(weight.load())
        finish_operand_copies()
        multiply_tile(mma_value, thread_value, acc_value, s_probability, s_value, group_idx > 0)
    for row_idx in cutlass.range_constexpr(cute.size(weight)):
        for col_idx in cutlass.range_constexpr(cute.size(value_rows, mode=[1])):
            shared_rows[row_idx, col_idx] = shared_rows[row_idx, col_idx] * rescale[row_idx] + value_rows[row_idx, col_idx] * weight[row_idx]


@cute.jit
def store_attention(mOut, mLse, s_output, row_sum, row_max, thread_value, query_block_idx, head_idx, tidx):
    D = cute.size(s_output, mode=[1])
    H = cute.size(mOut, mode=[2])
    out_tile = cute.local_tile(mOut[head_idx // H, None, head_idx % H, None], (BLOCK_QUERY, D), (query_block_idx, 0))  # (query row, channel)
    output = reshape_acc_to_mn(thread_value.partition_C(out_tile))  # (query rows, channels)
    accum = reshape_acc_to_mn(thread_value.partition_C(s_output))  # (query rows, channels)
    positions = reshape_acc_to_mn(thread_value.partition_C(cute.make_identity_tensor((BLOCK_QUERY, D))))  # (query rows, channels)
    for row_idx in cutlass.range_constexpr(cute.size(accum, mode=[0])):
        normalizer = row_sum[row_idx]
        if tidx % 4 == 0:
            mLse[head_idx, query_block_idx * BLOCK_QUERY + positions[row_idx, 0][0]] = (row_max[row_idx] + cute.math.log2(normalizer, fastmath=True)) * (1.0 / LOG2_E)
        for col_idx in cutlass.range_constexpr(cute.size(accum, mode=[1])):
            output[row_idx, col_idx] = (accum[row_idx, col_idx] / normalizer).to(BFloat16)


@cute.jit
def attend_tiles(tensors, buffers, query_block_idx, head_idx, tidx, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr, QUERY_SCALE_ROWS: cutlass.Constexpr):
    mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse = tensors
    s_query, s_key, s_value, s_probability, s_output = buffers
    D = cute.size(s_query, mode=[1])
    mma_score, mma_value = make_mma(TILE_KEYS), make_mma(D)
    thread_score, thread_value = mma_score.get_slice(tidx), mma_value.get_slice(tidx)
    score_shape = (BLOCK_QUERY, TILE_KEYS)
    score_coords = thread_score.partition_C(cute.make_identity_tensor(score_shape))  # (MMA, MMA_M, MMA_N)
    scores = cute.make_rmem_tensor(thread_score.partition_shape_C(score_shape), Float32)
    row_max, row_sum, rescale = (cute.make_rmem_tensor(2, Float32) for _ in range(3))
    row_max.fill(float("-inf"))
    row_sum.fill(0.0)

    query_start = query_block_idx * BLOCK_QUERY
    key_head_idx = head_idx // (cute.size(mQueryScale, mode=[0]) // cute.size(mKey, mode=[0]))
    scale = mQueryScale[head_idx, query_start // QUERY_SCALE_ROWS]
    thread_value.partition_C(s_output).fill(0.0)
    last_tile = query_start // TILE_KEYS
    for tile_offset in range(last_tile + 1):
        tile_idx = last_tile - tile_offset
        copy_operand(mKey, s_key, key_head_idx, tile_idx, 0, tidx)
        finish_operand_copies()
        multiply_tile(mma_score, thread_score, scores, s_query, s_key, False)
        scale_causal_scores(scores, score_coords, mKeyScale, key_head_idx, query_start, tile_idx * TILE_KEYS, scale, GROUP_KEYS)
        update_softmax(scores, row_max, row_sum, rescale)
        accumulate_value_tile(scores, score_coords, s_probability, s_value, s_output, mma_value, thread_value, mValueT, mValueScale, key_head_idx, tile_idx, tidx, row_max, rescale, GROUP_KEYS, TILE_KEYS // GROUP_KEYS)

    store_attention(mOut, mLse, s_output, row_sum, row_max, thread_value, query_block_idx, head_idx, tidx)


@cute.kernel
def forward_attention(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    D = cute.size(mQuery, mode=[2])
    allocator = SmemAllocator()
    s_query, s_key = allocate_operand(allocator, BLOCK_QUERY, D), allocate_operand(allocator, TILE_KEYS, D)
    s_value, s_probability = allocate_operand(allocator, D, GROUP_KEYS), allocate_operand(allocator, BLOCK_QUERY, GROUP_KEYS)
    s_output = allocator.allocate_tensor(Float32, cute.make_layout((BLOCK_QUERY, D), stride=(D, 1)), 16)  # (query row, value column)

    copy_operand(mQuery, s_query, by, bx, 0, tidx)
    finish_operand_copies()
    attend_tiles((mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse), (s_query, s_key, s_value, s_probability, s_output), bx, by, tidx, GROUP_KEYS, TILE_KEYS, GROUP_KEYS * 2)


@cute.jit
def launch_forward_attention(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, stream):
    H, N, D = mQuery.shape
    forward_attention(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, 8192 // D, 32768 // D).launch(grid=(N // BLOCK_QUERY, H, 1), block=(128, 1, 1), stream=stream)


def delta_matching_fwd_sm90(query, key, value, softmax_scale):
    """
    Quantize inputs and compute causal FP8 attention and its backward state on SM90.
    """
    B, N, H, D = query.shape
    group_keys = 8192 // D
    query_codes, query_scales, query_t = quantize_delta_matching(query, group_keys * 2, input_scale=softmax_scale)
    key_codes, key_scales, key_t = quantize_delta_matching(key, group_keys, center_key=True)
    value_codes, value_scales, value_t = quantize_delta_matching(value, group_keys)
    out = torch.empty((B, N, H, D), dtype=query.dtype, device=query.device)
    lse = torch.empty((B * H, N), dtype=torch.float32, device=query.device)
    launch_delta_matching_kernel("delta_matching_forward", launch_forward_attention, (query_codes, query_scales, key_codes, key_scales, value_t, value_scales, out, lse))
    return (out, lse, query_codes, query_scales, query_t, key_codes, key_scales, key_t, value_codes, value_scales)
