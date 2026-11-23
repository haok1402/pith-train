"""
Causal E4M3 QK and PV with online FP32 softmax and row-wise probability group scales.
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32
from cutlass.utils import SmemAllocator
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.delta_matching_sm90 import BLOCK_QUERY, LOG2_E, allocate_staged_operand, copy_tma_operand, initialize_attention_barriers, launch_delta_matching_kernel, make_mma, make_operand_tma, multiply_tile, quantize_delta_matching, reduce_mma_row

__all__ = ["delta_matching_fwd_sm90"]


@cute.jit
def scale_causal_scores(scores, coords, mKeyScale, head_idx, query_start, key_start, query_scale, is_diagonal, GROUP_KEYS: cutlass.Constexpr):
    rows, positions = reshape_acc_to_mn(scores), reshape_acc_to_mn(coords)  # (query rows, key columns)
    COLS_PER_GROUP = GROUP_KEYS // 4
    for group_idx in cutlass.range_constexpr(cute.size(rows, mode=[1]) // COLS_PER_GROUP):
        scale = (query_scale * mKeyScale[head_idx, key_start // GROUP_KEYS + group_idx]) * LOG2_E
        for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
            for col_idx in cutlass.range_constexpr(COLS_PER_GROUP):
                elem_col = group_idx * COLS_PER_GROUP + col_idx
                query_row, key_col = positions[row_idx, elem_col]
                value = rows[row_idx, elem_col] * scale
                if is_diagonal and key_start + key_col > query_start + query_row:
                    value = Float32(float("-inf"))
                rows[row_idx, elem_col] = value


@cute.jit
def update_softmax(scores, row_max, row_sum, rescale):
    rows = reshape_acc_to_mn(scores)  # (query rows, key columns)
    # The normalizer uses FP32 probabilities so the backward can reconstruct them from LSE.
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
def pack_probability_operand(rows, basis, fragment, group_idx: cutlass.Constexpr, GROUP_KEYS: cutlass.Constexpr, tidx):
    # WGMMA C assigns two consecutive keys per lane, while register A uses four.
    # Each four-lane group exchanges two packed words per 16-key half.
    quad_lane = tidx % 4
    choose_first = quad_lane == 0 or quad_lane == 3
    lane_first = (quad_lane // 2) ^ (3 * (quad_lane % 2))
    selector_low = 0x5410 if choose_first else 0x1054
    selector_high = 0x7632 if choose_first else 0x3276
    quad_mask = (28 << 8) | 31
    words = cute.recast_tensor(fragment, Int32)
    for half_idx in cutlass.range_constexpr(GROUP_KEYS // 16):
        values = cute.make_rmem_tensor(8, Float32)
        for elem_idx in cutlass.range_constexpr(8):
            row_idx = (elem_idx // 2) % 2
            col_idx = (GROUP_KEYS // 4) * group_idx + half_idx * 4 + (elem_idx // 4) * 2 + elem_idx % 2
            values[elem_idx] = 448.0 * cute.math.exp2(rows[row_idx, col_idx] - basis[row_idx], fastmath=True)
        codes = cute.make_rmem_tensor(8, Float8E4M3FN)
        codes.store(values.load().to(Float8E4M3FN))
        packed = cute.recast_tensor(codes, Int32)
        low = packed[0] if choose_first else packed[1]
        high = packed[1] if choose_first else packed[0]
        low = cute.arch.shuffle_sync(low, offset=lane_first, mask_and_clamp=quad_mask)
        high = cute.arch.shuffle_sync(high, offset=lane_first ^ 1, mask_and_clamp=quad_mask)
        words[2 * half_idx] = Int32(cute.arch.prmt(low, high, selector_low))
        words[2 * half_idx + 1] = Int32(cute.arch.prmt(low, high, selector_high))


@cute.jit
def encode_probability_group(scores, fragment, row_max, weight, value_scale, group_idx: cutlass.Constexpr, GROUP_KEYS: cutlass.Constexpr, tidx):
    rows = reshape_acc_to_mn(scores)  # (query rows, key columns)
    GROUP_COLUMNS = GROUP_KEYS // 4
    basis = cute.make_rmem_tensor(2, Float32)
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        group_max = Float32(float("-inf"))
        for col_idx in cutlass.range_constexpr(GROUP_COLUMNS):
            group_max = cute.arch.fmax(group_max, rows[row_idx, group_idx * GROUP_COLUMNS + col_idx])
        group_max = warp_reduce(group_max, cute.arch.fmax, threads_in_group=4)
        # The floor limits the spread between group bases in a promoted tile.
        basis[row_idx] = cute.arch.fmax(group_max, row_max[row_idx] - 12.0)
        weight[row_idx] = cute.math.exp2(basis[row_idx] - row_max[row_idx], fastmath=True) * (value_scale / 448.0)
    pack_probability_operand(rows, basis, fragment, group_idx, GROUP_KEYS, tidx)


@cute.jit
def encode_probability_tile(scores, weights, operands, mValueScale, head_idx, tile_idx, row_max, tidx, GROUP_KEYS: cutlass.Constexpr, GROUPS: cutlass.Constexpr):
    for group_idx in cutlass.range_constexpr(GROUPS):
        scale = mValueScale[head_idx, tile_idx * GROUPS + group_idx]
        encode_probability_group(scores, operands[None, None, None, group_idx], row_max, weights[None, group_idx], scale, group_idx, GROUP_KEYS, tidx)


@cute.jit
def rebase_value_accumulator(accumulator, weights, group_idx: cutlass.Constexpr):
    rows = reshape_acc_to_mn(accumulator)  # (per-thread query rows, value columns)
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        ratio = weights[row_idx, group_idx - 1] * cute.arch.rcp_approx(weights[row_idx, group_idx])
        ratio = cute.arch.fmin(cute.arch.fmax(ratio, 1e-30), 1e30)
        rows[row_idx, None].store(rows[row_idx, None].load() * ratio)


@cute.jit
def accumulate_value_tile(acc_output, mma, mma_thread, s_values, weights, operands, rescale, stage_idx, GROUP_KEYS: cutlass.Constexpr, GROUPS: cutlass.Constexpr, VALUE_COLUMNS: cutlass.Constexpr):
    for slice_idx in cutlass.range_constexpr(cute.size(acc_output, mode=[3])):
        acc_tile = cute.make_rmem_tensor(mma_thread.partition_shape_C((BLOCK_QUERY, VALUE_COLUMNS)), Float32)
        for group_idx in cutlass.range_constexpr(GROUPS):
            if cutlass.const_expr(group_idx > 0):
                rebase_value_accumulator(acc_tile, weights, group_idx)
            value_group = s_values[None, None, stage_idx * GROUPS + group_idx]  # (value channels, key rows)
            value_tile = cute.local_tile(value_group, (VALUE_COLUMNS, GROUP_KEYS), (slice_idx, 0))  # (value channel tile, key rows)
            multiply_tile(mma, mma_thread, acc_tile, operands[None, None, None, group_idx], value_tile, group_idx > 0)
        output_rows = reshape_acc_to_mn(acc_output[None, None, None, slice_idx])  # (per-thread query rows, value columns)
        value_rows = reshape_acc_to_mn(acc_tile)  # (per-thread query rows, value columns)
        for row_idx in cutlass.range_constexpr(cute.size(value_rows, mode=[0])):
            promoted = value_rows[row_idx, None].load() * weights[row_idx, GROUPS - 1]
            output_rows[row_idx, None].store(output_rows[row_idx, None].load() * rescale[row_idx] + promoted)


@cute.jit
def store_attention(mOut, mLse, acc_output, row_sum, row_max, mma_thread, query_block_idx, head_idx, tidx, VALUE_COLUMNS: cutlass.Constexpr):
    H = cute.size(mOut, mode=[2])
    target = mOut[head_idx // H, None, head_idx % H, None]  # (sequence, channel)
    positions = reshape_acc_to_mn(mma_thread.partition_C(cute.make_identity_tensor((BLOCK_QUERY, VALUE_COLUMNS))))  # (per-thread query rows, value columns)
    if tidx % 4 == 0:
        for row_idx in cutlass.range_constexpr(cute.size(row_sum)):
            token_idx = query_block_idx * BLOCK_QUERY + positions[row_idx, 0][0]
            mLse[head_idx, token_idx] = (row_max[row_idx] + cute.math.log2(row_sum[row_idx], fastmath=True)) * (1.0 / LOG2_E)
    for slice_idx in cutlass.range_constexpr(cute.size(acc_output, mode=[3])):
        out_tile = cute.local_tile(target, (BLOCK_QUERY, VALUE_COLUMNS), (query_block_idx, slice_idx))  # (query row, channel)
        output = reshape_acc_to_mn(mma_thread.partition_C(out_tile))  # (per-thread query rows, value columns)
        accum = reshape_acc_to_mn(acc_output[None, None, None, slice_idx])  # (per-thread query rows, value columns)
        for row_idx in cutlass.range_constexpr(cute.size(row_sum)):
            rcp = cute.arch.rcp_approx(row_sum[row_idx])
            output[row_idx, None].store((accum[row_idx, None].load() * rcp).to(BFloat16))


class CausalAttention:
    """
    Stage a shared causal K/V tile for two independent query warpgroups.
    """
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.group_keys, self.tile_keys = 8192 // head_dim, 32768 // head_dim
        self.groups = self.tile_keys // self.group_keys
        self.value_columns = min(head_dim, 128)
        self.stages = 2
        self.consumer_groups, self.load_group = 2, 0
        self.consumer_registers, self.load_registers = 232, 24

    @cute.jit
    def __call__(self, mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, stream):
        H, N, D = mQuery.shape
        query_copy, key_copy = make_operand_tma(mQuery, BLOCK_QUERY, D), make_operand_tma(mKey, self.tile_keys, D)
        value_copy = make_operand_tma(mValueT, D, self.group_keys)
        inputs = (mQueryScale, mKeyScale, mValueScale, mOut, mLse)
        self.kernel(inputs, (query_copy, key_copy, value_copy)).launch(grid=(N // (BLOCK_QUERY * self.consumer_groups), H, 1), block=((1 + self.consumer_groups) * 128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, inputs, copies):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = warp // 4
        query_block = bx * self.consumer_groups
        last_tile = query_block * BLOCK_QUERY // self.tile_keys
        key_head = by // (cute.size(inputs[0], mode=[0]) // cute.size(inputs[1], mode=[0]))
        allocator = SmemAllocator()
        s_query = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.consumer_groups)
        s_keys = allocate_staged_operand(allocator, self.tile_keys, self.head_dim, self.stages)
        s_values = allocate_staged_operand(allocator, self.head_dim, self.group_keys, self.stages * self.groups)
        buffers = (s_query, s_keys, s_values)
        barriers = initialize_attention_barriers(allocator, tidx, self.stages, self.consumer_groups)

        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if tidx == 0:
                self.load_loop(copies, buffers, barriers, query_block, by, key_head, last_tile)
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            consumer_group = warp_group - 1
            self.compute_loop(inputs, buffers, barriers, query_block + consumer_group, by, key_head, last_tile, consumer_group, tidx % 128)

    @cute.jit
    def load_loop(self, copies, buffers, barriers, query_block, head_idx, key_head, last_tile):
        query_copy, key_copy, value_copy = copies
        s_query, s_keys, s_values = buffers
        query_ready, tile_ready, tile_released = barriers
        cute.arch.mbarrier_arrive_and_expect_tx(query_ready, cute.size(s_query))
        for group_idx in cutlass.range_constexpr(self.consumer_groups):
            copy_tma_operand(query_copy, s_query[None, None, group_idx], query_block + group_idx, 0, head_idx, query_ready)
        for tile_offset in range(last_tile + 1):
            stage_idx, phase = tile_offset % self.stages, (tile_offset // self.stages) % 2
            tile_idx = last_tile - tile_offset
            if tile_offset >= self.stages:
                cute.arch.mbarrier_wait(tile_released + stage_idx, phase ^ 1)
            ready = tile_ready + stage_idx
            cute.arch.mbarrier_arrive_and_expect_tx(ready, 2 * self.tile_keys * self.head_dim)
            copy_tma_operand(key_copy, s_keys[None, None, stage_idx], tile_idx, 0, key_head, ready)
            for group_idx in cutlass.range_constexpr(self.groups):
                s_value = s_values[None, None, stage_idx * self.groups + group_idx]  # (value channels, key rows)
                copy_tma_operand(value_copy, s_value, 0, tile_idx * self.groups + group_idx, key_head, ready)

    @cute.jit
    def compute_loop(self, inputs, buffers, barriers, query_block, head_idx, key_head, last_tile, consumer_group, tidx):
        mQueryScale, mKeyScale, mValueScale, mOut, mLse = inputs
        s_queries, s_keys, s_values = buffers
        query_ready, tile_ready, tile_released = barriers
        s_query = s_queries[None, None, consumer_group]  # (query row, channel)
        mma_score, mma_value = make_mma(self.tile_keys), make_mma(self.value_columns, use_register_operand=True)
        thread_score, thread_value = mma_score.get_slice(tidx), mma_value.get_slice(tidx)
        score_shape = (BLOCK_QUERY, self.tile_keys)
        coords = thread_score.partition_C(cute.make_identity_tensor(score_shape))  # (MMA, MMA_M, MMA_N)
        scores = cute.make_rmem_tensor(thread_score.partition_shape_C(score_shape), Float32)
        out_shape = thread_value.partition_shape_C((BLOCK_QUERY, self.value_columns))
        acc_output = cute.make_rmem_tensor((*out_shape, self.head_dim // self.value_columns), Float32)
        acc_output.fill(0.0)
        row_max, row_sum, rescale = (cute.make_rmem_tensor(2, Float32) for _ in range(3))
        row_max.fill(float("-inf"))
        row_sum.fill(0.0)
        prob_shape = thread_value.partition_shape_A((BLOCK_QUERY, self.group_keys))
        probabilities = cute.make_rmem_tensor((*prob_shape, self.groups), Float8E4M3FN)
        weights = cute.make_rmem_tensor((2, self.groups), Float32)
        query_start = query_block * BLOCK_QUERY
        scale = mQueryScale[head_idx, query_start // (2 * self.group_keys)]

        cute.arch.mbarrier_wait(query_ready, 0)
        for tile_offset in range(last_tile + 1):
            stage_idx, phase = tile_offset % self.stages, (tile_offset // self.stages) % 2
            tile_idx = last_tile - tile_offset
            cute.arch.mbarrier_wait(tile_ready + stage_idx, phase)
            multiply_tile(mma_score, thread_score, scores, s_query, s_keys[None, None, stage_idx], False)
            scale_causal_scores(scores, coords, mKeyScale, key_head, query_start, tile_idx * self.tile_keys, scale, tile_offset == 0, self.group_keys)
            update_softmax(scores, row_max, row_sum, rescale)
            encode_probability_tile(scores, weights, probabilities, mValueScale, key_head, tile_idx, row_max, tidx, self.group_keys, self.groups)
            accumulate_value_tile(acc_output, mma_value, thread_value, s_values, weights, probabilities, rescale, stage_idx, self.group_keys, self.groups, self.value_columns)
            if tidx == 0:
                cute.arch.mbarrier_arrive(tile_released + stage_idx)
        store_attention(mOut, mLse, acc_output, row_sum, row_max, thread_value, query_block, head_idx, tidx, self.value_columns)


@cute.jit
def launch_forward_attention(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, stream):
    CausalAttention(mQuery.shape[2])(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, stream)


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
