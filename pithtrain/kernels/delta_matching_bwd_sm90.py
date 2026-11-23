"""
Matched correction, key/value gradients and query gradients on SM90.
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Float64
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import SmemAllocator
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.delta_matching_sm90 import BLOCK_QUERY, LOG2_E, allocate_operand, allocate_staged_operand, copy_tma_operand, initialize_attention_barriers, issue_multiply_tile, launch_delta_matching_kernel, make_mma, make_operand_tma, multiply_tile, quantize_delta_matching

__all__ = ["delta_matching_bwd_prepare_sm90", "delta_matching_bwd_sm90"]

PROBABILITY_SCALE = 1.0 / 256.0
PROMOTION_ROWS = 512


@cute.jit
def reconstruct_probabilities(probabilities, coords, mKeyScale, mLse, query_head_idx, key_head_idx, query_start, key_start, query_scale, GROUP_KEYS: cutlass.Constexpr):
    rows, positions = reshape_acc_to_mn(probabilities), reshape_acc_to_mn(coords)  # (query rows, key columns)
    COLS_PER_GROUP = GROUP_KEYS // 4
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        query_pos = query_start + positions[row_idx, 0][0]
        log2_lse = mLse[query_head_idx, query_pos] * LOG2_E
        for group_idx in cutlass.range_constexpr(cute.size(rows, mode=[1]) // COLS_PER_GROUP):
            log2_scale = (query_scale * mKeyScale[key_head_idx, key_start // GROUP_KEYS + group_idx]) * LOG2_E
            for col_idx in cutlass.range_constexpr(COLS_PER_GROUP):
                elem_col = group_idx * COLS_PER_GROUP + col_idx
                key_pos = key_start + positions[row_idx, elem_col][1]
                exponent = rows[row_idx, elem_col] * log2_scale - log2_lse + 8.0
                rows[row_idx, elem_col] = cute.math.exp2(cute.arch.fmin(exponent, 12.0), fastmath=True) if key_pos <= query_pos else Float32(0.0)


@cute.jit
def compute_backward_products(mma, mma_thread, probabilities, probability_grad, buffers):
    s_query, s_key, s_grad_out, s_value = buffers
    issue_multiply_tile(mma, mma_thread, probabilities, s_query, s_key, False)
    issue_multiply_tile(mma, mma_thread, probability_grad, s_grad_out, s_value, False)
    warpgroup.wait_group(0)


@cute.jit
def reduce_delta_tile(probabilities, probability_grad, coords, acc_delta, mValueScale, key_head_idx, key_start, grad_out_scale, GROUP_KEYS: cutlass.Constexpr):
    prob_rows, grad_rows = reshape_acc_to_mn(probabilities), reshape_acc_to_mn(probability_grad)  # (query rows, key columns)
    positions = reshape_acc_to_mn(coords)  # (query rows, key columns)
    COLS_PER_GROUP = 32 // 4
    for row_idx in cutlass.range_constexpr(cute.size(prob_rows, mode=[0])):
        for group_idx in cutlass.range_constexpr(cute.size(prob_rows, mode=[1]) // COLS_PER_GROUP):
            group_col = group_idx * COLS_PER_GROUP
            key_pos = key_start + positions[row_idx, group_col][1]
            product_scale = grad_out_scale * mValueScale[key_head_idx, key_pos // GROUP_KEYS]
            partial = Float32(0.0)
            for col_idx in cutlass.range_constexpr(COLS_PER_GROUP):
                elem_idx = group_col + col_idx
                partial += prob_rows[row_idx, elem_idx] * (grad_rows[row_idx, elem_idx] * product_scale)
            acc_delta[row_idx] += (warp_reduce(partial, operator.add, threads_in_group=4) * PROBABILITY_SCALE).to(Float64)


class MatchedCorrection:
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.group_keys, self.tile_keys = 8192 // head_dim, 128
        self.stages, self.consumer_groups, self.load_group = 2, 2, 0
        self.consumer_registers, self.load_registers = 200, 24

    @cute.jit
    def __call__(self, mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, stream):
        H, N, D = mQuery.shape
        query_copy, grad_copy = make_operand_tma(mQuery, BLOCK_QUERY, D), make_operand_tma(mGrad, BLOCK_QUERY, D)
        key_copy, value_copy = make_operand_tma(mKey, self.tile_keys, D), make_operand_tma(mValue, self.tile_keys, D)
        tensors = (mQueryScale, mKeyScale, mValueScale, mGradScale, mLse, mDelta)
        self.kernel(tensors, (query_copy, grad_copy, key_copy, value_copy)).launch(grid=(N // (BLOCK_QUERY * self.consumer_groups), H, 1), block=(384, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, tensors, copies):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp_group = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        query_block = bx * self.consumer_groups
        last_tile = query_block * BLOCK_QUERY // self.tile_keys
        key_head = by // (cute.size(tensors[0], mode=[0]) // cute.size(tensors[1], mode=[0]))
        allocator = SmemAllocator()
        s_query = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.consumer_groups)
        s_grad = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.consumer_groups)
        s_keys = allocate_staged_operand(allocator, self.tile_keys, self.head_dim, self.stages)
        s_values = allocate_staged_operand(allocator, self.tile_keys, self.head_dim, self.stages)
        buffers = (s_query, s_grad, s_keys, s_values)
        barriers = initialize_attention_barriers(allocator, tidx, self.stages, self.consumer_groups)

        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if tidx == 0:
                self.load_loop(copies, buffers, barriers, query_block, by, key_head, last_tile)
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            consumer_idx = warp_group - 1
            self.compute_loop(tensors, buffers, barriers, query_block + consumer_idx, by, key_head, last_tile, consumer_idx, tidx % 128)

    @cute.jit
    def load_loop(self, copies, buffers, barriers, query_block_idx, query_head_idx, key_head_idx, last_tile):
        query_copy, grad_copy, key_copy, value_copy = copies
        s_queries, s_grads, s_keys, s_values = buffers
        resident_ready, tile_ready, tile_released = barriers
        cute.arch.mbarrier_arrive_and_expect_tx(resident_ready, 2 * cute.size(s_queries))
        for query_idx in cutlass.range_constexpr(self.consumer_groups):
            copy_tma_operand(query_copy, s_queries[None, None, query_idx], query_block_idx + query_idx, 0, query_head_idx, resident_ready)
            copy_tma_operand(grad_copy, s_grads[None, None, query_idx], query_block_idx + query_idx, 0, query_head_idx, resident_ready)
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            if key_block_idx >= self.stages:
                cute.arch.mbarrier_wait(tile_released + stage_idx, phase ^ 1)
            ready = tile_ready + stage_idx
            cute.arch.mbarrier_arrive_and_expect_tx(ready, 2 * self.tile_keys * self.head_dim)
            copy_tma_operand(key_copy, s_keys[None, None, stage_idx], key_block_idx, 0, key_head_idx, ready)
            copy_tma_operand(value_copy, s_values[None, None, stage_idx], key_block_idx, 0, key_head_idx, ready)

    @cute.jit
    def compute_loop(self, tensors, buffers, barriers, query_block_idx, query_head_idx, key_head_idx, last_tile, consumer_idx, tidx):
        mQueryScale, mKeyScale, mValueScale, mGradScale, mLse, mDelta = tensors
        s_queries, s_grads, s_keys, s_values = buffers
        resident_ready, tile_ready, tile_released = barriers
        s_query, s_grad = s_queries[None, None, consumer_idx], s_grads[None, None, consumer_idx]
        query_start = query_block_idx * BLOCK_QUERY
        scale_idx = query_start // (self.group_keys * 2)
        query_scale, grad_out_scale = mQueryScale[query_head_idx, scale_idx], mGradScale[query_head_idx, scale_idx]
        mma = make_mma(self.tile_keys)
        mma_thread = mma.get_slice(tidx)
        shape = (BLOCK_QUERY, self.tile_keys)
        coords = mma_thread.partition_C(cute.make_identity_tensor(shape))  # (MMA, MMA_M, MMA_N)
        probabilities, probability_grad = (cute.make_rmem_tensor(mma_thread.partition_shape_C(shape), Float32) for _ in range(2))
        acc_delta = cute.make_rmem_tensor(2, Float64)
        acc_delta.fill(0.0)

        cute.arch.mbarrier_wait(resident_ready, 0)
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            cute.arch.mbarrier_wait(tile_ready + stage_idx, phase)
            products = (s_query, s_keys[None, None, stage_idx], s_grad, s_values[None, None, stage_idx])
            compute_backward_products(mma, mma_thread, probabilities, probability_grad, products)
            if tidx == 0:
                cute.arch.mbarrier_arrive(tile_released + stage_idx)
            key_start = key_block_idx * self.tile_keys
            reconstruct_probabilities(probabilities, coords, mKeyScale, mLse, query_head_idx, key_head_idx, query_start, key_start, query_scale, self.group_keys)
            reduce_delta_tile(probabilities, probability_grad, coords, acc_delta, mValueScale, key_head_idx, key_start, grad_out_scale, self.group_keys)

        if tidx % 4 == 0:
            positions = reshape_acc_to_mn(coords)
            for row_idx in cutlass.range_constexpr(cute.size(acc_delta)):
                mDelta[query_head_idx, query_start + positions[row_idx, 0][0]] = acc_delta[row_idx].to(Float32)


@cute.jit
def launch_matched_delta(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, stream):
    MatchedCorrection(mQuery.shape[2])(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, stream)


@cute.jit
def center_score_gradient(probabilities, probability_grad, coords, scales, mDelta, query_head_idx, key_head_idx, query_start, key_start, grad_out_scale, GROUP_KEYS: cutlass.Constexpr):
    mKeyScale, mValueScale = scales
    amax = Float32(0.0)
    for elem_idx in cutlass.range_constexpr(cute.size(probabilities)):
        query_row, key_col = coords[elem_idx]
        scale_idx = (key_start + key_col) // GROUP_KEYS
        key_scale, value_scale = mKeyScale[key_head_idx, scale_idx], mValueScale[key_head_idx, scale_idx]
        delta = mDelta[query_head_idx, query_start + query_row]
        product_scale = grad_out_scale * value_scale * key_scale
        centered = probability_grad[elem_idx] * product_scale - delta * key_scale
        score_grad = probabilities[elem_idx] * centered
        probability_grad[elem_idx] = score_grad
        amax = cute.arch.fmax(amax, abs(score_grad))
    return amax


@cute.jit
def choose_score_scale(amax, s_amax, tidx, max_weight, query_scale):
    amax = warp_reduce(amax, cute.arch.fmax)
    if tidx % 32 == 0:
        s_amax[tidx // 32] = amax
    cute.arch.barrier(barrier_id=1, number_of_threads=256)
    for warp_idx in cutlass.range_constexpr(8):
        amax = cute.arch.fmax(amax, s_amax[warp_idx])
    base_scale = amax * (1.0 / 448.0)
    floor_scale = (max_weight * 256.0e-8) * cute.arch.rcp_approx(query_scale)
    return cute.arch.fmax(base_scale, floor_scale) if base_scale >= 1e-30 else Float32(0.0)


@cute.jit
def store_score_operands(probabilities, score_grad, coords, s_probability, s_score_grad, mScoreGrad, score_head_idx, key_start, scale):
    rcp = cute.arch.rcp_approx(scale) if scale > 0.0 else Float32(0.0)
    for elem_idx in cutlass.range_constexpr(cute.size(probabilities)):
        row_idx, col_idx = coords[elem_idx]
        s_probability[col_idx, row_idx] = cute.arch.fmin(probabilities[elem_idx], 448.0).to(Float8E4M3FN)
        grad_code = (score_grad[elem_idx] * rcp).to(Float8E4M3FN)
        s_score_grad[col_idx, row_idx] = grad_code
        mScoreGrad[score_head_idx, row_idx, key_start + col_idx] = grad_code
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier(barrier_id=1, number_of_threads=256)


@cute.jit
def update_key_value_gradients(mma, mma_thread, inputs, operands, accumulators, channel_block, weights, is_first, is_boundary, TILE_KEYS: cutlass.Constexpr):
    s_query_t, s_grad_t = inputs
    s_probability, s_score_grad = operands
    acc_key, acc_value, value_window = accumulators
    weight_key, weight_value, previous_weight = weights
    ratio = cute.arch.fmin(cute.arch.fmax(previous_weight * cute.arch.rcp_approx(weight_value), 1e-8), 1e8)
    for channel_idx in cutlass.range_constexpr(cute.size(acc_key, mode=[3])):
        channel_tile_idx = channel_block + channel_idx
        query_tile = cute.local_tile(s_query_t, (64, BLOCK_QUERY), (channel_tile_idx, 0))  # (channel, query row)
        grad_tile = cute.local_tile(s_grad_t, (64, BLOCK_QUERY), (channel_tile_idx, 0))  # (channel, query row)
        key_gradient = acc_key[None, None, None, channel_idx]  # (MMA, MMA_M, MMA_N)
        value_gradient = acc_value[None, None, None, channel_idx]  # (MMA, MMA_M, MMA_N)
        accumulated = value_window[None, None, None, channel_idx]  # (MMA, MMA_M, MMA_N)
        tile_key = cute.make_rmem_tensor(mma_thread.partition_shape_C((64, TILE_KEYS)), Float32)
        multiply_tile(mma, mma_thread, tile_key, query_tile, s_score_grad, False)
        key_gradient.store(key_gradient.load() + tile_key.load() * weight_key)
        accumulated.store(accumulated.load() * ratio)
        multiply_tile(mma, mma_thread, accumulated, grad_tile, s_probability, not is_first)
        if is_boundary:
            value_gradient.store(value_gradient.load() + accumulated.load() * weight_value)


@cute.jit
def write_key_value_gradients(mGradKey, mGradValue, mKeyScale, accumulators, mma_thread, key_block_idx, query_head_idx, key_head_idx, channel_block, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr):
    grad_key, grad_value, _ = accumulators
    N, D = cute.size(mGradKey, mode=[1]), cute.size(mGradKey, mode=[2])
    key_matrix = cute.make_tensor(mGradKey[query_head_idx, None, None].iterator, cute.make_layout((D, N), stride=(1, D)))  # (channel, key)
    value_matrix = cute.make_tensor(mGradValue[query_head_idx, None, None].iterator, cute.make_layout((D, N), stride=(1, D)))  # (channel, key)
    coords = mma_thread.partition_C(cute.make_identity_tensor((64, TILE_KEYS)))  # (MMA, MMA_M, MMA_N)
    for slice_idx in cutlass.range_constexpr(cute.size(grad_key, mode=[3])):
        channel_tile_idx = channel_block + slice_idx
        key_tile = cute.local_tile(key_matrix, (64, TILE_KEYS), (channel_tile_idx, key_block_idx))  # (channel, key)
        value_tile = cute.local_tile(value_matrix, (64, TILE_KEYS), (channel_tile_idx, key_block_idx))  # (channel, key)
        key_out, value_out = mma_thread.partition_C(key_tile), mma_thread.partition_C(value_tile)  # (MMA, MMA_M, MMA_N)
        keys, values = grad_key[None, None, None, slice_idx], grad_value[None, None, None, slice_idx]
        for elem_idx in cutlass.range_constexpr(cute.size(keys)):
            scale = mKeyScale[key_head_idx, key_block_idx * 2 + coords[elem_idx][1] // GROUP_KEYS]
            key_out[elem_idx], value_out[elem_idx] = keys[elem_idx] * cute.arch.rcp_approx(scale), values[elem_idx]


class KeyValueGradients:
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.group_keys = 8192 // head_dim
        self.tile_keys = 2 * self.group_keys
        self.stages, self.consumer_groups, self.load_group = 2, 2, 0
        self.consumer_registers, self.load_registers = 232, 24

    @cute.jit
    def __call__(self, mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream):
        H, N, D = mQuery.shape
        key_copy, value_copy = make_operand_tma(mKey, self.tile_keys, D), make_operand_tma(mValue, self.tile_keys, D)
        query_copy, grad_copy = make_operand_tma(mQuery, BLOCK_QUERY, D), make_operand_tma(mGrad, BLOCK_QUERY, D)
        query_t_copy, grad_t_copy = make_operand_tma(mQueryT, D, BLOCK_QUERY), make_operand_tma(mGradT, D, BLOCK_QUERY)
        copies = (key_copy, value_copy, query_copy, grad_copy, query_t_copy, grad_t_copy)
        tensors = (mQueryScale, mKeyScale, mValueScale, mGradScale, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue)
        self.kernel(tensors, copies).launch(grid=(N // self.tile_keys, H, 1), block=(384, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, tensors, copies):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = warp // 4
        mQueryScale, mKeyScale, _, _, mLse, _, _, _, _, _ = tensors
        num_queries = cute.size(mLse, mode=[1]) // BLOCK_QUERY
        key_head = by // (cute.size(mQueryScale, mode=[0]) // cute.size(mKeyScale, mode=[0]))
        first_query = bx * self.tile_keys // BLOCK_QUERY
        allocator = SmemAllocator()
        s_keys, s_values = allocate_operand(allocator, self.tile_keys, self.head_dim), allocate_operand(allocator, self.tile_keys, self.head_dim)
        s_queries = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.stages)
        s_grads = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.stages)
        s_queries_t = allocate_staged_operand(allocator, self.head_dim, BLOCK_QUERY, self.stages)
        s_grads_t = allocate_staged_operand(allocator, self.head_dim, BLOCK_QUERY, self.stages)
        s_probability, s_score_grad = allocate_operand(allocator, self.tile_keys, BLOCK_QUERY), allocate_operand(allocator, self.tile_keys, BLOCK_QUERY)
        s_amax = allocator.allocate_tensor(Float32, cute.make_layout(8), 16)
        resident = (s_keys, s_values)
        streamed = (s_queries, s_grads, s_queries_t, s_grads_t)
        barriers = initialize_attention_barriers(allocator, tidx, self.stages, self.consumer_groups)

        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if tidx == 0:
                self.load_loop(copies, resident, streamed, barriers, bx, by, key_head, first_query, num_queries)
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            self.compute_loop(tensors, resident, streamed, (s_probability, s_score_grad, s_amax), barriers, bx, by, key_head, first_query, num_queries, tidx - 128, warp_group - 1)

    @cute.jit
    def load_loop(self, copies, resident, streamed, barriers, key_block_idx, query_head_idx, key_head_idx, first_query, num_queries):
        key_copy, value_copy, query_copy, grad_copy, query_t_copy, grad_t_copy = copies
        s_keys, s_values = resident
        s_queries, s_grads, s_queries_t, s_grads_t = streamed
        resident_ready, tile_ready, tile_released = barriers
        cute.arch.mbarrier_arrive_and_expect_tx(resident_ready, 2 * self.tile_keys * self.head_dim)
        copy_tma_operand(key_copy, s_keys, key_block_idx, 0, key_head_idx, resident_ready)
        copy_tma_operand(value_copy, s_values, key_block_idx, 0, key_head_idx, resident_ready)
        for query_block_idx in range(first_query, num_queries):
            offset = query_block_idx - first_query
            stage_idx, phase = offset % self.stages, (offset // self.stages) % 2
            if offset >= self.stages:
                cute.arch.mbarrier_wait(tile_released + stage_idx, phase ^ 1)
            ready = tile_ready + stage_idx
            cute.arch.mbarrier_arrive_and_expect_tx(ready, 4 * BLOCK_QUERY * self.head_dim)
            copy_tma_operand(query_copy, s_queries[None, None, stage_idx], query_block_idx, 0, query_head_idx, ready)
            copy_tma_operand(grad_copy, s_grads[None, None, stage_idx], query_block_idx, 0, query_head_idx, ready)
            copy_tma_operand(query_t_copy, s_queries_t[None, None, stage_idx], 0, query_block_idx, query_head_idx, ready)
            copy_tma_operand(grad_t_copy, s_grads_t[None, None, stage_idx], 0, query_block_idx, query_head_idx, ready)

    @cute.jit
    def compute_loop(self, tensors, resident, streamed, scratch, barriers, key_block_idx, query_head_idx, key_head_idx, first_query, num_queries, tidx, consumer_idx):
        mQueryScale, mKeyScale, mValueScale, mGradScale, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue = tensors
        s_keys, s_values = resident
        s_queries, s_grads, s_queries_t, s_grads_t = streamed
        s_probability, s_score_grad, s_amax = scratch
        resident_ready, tile_ready, tile_released = barriers
        s_key = cute.local_tile(s_keys, (self.group_keys, self.head_dim), (consumer_idx, 0))  # (key, channel)
        s_value = cute.local_tile(s_values, (self.group_keys, self.head_dim), (consumer_idx, 0))  # (key, channel)
        s_prob_group = cute.local_tile(s_probability, (self.group_keys, BLOCK_QUERY), (consumer_idx, 0))  # (key, query)
        s_grad_group = cute.local_tile(s_score_grad, (self.group_keys, BLOCK_QUERY), (consumer_idx, 0))  # (key, query)
        mma_score, mma_grad = make_mma(self.group_keys), make_mma(self.tile_keys)
        thread_score, thread_grad = mma_score.get_slice(tidx % 128), mma_grad.get_slice(tidx % 128)
        score_shape = (BLOCK_QUERY, self.group_keys)
        coords = thread_score.partition_C(cute.make_identity_tensor(score_shape))  # (MMA, MMA_M, MMA_N)
        probabilities, probability_grad = (cute.make_rmem_tensor(thread_score.partition_shape_C(score_shape), Float32) for _ in range(2))
        channel_slices = self.head_dim // (2 * 64)
        grad_shape = (*thread_grad.partition_shape_C((64, self.tile_keys)), channel_slices)
        acc_key, acc_value, value_window = (cute.make_rmem_tensor(grad_shape, Float32) for _ in range(3))
        acc_key.fill(0.0)
        acc_value.fill(0.0)
        value_window.fill(0.0)
        accumulated = (acc_key, acc_value, value_window)
        max_weight, previous_weight = Float32(0.0), Float32(1.0)
        key_start = key_block_idx * self.tile_keys + consumer_idx * self.group_keys

        cute.arch.mbarrier_wait(resident_ready, 0)
        for query_block_idx in range(first_query, num_queries):
            offset = query_block_idx - first_query
            stage_idx, phase = offset % self.stages, (offset // self.stages) % 2
            cute.arch.mbarrier_wait(tile_ready + stage_idx, phase)
            s_query, s_grad_out = s_queries[None, None, stage_idx], s_grads[None, None, stage_idx]
            products = (s_query, s_key, s_grad_out, s_value)
            compute_backward_products(mma_score, thread_score, probabilities, probability_grad, products)
            query_start = query_block_idx * BLOCK_QUERY
            scale_idx = query_start // (2 * self.group_keys)
            query_scale, grad_out_scale = mQueryScale[query_head_idx, scale_idx], mGradScale[query_head_idx, scale_idx]
            reconstruct_probabilities(probabilities, coords, mKeyScale, mLse, query_head_idx, key_head_idx, query_start, key_start, query_scale, self.group_keys)
            amax = center_score_gradient(probabilities, probability_grad, coords, (mKeyScale, mValueScale), mDelta, query_head_idx, key_head_idx, query_start, key_start, grad_out_scale, self.group_keys)
            score_scale = choose_score_scale(amax, s_amax, tidx, max_weight, query_scale)
            score_head_idx = query_head_idx * num_queries + query_block_idx
            if tidx == 0:
                mScoreScale[score_head_idx, key_block_idx] = score_scale
            store_score_operands(probabilities, probability_grad, coords, s_prob_group, s_grad_group, mScoreGrad, score_head_idx, key_start, score_scale)

            weight_key, weight_value = (score_scale * PROBABILITY_SCALE) * query_scale, grad_out_scale * PROBABILITY_SCALE
            is_boundary = (query_start + BLOCK_QUERY) % PROMOTION_ROWS == 0 or query_block_idx + 1 == num_queries
            is_first = query_start % PROMOTION_ROWS == 0 or query_block_idx == first_query
            twins = (s_queries_t[None, None, stage_idx], s_grads_t[None, None, stage_idx])
            update_key_value_gradients(mma_grad, thread_grad, twins, (s_probability, s_score_grad), accumulated, consumer_idx * channel_slices, (weight_key, weight_value, previous_weight), is_first, is_boundary, self.tile_keys)
            max_weight, previous_weight = cute.arch.fmax(max_weight, weight_key), weight_value
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            if tidx % 128 == 0:
                cute.arch.mbarrier_arrive(tile_released + stage_idx)
        write_key_value_gradients(mGradKey, mGradValue, mKeyScale, accumulated, thread_grad, key_block_idx, query_head_idx, key_head_idx, consumer_idx * channel_slices, self.group_keys, self.tile_keys)


@cute.jit
def launch_key_value_gradients(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream):
    KeyValueGradients(mQuery.shape[2])(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream)


@cute.jit
def apply_query_tile(mma, mma_thread, acc_window, s_score_grad, s_keys, scale, window_scale):
    ratio = cute.arch.fmin(cute.arch.fmax(window_scale * cute.arch.rcp_approx(scale), 1e-8), 1e8)
    acc_window.store(acc_window.load() * ratio)
    multiply_tile(mma, mma_thread, acc_window, s_score_grad, s_keys, window_scale > 0.0)


class QueryGradients:
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.tile_keys = 16384 // head_dim
        self.consumer_groups, self.load_group, self.stages = 2, 0, 3
        self.consumer_registers, self.load_registers = 152, 24
        self.channel_width = head_dim // self.consumer_groups

    @cute.jit
    def __call__(self, mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale: cutlass.Constexpr, stream):
        H, N, D = mGradQuery.shape
        head_group = H // cute.size(mKeyT, mode=[0])
        score_copy, key_copy = make_operand_tma(mScoreGrad, BLOCK_QUERY, self.tile_keys), make_operand_tma(mKeyT, D, self.tile_keys)
        self.kernel(mScoreScale, mGradQuery, (score_copy, key_copy), head_group, softmax_scale).launch(grid=(N // BLOCK_QUERY, H, 1), block=(384, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mScoreScale, mGradQuery, copies, HEAD_GROUP: cutlass.Constexpr, softmax_scale: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp_group = cute.arch.make_warp_uniform(cute.arch.warp_idx()) // 4
        key_head_idx = by // HEAD_GROUP
        last_tile = bx * BLOCK_QUERY // self.tile_keys
        num_query_blocks = cute.size(mGradQuery, mode=[1]) // BLOCK_QUERY
        score_head_idx = by * num_query_blocks + bx
        allocator = SmemAllocator()
        s_score_grad = allocate_staged_operand(allocator, BLOCK_QUERY, self.tile_keys, self.stages)
        s_key_t = allocate_staged_operand(allocator, self.head_dim, self.tile_keys, self.stages)
        buffers = (s_score_grad, s_key_t)
        _, tile_ready, tile_released = initialize_attention_barriers(allocator, tidx, self.stages, self.consumer_groups)
        barriers = (tile_ready, tile_released)

        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if tidx == 0:
                self.load_loop(copies, buffers, barriers, score_head_idx, key_head_idx, last_tile)
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            self.compute_loop(mScoreScale, mGradQuery, buffers, barriers, bx, by, score_head_idx, last_tile, warp_group - 1, tidx % 128, softmax_scale)

    @cute.jit
    def load_loop(self, copies, buffers, barriers, score_head_idx, key_head_idx, last_tile):
        score_copy, key_copy = copies
        s_score_grad, s_key_t = buffers
        tile_ready, tile_released = barriers
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            if key_block_idx >= self.stages:
                cute.arch.mbarrier_wait(tile_released + stage_idx, phase ^ 1)
            ready = tile_ready + stage_idx
            cute.arch.mbarrier_arrive_and_expect_tx(ready, (BLOCK_QUERY + self.head_dim) * self.tile_keys)
            copy_tma_operand(score_copy, s_score_grad[None, None, stage_idx], 0, key_block_idx, score_head_idx, ready)
            copy_tma_operand(key_copy, s_key_t[None, None, stage_idx], 0, key_block_idx, key_head_idx, ready)

    @cute.jit
    def compute_loop(self, mScoreScale, mGradQuery, buffers, barriers, query_block_idx, query_head_idx, score_head_idx, last_tile, consumer_idx, tidx, softmax_scale: cutlass.Constexpr):
        s_score_grad, s_key_t = buffers
        tile_ready, tile_released = barriers
        mma = make_mma(self.channel_width)
        mma_thread = mma.get_slice(tidx)
        shape = mma_thread.partition_shape_C((BLOCK_QUERY, self.channel_width))
        acc_query, acc_window = cute.make_rmem_tensor(shape, Float32), cute.make_rmem_tensor(shape, Float32)
        acc_query.fill(0.0)
        acc_window.fill(0.0)
        max_scale, window_scale = Float32(0.0), Float32(0.0)
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            cute.arch.mbarrier_wait(tile_ready + stage_idx, phase)
            scale = mScoreScale[score_head_idx, key_block_idx]
            if scale > 0.0 and scale >= max_scale * 1e-8:
                key_tile = cute.local_tile(s_key_t[None, None, stage_idx], (self.channel_width, self.tile_keys), (consumer_idx, 0))  # (channel, key)
                apply_query_tile(mma, mma_thread, acc_window, s_score_grad[None, None, stage_idx], key_tile, scale, window_scale)
                max_scale, window_scale = cute.arch.fmax(max_scale, scale), scale
            if (key_block_idx + 1) % (PROMOTION_ROWS // self.tile_keys) == 0 or key_block_idx == last_tile:
                if window_scale > 0.0:
                    acc_query.store(acc_query.load() + acc_window.load() * window_scale)
                    window_scale = Float32(0.0)
            if tidx == 0:
                cute.arch.mbarrier_arrive(tile_released + stage_idx)
        tile = cute.local_tile(mGradQuery[query_head_idx, None, None], (BLOCK_QUERY, self.channel_width), (query_block_idx, consumer_idx))
        mma_thread.partition_C(tile).store((acc_query.load() * (softmax_scale * PROBABILITY_SCALE)).to(BFloat16))


@cute.jit
def launch_query_gradients(mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale: cutlass.Constexpr, stream):
    QueryGradients(mGradQuery.shape[2])(mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale, stream)


def delta_matching_bwd_prepare_sm90(grad_out, state):
    """
    Quantize dO and compute the matched correction from the backward probability and dP products.
    """
    lse, query, query_scale, _, key, key_scale, _, value, value_scale = state
    grad, grad_scale, grad_t = quantize_delta_matching(grad_out, 16384 // grad_out.shape[-1], is_gradient=True)
    delta = torch.empty_like(lse)
    launch_delta_matching_kernel("delta_matching_delta", launch_matched_delta, (query, query_scale, key, key_scale, value, value_scale, grad, grad_scale, lse, delta))
    return grad, grad_scale, grad_t, delta


def delta_matching_bwd_sm90(grad_out, state, backward_state, softmax_scale):
    """
    Compute the three BF16 gradients with E4M3 operands and deterministic GQA reduction.
    """
    lse, query, query_scale, query_t, key, key_scale, key_t, value, value_scale = state
    grad, grad_scale, grad_t, delta = backward_state
    B, N, H, D = grad_out.shape
    num_key_heads = key.shape[0] // B
    query_blocks = N // BLOCK_QUERY
    score_grad = torch.empty((B * H * query_blocks, BLOCK_QUERY, N), dtype=query.dtype, device=query.device)
    score_scale = torch.empty((B * H * query_blocks, N // (16384 // D)), dtype=torch.float32, device=query.device)
    grad_query = torch.empty_like(grad, dtype=torch.bfloat16)
    grad_key, grad_value = (torch.empty_like(grad, dtype=torch.float32) for _ in range(2))

    launch_delta_matching_kernel("delta_matching_kv_gradients", launch_key_value_gradients, (query, query_scale, key, key_scale, value, value_scale, query_t, grad, grad_scale, grad_t, lse, delta, score_grad, score_scale, grad_key, grad_value))
    launch_delta_matching_kernel("delta_matching_query_gradients", launch_query_gradients, (key_t, score_grad, score_scale, grad_query), (softmax_scale,))
    grad_query = grad_query.reshape(B, H, N, D).transpose(1, 2)
    grad_key, grad_value = (tensor.reshape(B, num_key_heads, H // num_key_heads, N, D).sum(dim=2).to(torch.bfloat16).transpose(1, 2) for tensor in (grad_key, grad_value))
    return grad_query, grad_key, grad_value
