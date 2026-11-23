"""
Matched correction, key/value gradients and query gradients on SM90.
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Float64, Int32, Int64, Uint8, Uint16, Uint32
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import SmemAllocator
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.delta_matching_sm90 import BLOCK_QUERY, LOG2_E, allocate_operand, allocate_staged_operand, copy_tma_operand, initialize_attention_barriers, issue_multiply_tile, launch_delta_matching_kernel, make_mma, make_operand_layout, make_operand_tma, multiply_tile, quantize_delta_matching, quantizer_input

__all__ = ["delta_matching_bwd_prepare_sm90", "delta_matching_bwd_sm90"]

# Pi is scaled by 2**8 until a gradient or correction is promoted.
PROBABILITY_SCALE = 1.0 / 256.0
PROMOTION_ROWS = 512
GRAD_QUANT_THREADS = 256


@cute.jit
def load_log2_lse(mLse, coords, query_head_idx, query_start):
    positions = reshape_acc_to_mn(coords)  # (query rows, key columns)
    lse_rows = cute.make_rmem_tensor(cute.size(positions, mode=[0]), Float32)
    for row_idx in cutlass.range_constexpr(cute.size(lse_rows)):
        query_pos = query_start + positions[row_idx, 0][0]
        lse_rows[row_idx] = mLse[query_head_idx, query_pos] * LOG2_E
    return lse_rows


@cute.jit
def reconstruct_probabilities(probabilities, coords, mKeyScale, lse_rows, key_head_idx, query_start, key_start, query_scale, GROUP_KEYS: cutlass.Constexpr):
    rows, positions = reshape_acc_to_mn(probabilities), reshape_acc_to_mn(coords)  # (query rows, key columns)
    COLS_PER_GROUP = GROUP_KEYS // 4
    for group_idx in cutlass.range_constexpr(cute.size(rows, mode=[1]) // COLS_PER_GROUP):
        log2_scale = (query_scale * mKeyScale[key_head_idx, key_start // GROUP_KEYS + group_idx]) * LOG2_E
        for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
            query_pos = query_start + positions[row_idx, 0][0]
            for col_idx in cutlass.range_constexpr(COLS_PER_GROUP):
                elem_col = group_idx * COLS_PER_GROUP + col_idx
                key_pos = key_start + positions[row_idx, elem_col][1]
                exponent = rows[row_idx, elem_col] * log2_scale - lse_rows[row_idx] + 8.0
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
    # Match the kernel's own FP32 Pi and dequantized dP, then combine 32-key partials in FP64.
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
    """
    Recompute FP8 code products and form the backward row correction.
    """
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.group_keys, self.tile_keys = 8192 // head_dim, 128
        self.stages, self.consumer_groups, self.load_group = 2, 2, 0
        self.consumer_registers, self.load_registers = 200, 24
        self.split_key_value = head_dim == 256
        self.key_load_warp, self.value_load_warp = 1, 2

    @cute.jit
    def __call__(self, mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, stream):
        H, N, D = mQuery.shape
        query_copy, grad_copy = make_operand_tma(mQuery, BLOCK_QUERY, D), make_operand_tma(mGrad, BLOCK_QUERY, D)
        key_copy, value_copy = make_operand_tma(mKey, self.tile_keys, D), make_operand_tma(mValue, self.tile_keys, D)
        tensors = (mQueryScale, mKeyScale, mValueScale, mGradScale, mLse, mDelta)
        query_tiles = N // (BLOCK_QUERY * self.consumer_groups)
        self.kernel(tensors, (query_copy, grad_copy, key_copy, value_copy), query_tiles).launch(grid=(H, query_tiles, 1), block=(384, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, tensors, copies, QUERY_TILES: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        head, tile, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = warp // 4
        query_block = (QUERY_TILES - 1 - tile) * self.consumer_groups
        last_tile = query_block * BLOCK_QUERY // self.tile_keys
        key_head = head // (cute.size(tensors[0], mode=[0]) // cute.size(tensors[1], mode=[0]))
        allocator = SmemAllocator()
        s_query = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.consumer_groups)
        s_grad = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.consumer_groups)
        s_keys = allocate_staged_operand(allocator, self.tile_keys, self.head_dim, self.stages)
        s_values = allocate_staged_operand(allocator, self.tile_keys, self.head_dim, self.stages)
        buffers = (s_query, s_grad, s_keys, s_values)
        ring_slots = 2 * self.stages if cutlass.const_expr(self.split_key_value) else self.stages
        readers = 4 * self.consumer_groups if cutlass.const_expr(self.split_key_value) else self.consumer_groups
        barriers = initialize_attention_barriers(allocator, tidx, ring_slots, readers)

        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if cutlass.const_expr(self.split_key_value):
                if warp == 0:
                    self.load_resident(copies, buffers, barriers[0], query_block, head, tidx % 32)
                elif warp == self.key_load_warp:
                    self.load_key_value_stream(copies[2], s_keys, barriers[1], barriers[2], key_head, last_tile, tidx % 32)
                elif warp == self.value_load_warp:
                    self.load_key_value_stream(copies[3], s_values, barriers[1] + self.stages,
                                               barriers[2] + self.stages, key_head, last_tile, tidx % 32)
            elif tidx == 0:
                self.load_loop(copies, buffers, barriers, query_block, head, key_head, last_tile)
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            consumer_idx = warp_group - 1
            self.compute_loop(tensors, buffers, barriers, query_block + consumer_idx, head, key_head, last_tile, consumer_idx, tidx % 128)

    @cute.jit
    def load_resident(self, copies, buffers, resident_ready, query_block_idx, query_head_idx, lane):
        query_copy, grad_copy, _, _ = copies
        s_queries, s_grads, _, _ = buffers
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(resident_ready, 2 * cute.size(s_queries))
        if cutlass.const_expr(self.split_key_value):
            cute.arch.sync_warp()
        for query_idx in cutlass.range_constexpr(self.consumer_groups):
            copy_tma_operand(query_copy, s_queries[None, None, query_idx], query_block_idx + query_idx, 0, query_head_idx, resident_ready)
            copy_tma_operand(grad_copy, s_grads[None, None, query_idx], query_block_idx + query_idx, 0, query_head_idx, resident_ready)

    @cute.jit
    def load_key_value_stream(self, copy, tiles, ready, released, key_head_idx, last_tile, lane):
        for key_block_idx in range(last_tile + 1):
            stage = key_block_idx % self.stages
            phase = (key_block_idx // self.stages) % 2
            barrier = ready + stage
            if lane == 0:
                if key_block_idx >= self.stages:
                    cute.arch.mbarrier_wait(released + stage, phase ^ 1)
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, self.tile_keys * self.head_dim)
            cute.arch.sync_warp()
            copy_tma_operand(copy, tiles[None, None, stage], key_block_idx, 0, key_head_idx, barrier)

    @cute.jit
    def load_loop(self, copies, buffers, barriers, query_block_idx, query_head_idx, key_head_idx, last_tile):
        _, _, key_copy, value_copy = copies
        _, _, s_keys, s_values = buffers
        resident_ready, tile_ready, tile_released = barriers
        self.load_resident(copies, buffers, resident_ready, query_block_idx, query_head_idx, 0)
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
        lse_rows = load_log2_lse(mLse, coords, query_head_idx, query_start)

        cute.arch.mbarrier_wait(resident_ready, 0)
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            cute.arch.mbarrier_wait(tile_ready + stage_idx, phase)
            if cutlass.const_expr(self.split_key_value):
                issue_multiply_tile(mma, mma_thread, probabilities, s_query, s_keys[None, None, stage_idx], False)
                cute.arch.mbarrier_wait(tile_ready + self.stages + stage_idx, phase)
                issue_multiply_tile(mma, mma_thread, probability_grad, s_grad, s_values[None, None, stage_idx], False)
                warpgroup.wait_group(1)
                cute.arch.sync_warp()
                if tidx % 32 == 0:
                    cute.arch.mbarrier_arrive(tile_released + stage_idx)
            else:
                products = (s_query, s_keys[None, None, stage_idx], s_grad, s_values[None, None, stage_idx])
                compute_backward_products(mma, mma_thread, probabilities, probability_grad, products)
                if tidx == 0:
                    cute.arch.mbarrier_arrive(tile_released + stage_idx)
            key_start = key_block_idx * self.tile_keys
            reconstruct_probabilities(probabilities, coords, mKeyScale, lse_rows, key_head_idx, query_start, key_start, query_scale, self.group_keys)
            if cutlass.const_expr(self.split_key_value):
                warpgroup.wait_group(0)
                cute.arch.sync_warp()
                if tidx % 32 == 0:
                    cute.arch.mbarrier_arrive(tile_released + self.stages + stage_idx)
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
    # Fold the key scale into dS so dQ can multiply its codes by the original key codes.
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
def choose_score_scale(amax, s_amax, tidx, max_weight, query_scale, CONSUMERS: cutlass.Constexpr = 256):
    if cutlass.const_expr(CONSUMERS == 128):
        # A nonnegative FP32 maximum has the same ordering as its unsigned bits.
        lane_max = cute.make_rmem_tensor(1, Float32)
        lane_bits = cute.recast_tensor(lane_max, Uint32)
        lane_max[0] = amax
        lane_bits[0] = cute.arch.warp_redux_sync(lane_bits[0], "umax")
        amax = lane_max[0]
    else:
        amax = warp_reduce(amax, cute.arch.fmax)
    if tidx % 32 == 0:
        s_amax[tidx // 32] = amax
    cute.arch.barrier(barrier_id=1, number_of_threads=CONSUMERS)
    for warp_idx in cutlass.range_constexpr(CONSUMERS // 32):
        amax = cute.arch.fmax(amax, s_amax[warp_idx])
    # Bound the weight change from the largest preceding dK contribution.
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
def write_key_value_gradients(mGradKey, mGradValue, mKeyScale, accumulators, mma_thread, key_block_idx, query_head_idx, key_head_idx, channel_block, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr, CHANNEL_STRIDE: cutlass.Constexpr = 1):
    grad_key, grad_value, _ = accumulators
    N, D = cute.size(mGradKey, mode=[1]), cute.size(mGradKey, mode=[2])
    key_matrix = cute.make_tensor(mGradKey[query_head_idx, None, None].iterator, cute.make_layout((D, N), stride=(1, D)))  # (channel, key)
    value_matrix = cute.make_tensor(mGradValue[query_head_idx, None, None].iterator, cute.make_layout((D, N), stride=(1, D)))  # (channel, key)
    coords = mma_thread.partition_C(cute.make_identity_tensor((64, TILE_KEYS)))  # (MMA, MMA_M, MMA_N)
    for slice_idx in cutlass.range_constexpr(cute.size(grad_key, mode=[3])):
        channel_tile_idx = channel_block + slice_idx * CHANNEL_STRIDE
        key_tile = cute.local_tile(key_matrix, (64, TILE_KEYS), (channel_tile_idx, key_block_idx))  # (channel, key)
        value_tile = cute.local_tile(value_matrix, (64, TILE_KEYS), (channel_tile_idx, key_block_idx))  # (channel, key)
        key_out, value_out = mma_thread.partition_C(key_tile), mma_thread.partition_C(value_tile)  # (MMA, MMA_M, MMA_N)
        keys, values = grad_key[None, None, None, slice_idx], grad_value[None, None, None, slice_idx]
        for elem_idx in cutlass.range_constexpr(cute.size(keys)):
            scale = mKeyScale[key_head_idx, key_block_idx * 2 + coords[elem_idx][1] // GROUP_KEYS]
            key_out[elem_idx], value_out[elem_idx] = keys[elem_idx] * cute.arch.rcp_approx(scale), values[elem_idx]


class KeyValueGradients:
    """
    Accumulate dK and dV over causal query tiles and save quantized dS for dQ.
    """
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
            lse_rows = load_log2_lse(mLse, coords, query_head_idx, query_start)
            reconstruct_probabilities(probabilities, coords, mKeyScale, lse_rows, key_head_idx, query_start, key_start, query_scale, self.group_keys)
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
def make_backward_row_tma(mRows):
    H, N = mRows.shape
    rows = cute.make_tensor(mRows.iterator, cute.make_layout((N, H), stride=(1, N)))  # (query row, head)
    tile = cute.make_layout((BLOCK_QUERY, 1))
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), rows, tile, (BLOCK_QUERY, 1))


@cute.jit
def load_backward_rows(descriptor, s_rows, query_block_idx, head_idx, ready):
    atom, rows = descriptor
    tile = cute.local_tile(rows, (BLOCK_QUERY, 1), (query_block_idx, head_idx))
    destination, source = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(s_rows, 0, 2), cute.group_modes(tile, 0, 2)
    )
    cute.copy(atom, source, destination, tma_bar_ptr=ready)


@cute.jit
def stage_backward_operand(copy, buffer, row_copy, row_buffer, ready, released, offset, query_block_idx, head_idx, STAGES: cutlass.Constexpr, TRANSPOSED: cutlass.Constexpr):
    stage_idx, phase = offset % STAGES, (offset // STAGES) % 2
    if offset >= STAGES:
        cute.arch.mbarrier_wait(released + stage_idx, phase ^ 1)
    barrier = ready + stage_idx
    tile = buffer[None, None, stage_idx]
    if cutlass.const_expr(TRANSPOSED):
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, cute.size(tile))
        copy_tma_operand(copy, tile, 0, query_block_idx, head_idx, barrier)
    else:
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, cute.size(tile) + BLOCK_QUERY * 4)
        copy_tma_operand(copy, tile, query_block_idx, 0, head_idx, barrier)
        load_backward_rows(row_copy, row_buffer[None, None, stage_idx], query_block_idx, head_idx, barrier)


@cute.jit
def allocate_score_exchange(allocator, STAGES: cutlass.Constexpr, STORE_STAGES: cutlass.Constexpr):
    tile = make_operand_layout(BLOCK_QUERY, BLOCK_QUERY)
    staged = cute.make_layout(
        (*tile.outer.shape, STAGES), stride=(*tile.outer.stride, BLOCK_QUERY * BLOCK_QUERY),
    )
    staged_output = cute.make_layout((*tile.outer.shape, STORE_STAGES), stride=staged.stride)
    return (
        allocator.allocate_tensor(Float8E4M3FN, staged, 512, swizzle=tile.inner),
        allocator.allocate_tensor(Float8E4M3FN, staged, 512, swizzle=tile.inner),
        allocator.allocate_tensor(Float8E4M3FN, staged_output, 512, swizzle=tile.inner),
    )


@dsl_user_op
def pack_e4m3_pair(low, high, *, loc=None, ip=None):
    bits = llvm.inline_asm(
        T.i16(), [high.ir_value(loc=loc, ip=ip), low.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;", "=h,f,f",
        has_side_effects=False, is_align_stack=False, loc=loc, ip=ip,
    )
    return Uint16(bits)


@dsl_user_op
def store_transposed_score_matrices(destination, words, *, loc=None, ip=None):
    address = destination.toint(loc=loc, ip=ip).to(Uint32)
    llvm.inline_asm(
        None, [address.ir_value(loc=loc, ip=ip), *(word.ir_value(loc=loc, ip=ip) for word in words)],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}", has_side_effects=True, is_align_stack=False, loc=loc, ip=ip,
    )


@cute.jit
def score_query_pairs(first_row, second_row, even_lane, selector):
    # Each C lane has one key pair in each of two rows eight queries apart.
    packed = cute.arch.prmt(first_row.to(Uint32), second_row.to(Uint32), 0x5410)
    even = cute.arch.shuffle_sync(packed, offset=even_lane, mask_and_clamp=31)
    odd = cute.arch.shuffle_sync(packed, offset=even_lane + 4, mask_and_clamp=31)
    return cute.arch.prmt(even, odd, selector)


@cute.jit
def stage_pipelined_score_operands(probabilities, score_grad, coords, buffers, scale, tidx):
    s_probability, s_score_grad, s_score_output = buffers
    output_pairs = cute.make_tensor(
        cute.recast_ptr(s_score_output.iterator, dtype=Uint16), cute.make_layout(BLOCK_QUERY * BLOCK_QUERY // 2)
    )
    output_offsets = cute.make_rmem_tensor(2, Int32)
    for row in cutlass.range_constexpr(2):
        query, key = coords[2 * row]
        offset = query * BLOCK_QUERY + key
        output_offsets[row] = (offset ^ ((offset >> 3) & 0x30)) // 2

    lane = tidx % 32
    query_pair = lane // 4
    even_lane = (query_pair % 4) * 8 + lane % 4
    selector = Uint32(0x5140 if query_pair < 4 else 0x7362)
    # Each STSM row starts at the aligned query stripe of this score warp.
    query_start = (coords[0][0] // 16) * 16
    matrix_offset = lane * BLOCK_QUERY + query_start
    matrix_offset = matrix_offset ^ ((matrix_offset >> 3) & 0x30)
    probability_ptr = cute.recast_ptr(s_probability.iterator, dtype=Uint8)
    grad_ptr = cute.recast_ptr(s_score_grad.iterator, dtype=Uint8)
    rcp = cute.arch.rcp_approx(scale) if scale > 0.0 else Float32(0.0)

    for section in cutlass.range_constexpr(2):
        prob_words = cute.make_rmem_tensor(4, Uint32)
        grad_words = cute.make_rmem_tensor(4, Uint32)
        for matrix in cutlass.range_constexpr(4):
            quad = 4 * section + matrix
            elem = 4 * quad
            p0 = pack_e4m3_pair(probabilities[elem], probabilities[elem + 1])
            p1 = pack_e4m3_pair(probabilities[elem + 2], probabilities[elem + 3])
            ds0 = pack_e4m3_pair(score_grad[elem] * rcp, score_grad[elem + 1] * rcp)
            ds1 = pack_e4m3_pair(score_grad[elem + 2] * rcp, score_grad[elem + 3] * rcp)
            output_pairs[output_offsets[0] ^ (4 * quad)] = ds0
            output_pairs[output_offsets[1] ^ (4 * quad)] = ds1
            prob_words[matrix] = score_query_pairs(p0, p1, even_lane, selector)
            grad_words[matrix] = score_query_pairs(ds0, ds1, even_lane, selector)
        offset = matrix_offset + section * 32 * BLOCK_QUERY
        store_transposed_score_matrices(probability_ptr + offset, tuple(prob_words[i] for i in range(4)))
        store_transposed_score_matrices(grad_ptr + offset, tuple(grad_words[i] for i in range(4)))

    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier(barrier_id=1, number_of_threads=128)


@cute.jit
def issue_gradient_stripes(mma, mma_thread, accumulators, s_twin, s_operand, consumer_idx, accumulate):
    right = mma_thread.make_fragment_B(mma_thread.partition_B(s_operand))  # (MMA, MMA_N, MMA_K)
    warpgroup.fence()
    atom = cute.make_mma_atom(mma.op)
    for stripe_idx in cutlass.range_constexpr(cute.size(accumulators, mode=[3])):
        channel_tile_idx = consumer_idx + 2 * stripe_idx
        left_tile = cute.local_tile(s_twin, (64, BLOCK_QUERY), (channel_tile_idx, 0))  # (channel, query)
        left = mma_thread.make_fragment_A(mma_thread.partition_A(left_tile))  # (MMA, MMA_M, MMA_K)
        acc = accumulators[None, None, None, stripe_idx]  # (MMA, MMA_M, MMA_N)
        atom.set(warpgroup.Field.ACCUMULATE, accumulate)
        for reduction_idx in cutlass.range_constexpr(cute.size(left, mode=[2])):
            cute.gemm(atom, acc, left[None, None, reduction_idx], right[None, None, reduction_idx], acc)
            atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()


@cute.jit
def store_pipelined_score_tile(score_store, s_score_output, score_tile_idx):
    atom, coordinates = score_store
    target = coordinates[None, None, score_tile_idx]
    source_part, target_part = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(s_score_output, 0, 2), cute.group_modes(target, 0, 2),
    )
    cute.copy(atom, source_part, target_part)
    cute.arch.cp_async_bulk_commit_group()


@dsl_user_op
def wait_score_tile(barrier_id, participants, *, loc=None, ip=None):
    nvvm.barrier_cta_sync(
        barrier_id=Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(participants).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


class PipelinedKeyValueGradients:
    """
    Overlap score reconstruction with dK/dV using independent TMA stream producers.
    """

    def __init__(self):
        self.head_dim = 256
        self.group_keys, self.tile_keys = 32, 64
        self.query_stages, self.grad_stages = 2, 2
        self.query_t_stages, self.grad_t_stages = 1, 2
        self.grad_slot = self.query_stages
        self.query_t_slot = self.grad_slot + self.grad_stages
        self.grad_t_slot = self.query_t_slot + self.query_t_stages
        self.num_stages = self.grad_t_slot + self.grad_t_stages
        self.head_section = 4
        self.load_group, self.score_group, self.gradient_group = 0, 1, 2
        self.num_threads, self.score_stages = 512, 3
        self.score_participants = self.num_threads - 128
        self.store_stages = 2
        self.load_registers, self.score_registers, self.gradient_registers = 24, 104, 192
        self.value_register_values = 16

    @cute.jit
    def __call__(self, mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream):
        H, N, D = mQuery.shape
        key_copy, value_copy = make_operand_tma(mKey, self.tile_keys, D), make_operand_tma(mValue, self.tile_keys, D)
        query_copies = (make_operand_tma(mQuery, BLOCK_QUERY, D), make_operand_tma(mQueryT, D, BLOCK_QUERY), make_backward_row_tma(mLse))
        grad_copies = (make_operand_tma(mGrad, BLOCK_QUERY, D), make_operand_tma(mGradT, D, BLOCK_QUERY), make_backward_row_tma(mDelta))
        score_layout = cute.make_ordered_layout((BLOCK_QUERY, self.tile_keys, mScoreGrad.shape[0]), order=(1, 0, 2))
        score_tensor = cute.make_tensor(mScoreGrad.iterator, score_layout)  # (query row, key row, score tile)
        score_store = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), score_tensor, make_operand_layout(BLOCK_QUERY, self.tile_keys),
            (BLOCK_QUERY, self.tile_keys),
        )
        tensors = (mQueryScale, mKeyScale, mValueScale, mGradScale, mScoreScale, mGradKey, mGradValue)
        copies = ((key_copy, value_copy), query_copies, grad_copies, score_store)
        self.kernel(tensors, copies, N // BLOCK_QUERY).launch(
            grid=(H * (N // self.tile_keys), 1, 1), block=(self.num_threads, 1, 1),
            max_number_threads=(self.num_threads, 1, 1), min_blocks_per_mp=1, stream=stream,
        )

    @cute.kernel
    def kernel(self, tensors, copies, NUM_QUERIES: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        tile_idx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = warp // 4
        mQueryScale, mKeyScale, _, _, _, _, _ = tensors
        H = cute.size(mQueryScale, mode=[0])
        section = tile_idx // (NUM_QUERIES * self.head_section)
        tile_in_section = tile_idx % (NUM_QUERIES * self.head_section)
        heads_in_section = Int32(self.head_section)
        if cutlass.const_expr(H % self.head_section != 0):
            if section == H // self.head_section:
                heads_in_section = H % self.head_section
        key_block_idx = tile_in_section // heads_in_section
        query_head_idx = section * self.head_section + tile_in_section % heads_in_section
        key_head_idx = query_head_idx // (H // cute.size(mKeyScale, mode=[0]))
        allocator = SmemAllocator()
        resident = (
            allocate_operand(allocator, self.tile_keys, self.head_dim),
            allocate_operand(allocator, self.tile_keys, self.head_dim),
        )
        query_buffers = (
            allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.query_stages),
            allocate_staged_operand(allocator, self.head_dim, BLOCK_QUERY, self.query_t_stages),
            allocator.allocate_tensor(Float32, cute.make_ordered_layout((BLOCK_QUERY, 1, self.query_stages), order=(0, 1, 2)), 128),
        )
        grad_buffers = (
            allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.grad_stages),
            allocate_staged_operand(allocator, self.head_dim, BLOCK_QUERY, self.grad_t_stages),
            allocator.allocate_tensor(Float32, cute.make_ordered_layout((BLOCK_QUERY, 1, self.grad_stages), order=(0, 1, 2)), 128),
        )
        score_buffers = (
            *allocate_score_exchange(allocator, self.score_stages, self.store_stages),
            allocator.allocate_tensor(Float32, cute.make_layout(4), 16),
            allocator.allocate_tensor(Float32, cute.make_layout(self.score_stages), 16),
        )
        # Keep four adjacent FP32 promotion values contiguous for each lane.
        value_vectors = (self.tile_keys * self.head_dim // 256 - self.value_register_values) // 4
        value_layout = cute.make_ordered_layout((4, 256, value_vectors), order=(0, 1, 2))
        s_grad_value = allocator.allocate_tensor(Float32, value_layout, 128)
        resident_ready = allocator.allocate_array(Int64, 1)
        ready = allocator.allocate_array(Int64, self.num_stages)
        released = allocator.allocate_array(Int64, self.num_stages)
        score_released = allocator.allocate_array(Int64, self.score_stages)
        if tidx == 0:
            cute.arch.mbarrier_init(resident_ready, 1)
            for stage_idx in cutlass.range_constexpr(self.num_stages):
                cute.arch.mbarrier_init(ready + stage_idx, 1)
                consumers = 4 if stage_idx < self.query_t_slot else 8
                cute.arch.mbarrier_init(released + stage_idx, consumers)
            for stage_idx in cutlass.range_constexpr(self.score_stages):
                cute.arch.mbarrier_init(score_released + stage_idx, 8)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        barriers = (resident_ready, ready, released)
        coords = (key_block_idx, query_head_idx, key_head_idx)
        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if tidx % 32 == 0:
                self.load_loop(copies, resident, query_buffers, grad_buffers, barriers, coords, warp, NUM_QUERIES)
        elif warp_group == self.score_group:
            cute.arch.setmaxregister_decrease(self.score_registers)
            self.score_loop(tensors, resident, query_buffers, grad_buffers, score_buffers,
                            copies[3], barriers, score_released, coords, NUM_QUERIES, tidx % 128)
        else:
            cute.arch.setmaxregister_increase(self.gradient_registers)
            self.gradient_loop(tensors, query_buffers, grad_buffers, score_buffers, s_grad_value,
                               barriers, score_released, coords, NUM_QUERIES,
                               tidx - 128 * self.gradient_group, warp_group - self.gradient_group)

    @cute.jit
    def load_loop(self, copies, resident, query_buffers, grad_buffers, barriers, coords, warp, NUM_QUERIES: cutlass.Constexpr):
        (key_copy, value_copy), query_copies, grad_copies, _ = copies
        s_key, s_value = resident
        resident_ready, ready, released = barriers
        key_block_idx, query_head_idx, key_head_idx = coords
        # Each producer can fill its ring as soon as its own consumer releases a slot.
        if warp == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(resident_ready, 2 * self.tile_keys * self.head_dim)
            copy_tma_operand(key_copy, s_key, key_block_idx, 0, key_head_idx, resident_ready)
            copy_tma_operand(value_copy, s_value, key_block_idx, 0, key_head_idx, resident_ready)
            self.load_stream(query_copies, query_buffers, ready, released, key_block_idx, query_head_idx, NUM_QUERIES, self.query_stages, False)
        elif warp == 1:
            slot = self.query_t_slot
            self.load_stream(query_copies, query_buffers, ready + slot, released + slot, key_block_idx, query_head_idx, NUM_QUERIES, self.query_t_stages, True)
        elif warp == 2:
            slot = self.grad_slot
            self.load_stream(grad_copies, grad_buffers, ready + slot, released + slot, key_block_idx, query_head_idx, NUM_QUERIES, self.grad_stages, False)
        else:
            slot = self.grad_t_slot
            self.load_stream(grad_copies, grad_buffers, ready + slot, released + slot, key_block_idx, query_head_idx, NUM_QUERIES, self.grad_t_stages, True)

    @cute.jit
    def load_stream(self, copies, buffers, ready, released, first_query, head_idx, NUM_QUERIES: cutlass.Constexpr, STAGES: cutlass.Constexpr, TRANSPOSED: cutlass.Constexpr):
        operand_idx = 1 if cutlass.const_expr(TRANSPOSED) else 0
        for query_block_idx in range(first_query, NUM_QUERIES):
            stage_backward_operand(
                copies[operand_idx], buffers[operand_idx], copies[2], buffers[2], ready, released,
                query_block_idx - first_query, query_block_idx, head_idx, STAGES, TRANSPOSED,
            )

    @cute.jit
    def reconstruct_scores(self, probabilities, coords, lse_rows, query_start, key_start, log2_scales):
        GROUP_COLUMNS = self.group_keys // 4
        for group_idx in cutlass.range_constexpr(2):
            for row_idx in cutlass.range_constexpr(cute.size(probabilities, mode=[0])):
                query_pos = query_start + coords[row_idx, 0][0]
                for col_idx in cutlass.range_constexpr(GROUP_COLUMNS):
                    local_col = group_idx * GROUP_COLUMNS + col_idx
                    key_pos = key_start + coords[row_idx, local_col][1]
                    exponent = probabilities[row_idx, local_col] * log2_scales[group_idx] - lse_rows[row_idx] + 8.0
                    probabilities[row_idx, local_col] = (
                        cute.math.exp2(cute.arch.fmin(exponent, 12.0), fastmath=True)
                        if key_pos <= query_pos else Float32(0.0)
                    )

    @cute.jit
    def center_scores(self, prob_rows, grad_rows, delta_rows, key_scales, value_scales, grad_scale):
        amax = Float32(0.0)
        GROUP_COLUMNS = self.group_keys // 4
        for group_idx in cutlass.range_constexpr(2):
            key_scale, value_scale = key_scales[group_idx], value_scales[group_idx]
            product_scale = (grad_scale * value_scale) * key_scale
            for row_idx in cutlass.range_constexpr(cute.size(prob_rows, mode=[0])):
                scaled_delta = delta_rows[row_idx] * key_scale
                for col_idx in cutlass.range_constexpr(GROUP_COLUMNS):
                    local_col = group_idx * GROUP_COLUMNS + col_idx
                    centered = grad_rows[row_idx, local_col] * product_scale - scaled_delta
                    score_grad = prob_rows[row_idx, local_col] * centered
                    grad_rows[row_idx, local_col] = score_grad
                    amax = cute.arch.fmax(amax, abs(score_grad))
        return amax

    @cute.jit
    def score_loop(self, tensors, resident, query_buffers, grad_buffers, score_buffers, score_store, barriers, score_released, coords, NUM_QUERIES: cutlass.Constexpr, tidx):
        mQueryScale, mKeyScale, mValueScale, mGradScale, mScoreScale, _, _ = tensors
        key_block_idx, query_head_idx, key_head_idx = coords
        s_keys, s_values = resident
        s_queries, _, s_lse = query_buffers
        s_grads, _, s_delta = grad_buffers
        s_prob, s_score_grad, s_score_output, s_amax, s_key_weights = score_buffers
        resident_ready, ready, released = barriers
        score_coords = cute.make_identity_tensor((BLOCK_QUERY, self.tile_keys))
        mma = make_mma(self.tile_keys)
        mma_thread = mma.get_slice(tidx)
        positions = mma_thread.partition_C(score_coords)
        positions_rows = reshape_acc_to_mn(positions)
        shape = mma_thread.partition_shape_C(score_coords.shape)
        key_start = key_block_idx * self.tile_keys
        key_scales = (mKeyScale[key_head_idx, 2 * key_block_idx], mKeyScale[key_head_idx, 2 * key_block_idx + 1])
        value_scales = (mValueScale[key_head_idx, 2 * key_block_idx], mValueScale[key_head_idx, 2 * key_block_idx + 1])
        next_query_scale = mQueryScale[query_head_idx, key_block_idx]
        next_grad_scale = mGradScale[query_head_idx, key_block_idx]
        max_weight = Float32(0.0)

        cute.arch.mbarrier_wait(resident_ready, 0)
        for query_block_idx in range(key_block_idx, NUM_QUERIES):
            offset = query_block_idx - key_block_idx
            query_stage, query_phase = offset % self.query_stages, (offset // self.query_stages) % 2
            grad_stage, grad_phase = offset % self.grad_stages, (offset // self.grad_stages) % 2
            cute.arch.mbarrier_wait(ready + query_stage, query_phase)
            probabilities = cute.make_rmem_tensor(shape, Float32)
            probability_grad = cute.make_rmem_tensor(shape, Float32)
            prob_rows, grad_rows = reshape_acc_to_mn(probabilities), reshape_acc_to_mn(probability_grad)
            issue_multiply_tile(mma, mma_thread, probabilities, s_queries[None, None, query_stage], s_keys, False)
            lse_rows = cute.make_rmem_tensor(cute.size(prob_rows, mode=[0]), Float32)
            for row_idx in cutlass.range_constexpr(cute.size(lse_rows)):
                lse_rows[row_idx] = s_lse[positions_rows[row_idx, 0][0], 0, query_stage] * LOG2_E
            query_scale, grad_scale = next_query_scale, next_grad_scale
            if query_block_idx + 1 < NUM_QUERIES:
                next_query_scale = mQueryScale[query_head_idx, query_block_idx + 1]
                next_grad_scale = mGradScale[query_head_idx, query_block_idx + 1]
            cute.arch.mbarrier_wait(ready + self.grad_slot + grad_stage, grad_phase)
            issue_multiply_tile(mma, mma_thread, probability_grad, s_grads[None, None, grad_stage], s_values, False)
            warpgroup.wait_group(1)
            cute.arch.sync_warp()
            if tidx % 32 == 0:
                cute.arch.mbarrier_arrive(released + query_stage)
            log2_scales = ((query_scale * key_scales[0]) * LOG2_E, (query_scale * key_scales[1]) * LOG2_E)
            self.reconstruct_scores(prob_rows, positions_rows, lse_rows, query_block_idx * BLOCK_QUERY, key_start, log2_scales)
            delta_rows = cute.make_rmem_tensor(cute.size(prob_rows, mode=[0]), Float32)
            for row_idx in cutlass.range_constexpr(cute.size(delta_rows)):
                delta_rows[row_idx] = s_delta[positions_rows[row_idx, 0][0], 0, grad_stage]
            warpgroup.wait_group(0)
            cute.arch.sync_warp()
            if tidx % 32 == 0:
                cute.arch.mbarrier_arrive(released + self.grad_slot + grad_stage)
            amax = self.center_scores(prob_rows, grad_rows, delta_rows, key_scales, value_scales, grad_scale)
            score_stage, score_phase = offset % self.score_stages, (offset // self.score_stages) % 2
            if offset >= self.score_stages:
                cute.arch.mbarrier_wait(score_released + score_stage, score_phase ^ 1)
            output_stage = offset % self.store_stages
            if tidx == 0 and offset >= self.store_stages:
                cute.arch.cp_async_bulk_wait_group(self.store_stages - 1, read=True)
            scale = choose_score_scale(amax, s_amax, tidx, max_weight, query_scale, 128)
            key_weight = (scale * PROBABILITY_SCALE) * query_scale
            score_head_idx = query_head_idx * NUM_QUERIES + query_block_idx
            if tidx == 0:
                mScoreScale[score_head_idx, key_block_idx] = scale
                s_key_weights[score_stage] = key_weight
            s_output = s_score_output[None, None, output_stage]
            stage_pipelined_score_operands(
                probabilities, probability_grad, positions,
                (s_prob[None, None, score_stage], s_score_grad[None, None, score_stage], s_output), scale, tidx,
            )
            # The two gradient warpgroups join the score warpgroup at one
            # named barrier per slot. Only the gradient participants wait.
            cute.arch.barrier_arrive(
                barrier_id=2 + score_stage, number_of_threads=self.score_participants, aligned=False,
            )
            if tidx == 0:
                score_tile_idx = score_head_idx * NUM_QUERIES + key_block_idx
                store_pipelined_score_tile(score_store, s_output, score_tile_idx)
            max_weight = cute.arch.fmax(max_weight, key_weight)
        if tidx == 0:
            cute.arch.cp_async_bulk_wait_group(0)

    @cute.jit
    def gradient_loop(self, tensors, query_buffers, grad_buffers, score_buffers, s_grad_value, barriers, score_released, coords, NUM_QUERIES: cutlass.Constexpr, tidx, consumer_idx):
        _, mKeyScale, _, mGradScale, _, mGradKey, mGradValue = tensors
        key_block_idx, query_head_idx, key_head_idx = coords
        _, s_queries_t, _ = query_buffers
        _, s_grads_t, _ = grad_buffers
        s_prob, s_score_grad, _, _, s_key_weights = score_buffers
        _, ready, released = barriers
        mma = make_mma(self.tile_keys)
        mma_thread = mma.get_slice(tidx % 128)
        stripe_shape = mma_thread.partition_shape_C((64, self.tile_keys))
        grad_shape = (*stripe_shape, 2)
        acc_key = cute.make_rmem_tensor(grad_shape, Float32)
        acc_window = cute.make_rmem_tensor(grad_shape, Float32)
        value_registers = cute.make_rmem_tensor(self.value_register_values, Float32)
        s_value = s_grad_value[None, tidx, None]
        shared_shape = cute.make_layout(s_value.shape)
        register_window = cute.make_tensor(acc_window.iterator, value_registers.layout)
        shared_window = cute.make_tensor(acc_window.iterator + self.value_register_values, shared_shape)
        acc_key.fill(0.0)
        acc_window.fill(0.0)
        value_registers.fill(0.0)
        s_value.fill(0.0)
        previous_weight = Float32(1.0)
        next_value_weight = mGradScale[query_head_idx, key_block_idx] * PROBABILITY_SCALE
        num_query_tiles = NUM_QUERIES - key_block_idx
        for ring_idx in range(cute.ceil_div(num_query_tiles, self.score_stages)):
            # Fixed score slots fix the score operands' WGMMA descriptor offsets.
            for score_stage in cutlass.range_constexpr(self.score_stages):
                offset = ring_idx * self.score_stages + score_stage
                query_block_idx = key_block_idx + offset
                if offset < num_query_tiles:
                    query_stage, query_phase = offset % self.query_t_stages, (offset // self.query_t_stages) % 2
                    grad_stage, grad_phase = offset % self.grad_t_stages, (offset // self.grad_t_stages) % 2
                    value_weight = next_value_weight
                    if query_block_idx + 1 < NUM_QUERIES:
                        next_value_weight = mGradScale[query_head_idx, query_block_idx + 1] * PROBABILITY_SCALE
                    query_start = query_block_idx * BLOCK_QUERY
                    is_first = query_start % PROMOTION_ROWS == 0 or query_block_idx == key_block_idx
                    if not is_first:
                        ratio = cute.arch.fmin(cute.arch.fmax(previous_weight * cute.arch.rcp_approx(value_weight), 1e-8), 1e8)
                        acc_window.store(acc_window.load() * ratio)
                    wait_score_tile(2 + score_stage, self.score_participants)
                    key_weight = s_key_weights[score_stage]
                    cute.arch.mbarrier_wait(ready + self.grad_t_slot + grad_stage, grad_phase)
                    issue_gradient_stripes(mma, mma_thread, acc_window, s_grads_t[None, None, grad_stage], s_prob[None, None, score_stage], consumer_idx, not is_first)
                    cute.arch.mbarrier_wait(ready + self.query_t_slot + query_stage, query_phase)
                    self.accumulate_key_stripes(
                        mma, mma_thread, acc_key, s_queries_t[None, None, query_stage], s_score_grad[None, None, score_stage],
                        key_weight, released + self.grad_t_slot + grad_stage, consumer_idx, tidx % 128,
                    )
                    cute.arch.sync_warp()
                    if tidx % 32 == 0:
                        cute.arch.mbarrier_arrive(released + self.query_t_slot + query_stage)
                        cute.arch.mbarrier_arrive(score_released + score_stage)
                    if (query_start + BLOCK_QUERY) % PROMOTION_ROWS == 0 or query_block_idx + 1 == NUM_QUERIES:
                        value_registers.store(value_registers.load() + register_window.load() * value_weight)
                        s_value.store(s_value.load() + shared_window.load() * value_weight)
                        acc_window.fill(0.0)
                    previous_weight = value_weight
        register_window.store(value_registers.load())
        shared_window.store(s_value.load())
        write_key_value_gradients(
            mGradKey, mGradValue, mKeyScale, (acc_key, acc_window, acc_window), mma_thread,
            key_block_idx, query_head_idx, key_head_idx, consumer_idx, self.group_keys, self.tile_keys, 2,
        )

    @cute.jit
    def accumulate_key_stripes(self, mma, mma_thread, acc_key, s_query_t, s_score_grad, key_weight, value_released, consumer_idx, lane):
        stripe_shape = mma_thread.partition_shape_C((64, self.tile_keys))
        for stripe_idx in cutlass.range_constexpr(2):
            query_twin = cute.local_tile(s_query_t, (64, BLOCK_QUERY), (consumer_idx + 2 * stripe_idx, 0))
            acc_tile = cute.make_rmem_tensor(stripe_shape, Float32)
            issue_multiply_tile(mma, mma_thread, acc_tile, query_twin, s_score_grad, False)
            if cutlass.const_expr(stripe_idx == 0):
                warpgroup.wait_group(1)
                cute.arch.sync_warp()
                if lane % 32 == 0:
                    cute.arch.mbarrier_arrive(value_released)
            warpgroup.wait_group(0)
            acc_stripe = acc_key[None, None, None, stripe_idx]
            acc_stripe.store(acc_stripe.load() + acc_tile.load() * key_weight)


@cute.jit
def launch_key_value_gradients(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream):
    if cutlass.const_expr(mQuery.shape[2] == 256):
        mainloop = PipelinedKeyValueGradients()
    else:
        mainloop = KeyValueGradients(mQuery.shape[2])
    mainloop(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream)


@cute.jit
def apply_query_tile(mma, mma_thread, acc_window, s_score_grad, s_keys, scale, window_scale):
    ratio = cute.arch.fmin(cute.arch.fmax(window_scale * cute.arch.rcp_approx(scale), 1e-8), 1e8)
    acc_window.store(acc_window.load() * ratio)
    multiply_tile(mma, mma_thread, acc_window, s_score_grad, s_keys, window_scale > 0.0)


class QueryGradients:
    """
    Apply the saved dS scales and promote the FP8 dQ products in 512-key windows.
    """
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
                self.load_loop(copies, buffers, barriers, score_head_idx, key_head_idx, last_tile, num_query_blocks)
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            self.compute_loop(mScoreScale, mGradQuery, buffers, barriers, bx, by, score_head_idx, last_tile, warp_group - 1, tidx % 128, softmax_scale)

    @cute.jit
    def load_loop(self, copies, buffers, barriers, score_head_idx, key_head_idx, last_tile, num_query_blocks):
        score_copy, key_copy = copies
        s_score_grad, s_key_t = buffers
        tile_ready, tile_released = barriers
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            if key_block_idx >= self.stages:
                cute.arch.mbarrier_wait(tile_released + stage_idx, phase ^ 1)
            ready = tile_ready + stage_idx
            cute.arch.mbarrier_arrive_and_expect_tx(ready, (BLOCK_QUERY + self.head_dim) * self.tile_keys)
            if cutlass.const_expr(self.head_dim == 256):
                score_tile_idx = score_head_idx * num_query_blocks + key_block_idx
                copy_tma_operand(score_copy, s_score_grad[None, None, stage_idx], 0, 0, score_tile_idx, ready)
            else:
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


class PipelinedQueryGradients:
    """
    Own one dQ tile per warpgroup and promote its 512-key windows in FP32 shared memory.
    """

    def __init__(self):
        self.head_dim = 256
        self.tile_keys = 64
        self.stages = 2
        self.math_threads, self.num_threads = 128, 160

    @cute.jit
    def __call__(self, mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale: cutlass.Constexpr, stream):
        H, N, D = mGradQuery.shape
        head_group = H // cute.size(mKeyT, mode=[0])
        copies = make_operand_tma(mScoreGrad, BLOCK_QUERY, self.tile_keys), make_operand_tma(mKeyT, D, self.tile_keys)
        self.kernel(mScoreScale, mGradQuery, copies, head_group, N // BLOCK_QUERY, softmax_scale).launch(
            grid=(N // BLOCK_QUERY, H, 1), block=(self.num_threads, 1, 1), stream=stream, min_blocks_per_mp=2,
        )

    @cute.kernel
    def kernel(self, mScoreScale, mGradQuery, copies, HEAD_GROUP: cutlass.Constexpr, NUM_QUERIES: cutlass.Constexpr, softmax_scale: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        query_block_idx, query_head_idx, _ = cute.arch.block_idx()
        score_head_idx = query_head_idx * NUM_QUERIES + query_block_idx
        coords = (query_block_idx, score_head_idx, query_head_idx // HEAD_GROUP)
        allocator = SmemAllocator()
        buffers = (
            allocate_staged_operand(allocator, BLOCK_QUERY, self.tile_keys, self.stages),
            allocate_staged_operand(allocator, self.head_dim, self.tile_keys, self.stages),
        )
        _, ready, released = initialize_attention_barriers(allocator, tidx, self.stages, 1)
        tile_shape = (BLOCK_QUERY, self.head_dim)
        # Store adjacent register pairs contiguously and distribute pairs across the shared banks.
        num_pairs = BLOCK_QUERY * self.head_dim // (2 * self.math_threads)
        lane_pairs = cute.make_ordered_layout((2, self.math_threads, num_pairs), order=(0, 1, 2))
        s_query_pairs = allocator.allocate_tensor(Float32, lane_pairs, 128)
        barriers = (ready, released)
        if tidx < self.math_threads:
            mma = make_mma(self.head_dim)
            mma_thread = mma.get_slice(tidx)
            fragment_shape = mma_thread.partition_shape_C(tile_shape)
            lane_acc = s_query_pairs[None, tidx, None]
            s_query = cute.make_tensor(lane_acc.iterator, cute.composition(lane_acc.layout, cute.make_layout(fragment_shape)))
            s_query.fill(0.0)
            self.compute_loop(mScoreScale, buffers, barriers, s_query, (mma, mma_thread), coords, tidx)
            output = cute.local_tile(mGradQuery[query_head_idx, None, None], tile_shape, (query_block_idx, 0))
            mma_thread.partition_C(output).store((s_query.load() * (softmax_scale * PROBABILITY_SCALE)).to(BFloat16))
        elif tidx == self.math_threads:
            self.load_loop(copies, buffers, barriers, coords, NUM_QUERIES)

    @cute.jit
    def load_loop(self, copies, buffers, barriers, coords, NUM_QUERIES: cutlass.Constexpr):
        score_copy, key_copy = copies
        s_score_grad, s_key_t = buffers
        ready, released = barriers
        last_tile, score_head_idx, key_head_idx = coords
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            if key_block_idx >= self.stages:
                cute.arch.mbarrier_wait(released + stage_idx, phase ^ 1)
            barrier = ready + stage_idx
            score_tile_idx = score_head_idx * NUM_QUERIES + key_block_idx
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, (BLOCK_QUERY + self.head_dim) * self.tile_keys)
            copy_tma_operand(score_copy, s_score_grad[None, None, stage_idx], 0, 0, score_tile_idx, barrier)
            copy_tma_operand(key_copy, s_key_t[None, None, stage_idx], 0, key_block_idx, key_head_idx, barrier)

    @cute.jit
    def compute_loop(self, mScoreScale, buffers, barriers, s_query, mma_parts, coords, tidx):
        last_tile, score_head_idx, _ = coords
        s_score_grad, s_key_t = buffers
        ready, released = barriers
        mma, mma_thread = mma_parts
        acc_window = cute.make_rmem_tensor(s_query.shape, Float32)
        acc_window.fill(0.0)
        max_scale, window_scale = Float32(0.0), Float32(0.0)
        next_scale = mScoreScale[score_head_idx, 0]
        for key_block_idx in range(last_tile + 1):
            stage_idx, phase = key_block_idx % self.stages, (key_block_idx // self.stages) % 2
            scale = next_scale
            if key_block_idx < last_tile:
                next_scale = mScoreScale[score_head_idx, key_block_idx + 1]
            cute.arch.mbarrier_wait(ready + stage_idx, phase)
            if scale > 0.0 and scale >= max_scale * 1e-8:
                apply_query_tile(mma, mma_thread, acc_window, s_score_grad[None, None, stage_idx], s_key_t[None, None, stage_idx], scale, window_scale)
                max_scale, window_scale = cute.arch.fmax(max_scale, scale), scale
            # A skipped tile still needs all four warps to finish waiting on its phase.
            cute.arch.barrier(barrier_id=1, number_of_threads=self.math_threads)
            if tidx == 0:
                cute.arch.mbarrier_arrive(released + stage_idx)
            if (key_block_idx + 1) % (PROMOTION_ROWS // self.tile_keys) == 0 or key_block_idx == last_tile:
                if window_scale > 0.0:
                    s_query.store(s_query.load() + acc_window.load() * window_scale)
                    window_scale = Float32(0.0)


@cute.jit
def launch_query_gradients(mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale: cutlass.Constexpr, stream):
    if cutlass.const_expr(mGradQuery.shape[2] == 256):
        mainloop = PipelinedQueryGradients()
    else:
        mainloop = QueryGradients(mGradQuery.shape[2])
    mainloop(mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale, stream)


@cute.jit
def load_grad_block(mGradOut, block_idx, head_idx, tidx, s_amax):
    _, _, H, D = mGradOut.shape
    VECTOR = 8
    REPEATS = BLOCK_QUERY * D // (GRAD_QUANT_THREADS * VECTOR)
    values = cute.make_rmem_tensor((VECTOR, REPEATS), Float32)
    col, first_row = tidx % (D // VECTOR), tidx // (D // VECTOR)
    amax = Float32(0.0)
    for repeat_idx in cutlass.range_constexpr(REPEATS):
        row = block_idx * BLOCK_QUERY + first_row + repeat_idx * (GRAD_QUANT_THREADS * VECTOR // D)
        source = cute.local_tile(mGradOut[head_idx // H, row, head_idx % H, None], (VECTOR,), (col,))
        raw = source.load().to(Float32)
        values[None, repeat_idx].store(raw)
        for elem_idx in cutlass.range_constexpr(VECTOR):
            amax = cute.arch.fmax(amax, abs(values[elem_idx, repeat_idx]))
    amax = warp_reduce(amax, cute.arch.fmax)
    if tidx % 32 == 0:
        s_amax[tidx // 32] = amax
    cute.arch.barrier()
    for warp_idx in cutlass.range_constexpr(GRAD_QUANT_THREADS // 32):
        amax = cute.arch.fmax(amax, s_amax[warp_idx])
    return values, cute.arch.fmax(amax, 1e-30) * (1.0 / 448.0)


@cute.jit
def store_grad_codes(mCodes, values, s_codes, block_idx, head_idx, tidx, scale):
    D = cute.size(mCodes, mode=[2])
    VECTOR = 8
    channel_idx = tidx % (D // VECTOR) * VECTOR
    first_row = tidx // (D // VECTOR)
    for repeat_idx in cutlass.range_constexpr(cute.size(values, mode=[1])):
        row_idx = first_row + repeat_idx * (GRAD_QUANT_THREADS * VECTOR // D)
        scaled = cute.make_rmem_tensor(VECTOR, Float32)
        for elem_idx in cutlass.range_constexpr(VECTOR):
            scaled[elem_idx] = values[elem_idx, repeat_idx] / scale
        quantized = scaled.load().to(Float8E4M3FN)
        output = cute.local_tile(mCodes[head_idx, block_idx * BLOCK_QUERY + row_idx, None], (VECTOR,), (channel_idx // VECTOR,))
        output.store(quantized)
        # Each group of eight source rows shifts the shared bank of a transpose read.
        shared_channel = channel_idx ^ ((row_idx // VECTOR) * VECTOR)
        scratch = cute.local_tile(s_codes[row_idx, None], (VECTOR,), (shared_channel // VECTOR,))
        scratch.store(quantized)


@cute.jit
def store_grad_twin(mTwin, s_codes, block_idx, head_idx, tidx):
    D = cute.size(mTwin, mode=[1])
    VECTOR = 8
    for repeat_idx in cutlass.range_constexpr(BLOCK_QUERY * D // (GRAD_QUANT_THREADS * VECTOR)):
        vector_idx = tidx + repeat_idx * GRAD_QUANT_THREADS
        channel_idx = vector_idx // (BLOCK_QUERY // VECTOR)
        row_group = vector_idx % (BLOCK_QUERY // VECTOR)
        row_idx = row_group * VECTOR
        shared_channel = channel_idx ^ row_idx
        codes = cute.make_rmem_tensor(VECTOR, Float8E4M3FN)
        for elem_idx in cutlass.range_constexpr(VECTOR):
            codes[elem_idx] = s_codes[row_idx + elem_idx, shared_channel]
        output = cute.local_tile(mTwin[head_idx, channel_idx, None], (VECTOR,), (block_idx * (BLOCK_QUERY // VECTOR) + row_group,))
        output.store(codes.load())


@cute.kernel
def quantize_grad_block(mGradOut, mCodes, mScale, mTwin):
    tidx, _, _ = cute.arch.thread_idx()
    block_idx, head_idx, _ = cute.arch.block_idx()
    allocator = SmemAllocator()
    s_amax = allocator.allocate_tensor(Float32, cute.make_layout(GRAD_QUANT_THREADS // 32), 16)
    D = cute.size(mGradOut, mode=[3])
    s_codes = allocator.allocate_tensor(Float8E4M3FN, cute.make_ordered_layout((BLOCK_QUERY, D), order=(1, 0)), 128)
    values, scale = load_grad_block(mGradOut, block_idx, head_idx, tidx, s_amax)
    if tidx == 0:
        mScale[head_idx, block_idx] = scale
    store_grad_codes(mCodes, values, s_codes, block_idx, head_idx, tidx, scale)
    cute.arch.barrier()
    store_grad_twin(mTwin, s_codes, block_idx, head_idx, tidx)


@cute.jit
def launch_grad_quantization(mGradOut, mCodes, mScale, mTwin, stream):
    B, N, H, _ = mGradOut.shape
    quantize_grad_block(mGradOut, mCodes, mScale, mTwin).launch(
        grid=(N // BLOCK_QUERY, B * H, 1), block=(GRAD_QUANT_THREADS, 1, 1), stream=stream,
    )


def quantize_output_gradient(grad_out):
    B, N, H, D = grad_out.shape
    if D != 256:
        return quantize_delta_matching(grad_out, 16384 // D, is_gradient=True)
    grad_out = quantizer_input(grad_out)
    codes = torch.empty((B * H, N, D), dtype=torch.float8_e4m3fn, device=grad_out.device)
    scales = torch.empty((B * H, N // BLOCK_QUERY), dtype=torch.float32, device=grad_out.device)
    twin = torch.empty((B * H, D, N), dtype=codes.dtype, device=grad_out.device)
    launch_delta_matching_kernel("delta_matching_quantize_grad_out", launch_grad_quantization, (grad_out, codes, scales, twin))
    return codes, scales, twin


def delta_matching_bwd_prepare_sm90(grad_out, state):
    """
    Quantize dO and compute the matched correction from the backward probability and dP products.
    """
    lse, query, query_scale, _, key, key_scale, _, value, value_scale = state
    grad, grad_scale, grad_t = quantize_output_gradient(grad_out)
    delta = torch.empty_like(lse)
    launch_delta_matching_kernel("delta_matching_delta", launch_matched_delta, (query, query_scale, key, key_scale, value, value_scale, grad, grad_scale, lse, delta))
    return grad, grad_scale, grad_t, delta


@cute.jit
def fold_six_heads(mInput, mOutput, output_head, vector_idx):
    first_head = output_head * 6
    head0 = cute.local_tile(mInput[first_head, None], (4,), (vector_idx,)).load()
    head1 = cute.local_tile(mInput[first_head + 1, None], (4,), (vector_idx,)).load()
    head2 = cute.local_tile(mInput[first_head + 2, None], (4,), (vector_idx,)).load()
    head3 = cute.local_tile(mInput[first_head + 3, None], (4,), (vector_idx,)).load()
    head4 = cute.local_tile(mInput[first_head + 4, None], (4,), (vector_idx,)).load()
    head5 = cute.local_tile(mInput[first_head + 5, None], (4,), (vector_idx,)).load()
    # CUDA sum groups six inputs in four zero-initialized FP32 lanes.
    partial0, partial1 = (head0 + 0.0) + head4, (head1 + 0.0) + head5
    partial2, partial3 = head2 + 0.0, head3 + 0.0
    folded = ((partial0 + partial1) + partial2) + partial3
    output = cute.local_tile(mOutput[output_head, None], (4,), (vector_idx,))  # (channel vector,)
    output.store(folded.to(BFloat16))


@cute.kernel
def fold_attention_gradients(mGradKey, mGradValue, mKey, mValue):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    vector_idx = bx * 128 + tidx
    if bz == 0:
        fold_six_heads(mGradKey, mKey, by, vector_idx)
    else:
        fold_six_heads(mGradValue, mValue, by, vector_idx)


@cute.jit
def launch_fold_attention_gradients(mGradKey, mGradValue, mKey, mValue, stream):
    output_heads, elements = mKey.shape
    fold_attention_gradients(mGradKey, mGradValue, mKey, mValue).launch(grid=(elements // 512, output_heads, 2), block=(128, 1, 1), stream=stream)


def delta_matching_bwd_sm90(grad_out, state, backward_state, softmax_scale):
    """
    Compute the three BF16 gradients with E4M3 operands and deterministic GQA reduction.
    """
    lse, query, query_scale, query_t, key, key_scale, key_t, value, value_scale = state
    grad, grad_scale, grad_t, delta = backward_state
    B, N, H, D = grad_out.shape
    num_key_heads = key.shape[0] // B
    query_blocks = N // BLOCK_QUERY
    key_tile_rows = 16384 // D
    num_key_tiles = N // key_tile_rows
    if D == 256:
        # Each bulk dS store and its matching dQ load see a contiguous FP8 tile.
        score_shape = (B * H * query_blocks * num_key_tiles, BLOCK_QUERY, key_tile_rows)
    else:
        score_shape = (B * H * query_blocks, BLOCK_QUERY, N)
    score_grad = torch.empty(score_shape, dtype=query.dtype, device=query.device)
    score_scale = torch.empty((B * H * query_blocks, num_key_tiles), dtype=torch.float32, device=query.device)
    grad_query = torch.empty_like(grad, dtype=torch.bfloat16)
    grad_key, grad_value = (torch.empty_like(grad, dtype=torch.float32) for _ in range(2))

    launch_delta_matching_kernel("delta_matching_kv_gradients", launch_key_value_gradients, (query, query_scale, key, key_scale, value, value_scale, query_t, grad, grad_scale, grad_t, lse, delta, score_grad, score_scale, grad_key, grad_value))
    launch_delta_matching_kernel("delta_matching_query_gradients", launch_query_gradients, (key_t, score_grad, score_scale, grad_query), (softmax_scale,))
    grad_query = grad_query.reshape(B, H, N, D).transpose(1, 2)
    if D == 256 and H == 6 * num_key_heads:
        folded_shape = (B * num_key_heads, N * D)
        key_out, value_out = (torch.empty(folded_shape, dtype=torch.bfloat16, device=query.device) for _ in range(2))
        launch_delta_matching_kernel("delta_matching_fold_gradients", launch_fold_attention_gradients, (grad_key.flatten(1), grad_value.flatten(1), key_out, value_out))
        return grad_query, key_out.reshape(B, num_key_heads, N, D).transpose(1, 2), value_out.reshape(B, num_key_heads, N, D).transpose(1, 2)
    grad_key, grad_value = (tensor.reshape(B, num_key_heads, H // num_key_heads, N, D).sum(dim=2).to(torch.bfloat16).transpose(1, 2) for tensor in (grad_key, grad_value))
    return grad_query, grad_key, grad_value
