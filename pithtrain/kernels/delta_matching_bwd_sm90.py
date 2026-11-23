"""
Matched correction, key/value gradients and query gradients on SM90.
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Float64
from cutlass.utils import SmemAllocator
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.delta_matching_sm90 import BLOCK_QUERY, LOG2_E, allocate_operand, copy_operand, finish_operand_copies, launch_delta_matching_kernel, make_mma, multiply_tile, quantize_delta_matching, share_mma_operand

__all__ = ["delta_matching_bwd_prepare_sm90", "delta_matching_bwd_sm90"]

PROBABILITY_SCALE = 1.0 / 256.0
PROMOTION_ROWS = 512
THREADS = 128


@cute.jit
def reconstruct_probabilities(probabilities, coords, mKeyScale, mLse, query_head_idx, key_head_idx, query_start, key_start, query_scale, GROUP_KEYS: cutlass.Constexpr):
    for elem_idx in cutlass.range_constexpr(cute.size(probabilities)):
        query_row, key_col = coords[elem_idx]
        query_pos, key_pos = query_start + query_row, key_start + key_col
        log2_lse = mLse[query_head_idx, query_pos] * LOG2_E
        log2_scale = (query_scale * mKeyScale[key_head_idx, key_pos // GROUP_KEYS]) * LOG2_E
        exponent = probabilities[elem_idx] * log2_scale - log2_lse + 8.0
        probabilities[elem_idx] = cute.math.exp2(cute.arch.fmin(exponent, 12.0), fastmath=True) if key_pos <= query_pos else Float32(0.0)


@cute.jit
def compute_backward_products(mma, mma_thread, probabilities, probability_grad, buffers):
    s_query, s_key, s_grad_out, s_value = buffers
    multiply_tile(mma, mma_thread, probabilities, s_query, s_key, False)
    multiply_tile(mma, mma_thread, probability_grad, s_grad_out, s_value, False)


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


@cute.jit
def compute_delta(tensors, buffers, query_block_idx, query_head_idx, tidx, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr):
    mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGradScale, mLse, mDelta = tensors
    s_query, s_key, s_grad_out, s_value = buffers
    query_start = query_block_idx * BLOCK_QUERY
    key_head_idx = query_head_idx // (cute.size(mQueryScale, mode=[0]) // cute.size(mKey, mode=[0]))
    scale_idx = query_start // (GROUP_KEYS * 2)
    query_scale, grad_out_scale = mQueryScale[query_head_idx, scale_idx], mGradScale[query_head_idx, scale_idx]
    mma = make_mma(TILE_KEYS)
    mma_thread = mma.get_slice(tidx)
    tile_shape = (BLOCK_QUERY, TILE_KEYS)
    coords = mma_thread.partition_C(cute.make_identity_tensor(tile_shape))  # (MMA, MMA_M, MMA_N)
    probabilities, probability_grad = (cute.make_rmem_tensor(mma_thread.partition_shape_C(tile_shape), Float32) for _ in range(2))
    acc_delta = cute.make_rmem_tensor(2, Float64)
    acc_delta.fill(0.0)
    for key_block_idx in range(query_start // TILE_KEYS + 1):
        copy_operand(mKey, s_key, key_head_idx, key_block_idx, 0, tidx)
        copy_operand(mValue, s_value, key_head_idx, key_block_idx, 0, tidx)
        finish_operand_copies()
        compute_backward_products(mma, mma_thread, probabilities, probability_grad, buffers)
        key_start = key_block_idx * TILE_KEYS
        reconstruct_probabilities(probabilities, coords, mKeyScale, mLse, query_head_idx, key_head_idx, query_start, key_start, query_scale, GROUP_KEYS)
        reduce_delta_tile(probabilities, probability_grad, coords, acc_delta, mValueScale, key_head_idx, key_start, grad_out_scale, GROUP_KEYS)

    if tidx % 4 == 0:
        positions = reshape_acc_to_mn(coords)
        for row_idx in cutlass.range_constexpr(cute.size(acc_delta)):
            mDelta[query_head_idx, query_start + positions[row_idx, 0][0]] = acc_delta[row_idx].to(Float32)


@cute.kernel
def matched_delta(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    D = cute.size(mQuery, mode=[2])
    allocator = SmemAllocator()
    s_query, s_key = allocate_operand(allocator, BLOCK_QUERY, D), allocate_operand(allocator, TILE_KEYS, D)
    s_grad_out, s_value = allocate_operand(allocator, BLOCK_QUERY, D), allocate_operand(allocator, TILE_KEYS, D)

    copy_operand(mQuery, s_query, by, bx, 0, tidx)
    copy_operand(mGrad, s_grad_out, by, bx, 0, tidx)
    finish_operand_copies()
    compute_delta((mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGradScale, mLse, mDelta), (s_query, s_key, s_grad_out, s_value), bx, by, tidx, GROUP_KEYS, TILE_KEYS)


@cute.jit
def launch_matched_delta(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, stream):
    H, N, D = mQuery.shape
    matched_delta(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mGrad, mGradScale, mLse, mDelta, 8192 // D, 16384 // D).launch(grid=(N // BLOCK_QUERY, H, 1), block=(THREADS, 1, 1), stream=stream)


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
    cute.arch.barrier()
    for warp_idx in cutlass.range_constexpr(THREADS // 32):
        amax = cute.arch.fmax(amax, s_amax[warp_idx])
    base_scale = amax * (1.0 / 448.0)
    floor_scale = (max_weight * (256.0e-8)) / query_scale
    return cute.arch.fmax(base_scale, floor_scale) if base_scale >= 1e-30 else Float32(0.0)


@cute.jit
def store_score_operands(probabilities, score_grad, coords, s_probability, s_score_grad, mScoreGrad, score_head_idx, key_start, scale):
    rcp = 1.0 / scale if scale > 0.0 else Float32(0.0)
    for elem_idx in cutlass.range_constexpr(cute.size(probabilities)):
        row_idx, col_idx = coords[elem_idx]
        prob_code = cute.arch.fmin(probabilities[elem_idx], 448.0).to(Float8E4M3FN)
        grad_code = (score_grad[elem_idx] * rcp).to(Float8E4M3FN)
        s_probability[col_idx, row_idx], s_score_grad[col_idx, row_idx] = prob_code, grad_code
        mScoreGrad[score_head_idx, row_idx, key_start + col_idx] = grad_code
    share_mma_operand()


@cute.jit
def accumulate_key_gradient(mma, mma_thread, s_score_grad, s_twin, s_acc_key, weight, TILE_KEYS: cutlass.Constexpr):
    D = cute.size(s_twin, mode=[0])
    for key_idx in cutlass.range_constexpr(TILE_KEYS // 64):
        score_tile = cute.local_tile(s_score_grad, (64, BLOCK_QUERY), (key_idx, 0))  # (key rows, query rows)
        key_tile = cute.local_tile(s_acc_key, (64, D), (key_idx, 0))  # (key rows, channels)
        accumulated = mma_thread.partition_C(key_tile)  # (MMA, MMA_M, MMA_N)
        grad_key = cute.make_rmem_tensor(mma_thread.partition_shape_C((64, D)), Float32)
        multiply_tile(mma, mma_thread, grad_key, score_tile, s_twin, False)
        accumulated.store(accumulated.load() + grad_key.load() * weight)


@cute.jit
def accumulate_value_gradient(mma, mma_thread, s_probability, s_twin, s_acc_value, acc_value, weight, previous_weight, is_first, is_boundary, TILE_KEYS: cutlass.Constexpr):
    D = cute.size(s_twin, mode=[0])
    ratio = cute.arch.fmin(cute.arch.fmax(previous_weight / weight, 1e-8), 1e8)
    for key_idx in cutlass.range_constexpr(TILE_KEYS // 64):
        prob_tile = cute.local_tile(s_probability, (64, BLOCK_QUERY), (key_idx, 0))  # (key rows, query rows)
        accumulator = acc_value[None, None, None, key_idx]  # (MMA, MMA_M, MMA_N)
        accumulator.store(accumulator.load() * ratio)
        multiply_tile(mma, mma_thread, accumulator, prob_tile, s_twin, not is_first)
        if is_boundary:
            value_tile = cute.local_tile(s_acc_value, (64, D), (key_idx, 0))  # (key rows, channels)
            accumulated = mma_thread.partition_C(value_tile)  # (MMA, MMA_M, MMA_N)
            accumulated.store(accumulated.load() + accumulator.load() * weight)


@cute.jit
def write_kv_gradients(mGradKey, mGradValue, mKeyScale, s_acc_key, s_acc_value, key_block_idx, query_head_idx, key_head_idx, tidx, GROUP_KEYS: cutlass.Constexpr):
    TILE_KEYS, D = s_acc_key.shape
    for elem_idx in cutlass.range_constexpr(TILE_KEYS * D // THREADS):
        offset = tidx + elem_idx * THREADS
        row_idx, col_idx = offset // D, offset % D
        key_pos = key_block_idx * TILE_KEYS + row_idx
        mGradKey[query_head_idx, key_pos, col_idx] = s_acc_key[row_idx, col_idx] / mKeyScale[key_head_idx, key_pos // GROUP_KEYS]
        mGradValue[query_head_idx, key_pos, col_idx] = s_acc_value[row_idx, col_idx]


@cute.jit
def accumulate_kv_tiles(tensors, buffers, key_block_idx, query_head_idx, tidx, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr):
    mQuery, mQueryScale, mKeyScale, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale = tensors
    s_query, s_key, s_grad_out, s_value, s_twin, s_probability, s_score_grad, s_acc_key, s_acc_value, s_amax = buffers
    D = cute.size(s_twin, mode=[0])
    N = cute.size(mQuery, mode=[1])
    num_query_blocks = N // BLOCK_QUERY
    key_start = key_block_idx * TILE_KEYS
    key_head_idx = query_head_idx // (cute.size(mQueryScale, mode=[0]) // cute.size(mKeyScale, mode=[0]))
    mma_score, mma_grad = make_mma(TILE_KEYS), make_mma(D)
    thread_score, thread_grad = mma_score.get_slice(tidx), mma_grad.get_slice(tidx)
    score_shape = (BLOCK_QUERY, TILE_KEYS)
    coords = thread_score.partition_C(cute.make_identity_tensor(score_shape))  # (MMA, MMA_M, MMA_N)
    probabilities, probability_grad = (cute.make_rmem_tensor(thread_score.partition_shape_C(score_shape), Float32) for _ in range(2))
    acc_value = cute.make_rmem_tensor((*thread_grad.partition_shape_C((64, D)), TILE_KEYS // 64), Float32)
    acc_value.fill(0.0)
    max_weight, previous_weight = Float32(0.0), Float32(1.0)
    first_block = key_start // BLOCK_QUERY
    for query_block_idx in range(first_block, num_query_blocks):
        copy_operand(mQuery, s_query, query_head_idx, query_block_idx, 0, tidx)
        copy_operand(mGrad, s_grad_out, query_head_idx, query_block_idx, 0, tidx)
        copy_operand(mQueryT, s_twin, query_head_idx, 0, query_block_idx, tidx)
        finish_operand_copies()
        compute_backward_products(mma_score, thread_score, probabilities, probability_grad, (s_query, s_key, s_grad_out, s_value))
        query_start = query_block_idx * BLOCK_QUERY
        scale_idx = query_start // (GROUP_KEYS * 2)
        query_scale, grad_out_scale = mQueryScale[query_head_idx, scale_idx], mGradScale[query_head_idx, scale_idx]
        reconstruct_probabilities(probabilities, coords, mKeyScale, mLse, query_head_idx, key_head_idx, query_start, key_start, query_scale, GROUP_KEYS)
        amax = center_score_gradient(probabilities, probability_grad, coords, (mKeyScale, mValueScale), mDelta, query_head_idx, key_head_idx, query_start, key_start, grad_out_scale, GROUP_KEYS)
        score_scale = choose_score_scale(amax, s_amax, tidx, max_weight, query_scale)
        score_head_idx = query_head_idx * num_query_blocks + query_block_idx
        if tidx == 0:
            mScoreScale[score_head_idx, key_block_idx] = score_scale
        store_score_operands(probabilities, probability_grad, coords, s_probability, s_score_grad, mScoreGrad, score_head_idx, key_start, score_scale)

        weight = (score_scale * PROBABILITY_SCALE) * query_scale
        accumulate_key_gradient(mma_grad, thread_grad, s_score_grad, s_twin, s_acc_key, weight, TILE_KEYS)
        max_weight = cute.arch.fmax(max_weight, weight)
        copy_operand(mGradT, s_twin, query_head_idx, 0, query_block_idx, tidx)
        finish_operand_copies()
        value_weight = grad_out_scale * PROBABILITY_SCALE
        is_boundary = (query_start + BLOCK_QUERY) % PROMOTION_ROWS == 0 or query_block_idx + 1 == num_query_blocks
        is_first = query_start % PROMOTION_ROWS == 0 or query_block_idx == first_block
        accumulate_value_gradient(mma_grad, thread_grad, s_probability, s_twin, s_acc_value, acc_value, value_weight, previous_weight, is_first, is_boundary, TILE_KEYS)
        previous_weight = value_weight


@cute.kernel
def key_value_gradients(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, GROUP_KEYS: cutlass.Constexpr, TILE_KEYS: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    D = cute.size(mKey, mode=[2])
    key_head_idx = by // (cute.size(mQuery, mode=[0]) // cute.size(mKey, mode=[0]))
    allocator = SmemAllocator()
    s_query, s_key = allocate_operand(allocator, BLOCK_QUERY, D), allocate_operand(allocator, TILE_KEYS, D)
    s_grad_out, s_value = allocate_operand(allocator, BLOCK_QUERY, D), allocate_operand(allocator, TILE_KEYS, D)
    s_twin = allocate_operand(allocator, D, BLOCK_QUERY)
    s_probability, s_score_grad = allocate_operand(allocator, TILE_KEYS, BLOCK_QUERY), allocate_operand(allocator, TILE_KEYS, BLOCK_QUERY)
    grad_layout = cute.make_layout((TILE_KEYS, D), stride=(D, 1))
    s_acc_key, s_acc_value = allocator.allocate_tensor(Float32, grad_layout, 16), allocator.allocate_tensor(Float32, grad_layout, 16)
    s_amax = allocator.allocate_tensor(Float32, cute.make_layout(THREADS // 32), 16)

    for elem_idx in cutlass.range_constexpr(TILE_KEYS * D // THREADS):
        offset = tidx + elem_idx * THREADS
        s_acc_key[offset], s_acc_value[offset] = Float32(0.0), Float32(0.0)
    copy_operand(mKey, s_key, key_head_idx, bx, 0, tidx)
    copy_operand(mValue, s_value, key_head_idx, bx, 0, tidx)
    finish_operand_copies()
    accumulate_kv_tiles((mQuery, mQueryScale, mKeyScale, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale), (s_query, s_key, s_grad_out, s_value, s_twin, s_probability, s_score_grad, s_acc_key, s_acc_value, s_amax), bx, by, tidx, GROUP_KEYS, TILE_KEYS)
    cute.arch.barrier()
    write_kv_gradients(mGradKey, mGradValue, mKeyScale, s_acc_key, s_acc_value, bx, by, key_head_idx, tidx, GROUP_KEYS)


@cute.jit
def launch_key_value_gradients(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, stream):
    H, N, D = mQuery.shape
    TILE_KEYS = 16384 // D
    key_value_gradients(mQuery, mQueryScale, mKey, mKeyScale, mValue, mValueScale, mQueryT, mGrad, mGradScale, mGradT, mLse, mDelta, mScoreGrad, mScoreScale, mGradKey, mGradValue, 8192 // D, TILE_KEYS).launch(grid=(N // TILE_KEYS, H, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def accumulate_query_tiles(mKeyT, mScoreGrad, mScoreScale, mGradQuery, buffers, query_block_idx, query_head_idx, tidx, softmax_scale: cutlass.Constexpr):
    s_key_t, s_score_grad, s_acc_query = buffers
    D, TILE_KEYS = cute.size(s_key_t, mode=[0]), cute.size(s_key_t, mode=[1])
    num_query_blocks = cute.size(mGradQuery, mode=[1]) // BLOCK_QUERY
    key_head_idx = query_head_idx // (cute.size(mGradQuery, mode=[0]) // cute.size(mKeyT, mode=[0]))
    mma = make_mma(D)
    mma_thread = mma.get_slice(tidx)
    accumulated = mma_thread.partition_C(s_acc_query)  # (MMA, MMA_M, MMA_N)
    accumulated.fill(0.0)
    acc_window = cute.make_rmem_tensor(mma_thread.partition_shape_C((BLOCK_QUERY, D)), Float32)
    acc_window.fill(0.0)
    score_head_idx = query_head_idx * num_query_blocks + query_block_idx
    max_scale, window_scale = Float32(0.0), Float32(0.0)
    last_tile = query_block_idx * BLOCK_QUERY // TILE_KEYS
    for key_block_idx in range(last_tile + 1):
        scale = mScoreScale[score_head_idx, key_block_idx]
        if scale > 0.0 and scale >= max_scale * 1e-8:
            copy_operand(mScoreGrad, s_score_grad, score_head_idx, 0, key_block_idx, tidx)
            copy_operand(mKeyT, s_key_t, key_head_idx, 0, key_block_idx, tidx)
            ratio = cute.arch.fmin(cute.arch.fmax(window_scale / scale, 1e-8), 1e8)
            acc_window.store(acc_window.load() * ratio)
            finish_operand_copies()
            multiply_tile(mma, mma_thread, acc_window, s_score_grad, s_key_t, window_scale > 0.0)
            max_scale, window_scale = cute.arch.fmax(max_scale, scale), scale
        if (key_block_idx + 1) % (PROMOTION_ROWS // TILE_KEYS) == 0 or key_block_idx == last_tile:
            if window_scale > 0.0:
                accumulated.store(accumulated.load() + acc_window.load() * window_scale)
                window_scale = Float32(0.0)
    output_tile = cute.local_tile(mGradQuery[query_head_idx, None, None], (BLOCK_QUERY, D), (query_block_idx, 0))  # (query row, channel)
    output = mma_thread.partition_C(output_tile)  # (MMA, MMA_M, MMA_N)
    output.store((accumulated.load() * (softmax_scale * PROBABILITY_SCALE)).to(BFloat16))


@cute.kernel
def query_gradients(mKeyT, mScoreGrad, mScoreScale, mGradQuery, TILE_KEYS: cutlass.Constexpr, softmax_scale: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    D = cute.size(mGradQuery, mode=[2])
    allocator = SmemAllocator()
    s_key_t, s_score_grad = allocate_operand(allocator, D, TILE_KEYS), allocate_operand(allocator, BLOCK_QUERY, TILE_KEYS)
    s_acc_query = allocator.allocate_tensor(Float32, cute.make_layout((BLOCK_QUERY, D), stride=(D, 1)), 16)

    accumulate_query_tiles(mKeyT, mScoreGrad, mScoreScale, mGradQuery, (s_key_t, s_score_grad, s_acc_query), bx, by, tidx, softmax_scale)


@cute.jit
def launch_query_gradients(mKeyT, mScoreGrad, mScoreScale, mGradQuery, softmax_scale: cutlass.Constexpr, stream):
    H, N, D = mGradQuery.shape
    query_gradients(mKeyT, mScoreGrad, mScoreScale, mGradQuery, 16384 // D, softmax_scale).launch(grid=(N // BLOCK_QUERY, H, 1), block=(THREADS, 1, 1), stream=stream)


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
