"""
Causal E4M3 QK and PV with online FP32 softmax and row-wise probability group scales.
"""

import operator

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import SmemAllocator
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_ptx_sm90 import asm, to_ir
from pithtrain.kernels.delta_matching_sm90 import BLOCK_QUERY, LOG2_E, allocate_staged_operand, copy_tma_operand, issue_multiply_tile, launch_delta_matching_kernel, make_mma, make_operand_tma, multiply_tile, quantize_delta_matching

__all__ = ["delta_matching_fwd_sm90"]


@cute.jit
def read_tile_scales(s_key_scale, s_value_scale, stage, query_scale, GROUPS: cutlass.Constexpr):
    key_scales = cute.make_rmem_tensor(GROUPS, Float32)
    value_scales = cute.make_rmem_tensor(GROUPS, Float32)
    for group in cutlass.range_constexpr(GROUPS):
        key_scales[group] = (query_scale * s_key_scale[group, stage]) * LOG2_E
        value_scales[group] = s_value_scale[group, stage]
    return key_scales, value_scales


@cute.jit
def scale_causal_scores(scores, coords, key_scales, query_start, key_start,
                        IS_DIAGONAL: cutlass.Constexpr, GROUP_KEYS: cutlass.Constexpr):
    rows, positions = reshape_acc_to_mn(scores), reshape_acc_to_mn(coords)
    COLS_PER_GROUP = GROUP_KEYS // 4
    for group_idx in cutlass.range_constexpr(cute.size(key_scales)):
        for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
            for col_idx in cutlass.range_constexpr(COLS_PER_GROUP):
                elem_col = group_idx * COLS_PER_GROUP + col_idx
                value = rows[row_idx, elem_col] * key_scales[group_idx]
                if cutlass.const_expr(IS_DIAGONAL):
                    query_row, key_col = positions[row_idx, elem_col]
                    if key_start + key_col > query_start + query_row:
                        value = Float32(float("-inf"))
                rows[row_idx, elem_col] = value


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
        if cutlass.const_expr(GROUP_KEYS == 32):
            # The value-twin producer already arranges these keys in register-A order.
            words[2 * half_idx] = Int32(cute.arch.prmt(packed[0], packed[1], 0x5410))
            words[2 * half_idx + 1] = Int32(cute.arch.prmt(packed[0], packed[1], 0x7632))
        else:
            low = packed[0] if choose_first else packed[1]
            high = packed[1] if choose_first else packed[0]
            low = cute.arch.shuffle_sync(low, offset=lane_first, mask_and_clamp=quad_mask)
            high = cute.arch.shuffle_sync(high, offset=lane_first ^ 1, mask_and_clamp=quad_mask)
            words[2 * half_idx] = Int32(cute.arch.prmt(low, high, selector_low))
            words[2 * half_idx + 1] = Int32(cute.arch.prmt(low, high, selector_high))


@cute.jit
def encode_probability_group(scores, fragment, row_max, maxima, weight, value_scale, group_idx: cutlass.Constexpr, GROUP_KEYS: cutlass.Constexpr, tidx):
    rows = reshape_acc_to_mn(scores)  # (query rows, key columns)
    basis = cute.make_rmem_tensor(2, Float32)
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        basis[row_idx] = cute.arch.fmax(maxima[row_idx, group_idx], row_max[row_idx] - 12.0)
        weight[row_idx] = cute.math.exp2(basis[row_idx] - row_max[row_idx], fastmath=True) * (value_scale / 448.0)
    pack_probability_operand(rows, basis, fragment, group_idx, GROUP_KEYS, tidx)


@cute.jit
def update_probability_tile(scores, weights, operands, value_scales, row_max, row_sum, rescale,
                            tidx, GROUP_KEYS: cutlass.Constexpr, GROUPS: cutlass.Constexpr):
    rows = reshape_acc_to_mn(scores)
    GROUP_COLUMNS = GROUP_KEYS // 4
    maxima = cute.make_rmem_tensor((2, GROUPS), Float32)
    tile_sums = cute.make_rmem_tensor(2, Float32)
    tile_sums.fill(0.0)
    for row in cutlass.range_constexpr(2):
        tile_max = Float32(float("-inf"))
        for group in cutlass.range_constexpr(GROUPS):
            group_max = Float32(float("-inf"))
            for col in cutlass.range_constexpr(GROUP_COLUMNS):
                group_max = cute.arch.fmax(group_max, rows[row, group * GROUP_COLUMNS + col])
            maximum = warp_reduce(group_max, cute.arch.fmax, threads_in_group=4)
            maxima[row, group] = maximum
            tile_max = cute.arch.fmax(tile_max, maximum)
        next_max = cute.arch.fmax(row_max[row], tile_max)
        rescale[row] = cute.math.exp2(row_max[row] - next_max, fastmath=True)
        row_max[row] = next_max

    for group in cutlass.range_constexpr(GROUPS):
        # Each row sums the same columns in the same order as the port.
        # Once a group is summed and packed, its FP32 scores can be retired.
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(GROUP_COLUMNS):
                value = rows[row, group * GROUP_COLUMNS + col]
                tile_sums[row] += cute.math.exp2(value - row_max[row], fastmath=True)
        encode_probability_group(scores, operands[None, None, None, group], row_max, maxima,
                                 weights[None, group], value_scales[group], group, GROUP_KEYS, tidx)
    for row in cutlass.range_constexpr(2):
        tile_sum = warp_reduce(tile_sums[row], operator.add, threads_in_group=4)
        row_sum[row] = row_sum[row] * rescale[row] + tile_sum


@cute.jit
def value_basis_ratios(weights, GROUPS: cutlass.Constexpr):
    ratios = cute.make_rmem_tensor((2, GROUPS - 1), Float32)
    for group_idx in cutlass.range_constexpr(1, GROUPS):
        for row_idx in cutlass.range_constexpr(2):
            ratio = weights[row_idx, group_idx - 1] * cute.arch.rcp_approx(weights[row_idx, group_idx])
            ratios[row_idx, group_idx - 1] = cute.arch.fmin(cute.arch.fmax(ratio, 1e-30), 1e30)
    return ratios


@cute.jit
def rebase_value_accumulator(accumulator, ratios, group_idx: cutlass.Constexpr):
    rows = reshape_acc_to_mn(accumulator)
    for row_idx in cutlass.range_constexpr(cute.size(rows, mode=[0])):
        rows[row_idx, None].store(rows[row_idx, None].load() * ratios[row_idx, group_idx - 1])


@dsl_user_op
def promote_output_f32(previous, partial, rescale, weight, FUSE_PREVIOUS, *, loc=None, ip=None):
    if FUSE_PREVIOUS:
        instructions = "{ .reg .f32 promoted; mul.rn.f32 promoted, $2, $4; fma.rn.f32 $0, $1, $3, promoted; }"
    else:
        instructions = "{ .reg .f32 scaled; mul.rn.f32 scaled, $1, $3; fma.rn.f32 $0, $2, $4, scaled; }"
    operands = [to_ir(value, Float32, loc, ip) for value in (previous, partial, rescale, weight)]
    return Float32(asm(T.f32(), operands, instructions, "=f,f,f,f,f"))


@cute.jit
def promote_output_row(output, partial, rescale, weight, FUSE_PREVIOUS: cutlass.Constexpr):
    for col_idx in cutlass.range_constexpr(cute.size(output)):
        output[col_idx] = promote_output_f32(output[col_idx], partial[col_idx], rescale, weight, FUSE_PREVIOUS)


@cute.jit
def promote_cached_output_row(output, partial, rescale, weight, FUSE_PREVIOUS: cutlass.Constexpr):
    for pair_idx in cutlass.range_constexpr(cute.size(output) // 2):
        destination = cute.local_tile(output, (2,), (pair_idx,))
        previous = destination.load()
        updated = cute.make_rmem_tensor(2, Float32)
        for elem_idx in cutlass.range_constexpr(2):
            updated[elem_idx] = promote_output_f32(
                previous[elem_idx], partial[pair_idx * 2 + elem_idx], rescale, weight, FUSE_PREVIOUS,
            )
        destination.store(updated.load())


@cute.jit
def output_slab(acc_output, cached_output, slab: cutlass.Constexpr, SLABS: cutlass.Constexpr):
    if cutlass.const_expr(SLABS == 2 and slab == SLABS - 1):
        return cached_output
    else:
        return acc_output[None, None, None, slab]


@cute.jit
def accumulate_value_tile(acc_output, cached_output, mma, mma_thread, s_values, weights, operands,
                          rescale, stage_idx, GROUP_KEYS: cutlass.Constexpr, GROUPS: cutlass.Constexpr,
                          VALUE_COLUMNS: cutlass.Constexpr, SLABS: cutlass.Constexpr):
    ratios = value_basis_ratios(weights, GROUPS)
    for slab in cutlass.range_constexpr(SLABS):
        acc_tile = cute.make_rmem_tensor(mma_thread.partition_shape_C((BLOCK_QUERY, VALUE_COLUMNS)), Float32)
        for group in cutlass.range_constexpr(GROUPS):
            if cutlass.const_expr(group > 0):
                rebase_value_accumulator(acc_tile, ratios, group)
            source = s_values[None, None, stage_idx * GROUPS + group]
            value_tile = cute.local_tile(source, (VALUE_COLUMNS, GROUP_KEYS), (slab, 0))
            multiply_tile(mma, mma_thread, acc_tile, operands[None, None, None, group], value_tile, group > 0)
        output = reshape_acc_to_mn(output_slab(acc_output, cached_output, slab, SLABS))
        partial = reshape_acc_to_mn(acc_tile)
        for row in cutlass.range_constexpr(2):
            if cutlass.const_expr(SLABS == 2):
                fuse_previous = slab != 0 or row != 1
                if cutlass.const_expr(slab == 1):
                    promote_cached_output_row(output[row, None], partial[row, None], rescale[row],
                                              weights[row, GROUPS - 1], fuse_previous)
                else:
                    promote_output_row(output[row, None], partial[row, None], rescale[row],
                                       weights[row, GROUPS - 1], fuse_previous)
            else:
                promoted = partial[row, None].load() * weights[row, GROUPS - 1]
                output[row, None].store(output[row, None].load() * rescale[row] + promoted)


@cute.jit
def store_attention(mOut, mLse, acc_output, cached_output, row_sum, row_max, mma_thread, query_block_idx,
                    head_idx, tidx, VALUE_COLUMNS: cutlass.Constexpr, SLABS: cutlass.Constexpr):
    H = cute.size(mOut, mode=[2])
    target = mOut[head_idx // H, None, head_idx % H, None]
    positions = reshape_acc_to_mn(mma_thread.partition_C(cute.make_identity_tensor((BLOCK_QUERY, VALUE_COLUMNS))))
    if tidx % 4 == 0:
        for row_idx in cutlass.range_constexpr(cute.size(row_sum)):
            token_idx = query_block_idx * BLOCK_QUERY + positions[row_idx, 0][0]
            mLse[head_idx, token_idx] = (row_max[row_idx] + cute.math.log2(row_sum[row_idx], fastmath=True)) * (1.0 / LOG2_E)
    for slab in cutlass.range_constexpr(SLABS):
        out_tile = cute.local_tile(target, (BLOCK_QUERY, VALUE_COLUMNS), (query_block_idx, slab))
        output = reshape_acc_to_mn(mma_thread.partition_C(out_tile))
        accum = reshape_acc_to_mn(output_slab(acc_output, cached_output, slab, SLABS))
        for row_idx in cutlass.range_constexpr(cute.size(row_sum)):
            rcp = cute.arch.rcp_approx(row_sum[row_idx])
            output[row_idx, None].store((accum[row_idx, None].load() * rcp).to(BFloat16))


class CausalAttention:
    """
    Persistent 128-row CTAs with independent K/V stages and a PV-first consumer schedule.
    """
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.group_keys, self.tile_keys = 8192 // head_dim, 32768 // head_dim
        self.groups = self.tile_keys // self.group_keys
        self.value_columns = min(head_dim, 128)
        self.stages = 2
        self.consumer_groups, self.load_group = 2, 0
        self.consumer_registers, self.load_registers = 240, 24

    @cute.jit
    def __call__(self, mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, mTicket, stream):
        H, N, D = mQuery.shape
        query_copy, key_copy = make_operand_tma(mQuery, BLOCK_QUERY, D), make_operand_tma(mKey, self.tile_keys, D)
        value_copy = make_operand_tma(mValueT, D, self.group_keys)
        inputs = (mQueryScale, mKeyScale, mValueScale, mOut, mLse)
        query_groups = N // (BLOCK_QUERY * self.consumer_groups)
        ctas = min(SM_COUNT, query_groups * H)
        self.kernel(inputs, (query_copy, key_copy, value_copy), mTicket, query_groups, ctas).launch(
            grid=(ctas, 1, 1), block=((1 + self.consumer_groups) * 128, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, inputs, copies, mTicket, QUERY_GROUPS: cutlass.Constexpr, CTAS: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group = warp // 4
        query_heads = cute.size(inputs[0], mode=[0])
        key_heads = cute.size(inputs[1], mode=[0])
        allocator = SmemAllocator()
        s_query = allocate_staged_operand(allocator, BLOCK_QUERY, self.head_dim, self.consumer_groups)
        s_keys = allocate_staged_operand(allocator, self.tile_keys, self.head_dim, self.stages)
        s_values = allocate_staged_operand(allocator, self.head_dim, self.group_keys, self.stages * self.groups)
        s_output_cache = None
        if cutlass.const_expr(self.head_dim == 256):
            # Each lane accesses pairs of adjacent FP32 values.
            cache_layout = cute.make_layout((128, (2, 32), self.consumer_groups), stride=(2, (1, 256), 8192))
            s_output_cache = allocator.allocate_tensor(Float32, cache_layout, 16)
        scales_layout = cute.make_layout((self.groups, self.stages), stride=(1, self.groups))
        s_key_scale = allocator.allocate_tensor(Float32, scales_layout, 16)
        s_value_scale = allocator.allocate_tensor(Float32, scales_layout, 16)
        buffers = (s_query, s_keys, s_values, s_key_scale, s_value_scale)
        work_slot = allocator.allocate_tensor(Int32, cute.make_layout(1), 4)
        barriers = self.initialize_barriers(allocator, tidx)

        task_count = QUERY_GROUPS * query_heads
        if warp_group == self.load_group:
            cute.arch.setmaxregister_decrease(self.load_registers)
            if tidx == 0:
                task_idx, tile_base, wave = bx, Int32(0), Int32(0)
                while task_idx < task_count:
                    head_idx, query_block, last_tile, key_head = self.work_item(
                        task_idx, query_heads, key_heads, QUERY_GROUPS,
                    )
                    work_slot[0] = task_idx
                    self.load_loop(copies, buffers, barriers, inputs[1], inputs[2], query_block, head_idx, key_head,
                                   last_tile, tile_base)
                    task_idx = CTAS + cute.arch.atomic_add(mTicket.iterator, Int32(1), sem="relaxed", scope="gpu")
                    tile_base += last_tile + 1
                    cute.arch.mbarrier_wait(barriers[5], wave % 2)
                    wave += 1
                # The sentinel releases the consumers from their next query wait.
                work_slot[0] = task_count
                cute.arch.mbarrier_arrive(barriers[0])
        else:
            cute.arch.setmaxregister_increase(self.consumer_registers)
            consumer_group = warp_group - 1
            tile_base, wave = Int32(0), Int32(0)
            task_idx = self.read_work(work_slot, barriers[0], 0)
            while task_idx < task_count:
                head_idx, query_block, last_tile, key_head = self.work_item(
                    task_idx, query_heads, key_heads, QUERY_GROUPS,
                )
                self.compute_loop(inputs, buffers, s_output_cache, barriers, query_block + consumer_group, head_idx,
                                  key_head, last_tile, consumer_group, tidx % 128, tile_base)
                tile_base += last_tile + 1
                wave += 1
                task_idx = self.read_work(work_slot, barriers[0], wave % 2)

    @cute.jit
    def read_work(self, work_slot, ready, phase):
        cute.arch.mbarrier_wait(ready, phase)
        return work_slot[0]

    @cute.jit
    def work_item(self, task_idx, query_heads, key_heads, QUERY_GROUPS: cutlass.Constexpr):
        head_idx = task_idx % query_heads
        query_block = (QUERY_GROUPS - 1 - task_idx // query_heads) * self.consumer_groups
        last_tile = query_block * BLOCK_QUERY // self.tile_keys
        key_head = head_idx // (query_heads // key_heads)
        return head_idx, query_block, last_tile, key_head

    @cute.jit
    def initialize_barriers(self, allocator, tidx):
        base = allocator.allocate_tensor(Int64, cute.make_layout(2 + 4 * self.stages), 8).iterator
        query_ready = base
        key_ready, key_empty = base + 1, base + 1 + self.stages
        value_ready, value_empty = base + 1 + 2 * self.stages, base + 1 + 3 * self.stages
        query_empty = base + 1 + 4 * self.stages
        if tidx == 0:
            cute.arch.mbarrier_init(query_ready, 1)
            cute.arch.mbarrier_init(query_empty, self.consumer_groups)
            for stage in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(key_ready + stage, 2)
                cute.arch.mbarrier_init(value_ready + stage, 1)
                cute.arch.mbarrier_init(key_empty + stage, self.consumer_groups)
                cute.arch.mbarrier_init(value_empty + stage, self.consumer_groups)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        return query_ready, key_ready, key_empty, value_ready, value_empty, query_empty

    @cute.jit
    def load_loop(self, copies, buffers, barriers, mKeyScale, mValueScale, query_block, head_idx, key_head, last_tile, tile_base):
        query_copy, key_copy, value_copy = copies
        s_query, s_keys, s_values, s_key_scale, s_value_scale = buffers
        query_ready, key_ready, key_empty, value_ready, value_empty, query_empty = barriers
        cute.arch.mbarrier_arrive_and_expect_tx(query_ready, cute.size(s_query))
        for group_idx in cutlass.range_constexpr(self.consumer_groups):
            copy_tma_operand(query_copy, s_query[None, None, group_idx], query_block + group_idx, 0, head_idx, query_ready)
        for tile_offset in range(last_tile + 1):
            stream_offset = tile_base + tile_offset
            stage, phase = stream_offset % self.stages, (stream_offset // self.stages) % 2
            tile_idx = last_tile - tile_offset
            if stream_offset >= self.stages:
                cute.arch.mbarrier_wait(key_empty + stage, phase ^ 1)
            k_ready = key_ready + stage
            cute.arch.mbarrier_arrive_and_expect_tx(k_ready, self.tile_keys * self.head_dim)
            copy_tma_operand(key_copy, s_keys[None, None, stage], tile_idx, 0, key_head, k_ready)
            key_scale_tile = cute.local_tile(mKeyScale[key_head, None], (self.groups,), (tile_idx,))
            value_scale_tile = cute.local_tile(mValueScale[key_head, None], (self.groups,), (tile_idx,))
            s_key_scale[None, stage].store(key_scale_tile.load())
            s_value_scale[None, stage].store(value_scale_tile.load())
            # K readiness includes the TMA transaction and both ordinary stores.
            cute.arch.mbarrier_arrive(k_ready)
            if stream_offset >= self.stages:
                cute.arch.mbarrier_wait(value_empty + stage, phase ^ 1)
            v_ready = value_ready + stage
            cute.arch.mbarrier_arrive_and_expect_tx(v_ready, self.tile_keys * self.head_dim)
            for group_idx in cutlass.range_constexpr(self.groups):
                s_value = s_values[None, None, stage * self.groups + group_idx]
                copy_tma_operand(value_copy, s_value, 0, tile_idx * self.groups + group_idx, key_head, v_ready)

    @cute.jit
    def build_probabilities(self, s_key_scale, s_value_scale, stage, query, key, mma, thread, row_max, row_sum,
                            rescale, weights, probabilities, query_start, tile_idx,
                            scale, tidx, key_empty, peer_barrier, MASKED: cutlass.Constexpr):
        shape = (BLOCK_QUERY, self.tile_keys)
        # The QK destination is born after the previous tile's PV fragment dies.
        scores = cute.make_rmem_tensor(thread.partition_shape_C(shape), Float32)
        issue_multiply_tile(mma, thread, scores, query, key, False)
        if cutlass.const_expr(not MASKED):
            cute.arch.barrier_arrive(barrier_id=peer_barrier, number_of_threads=256)
        key_scales, value_scales = read_tile_scales(s_key_scale, s_value_scale, stage, scale, self.groups)
        warpgroup.wait_group(0)
        if tidx == 0:
            cute.arch.mbarrier_arrive(key_empty)
        coords = thread.partition_C(cute.make_identity_tensor(shape))
        scale_causal_scores(scores, coords, key_scales, query_start, tile_idx * self.tile_keys, MASKED, self.group_keys)
        update_probability_tile(scores, weights, probabilities, value_scales, row_max, row_sum, rescale,
                                tidx, self.group_keys, self.groups)

    @cute.jit
    def compute_loop(self, inputs, buffers, s_output_cache, barriers, query_block, head_idx, key_head, last_tile,
                     consumer_group, tidx, tile_base):
        mQueryScale, mKeyScale, mValueScale, mOut, mLse = inputs
        s_queries, s_keys, s_values, s_key_scale, s_value_scale = buffers
        query_ready, key_ready, key_empty, value_ready, value_empty, query_empty = barriers
        query = s_queries[None, None, consumer_group]
        mma_score, mma_value = make_mma(self.tile_keys), make_mma(self.value_columns, use_register_operand=True)
        score_thread, value_thread = mma_score.get_slice(tidx), mma_value.get_slice(tidx)
        out_shape = value_thread.partition_shape_C((BLOCK_QUERY, self.value_columns))
        SLABS = self.head_dim // self.value_columns
        REGISTER_SLABS = SLABS - 1 if SLABS == 2 else SLABS
        acc_output = cute.make_rmem_tensor((*out_shape, REGISTER_SLABS), Float32)
        acc_output.fill(0.0)
        cached_output = None
        if cutlass.const_expr(SLABS == 2):
            lane_cache = s_output_cache[tidx, None, consumer_group]
            fragment_layout = acc_output[None, None, None, 0].layout
            cached_output = cute.make_tensor(lane_cache.iterator, cute.composition(lane_cache.layout, fragment_layout))
            cached_output.fill(0.0)
        row_max, row_sum, rescale = (cute.make_rmem_tensor(2, Float32) for _ in range(3))
        row_max.fill(float("-inf"))
        row_sum.fill(0.0)
        prob_shape = value_thread.partition_shape_A((BLOCK_QUERY, self.group_keys))
        probabilities = cute.make_rmem_tensor((*prob_shape, self.groups), Float8E4M3FN)
        weights = cute.make_rmem_tensor((2, self.groups), Float32)
        query_start = query_block * BLOCK_QUERY
        scale = mQueryScale[head_idx, query_start // (2 * self.group_keys)]
        own_barrier, peer_barrier = 1 + consumer_group, 2 - consumer_group

        first_stage, first_phase = tile_base % self.stages, (tile_base // self.stages) % 2
        cute.arch.mbarrier_wait(key_ready + first_stage, first_phase)
        self.build_probabilities(s_key_scale, s_value_scale, first_stage, query, s_keys[None, None, first_stage],
                                 mma_score, score_thread, row_max, row_sum, rescale, weights, probabilities,
                                 query_start, last_tile, scale, tidx, key_empty + first_stage,
                                 peer_barrier, True)
        if consumer_group == 1:
            cute.arch.barrier_arrive(barrier_id=1, number_of_threads=256)

        for tile_offset in range(last_tile):
            stream_offset = tile_base + tile_offset
            stage, phase = stream_offset % self.stages, (stream_offset // self.stages) % 2
            tile_idx = last_tile - tile_offset
            cute.arch.mbarrier_wait(value_ready + stage, phase)
            cute.arch.barrier(barrier_id=own_barrier, number_of_threads=256)
            accumulate_value_tile(acc_output, cached_output, mma_value, value_thread, s_values, weights, probabilities,
                                  rescale, stage, self.group_keys, self.groups, self.value_columns, SLABS)
            if tidx == 0:
                cute.arch.mbarrier_arrive(value_empty + stage)
            next_offset = stream_offset + 1
            next_stage, next_phase = next_offset % self.stages, (next_offset // self.stages) % 2
            cute.arch.mbarrier_wait(key_ready + next_stage, next_phase)
            self.build_probabilities(s_key_scale, s_value_scale, next_stage, query, s_keys[None, None, next_stage],
                                     mma_score, score_thread, row_max, row_sum, rescale, weights, probabilities,
                                     query_start, tile_idx - 1, scale, tidx, key_empty + next_stage,
                                     peer_barrier, False)

        # Drain the last turn so the named barriers start empty for the next query pair.
        if consumer_group == 0:
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
        if tidx == 0:
            cute.arch.mbarrier_arrive(query_empty)
        tail_offset = tile_base + last_tile
        stage, phase = tail_offset % self.stages, (tail_offset // self.stages) % 2
        cute.arch.mbarrier_wait(value_ready + stage, phase)
        accumulate_value_tile(acc_output, cached_output, mma_value, value_thread, s_values, weights, probabilities,
                              rescale, stage, self.group_keys, self.groups, self.value_columns, SLABS)
        if tidx == 0:
            cute.arch.mbarrier_arrive(value_empty + stage)
        store_attention(mOut, mLse, acc_output, cached_output, row_sum, row_max, value_thread,
                        query_block, head_idx, tidx, self.value_columns, SLABS)


@cute.jit
def launch_forward_attention(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, mTicket, stream):
    CausalAttention(mQuery.shape[2])(mQuery, mQueryScale, mKey, mKeyScale, mValueT, mValueScale, mOut, mLse, mTicket, stream)


def delta_matching_fwd_sm90(query, key, value, softmax_scale):
    """
    Quantize inputs and compute causal FP8 attention and its backward state on SM90.
    """
    B, N, H, D = query.shape
    group_keys = 8192 // D
    query_codes, query_scales, query_t = quantize_delta_matching(query, group_keys * 2, input_scale=softmax_scale, is_forward=True)
    key_codes, key_scales, key_t = quantize_delta_matching(key, group_keys, center_key=True, is_forward=True)
    value_codes, value_scales, value_t = quantize_delta_matching(value, group_keys, is_forward=True)
    out = torch.empty((B, N, H, D), dtype=query.dtype, device=query.device)
    lse = torch.empty((B * H, N), dtype=torch.float32, device=query.device)
    ticket = torch.zeros(1, dtype=torch.int32, device=query.device)
    launch_delta_matching_kernel("delta_matching_forward", launch_forward_attention,
                                 (query_codes, query_scales, key_codes, key_scales, value_t, value_scales, out, lse, ticket))
    return (out, lse, query_codes, query_scales, query_t, key_codes, key_scales, key_t, value_codes, value_scales)
