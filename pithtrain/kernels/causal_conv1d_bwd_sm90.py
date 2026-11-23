"""
SM90 input and weight gradients for depthwise causal convolution.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Float64, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T as mlir_types, dsl_user_op

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, atom_add_u64, expect_slot_bytes, bf16_lo, f32_as_u32, fence_async_shared, fence_gpu, fma_f32, ld_b16, ld_v4, lds_b32, pack_bf16, red_or_shared, release_slot, st_b16, st_b64, st_f32, st_v2, st_v4, sts_b32, tma_load_rows, to_ir, u32_as_f32, wait_tma_slot
from pithtrain.kernels.causal_conv1d_sm90 import BLOCKS_PER_WARP, CHANNELS_PER_CTA, GROUPS_PER_WARP, MAX_TAPS, SMEM_LIMIT, TMA_CHANNELS, TMA_CONSUMERS, ThreadTile, TmaRing, TmaRingMemory, get_tma_descriptor_addr, initialize_tma_ring, is_tma_input, load_channel_segment, load_input_window, load_weights, make_fake_conv_vector, make_fake_tma_rows, prepare_weights, unpack_bf16_element

__all__ = ["causal_conv1d_bwd_sm90"]

COMPILED: dict = {}
BWD_TOKENS, BWD_WARPS = 8, 4
TARGET_CTAS = 8  # per SM, for the backward's token split
BWD_TMA_ROWS, BWD_TMA_STAGES = 64, 4
BWD_TMA_THREAD_CHANNELS = 4


@dsl_user_op
def fma_f64(first: Float64, second: Float64, acc: Float64, *, loc=None, ip=None) -> Float64:
    return Float64(asm(mlir_types.f64(), [to_ir(first, Float64, loc, ip), to_ir(second, Float64, loc, ip), to_ir(acc, Float64, loc, ip)], "fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d"))


@dsl_user_op
def max_abs_operand(prior: Float32, value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(asm(mlir_types.f32(), [to_ir(prior, Float32, loc, ip), to_ir(value, Float32, loc, ip)], "{ .reg .f32 mag; abs.f32 mag, $2; max.f32 $0, $1, mag; }", "=f,f,f",))


@dsl_user_op
def reduce_warp_max_magnitude(value: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(mlir_types.i32(), [to_ir(value, Uint32, loc, ip)], "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r",))


@dsl_user_op
def round_dw_bf16(value: Float64, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(mlir_types.i32(), [to_ir(value, Float64, loc, ip)], "{ .reg .b16 bits; cvt.rn.bf16.f64 bits, $1; cvt.u32.u16 $0, bits; }", "=r,d",))


@dsl_user_op
def lds_f64(addr: Int32, *, loc=None, ip=None) -> Float64:
    return Float64(asm(mlir_types.f64(), [to_ir(addr, Int32, loc, ip)], "ld.shared.f64 $0, [$1];", "=d,r", side_effects=True))


@dsl_user_op
def sts_f64(addr: Int32, value: Float64, access_ok: Int32, *, loc=None, ip=None) -> None:
    asm(None, [to_ir(addr, Int32, loc, ip), to_ir(value, Float64, loc, ip), to_ir(access_ok, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.f64 [$0], $1;\n\t}", "r,d,r", side_effects=True)


def store_dw_value(base: Int64, index: Int64, value: Float64, access_ok: Int32, dtype: cutlass.Constexpr) -> None:
    """
    Round the completed FP64 sum directly to the output type.
    """
    if dtype is Float64:
        st_b64(base + index * 8, value.bitcast(Int64), access_ok)
    elif dtype is Float32:
        st_f32(base + index * 4, value.to(Float32), access_ok)
    else:
        output_addr = base + index * 2
        ptx_dtype = "f16" if dtype is cutlass.Float16 else "bf16"
        asm(None, [to_ir(output_addr, Int64), to_ir(value, Float64), to_ir(access_ok, Int32)],
            "{\n\t.reg .pred p;\n\t.reg .b16 rounded;\n\tsetp.ne.b32 p, $2, 0;\n\t"
            f"cvt.rn.{ptx_dtype}.f64 rounded, $1;\n\t"
            "@p st.global.b16 [$0], rounded;\n\t}", "l,d,r", side_effects=True)


def store_channel_segment(addr: Int64, words: list, row_ok: Int32, channel_start: Int32, num_channels: cutlass.Constexpr, use_vector: cutlass.Constexpr) -> None:
    """
    Store one eight-channel segment in channels-last order.
    """
    if cutlass.const_expr(use_vector):
        st_v4(addr, tuple(words), row_ok & Int32(channel_start < num_channels))
    else:
        for word_idx in range(4):
            st_b16(addr + 4 * word_idx, words[word_idx], row_ok & Int32(channel_start + 2 * word_idx < num_channels))
            st_b16(addr + 4 * word_idx + 2, words[word_idx] >> 16, row_ok & Int32(channel_start + 2 * word_idx + 1 < num_channels))


def sum_over_token_blocks(value: Float64) -> Float64:
    """
    Butterfly sum over the lanes of a warp that share a channel group.
    """
    lane_mask = GROUPS_PER_WARP
    while lane_mask < 32:
        value = value + cute.arch.shuffle_sync_bfly(value, offset=lane_mask)
        lane_mask *= 2
    return value


class DirectBackwardTile:
    def __init__(self, grad_base, x_base, dx_base, thread_tile, split_idx, input_length, output_length, offset, grad_major_stride):
        self.grad_base, self.x_base, self.dx_base = grad_base, x_base, dx_base
        self.channel_start, self.block_row_offset = thread_tile.channel_start, thread_tile.block_row_offset
        self.split_idx, self.input_length, self.output_length = split_idx, input_length, output_length
        self.offset, self.grad_major_stride = offset, grad_major_stride


class DirectConvBackward:
    """
    Each CTA writes input gradients and a weight-gradient partial. The last CTA for a
    channel tile merges partials in order and resets the counter for reuse.
    """

    def __init__(self, num_channels: int, K: int, is_weight_fp32: bool, use_vector_input: bool, is_grad_channels_first: bool, use_vector_grad: bool, grad_vector_shift: int, dw_type: type, use_certificate: bool) -> None:
        self.num_channels, self.K, self.is_weight_fp32, self.use_vector_input = num_channels, K, is_weight_fp32, use_vector_input
        self.is_grad_channels_first, self.use_vector_grad, self.grad_vector_shift = is_grad_channels_first, use_vector_grad, grad_vector_shift
        self.dw_type, self.use_certificate = dw_type, use_certificate
        self.dw_bytes = BWD_WARPS * CHANNELS_PER_CTA * K * 8
        self.chunk_tokens = BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS
        self.window = BWD_TOKENS + K - 1

    @cute.jit
    def __call__(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, mPartials: cute.Tensor, mCounters: cute.Tensor, B: Int32, input_length: Int32, output_length: Int32, offset: Int32, grad_batch_stride: Int64, grad_major_stride: Int64, num_splits: Int32, num_iterations: Int32, stream) -> None:
        num_channel_tiles = (self.num_channels + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
        self.kernel(mG, mX, mW, mDx, mDw, mPartials, mCounters, B, input_length, output_length, offset, grad_batch_stride, grad_major_stride, num_splits, num_iterations).launch(grid=[num_channel_tiles, num_splits, B], block=[BWD_WARPS * 32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, mPartials: cute.Tensor, mCounters: cute.Tensor, B: Int32, input_length: Int32, output_length: Int32, offset: Int32, grad_batch_stride: Int64, grad_major_stride: Int64, num_splits: Int32, num_iterations: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        thread_tile = ThreadTile(tidx, bx, BWD_TOKENS)
        smem_allocator = cutlass.memory.SmemAllocator()
        if cutlass.const_expr(self.use_certificate):
            s_dw = smem_allocator.allocate(self.dw_bytes + BWD_WARPS * 4, byte_alignment=16).toint()
            s_last_partial = Int32(0)
        else:
            s_dw = smem_allocator.allocate(self.dw_bytes, byte_alignment=16).toint()
            s_last_partial = smem_allocator.allocate(16, byte_alignment=16).toint()
        grad_base, x_base = mG.iterator.toint() + Int64(bz) * grad_batch_stride * 2, mX.iterator.toint() + Int64(bz) * input_length * self.num_channels * 2
        dx_base = mDx.iterator.toint() + Int64(bz) * input_length * self.num_channels * 2
        tile = DirectBackwardTile(grad_base, x_base, dx_base, thread_tile, by, input_length, output_length, offset, grad_major_stride)

        weights = load_weights(mW.iterator.toint(), thread_tile.channel_start, self.num_channels, self.K, self.is_weight_fp32)
        acc_dw = cute.make_rmem_tensor((8 * self.K,), Float64)  # (CHANNEL * TAP,)
        for i in cutlass.range_constexpr(8 * self.K):
            acc_dw[i] = Float64(0.0)

        amax_operand = self.compute_chunks(tile, weights, acc_dw, num_iterations)

        self.reduce_dw_in_cta(acc_dw, tidx, s_dw, amax_operand)
        partial_idx = bz * num_splits + by
        self.store_dw_partial(mPartials.iterator.toint(), s_dw, partial_idx, bx, tidx)
        if cutlass.const_expr(self.use_certificate):
            self.store_dw_certificate(mPartials, s_dw, partial_idx, bx, tidx, B, num_splits)
        else:
            counter_addr = mCounters.iterator.toint() + Int64(bx) * 8
            is_last = self.count_dw_partial(counter_addr, B * num_splits, tidx, s_last_partial)
            if is_last != 0:
                self.store_dw(mPartials.iterator.toint(), mDw.iterator.toint(), counter_addr, B * num_splits, bx, tidx)

    @cute.jit
    def compute_chunks(self, tile: DirectBackwardTile, weights: list, acc_dw, num_iterations: Int32) -> Float32:
        grad_base, x_base, dx_base = tile.grad_base, tile.x_base, tile.dx_base
        channel_start, block_row_offset = tile.channel_start, tile.block_row_offset
        split_idx, input_length, output_length = tile.split_idx, tile.input_length, tile.output_length
        offset, grad_major_stride = tile.offset, tile.grad_major_stride
        amax_operand = Float32(0.0)
        for iteration_idx in cutlass.range(num_iterations, unroll=1):
            row_start = (split_idx * num_iterations + iteration_idx) * self.chunk_tokens + block_row_offset
            grad_window = self.load_grad_window(grad_base, row_start - offset, output_length, grad_major_stride, channel_start)
            input_rows = load_input_window(x_base, row_start - (self.K - 1), input_length, channel_start, self.num_channels, self.window, self.use_vector_input)
            self.store_dx(dx_base, grad_window, weights, row_start, input_length, channel_start)
            self.accumulate_dw(acc_dw, grad_window, input_rows)
            if cutlass.const_expr(self.use_certificate):
                for j in cutlass.range_constexpr(BWD_TOKENS):
                    for channel_idx in cutlass.range_constexpr(8):
                        amax_operand = max_abs_operand(amax_operand, grad_window[j][channel_idx])
                for j in cutlass.range_constexpr(self.window):
                    for channel_idx in cutlass.range_constexpr(8):
                        amax_operand = max_abs_operand(amax_operand, unpack_bf16_element(input_rows[j], channel_idx))
        return amax_operand

    @cute.jit
    def store_dw_certificate(self, mPartials: cute.Tensor, s_dw: Int32, partial_idx: Int32, channel_tile_idx: Int32, tidx: Int32, B: Int32, num_splits: Int32) -> None:
        if tidx == 0:
            acc_tile_amax = Float32(0.0)
            for warp in cutlass.range_constexpr(BWD_WARPS):
                warp_magnitude = u32_as_f32(lds_b32(s_dw + self.dw_bytes + warp * 4))
                acc_tile_amax = max_abs_operand(acc_tile_amax, warp_magnitude)
            tiles = (self.num_channels + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
            index = Int64(B) * num_splits * self.num_channels * self.K + Int64(partial_idx) * tiles + channel_tile_idx
            st_b64(mPartials.iterator.toint() + index * 8, acc_tile_amax.to(Float64).bitcast(Int64), Int32(1))

    def load_grad_window(self, grad_base: Int64, output_start: Int32, output_length: Int32, grad_major_stride: Int64, channel_start: Int32) -> list:
        """
        Load the thread's output-gradient history as FP32 with out-of-range rows zeroed.
        """
        num_channels, window = self.num_channels, self.window
        grad_window = [[None] * 8 for _ in range(window)]
        if cutlass.const_expr(not self.is_grad_channels_first):
            for i in range(window):
                out_idx = output_start + i
                row_ok = Int32(out_idx >= 0) & Int32(out_idx < output_length)
                words = load_channel_segment(grad_base + (Int64(out_idx) * grad_major_stride + channel_start) * 2, row_ok, channel_start, num_channels, self.use_vector_grad)
                for channel_idx in range(8):
                    grad_window[i][channel_idx] = unpack_bf16_element(words, channel_idx)
        elif cutlass.const_expr(self.use_vector_grad):
            # out_start - g_shift and S are multiples of 8: each 8-token vector is in or out whole
            vector_start = output_start - self.grad_vector_shift
            num_vectors = (self.grad_vector_shift + window + 7) // 8
            for channel_idx in range(8):
                channel_ok = Int32(channel_start + channel_idx < num_channels)
                channel_addr = grad_base + Int64(channel_start + channel_idx) * grad_major_stride * 2
                words = []
                for vector_idx in range(num_vectors):
                    token_idx = vector_start + 8 * vector_idx
                    words += list(ld_v4(channel_addr + Int64(token_idx) * 2, channel_ok & Int32(token_idx >= 0) & Int32(token_idx < output_length)))
                for i in range(window):
                    grad_window[i][channel_idx] = unpack_bf16_element(words, self.grad_vector_shift + i)
        else:
            for channel_idx in range(8):
                channel_ok = Int32(channel_start + channel_idx < num_channels)
                channel_addr = grad_base + Int64(channel_start + channel_idx) * grad_major_stride * 2
                for i in range(window):
                    out_idx = output_start + i
                    grad_window[i][channel_idx] = bf16_lo(ld_b16(channel_addr + Int64(out_idx) * 2, channel_ok & Int32(out_idx >= 0) & Int32(out_idx < output_length)))
        return grad_window

    def store_dx(self, dx_base: Int64, grad_window: list, weights: list, row_start: Int32, input_length: Int32, channel_start: Int32) -> None:
        """
        Accumulate input gradients in ascending tap order before rounding to BF16.
        """
        K = self.K
        for token_idx in range(BWD_TOKENS):
            words = []
            for pair_idx in range(4):
                input_grad_pair = []
                for channel_idx in (2 * pair_idx, 2 * pair_idx + 1):
                    acc_input_grad = Float32(0.0)
                    for k in range(K):
                        acc_input_grad = fma_f32(weights[channel_idx][k], grad_window[token_idx + K - 1 - k][channel_idx], acc_input_grad)
                    input_grad_pair.append(acc_input_grad)
                words.append(pack_bf16(input_grad_pair[0], input_grad_pair[1]))
            row_idx = row_start + token_idx
            store_channel_segment(dx_base + (Int64(row_idx) * self.num_channels + channel_start) * 2, words, Int32(row_idx < input_length), channel_start, self.num_channels, self.use_vector_input)

    def accumulate_dw(self, acc_dw, grad_window: list, input_rows: list) -> None:
        """
        Accumulate each channel and tap in FP64 in output row order.
        """
        K = self.K
        for channel_idx in range(8):
            for k in range(K):
                acc = acc_dw[channel_idx * K + k]
                for j in range(BWD_TOKENS):
                    acc = fma_f64(grad_window[j][channel_idx].to(Float64), unpack_bf16_element(input_rows[j + k], channel_idx).to(Float64), acc)
                acc_dw[channel_idx * K + k] = acc

    @cute.jit
    def reduce_dw_in_cta(self, acc_dw, tidx, s_dw: Int32, amax_operand: Float32) -> None:
        """
        Merge lanes with the same channels and store the warp partials in shared memory.
        """
        K = self.K
        lane, warp = tidx % 32, tidx // 32
        for i in cutlass.range_constexpr(8 * K):
            acc_dw[i] = sum_over_token_blocks(acc_dw[i])
        for i in cutlass.range_constexpr(8 * K):
            entry_idx = (warp * CHANNELS_PER_CTA + lane * 8) * K + i
            sts_f64(s_dw + entry_idx * 8, acc_dw[i], Int32(lane < GROUPS_PER_WARP))
        if cutlass.const_expr(self.use_certificate):
            warp_amax = reduce_warp_max_magnitude(f32_as_u32(amax_operand))
            sts_b32(s_dw + self.dw_bytes + warp * 4, warp_amax, Int32(lane == 0))
        cute.arch.sync_threads()

    @cute.jit
    def store_dw_partial(self, partials_base: Int64, s_dw: Int32, partial_idx: Int32, channel_tile_idx: Int32, tidx) -> None:
        """
        Sum warp partials in warp order into the CTA's weight-gradient partial.
        """
        K, num_channels = self.K, self.num_channels
        for i in cutlass.range_constexpr((CHANNELS_PER_CTA * K + BWD_WARPS * 32 - 1) // (BWD_WARPS * 32)):
            entry_idx = i * BWD_WARPS * 32 + tidx
            entry_ok = Int32(entry_idx < CHANNELS_PER_CTA * K)
            acc = Float64(0.0)
            for warp in cutlass.range_constexpr(BWD_WARPS):
                acc = acc + lds_f64(s_dw + (warp * CHANNELS_PER_CTA * K + entry_idx * entry_ok) * 8)
            channel_idx = channel_tile_idx * CHANNELS_PER_CTA + entry_idx // K
            st_b64(partials_base + ((Int64(partial_idx) * num_channels + channel_idx) * K + entry_idx % K) * 8, acc.bitcast(Int64), entry_ok & Int32(channel_idx < num_channels))

    @cute.jit
    def count_dw_partial(self, counter_addr: Int64, num_partials: Int32, tidx, s_last_partial: Int32) -> Int32:
        """
        Count this CTA's partial as written; 1 in every thread of the CTA that counts the last one.
        """
        fence_gpu()  # each thread's partial stores before the barrier and the count
        cute.arch.sync_threads()
        if tidx == 0:
            num_done = atom_add_u64(counter_addr, Int64(1))
            sts_b32(s_last_partial, Uint32(Int32(num_done == Int64(num_partials) - 1)), Int32(1))
        cute.arch.sync_threads()
        return Int32(lds_b32(s_last_partial))

    @cute.jit
    def store_dw(self, partials_base: Int64, dw_base: Int64, counter_addr: Int64, num_partials: Int32, channel_tile_idx: Int32, tidx) -> None:
        """
        Merge CTA partials in order and reset the counter for the next launch.
        """
        K, num_channels = self.K, self.num_channels
        fence_gpu()  # the other CTAs' partials, published before their count
        for i in cutlass.range_constexpr((CHANNELS_PER_CTA * K + BWD_WARPS * 32 - 1) // (BWD_WARPS * 32)):
            entry_idx = i * BWD_WARPS * 32 + tidx
            channel_idx = channel_tile_idx * CHANNELS_PER_CTA + entry_idx // K
            entry_ok = Int32(entry_idx < CHANNELS_PER_CTA * K) & Int32(channel_idx < num_channels)
            acc = Float64(0.0)
            for partial_idx in cutlass.range(num_partials, unroll=1):
                partial_addr = partials_base + ((Int64(partial_idx) * num_channels + channel_idx) * K + entry_idx % K) * 8
                acc = acc + Float64(asm(mlir_types.f64(), [to_ir(partial_addr, Int64), to_ir(entry_ok, Int32)],
                    "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, 0;\n\t@p ld.relaxed.gpu.global.f64 $0, [$1];\n\t}",
                    "=d,l,r", side_effects=True))
            weight_idx = Int64(channel_idx) * K + entry_idx % K
            store_dw_value(dw_base, weight_idx, acc, entry_ok, self.dw_type)
        st_b64(counter_addr, Int64(0), Int32(tidx == 0))


class TmaBackwardStage:
    def __init__(self, s_grad_slot, s_input_slot, s_empty_barrier, local_row_start, group_idx, dx_base, row_start, input_length, channel_start, s_dx_slot) -> None:
        self.s_grad_slot, self.s_input_slot, self.s_empty_barrier = s_grad_slot, s_input_slot, s_empty_barrier
        self.local_row_start, self.group_idx, self.s_dx_slot = local_row_start, group_idx, s_dx_slot
        self.dx_base, self.row_start, self.input_length, self.channel_start = dx_base, row_start, input_length, channel_start


class TmaConvBackward:
    """
    The producer loads overlapping input and output-gradient windows. Consumers write input
    gradients and reduce weight gradients in token, lane, warp and span order.
    """

    def __init__(self, num_channels: int, K: int, is_weight_fp32: bool, num_sequence_splits: int, dw_type: type, batched: bool = False, certify_single_sequence: bool = False) -> None:
        self.num_channels, self.K, self.is_weight_fp32 = num_channels, K, is_weight_fp32
        self.num_sequence_splits = num_sequence_splits
        self.dw_type = dw_type
        self.batched = batched
        self.sequences = 2 if batched else 1
        self.use_certificate = batched or certify_single_sequence
        self.thread_channels = 2 if batched else BWD_TMA_THREAD_CHANNELS
        self.channel_groups = TMA_CHANNELS // self.thread_channels
        self.stage_rows = 32 if batched else BWD_TMA_ROWS
        self.tokens = self.stage_rows // (TMA_CONSUMERS // self.channel_groups)
        self.box_rows = self.stage_rows + K - 1
        self.box_bytes = self.box_rows * TMA_CHANNELS * 2
        self.box_stride = (self.box_bytes + 127) // 128 * 128
        self.ring = TmaRing(BWD_TMA_STAGES, 2 * self.box_stride, 1)
        self.dx_stage_bytes = self.stage_rows * TMA_CHANNELS * 2
        self.dx_slots = 3 if batched else 0
        self.producer_warp = 0
        self.dw_bytes = 4 * TMA_CHANNELS * K * 8
        self.reduction_bytes = self.dw_bytes + (TMA_CONSUMERS // 32 * 4 if self.use_certificate else 0)
        assert (self.ring.num_slots * self.ring.slot_bytes + self.reduction_bytes + self.dx_slots * self.dx_stage_bytes + 1024 <= SMEM_LIMIT)

    @cute.jit
    def __call__(self, mG: cute.Tensor, mX: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, input_length: Int32, offset: Int32, stream) -> None:
        box_layout = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        atom_g, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mG, box_layout, (self.box_rows, TMA_CHANNELS))
        atom_x, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX, box_layout, (self.box_rows, TMA_CHANNELS))
        self.kernel(atom_g, atom_x, atom_g, atom_x, atom_g, mW, mDx, mDw, input_length, offset).launch(grid=[self.num_channels // TMA_CHANNELS, self.num_sequence_splits, 1], block=[32 + TMA_CONSUMERS, 1, 1], stream=stream)

    @cute.jit
    def batch_two(self, mG0: cute.Tensor, mX0: cute.Tensor, mG1: cute.Tensor, mX1: cute.Tensor, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, input_length: Int32, offset: Int32, stream) -> None:
        box_layout = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        tile = (self.box_rows, TMA_CHANNELS)
        atom_g0, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mG0, box_layout, tile)
        atom_x0, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX0, box_layout, tile)
        atom_g1, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mG1, box_layout, tile)
        atom_x1, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX1, box_layout, tile)
        dx_tile = (self.stage_rows, TMA_CHANNELS)
        dx_box_layout = cute.make_layout(dx_tile, stride=(TMA_CHANNELS, 1))
        atom_dx, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDx, dx_box_layout, dx_tile)
        self.kernel(atom_g0, atom_x0, atom_g1, atom_x1, atom_dx, mW, mDx, mDw, input_length, offset).launch(grid=[self.num_channels // TMA_CHANNELS, self.num_sequence_splits, 2], block=[32 + TMA_CONSUMERS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, atom_g: cute.CopyAtom, atom_x: cute.CopyAtom, atom_g1: cute.CopyAtom, atom_x1: cute.CopyAtom, atom_dx: cute.CopyAtom, mW: cute.Tensor, mDx: cute.Tensor, mDw: cute.Tensor, input_length: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, batch = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        smem_allocator = cutlass.memory.SmemAllocator()
        ring_memory = self.ring.allocate(smem_allocator)
        s_dw, s_dx_slots = Int32(smem_allocator.allocate(self.reduction_bytes, byte_alignment=16).toint()), Int32(0)
        if cutlass.const_expr(self.batched):
            s_dx_slots = Int32(smem_allocator.allocate(self.dx_slots * self.dx_stage_bytes, byte_alignment=128).toint())
        if tidx == 0:
            initialize_tma_ring(ring_memory.s_full_barriers, ring_memory.s_empty_barriers, self.ring.num_slots)
        cute.arch.sync_threads()
        rows_per_stage_across_spans = self.stage_rows * self.num_sequence_splits
        num_stages = (input_length + rows_per_stage_across_spans - 1) // rows_per_stage_across_spans
        span_row_start = Int32(0)
        if cutlass.const_expr(self.num_sequence_splits > 1):
            span_row_start = by * num_stages * self.stage_rows
        tile_channel_start = bx * TMA_CHANNELS
        if warp == self.producer_warp:
            if tidx == 0:
                self.load_loop(atom_g, atom_x, atom_g1, atom_x1, batch, ring_memory, num_stages, tile_channel_start, span_row_start, offset)
        else:
            dx_base = mDx.iterator.toint()
            dx_descriptor_addr = Int64(0)
            span_idx = by
            if cutlass.const_expr(self.batched):
                dx_descriptor_addr = get_tma_descriptor_addr(atom_dx)
                span_idx += batch * self.num_sequence_splits
            self.compute_loop(mW, dx_base, mDw, ring_memory, s_dw, num_stages, tile_channel_start, span_row_start, span_idx, input_length, tidx - 32, s_dx_slots, dx_descriptor_addr, batch * input_length)

    @cute.jit
    def load_loop(self, atom_g: cute.CopyAtom, atom_x: cute.CopyAtom, atom_g1: cute.CopyAtom, atom_x1: cute.CopyAtom, batch: Int32, ring_memory: TmaRingMemory, num_stages: Int32, tile_channel_start: Int32, span_row_start: Int32, offset: Int32) -> None:
        grad_descriptor_addr, input_descriptor_addr = get_tma_descriptor_addr(atom_g), get_tma_descriptor_addr(atom_x)
        if cutlass.const_expr(self.batched):
            if batch == 1:
                cpasync.prefetch_descriptor(atom_g1)
                cpasync.prefetch_descriptor(atom_x1)
                grad_descriptor_addr, input_descriptor_addr = get_tma_descriptor_addr(atom_g1), get_tma_descriptor_addr(atom_x1)
            else:
                cpasync.prefetch_descriptor(atom_g)
                cpasync.prefetch_descriptor(atom_x)
        else:
            cpasync.prefetch_descriptor(atom_g)
            cpasync.prefetch_descriptor(atom_x)
        for stage_idx in cutlass.range(num_stages, unroll=1):
            self.load_stage(ring_memory, stage_idx, grad_descriptor_addr, input_descriptor_addr, tile_channel_start, span_row_start, offset)

    @cute.jit
    def load_stage(self, ring_memory: TmaRingMemory, stage_idx: Int32, grad_descriptor_addr: Int64, input_descriptor_addr: Int64, tile_channel_start: Int32, span_row_start: Int32, offset: Int32) -> None:
        s_slots, s_full_barriers, s_empty_barriers = ring_memory.s_slots, ring_memory.s_full_barriers, ring_memory.s_empty_barriers
        slot_idx, parity = self.ring.get_slot_coordinates(stage_idx)
        wait_tma_slot(s_empty_barriers + 8 * slot_idx, parity ^ 1)
        s_full_barrier = s_full_barriers + 8 * slot_idx
        expect_slot_bytes(s_full_barrier, 2 * self.box_bytes)
        s_slot, row_start = s_slots + slot_idx * self.ring.slot_bytes, span_row_start + stage_idx * self.stage_rows
        tma_load_rows(s_slot, grad_descriptor_addr, tile_channel_start, row_start - offset, s_full_barrier)
        tma_load_rows(s_slot + self.box_stride, input_descriptor_addr, tile_channel_start, row_start - (self.K - 1), s_full_barrier)

    @cute.jit
    def compute_loop(self, mW: cute.Tensor, dx_base: Int64, mDw: cute.Tensor, ring_memory: TmaRingMemory, s_dw: Int32, num_stages: Int32, tile_channel_start: Int32, span_row_start: Int32, span_idx: Int32, input_length: Int32, consumer_idx: Int32, s_dx_slots: Int32,
                     dx_descriptor_addr: Int64, batch_row: Int32) -> None:
        s_slots, s_full_barriers, s_empty_barriers = ring_memory.s_slots, ring_memory.s_full_barriers, ring_memory.s_empty_barriers
        group_idx, block_idx = consumer_idx % self.channel_groups, consumer_idx // self.channel_groups
        channel_start = tile_channel_start + group_idx * self.thread_channels
        weights = load_weights(mW.iterator.toint(), channel_start, self.num_channels, self.K, self.is_weight_fp32, self.thread_channels)
        acc_dw = cute.make_rmem_tensor((self.thread_channels * self.K,), Float64)  # (CHANNEL * TAP,)
        for i in cutlass.range_constexpr(self.thread_channels * self.K):
            acc_dw[i] = Float64(0.0)
        amax_operand = Float32(0.0)
        local_row_start = block_idx * self.tokens
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.get_slot_coordinates(stage_idx)
            wait_tma_slot(s_full_barriers + 8 * slot_idx, parity)
            s_grad_slot, stage_row = s_slots + slot_idx * self.ring.slot_bytes, span_row_start + stage_idx * self.stage_rows
            s_dx_slot = Int32(0)
            if cutlass.const_expr(self.batched):
                s_dx_slot = s_dx_slots + (stage_idx % self.dx_slots) * self.dx_stage_bytes
            stage = TmaBackwardStage(s_grad_slot, s_grad_slot + self.box_stride, s_empty_barriers + 8 * slot_idx, local_row_start, group_idx,
                                     dx_base, stage_row + local_row_start, input_length, channel_start, s_dx_slot)
            amax_operand = self.compute_stage(stage, weights, acc_dw, amax_operand)
            if cutlass.const_expr(self.batched):
                self.store_dx_stage(consumer_idx, dx_descriptor_addr, tile_channel_start, batch_row, stage_row, s_dx_slot)
        if cutlass.const_expr(self.batched):
            if consumer_idx == 0:
                asm(None, [], "cp.async.bulk.wait_group 0;", "", side_effects=True)
        self.reduce_dw(acc_dw, amax_operand, consumer_idx, s_dw, mDw.iterator.toint(), tile_channel_start, span_idx)

    @cute.jit
    def store_dx_stage(self, consumer_idx: Int32, dx_descriptor_addr: Int64, tile_channel_start: Int32, batch_row: Int32, stage_row: Int32, s_dx_slot: Int32) -> None:
        fence_async_shared()
        cute.arch.barrier(barrier_id=2, number_of_threads=TMA_CONSUMERS)
        if consumer_idx == 0:
            # The next stage's barrier waits for this thread. A slot is
            # reused only after the previous store's shared read retires.
            store_row = batch_row + stage_row
            asm(None, [to_ir(dx_descriptor_addr, Int64), to_ir(tile_channel_start, Int32), to_ir(store_row, Int32), to_ir(s_dx_slot, Int32)],
                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [$0, {$1, $2}], [$3];", "l,r,r,r", side_effects=True)
            asm(None, [], "cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 1;", "", side_effects=True)

    def load_box_row(self, s_slot: Int32, row_idx: Int32, group_idx: Int32) -> list:
        """
        Load the thread's consecutive BF16 channels from the box as FP32.
        """
        addr = s_slot + (row_idx * TMA_CHANNELS + group_idx * self.thread_channels) * 2
        if cutlass.const_expr(self.thread_channels == 2):
            words = (lds_b32(addr),)
        else:
            bits = lds_f64(addr).bitcast(Int64)
            words = (bits.to(Uint32), (bits >> 32).to(Uint32))
        return [unpack_bf16_element(words, channel_idx) for channel_idx in range(self.thread_channels)]

    def compute_stage(self, stage: TmaBackwardStage, weights: list, acc_dw, amax_operand: Float32) -> Float32:
        """
        Release the slot after the last shared load so the producer can refill it safely.
        """
        s_grad_slot, s_input_slot, s_empty_barrier = stage.s_grad_slot, stage.s_input_slot, stage.s_empty_barrier
        local_row_start, group_idx, s_dx_slot = stage.local_row_start, stage.group_idx, stage.s_dx_slot
        dx_base, row_start, input_length, channel_start = stage.dx_base, stage.row_start, stage.input_length, stage.channel_start
        K = self.K
        grad_rows = [self.load_box_row(s_grad_slot, local_row_start + i, group_idx) for i in range(K - 1)]
        input_rows = [self.load_box_row(s_input_slot, local_row_start + i, group_idx) for i in range(K - 1)]
        if cutlass.const_expr(self.use_certificate):
            for row in input_rows:
                for value in row:
                    amax_operand = max_abs_operand(amax_operand, value)
        for token_idx in range(self.tokens):
            grad_rows.append(self.load_box_row(s_grad_slot, local_row_start + token_idx + K - 1, group_idx))
            input_rows.append(self.load_box_row(s_input_slot, local_row_start + token_idx + K - 1, group_idx))
            if token_idx == self.tokens - 1:
                release_slot(s_empty_barrier)
            words = []
            for pair_idx in range(self.thread_channels // 2):
                input_grad_pair = []
                for channel_idx in (2 * pair_idx, 2 * pair_idx + 1):
                    acc_input_grad = Float32(0.0)
                    for k in range(K):
                        acc_input_grad = fma_f32(weights[channel_idx][k], grad_rows[token_idx + K - 1 - k][channel_idx], acc_input_grad)
                    input_grad_pair.append(acc_input_grad)
                words.append(pack_bf16(input_grad_pair[0], input_grad_pair[1]))
            row_idx = row_start + token_idx
            addr = dx_base + (Int64(row_idx) * self.num_channels + channel_start) * 2
            if cutlass.const_expr(self.batched):
                smem_elem = (local_row_start + token_idx) * TMA_CHANNELS + group_idx * self.thread_channels
                sts_b32(s_dx_slot + smem_elem * 2, words[0], Int32(1))
            else:
                st_v2(addr, words[0], words[1], Int32(row_idx < input_length))
            for channel_idx in range(self.thread_channels):
                if cutlass.const_expr(self.use_certificate):
                    amax_operand = max_abs_operand(amax_operand, grad_rows[token_idx][channel_idx])
                    amax_operand = max_abs_operand(amax_operand, input_rows[token_idx + K - 1][channel_idx])
                grad = grad_rows[token_idx][channel_idx].to(Float64)
                for k in range(K):
                    acc_dw[channel_idx * K + k] = fma_f64(grad, input_rows[token_idx + k][channel_idx].to(Float64), acc_dw[channel_idx * K + k])
        return amax_operand

    @cute.jit
    def reduce_dw(self, acc_dw, amax_operand: Float32, consumer_idx: Int32, s_dw: Int32, dw_base: Int64, tile_channel_start: Int32, span_idx: Int32) -> None:
        """
        Reduce in lane and warp order before writing one partial or final value per weight.
        """
        K = self.K
        lane, consumer_warp = consumer_idx % 32, consumer_idx // 32
        if cutlass.const_expr(self.channel_groups < 32):
            for i in cutlass.range_constexpr(self.thread_channels * K):
                value = acc_dw[i]
                acc_dw[i] = value + cute.arch.shuffle_sync_bfly(value, offset=self.channel_groups)
        for i in cutlass.range_constexpr(self.thread_channels * K):
            entry_idx = (consumer_warp * TMA_CHANNELS + lane * self.thread_channels) * K + i
            sts_f64(s_dw + entry_idx * 8, acc_dw[i], Int32(lane < self.channel_groups))
        if cutlass.const_expr(self.use_certificate):
            warp_amax = reduce_warp_max_magnitude(f32_as_u32(amax_operand))
            sts_b32(s_dw + self.dw_bytes + consumer_warp * 4, warp_amax, Int32(lane == 0))
        cute.arch.barrier(barrier_id=1, number_of_threads=TMA_CONSUMERS)
        if cutlass.const_expr(self.use_certificate):
            if consumer_idx == 0:
                acc_tile_amax = Float32(0.0)
                for warp_idx in cutlass.range_constexpr(TMA_CONSUMERS // 32):
                    warp_magnitude = u32_as_f32(lds_b32(s_dw + self.dw_bytes + warp_idx * 4))
                    acc_tile_amax = max_abs_operand(acc_tile_amax, warp_magnitude)
                tiles = self.num_channels // TMA_CHANNELS
                amax_idx = self.sequences * self.num_sequence_splits * self.num_channels * K + span_idx * tiles + tile_channel_start // TMA_CHANNELS
                st_b64(dw_base + Int64(amax_idx) * 8, acc_tile_amax.to(Float64).bitcast(Int64), Int32(1))
        for entry_start in cutlass.range_constexpr(0, TMA_CHANNELS * K, TMA_CONSUMERS):
            entry_idx = entry_start + consumer_idx
            entry_ok = Int32(entry_idx < TMA_CHANNELS * K)
            acc = Float64(0.0)
            for warp_idx in cutlass.range_constexpr(TMA_CONSUMERS // 32):
                acc = acc + lds_f64(s_dw + (warp_idx * TMA_CHANNELS * K + entry_idx * entry_ok) * 8)
            weight_idx = Int64(tile_channel_start) * K + entry_idx
            if cutlass.const_expr(self.num_sequence_splits > 1 or self.use_certificate):
                weight_idx += Int64(span_idx) * self.num_channels * K
                st_b64(dw_base + weight_idx * 8, acc.bitcast(Int64), entry_ok)
            else:
                store_dw_value(dw_base, weight_idx, acc, entry_ok, self.dw_type)


COUNTERS: dict = {}
DW_TYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16, torch.float32: Float32, torch.float64: Float64,}


@cute.kernel
def merge_dw_spans_kernel(mPartials: cute.Tensor, mDw: cute.Tensor, num_entries: Int32, num_spans: cutlass.Constexpr[int]) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    entry_idx = bx * 128 + tidx
    if entry_idx < num_entries:
        acc = sum_dw_spans(mPartials, entry_idx, num_entries, num_spans)
        store_dw_value(mDw.iterator.toint(), Int64(entry_idx), acc, Int32(1), mDw.element_type)


@cute.jit
def sum_dw_spans(mPartials: cute.Tensor, entry_idx: Int32, num_entries: Int32, num_spans: cutlass.Constexpr[int]) -> Float64:
    acc = Float64(0.0)
    for span_idx in cutlass.range_constexpr(num_spans):
        acc = acc + mPartials[span_idx * num_entries + entry_idx]
    return acc


@cute.jit
def merge_dw_spans(mPartials: cute.Tensor, mDw: cute.Tensor, num_entries: Int32, stream, num_spans: cutlass.Constexpr[int]) -> None:
    merge_dw_spans_kernel(mPartials, mDw, num_entries, num_spans).launch(grid=[(num_entries + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)


# Finite BF16 products are signed integers times 2**-266. An individual
# product needs 522 bits, and 37 base-2**16 digits also cover the sum over
# every row admitted by the existing batch and sequence dimensions.
DW_EXACT_DIGITS = 37
DW_EXACT_THREADS = 128
# The comparator scales the 2**-266 product bins by another 2**23.
dw_bound_slack_numerator, dw_bound_slack_denominator = (1e-30).as_integer_ratio()
dw_bound_scaled_slack = (dw_bound_slack_numerator << 289) // dw_bound_slack_denominator
DW_BOUND_SLACK_DIGITS = tuple((i, (dw_bound_scaled_slack >> (16 * i)) & 65535) for i in range(DW_EXACT_DIGITS) if (dw_bound_scaled_slack >> (16 * i)) & 65535)


@dsl_user_op
def add_dw_digit(addr: Int32, value: Int64, *, loc=None, ip=None) -> None:
    asm(None, [to_ir(addr, Int32, loc, ip), to_ir(value, Int64, loc, ip)], "{ .reg .b64 prior; atom.shared.add.u64 prior, [$0], $1; }", "r,l", side_effects=True)


@cute.jit
def normalize_dw_digits(s_digits: Int32) -> tuple[Int32, Int32]:
    """
    Carry the signed bins, then leave the magnitude in base 2**16.
    """
    carry = Int64(0)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        addr = s_digits + i * 8
        total = lds_f64(addr).bitcast(Int64) + carry
        sts_f64(addr, (total & Int64(65535)).bitcast(Float64), Int32(1))
        carry = total >> 16
    is_negative = Int32(carry < 0)
    inverse = Int64(is_negative) * 65535
    carry = Int64(is_negative)
    highest = Int32(-1)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        addr = s_digits + i * 8
        total = (lds_f64(addr).bitcast(Int64) ^ inverse) + carry
        digit = total & Int64(65535)
        sts_f64(addr, digit.bitcast(Float64), Int32(1))
        carry = total >> 16
        if digit != 0:
            highest = i
    top = Int32(-1)
    if highest >= 0:
        digit_start_bit = highest * 16
        digit = lds_f64(s_digits + highest * 8).bitcast(Int64).to(Uint32)
        digit_top = Int32(asm(mlir_types.i32(), [to_ir(digit, Uint32)], "{ .reg .b32 zeros; clz.b32 zeros, $1; sub.s32 $0, 31, zeros; }", "=r,r"))
        top = digit_start_bit + digit_top
    return is_negative, top


@cute.jit
def read_dw_rounding_bits(s_digits: Int32, shift: Int32, fraction_bits: cutlass.Constexpr[int]) -> tuple[Int64, Int64, Int64]:
    """
    Read the fraction, guard bit, and whether any lower bits are set.
    """
    base, bit_offset = (shift - 1) // 16, (shift - 1) % 16
    low = lds_f64(s_digits + base * 8).bitcast(Int64)
    window = low >> bit_offset
    sticky = Int64((low & ((Int64(1) << bit_offset) - 1)) != 0)
    for i in cutlass.range_constexpr(1, (fraction_bits + 16) // 16 + 1):
        left = i * 16 - bit_offset
        if left < fraction_bits + 1:
            window = window | (lds_f64(s_digits + (base + i) * 8).bitcast(Int64) << left)
    for i in cutlass.range(base, unroll=1):
        sticky = sticky | Int64(lds_f64(s_digits + i * 8).bitcast(Int64) != 0)
    return (window >> 1) & Int64((1 << fraction_bits) - 1), window & Int64(1), sticky


@cute.jit
def round_exact_dw_to_float(s_digits: Int32, is_negative: Int32, top: Int32, fraction_bits: cutlass.Constexpr[int], use_round_to_odd: cutlass.Constexpr[bool]) -> Float32:
    """
    Round the normalized sum at FP32 or BF16 precision and encode it in FP32.
    """
    magnitude = Uint32(0)
    if top >= 0:
        exponent = top - 266
        if exponent > 127:
            magnitude = Uint32(255 << fraction_bits)
        elif exponent < -127 - fraction_bits:
            if cutlass.const_expr(use_round_to_odd):
                magnitude = Uint32(1)
        else:
            shift = top - fraction_bits
            exponent_bits = exponent + 127
            if exponent_bits <= 0:
                shift, exponent_bits = Int32(140 - fraction_bits), Int32(0)
            fraction, guard, sticky = read_dw_rounding_bits(s_digits, shift, fraction_bits)
            magnitude = (Uint32(exponent_bits) << fraction_bits) | Uint32(fraction)
            if cutlass.const_expr(use_round_to_odd):
                magnitude = magnitude | Uint32(guard | sticky)
            else:
                magnitude = magnitude + Uint32(guard & (sticky | (fraction & Int64(1))))
    return u32_as_f32((Uint32(is_negative) << 31) | (magnitude << (23 - fraction_bits)))


@cute.jit
def initialize_dw_bound(s_digits: Int32, endpoint: Float32, epsilon: Float32) -> None:
    """
    Start the exact scaled difference (1 - epsilon) * abs(sum) - endpoint - slack.
    """
    scale = Int64(2**23) - (epsilon * Float32(2**23)).to(Int64)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        addr = s_digits + i * 8
        sts_f64(addr, (lds_f64(addr).bitcast(Int64) * scale).bitcast(Float64), Int32(1))
    bits = f32_as_u32(endpoint) & Uint32(0x7fffffff)
    exponent = Int32(bits >> 23)
    mantissa = Int64(bits & Uint32(0x7fffff)) + Int64(exponent != 0) * 2**23
    # The FP32 mantissa is scaled to integer units of 2**-289.
    shift = exponent + Int32(exponent == 0) + 139
    start = shift // 16
    shifted = mantissa << (shift % 16)
    for i in cutlass.range_constexpr(3):
        add_dw_digit(s_digits + (start + i) * 8, -((shifted >> (16 * i)) & Int64(65535)))
    for i in cutlass.range_constexpr(len(DW_BOUND_SLACK_DIGITS)):
        index, digit = DW_BOUND_SLACK_DIGITS[i]
        add_dw_digit(s_digits + index * 8, Int64(-digit))


@cute.jit
def is_dw_bound_difference_positive(s_digits: Int32) -> Int32:
    carry = Int64(0)
    is_nonzero = Int32(0)
    for i in cutlass.range(DW_EXACT_DIGITS, unroll=1):
        total = lds_f64(s_digits + i * 8).bitcast(Int64) + carry
        is_nonzero = is_nonzero | Int32((total & Int64(65535)) != 0)
        carry = total >> 16
    return Int32(carry > 0) | (Int32(carry == 0) & is_nonzero)


class DwGeometry:
    def __init__(self, B, input_length, output_length, num_channels, K, offset, grad_batch_stride, grad_row_stride, grad_channel_stride, num_splits, num_iterations):
        self.B, self.input_length, self.output_length = B, input_length, output_length
        self.num_channels, self.K, self.offset = num_channels, K, offset
        self.grad_batch_stride, self.grad_row_stride, self.grad_channel_stride = grad_batch_stride, grad_row_stride, grad_channel_stride
        self.num_splits, self.num_iterations = num_splits, num_iterations


class DwExactMemory:
    def __init__(self, s_digits: Int32, s_nonfinite_flag: Int32, s_overflow: Int32) -> None:
        self.s_digits, self.s_nonfinite_flag, self.s_overflow = s_digits, s_nonfinite_flag, s_overflow


@cute.jit
def accumulate_exact_dw(mGrad: cute.Tensor, mX: cute.Tensor, entry_idx: Int32, tidx: Int32, exact_memory: DwExactMemory, geometry: DwGeometry, subtract_bound: cutlass.Constexpr[bool]) -> None:
    B, input_length, output_length = geometry.B, geometry.input_length, geometry.output_length
    num_channels, K, offset = geometry.num_channels, geometry.K, geometry.offset
    grad_batch_stride, grad_row_stride, grad_channel_stride = geometry.grad_batch_stride, geometry.grad_row_stride, geometry.grad_channel_stride
    s_digits, s_nonfinite_flag = exact_memory.s_digits, exact_memory.s_nonfinite_flag
    channel, tap = entry_idx // K, entry_idx % K
    if cutlass.const_expr(not subtract_bound):
        if tidx < DW_EXACT_DIGITS:
            sts_f64(s_digits + tidx * 8, Float64(0.0), Int32(1))
        if tidx == 0:
            sts_b32(s_nonfinite_flag, Uint32(0), Int32(1))
    cute.arch.sync_threads()

    for index in cutlass.range(Int64(tidx), Int64(B) * output_length, DW_EXACT_THREADS, unroll=1):
        batch, token = index // output_length, index % output_length
        input_row, grad_index = token + offset - (K - 1) + tap, batch * grad_batch_stride + token * grad_row_stride + channel * grad_channel_stride
        input_index = (batch * input_length + input_row) * num_channels + channel
        grad_bits = ld_b16(mGrad.iterator.toint() + grad_index * 2, Int32(1))
        input_bits = ld_b16(mX.iterator.toint() + input_index * 2, Int32(input_row >= 0) & Int32(input_row < input_length))
        grad_exponent, input_exponent = Int32((grad_bits >> 7) & 255), Int32((input_bits >> 7) & 255)
        if grad_exponent == 255 or input_exponent == 255:
            red_or_shared(s_nonfinite_flag, Uint32(1), Int32(1))
        else:
            grad_mantissa = Int64(grad_bits & 127) + Int64(Int32(grad_exponent != 0)) * 128
            input_mantissa = Int64(input_bits & 127) + Int64(Int32(input_exponent != 0)) * 128
            product = grad_mantissa * input_mantissa
            if product != 0:
                # Subnormals share the scale of exponent field 1 but have no hidden bit.
                shift = grad_exponent + input_exponent - Int32(grad_exponent != 0) - Int32(input_exponent != 0)
                shifted = product << (shift % 16)
                sign = Int64(-4)
                if cutlass.const_expr(not subtract_bound):
                    sign = Int64(1) - Int64(((grad_bits ^ input_bits) >> 15) & 1) * 2
                addr = s_digits + (shift // 16) * 8
                add_dw_digit(addr, sign * (shifted & 65535))
                add_dw_digit(addr + 8, sign * (shifted >> 16))
    cute.arch.sync_threads()


@cute.jit
def finish_finite_dw(mGrad: cute.Tensor, mX: cute.Tensor, entry_idx: Int32, tidx: Int32, exact_memory: DwExactMemory, geometry: DwGeometry, lowest: Float32, highest: Float32, epsilon: Float32, fraction_bits: cutlass.Constexpr[int], use_round_to_odd: cutlass.Constexpr[bool]) -> tuple[Float32, Int32]:
    s_digits, s_overflow = exact_memory.s_digits, exact_memory.s_overflow
    value = Float32(0.0)
    if tidx == 0:
        is_negative, top = normalize_dw_digits(s_digits)
        value = round_exact_dw_to_float(s_digits, is_negative, top, fraction_bits, use_round_to_odd)
        clamped = Int32(value > highest) | (Int32(lowest < Float32(0.0)) & Int32(value < lowest))
        if value < lowest:
            value = lowest
        elif value > highest:
            value = highest
        if clamped != 0:
            initialize_dw_bound(s_digits, value, epsilon)
        sts_b32(s_overflow, Uint32(clamped), Int32(1))
    cute.arch.sync_threads()

    needs_replay = Int32(0)
    if lds_b32(s_overflow) != 0:
        accumulate_exact_dw(mGrad, mX, entry_idx, tidx, exact_memory, geometry, True)
        if tidx == 0:
            sts_b32(s_overflow, Uint32(is_dw_bound_difference_positive(s_digits)), Int32(1))
        cute.arch.sync_threads()
        needs_replay = Int32(lds_b32(s_overflow))
    return value, needs_replay


@cute.jit
def replay_pinned_dw(mGrad: cute.Tensor, mX: cute.Tensor, entry_idx: Int32, lane: Int32, geometry: DwGeometry, use_pinned_tma: cutlass.Constexpr[bool]) -> Float32:
    """
    One warp replays the pinned FP32 FMA and reduction order for an entry.
    """
    B, input_length, output_length = geometry.B, geometry.input_length, geometry.output_length
    num_channels, K, offset = geometry.num_channels, geometry.K, geometry.offset
    grad_batch_stride, grad_row_stride, grad_channel_stride = geometry.grad_batch_stride, geometry.grad_row_stride, geometry.grad_channel_stride
    num_splits, num_iterations = geometry.num_splits, geometry.num_iterations
    tokens = BWD_TMA_ROWS // 16 if use_pinned_tma else BWD_TOKENS
    blocks_per_warp = 4 if use_pinned_tma else BLOCKS_PER_WARP
    rows_per_iter = BWD_TMA_ROWS if use_pinned_tma else BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS
    channel, tap = entry_idx // K, entry_idx % K
    acc_spans = Float32(0.0)
    for partial_idx in cutlass.range(B * num_splits, unroll=1):
        batch, split_idx = partial_idx // num_splits, partial_idx % num_splits
        x_base = mX.iterator.toint() + (Int64(batch) * input_length * num_channels + channel) * 2
        g_base = mGrad.iterator.toint() + (Int64(batch) * grad_batch_stride + channel * grad_channel_stride) * 2
        acc_lane = Float32(0.0)
        if lane * tokens < rows_per_iter:
            for iteration_idx in cutlass.range(num_iterations, unroll=1):
                row_start = (split_idx * num_iterations + iteration_idx) * rows_per_iter + lane * tokens
                for token_idx in cutlass.range_constexpr(tokens):
                    row = row_start + token_idx
                    token, x_row = row - offset, row - (K - 1) + tap
                    grad_bits = ld_b16(g_base + Int64(token) * grad_row_stride * 2, Int32(token >= 0) & Int32(token < output_length))
                    input_bits = ld_b16(x_base + Int64(x_row) * num_channels * 2, Int32(x_row >= 0) & Int32(x_row < input_length))
                    acc_lane = fma_f32(bf16_lo(grad_bits), bf16_lo(input_bits), acc_lane)
        acc_lane = acc_lane + cute.arch.shuffle_sync_bfly(acc_lane, offset=1)
        acc_lane = acc_lane + cute.arch.shuffle_sync_bfly(acc_lane, offset=2)
        if cutlass.const_expr(not use_pinned_tma):
            acc_lane = acc_lane + cute.arch.shuffle_sync_bfly(acc_lane, offset=4)
        cta_sum = Float32(0.0)
        for warp in cutlass.range_constexpr(BWD_WARPS):
            cta_sum = cta_sum + cute.arch.shuffle_sync(acc_lane, offset=warp * blocks_per_warp)
        acc_spans = acc_spans + cta_sum
    return acc_spans


@cute.jit
def round_exact_dw_to_fp64(s_digits: Int32, is_negative: Int32, top: Int32) -> Float64:
    magnitude = Float64(0.0)
    if top >= 53:
        fraction, guard, sticky = read_dw_rounding_bits(s_digits, top - 52, 52)
        bits = (Int64(top - 266 + 1023) << 52) | fraction
        magnitude = (bits + (guard & (sticky | (fraction & Int64(1))))).bitcast(Float64)
    elif top >= 0:
        magnitude_bits = Int64(0)
        for i in cutlass.range_constexpr(4):
            magnitude_bits = magnitude_bits | (lds_f64(s_digits + 8 * i).bitcast(Int64) << (16 * i))
        magnitude = magnitude_bits.to(Float64) * Float64(2.0**-266)
    return -magnitude if is_negative != 0 else magnitude


@cute.kernel
def exact_narrow_dw_kernel(mGrad: cute.Tensor, mX: cute.Tensor, mApprox: cute.Tensor, mDw: cute.Tensor,
                           B: Int32, input_length: Int32, output_length: Int32, num_channels: Int32, K: Int32, offset: Int32,
                           grad_batch_stride: Int64, grad_row_stride: Int64, grad_channel_stride: Int64,
                           num_splits: Int32, num_iterations: Int32, lowest: Float32, highest: Float32, epsilon: Float32,
                           use_pinned_tma: cutlass.Constexpr[bool]) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    entry_idx, _, _ = cute.arch.block_idx()
    smem_allocator = cutlass.memory.SmemAllocator()
    s_digits = Int32(smem_allocator.allocate(DW_EXACT_DIGITS * 8, byte_alignment=8).toint())
    s_nonfinite_flag, s_overflow = Int32(smem_allocator.allocate(4, byte_alignment=4).toint()), Int32(smem_allocator.allocate(8, byte_alignment=8).toint())
    exact_memory = DwExactMemory(s_digits, s_nonfinite_flag, s_overflow)
    geometry = DwGeometry(B, input_length, output_length, num_channels, K, offset, grad_batch_stride, grad_row_stride, grad_channel_stride, num_splits, num_iterations)

    accumulate_exact_dw(mGrad, mX, entry_idx, tidx, exact_memory, geometry, False)
    store_exact_dw(mGrad, mX, mApprox, mDw, geometry, exact_memory, entry_idx, tidx, lowest, highest, epsilon, use_pinned_tma)


@cute.jit
def store_exact_dw(mGrad: cute.Tensor, mX: cute.Tensor, mApprox: cute.Tensor, mDw: cute.Tensor, geometry: DwGeometry, exact_memory: DwExactMemory, entry_idx: Int32, tidx: Int32, lowest: Float32, highest: Float32, epsilon: Float32, use_pinned_tma: cutlass.Constexpr[bool]) -> None:
    s_digits, s_nonfinite_flag = exact_memory.s_digits, exact_memory.s_nonfinite_flag
    prior = mApprox[entry_idx]
    needs_replay = Int32(lds_b32(s_nonfinite_flag) != 0) | Int32(prior != prior)
    if cutlass.const_expr(mDw.element_type is Float64):
        if needs_replay == 0:
            if tidx == 0:
                is_negative, top = normalize_dw_digits(s_digits)
                mDw[entry_idx] = round_exact_dw_to_fp64(s_digits, is_negative, top)
        elif tidx < 32:
            value = replay_pinned_dw(mGrad, mX, entry_idx, tidx, geometry, use_pinned_tma)
            if tidx == 0:
                mDw[entry_idx] = value.to(Float64)
    else:
        value = Float32(0.0)
        if needs_replay == 0:
            value, needs_replay = finish_finite_dw(mGrad, mX, entry_idx, tidx, exact_memory, geometry, lowest, highest, epsilon, 23, True)
        if needs_replay != 0 and tidx < 32:
            value = replay_pinned_dw(mGrad, mX, entry_idx, tidx, geometry, use_pinned_tma)
        if tidx == 0:
            mDw[entry_idx] = value


@cute.jit
def exact_narrow_dw(mGrad: cute.Tensor, mX: cute.Tensor, mApprox: cute.Tensor, mDw: cute.Tensor,
                    B: Int32, input_length: Int32, output_length: Int32, num_channels: Int32, K: Int32, offset: Int32,
                    grad_batch_stride: Int64, grad_row_stride: Int64, grad_channel_stride: Int64,
                    num_splits: Int32, num_iterations: Int32, lowest: Float32, highest: Float32, epsilon: Float32, stream,
                    use_pinned_tma: cutlass.Constexpr[bool]) -> None:
    exact_narrow_dw_kernel(mGrad, mX, mApprox, mDw, B, input_length, output_length, num_channels, K, offset, grad_batch_stride, grad_row_stride, grad_channel_stride, num_splits, num_iterations, lowest, highest, epsilon, use_pinned_tma).launch(grid=[num_channels * K, 1, 1], block=[DW_EXACT_THREADS, 1, 1], stream=stream)


@cute.kernel
def repair_wide_dw_kernel(mGrad: cute.Tensor, mX: cute.Tensor, mDw: cute.Tensor, mPartials: cute.Tensor,
                          B: Int32, input_length: Int32, output_length: Int32, num_channels: Int32, K: Int32, offset: Int32,
                          grad_batch_stride: Int64, grad_row_stride: Int64, grad_channel_stride: Int64,
                          num_splits: Int32, num_iterations: Int32, use_pinned_tma: cutlass.Constexpr[bool],
                          num_spans: cutlass.Constexpr[int], certificate_tile: cutlass.Constexpr[int],
                          generic_certificate: cutlass.Constexpr[bool]) -> None:
    """
    Optionally merge and certify FP64 spans before repairing exceptional entries.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    entry_idx = bx * DW_EXACT_THREADS + tidx
    smem_allocator = cutlass.memory.SmemAllocator()
    s_repair_count = Int32(smem_allocator.allocate(4, byte_alignment=4).toint())
    s_repair_entries = Int32(smem_allocator.allocate(DW_EXACT_THREADS * 4, byte_alignment=4).toint())
    s_digits = Int32(smem_allocator.allocate(DW_EXACT_DIGITS * 8, byte_alignment=8).toint())
    s_nonfinite_flag, s_overflow = Int32(smem_allocator.allocate(4, byte_alignment=4).toint()), Int32(smem_allocator.allocate(8, byte_alignment=8).toint())
    exact_memory = DwExactMemory(s_digits, s_nonfinite_flag, s_overflow)
    geometry = DwGeometry(B, input_length, output_length, num_channels, K, offset, grad_batch_stride, grad_row_stride, grad_channel_stride, num_splits, num_iterations)

    is_bf16 = mDw.element_type is cutlass.BFloat16
    limit = 0x7f7f0000 if is_bf16 else 0x7f7fffff
    fraction_bits = 7 if is_bf16 else 23

    total_spans = Int32(num_spans)
    if cutlass.const_expr(generic_certificate and not use_pinned_tma):
        total_spans = B * num_splits
    initialize_dw_certificate(mPartials, s_repair_count, s_overflow, bx, tidx, num_channels, K, num_spans, certificate_tile, generic_certificate)
    select_dw_repairs(mDw, mPartials, geometry, s_repair_count, s_repair_entries, s_overflow, entry_idx, total_spans, limit, num_spans, certificate_tile, generic_certificate)
    repair_dw_entries(mGrad, mX, mDw, geometry, exact_memory, s_repair_count, s_repair_entries, tidx, limit, fraction_bits, use_pinned_tma)


@cute.jit
def initialize_dw_certificate(mPartials: cute.Tensor, s_repair_count: Int32, s_overflow: Int32, bx: Int32, tidx: Int32, num_channels: Int32, K: Int32,
                              num_spans: cutlass.Constexpr[int], certificate_tile: cutlass.Constexpr[int], generic_certificate: cutlass.Constexpr[bool]) -> None:
    if tidx == 0:
        sts_b32(s_repair_count, Uint32(0), Int32(1))
        if cutlass.const_expr(num_spans > 0 and not generic_certificate):
            tile, tiles = (bx * DW_EXACT_THREADS) // (certificate_tile * K), num_channels // certificate_tile
            amax_idx = num_spans * num_channels * K + tile
            acc_tile_amax = Float32(0.0)
            for span_idx in cutlass.range_constexpr(num_spans):
                maximum = mPartials[amax_idx + span_idx * tiles].to(Float32)
                acc_tile_amax = max_abs_operand(acc_tile_amax, maximum)
            magnitude = acc_tile_amax.to(Float64)
            # At the production shape, depth <1024 and at most 8192 products give an error
            # below 2**-30 * max_operand**2. The extra factor covers
            # rounding in the interval endpoints.
            sts_f64(s_overflow, Float64(2.0**-29) * magnitude * magnitude, Int32(1))
    cute.arch.sync_threads()


@cute.jit
def select_dw_repairs(mDw: cute.Tensor, mPartials: cute.Tensor, geometry: DwGeometry, s_repair_count: Int32, s_repair_entries: Int32, s_overflow: Int32, entry_idx: Int32, total_spans: Int32,
                      limit: cutlass.Constexpr[int], num_spans: cutlass.Constexpr[int], certificate_tile: cutlass.Constexpr[int], generic_certificate: cutlass.Constexpr[bool]) -> None:
    B, output_length, num_channels, K = geometry.B, geometry.output_length, geometry.num_channels, geometry.K
    num_iterations = geometry.num_iterations
    is_bf16, is_fp64 = mDw.element_type is cutlass.BFloat16, mDw.element_type is Float64
    if entry_idx < num_channels * K:
        if cutlass.const_expr(generic_certificate):
            acc = Float64(0.0)
            acc_entry_amax = Float32(0.0)
            tiles = (num_channels + certificate_tile - 1) // certificate_tile
            amax_idx = Int64(total_spans) * num_channels * K + entry_idx // K // certificate_tile
            for span_idx in cutlass.range(total_spans, unroll=1):
                acc = acc + mPartials[Int64(span_idx) * num_channels * K + entry_idx]
                acc_entry_amax = max_abs_operand(acc_entry_amax, mPartials[amax_idx + Int64(span_idx) * tiles].to(Float32))
            # The depth covers the per-thread FMAs, warp and CTA sums, and
            # ordered span merge. The factor covers roundoff at both endpoints.
            terms = (Int64(B) * output_length).to(Float64)
            depth = (Int64(num_iterations) * BWD_TOKENS + total_spans + 8).to(Float64)
            magnitude = acc_entry_amax.to(Float64)
            radius = Float64(2.0**-51) * terms * depth * magnitude * magnitude
            if cutlass.const_expr(is_bf16):
                bits = round_dw_bf16(acc)
                st_b16(mDw.iterator.toint() + Int64(entry_idx) * 2, bits, Int32(1))
                needs_repair = (bits & Uint32(0x7fff)) >= Uint32(0x7f7f)
                needs_repair = needs_repair | (round_dw_bf16(acc - radius) != round_dw_bf16(acc + radius))
            else:
                rounded = acc.to(Float32)
                mDw[entry_idx] = rounded
                rounded_bits, lower, upper = f32_as_u32(rounded), f32_as_u32((acc - radius).to(Float32)), f32_as_u32((acc + radius).to(Float32))
                needs_repair = ((rounded_bits & Uint32(0x7fffffff)) >= Uint32(limit)) | (lower != upper)
        elif cutlass.const_expr(num_spans > 0):
            acc = Float64(0.0)
            for span_idx in cutlass.range_constexpr(num_spans):
                acc = acc + mPartials[span_idx * num_channels * K + entry_idx]
            bits = round_dw_bf16(acc)
            st_b16(mDw.iterator.toint() + Int64(entry_idx) * 2, bits, Int32(1))
            radius = lds_f64(s_overflow)
            needs_repair = (bits & Uint32(0x7fff)) >= Uint32(0x7f7f)
            needs_repair = needs_repair | (round_dw_bf16(acc - radius) != round_dw_bf16(acc + radius))
        elif cutlass.const_expr(is_fp64):
            magnitude = mDw[entry_idx].bitcast(Int64) & Int64(0x7fffffffffffffff)
            needs_repair = magnitude >= Int64(0x7ff0000000000000)
        else:
            magnitude = f32_as_u32(mDw[entry_idx].to(Float32)) & Uint32(0x7fffffff)
            needs_repair = magnitude >= Uint32(limit)
        if needs_repair:
            slot_idx = Int32(asm(mlir_types.i32(), [to_ir(s_repair_count, Int32)], "atom.shared.add.u32 $0, [$1], 1;", "=r,r", side_effects=True))
            sts_b32(s_repair_entries + slot_idx * 4, Uint32(entry_idx), Int32(1))
    cute.arch.sync_threads()


@cute.jit
def repair_dw_entries(mGrad: cute.Tensor, mX: cute.Tensor, mDw: cute.Tensor, geometry: DwGeometry, exact_memory: DwExactMemory, s_repair_count: Int32, s_repair_entries: Int32, tidx: Int32,
                      limit: cutlass.Constexpr[int], fraction_bits: cutlass.Constexpr[int], use_pinned_tma: cutlass.Constexpr[bool]) -> None:
    s_nonfinite_flag = exact_memory.s_nonfinite_flag
    is_fp64 = mDw.element_type is Float64
    for slot_idx in cutlass.range(Int32(lds_b32(s_repair_count)), unroll=1):
        repair_entry, needs_replay = Int32(lds_b32(s_repair_entries + slot_idx * 4)), Int32(1)
        value = Float32(0.0)
        if cutlass.const_expr(not is_fp64):
            prior = mDw[repair_entry]
            accumulate_exact_dw(mGrad, mX, repair_entry, tidx, exact_memory, geometry, False)
            needs_replay = Int32(lds_b32(s_nonfinite_flag) != 0) | Int32(prior != prior)
            if needs_replay == 0:
                highest = u32_as_f32(Uint32(limit))
                value, needs_replay = finish_finite_dw(mGrad, mX, repair_entry, tidx, exact_memory, geometry, -highest, highest, Float32(2.0 ** -fraction_bits), fraction_bits, False)
        if needs_replay != 0 and tidx < 32:
            value = replay_pinned_dw(mGrad, mX, repair_entry, tidx, geometry, use_pinned_tma)
        if tidx == 0:
            mDw[repair_entry] = value.to(mDw.element_type)
        cute.arch.sync_threads()


@cute.jit
def repair_wide_dw(mGrad: cute.Tensor, mX: cute.Tensor, mDw: cute.Tensor, mPartials: cute.Tensor,
                   B: Int32, input_length: Int32, output_length: Int32, num_channels: Int32, K: Int32, offset: Int32,
                   grad_batch_stride: Int64, grad_row_stride: Int64, grad_channel_stride: Int64,
                   num_splits: Int32, num_iterations: Int32, stream, use_pinned_tma: cutlass.Constexpr[bool],
                   num_spans: cutlass.Constexpr[int], certificate_tile: cutlass.Constexpr[int],
                   generic_certificate: cutlass.Constexpr[bool]) -> None:
    repair_wide_dw_kernel(mGrad, mX, mDw, mPartials, B, input_length, output_length, num_channels, K, offset,
                          grad_batch_stride, grad_row_stride, grad_channel_stride, num_splits, num_iterations,
                          use_pinned_tma, num_spans, certificate_tile, generic_certificate).launch(
        grid=[(num_channels * K + DW_EXACT_THREADS - 1) // DW_EXACT_THREADS, 1, 1],
        block=[DW_EXACT_THREADS, 1, 1], stream=stream,
    )


def finish_exact_dw(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, approx: torch.Tensor, offset: int, use_pinned_tma: bool, num_splits: int, num_iterations: int) -> torch.Tensor:
    """
    Round FP64 outputs directly and use an FP32 carrier for narrower outputs.
    """
    B, input_length, num_channels = x.shape
    K = w.shape[-1]
    is_fp64 = w.dtype == torch.float64
    carrier_type = Float64 if is_fp64 else Float32
    carrier = torch.empty(w.shape, dtype=torch.float64 if is_fp64 else torch.float32, device=w.device)
    key = ("conv_bwd_exact_dw", use_pinned_tma, carrier_type)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = cute.compile(
            exact_narrow_dw, make_fake_conv_vector(cutlass.BFloat16), make_fake_conv_vector(cutlass.BFloat16), make_fake_conv_vector(Float64), make_fake_conv_vector(carrier_type),
            *(Int32(1) for _ in range(6)), Int64(1), Int64(1), Int64(1),
            Int32(1), Int32(1), Float32(1), Float32(1), Float32(1), fake_stream(), use_pinned_tma,
            options="--enable-tvm-ffi",
        )
    info = torch.finfo(w.dtype)
    if is_fp64:
        limits = (0.0, 0.0, 0.0)
    else:
        lowest = info.tiny if w.dtype == getattr(torch, "float8_e8m0fnu", None) else info.min
        limits = (lowest, info.max, info.eps)
    fn(g.as_strided((1,), (1,)), x.view(-1), approx.view(-1), carrier.view(-1), B, input_length, input_length - offset, num_channels, K, offset, *g.stride(), num_splits, num_iterations, *limits)
    return carrier


def finalize_weight_gradient(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, dw: torch.Tensor, offset: int,
             exact_output: bool, use_pinned_tma: bool, num_splits: int, num_iterations: int,
             partials: torch.Tensor | None = None, num_spans: int = 0,
             certificate_tile: int = TMA_CHANNELS, generic_certificate: bool = False) -> torch.Tensor:
    if exact_output or w.dtype == torch.float64:
        dw = finish_exact_dw(g, x, w, dw, offset, use_pinned_tma, num_splits, num_iterations)
    elif w.dtype in DW_TYPES:
        B, input_length, num_channels = x.shape
        K = w.shape[-1]
        key = ("conv_bwd_repair_wide_dw", w.dtype, use_pinned_tma, num_spans, certificate_tile, generic_certificate)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = compile_dw_repair(*key[1:])
        partial_data = dw if partials is None else partials
        fn(g.as_strided((1,), (1,)), x.view(-1), dw.view(-1), partial_data.view(-1), B, input_length, input_length - offset, num_channels, K, offset, *g.stride(), num_splits, num_iterations)
    return dw.to(w.dtype).view(w.shape)


def compile_dw_repair(dtype: torch.dtype, use_pinned_tma: bool, num_spans: int, certificate_tile: int, generic_certificate: bool) -> Callable:
    dw_type = DW_TYPES[dtype]
    partial_type = Float64 if num_spans or generic_certificate else dw_type
    tensors = (make_fake_conv_vector(cutlass.BFloat16), make_fake_conv_vector(cutlass.BFloat16), make_fake_conv_vector(dw_type), make_fake_conv_vector(partial_type))
    return cute.compile(repair_wide_dw, *tensors, *(Int32(1) for _ in range(6)), Int64(1), Int64(1), Int64(1), Int32(1), Int32(1), fake_stream(),
                        use_pinned_tma, num_spans, certificate_tile, generic_certificate, options="--enable-tvm-ffi")


def compile_direct_backward(num_channels: int, K: int, is_weight_fp32: bool, use_vector_input: bool, is_grad_channels_first: bool, use_vector_grad: bool, grad_vector_shift: int, dw_type: type, use_certificate: bool) -> Callable:
    w = make_fake_conv_vector(Float32 if is_weight_fp32 else cutlass.BFloat16)
    partials = fake(Float64, (cute.sym_int(),), (1,), 8)
    tensors = (make_fake_conv_vector(cutlass.BFloat16), make_fake_conv_vector(cutlass.BFloat16), w, make_fake_conv_vector(cutlass.BFloat16), make_fake_conv_vector(dw_type), partials, fake(Int64, (cute.sym_int(),), (1,), 8))
    scalars = (Int32(1), Int32(1), Int32(1), Int32(0), Int64(0), Int64(0), Int32(1), Int32(1))
    return cute.compile(DirectConvBackward(num_channels, K, is_weight_fp32, use_vector_input, is_grad_channels_first, use_vector_grad, grad_vector_shift, dw_type, use_certificate), *tensors, *scalars, fake_stream(), options="--enable-tvm-ffi")


def compile_tma_backward(num_channels: int, K: int, is_weight_fp32: bool, num_sequence_splits: int, dw_type: type, batched: bool, certify_single_sequence: bool) -> Callable:
    w = make_fake_conv_vector(Float32 if is_weight_fp32 else cutlass.BFloat16)
    dw = make_fake_conv_vector(Float64 if num_sequence_splits > 1 or certify_single_sequence else dw_type)
    conv = TmaConvBackward(num_channels, K, is_weight_fp32, num_sequence_splits, dw_type, batched, certify_single_sequence)
    launcher = conv.batch_two if batched else conv
    inputs = [make_fake_tma_rows(cutlass.BFloat16, num_channels) for _ in range(4 if batched else 2)]
    dx = make_fake_tma_rows(cutlass.BFloat16, num_channels) if batched else make_fake_conv_vector(cutlass.BFloat16)
    return cute.compile(launcher, *inputs, w, dx, dw, Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def is_tma_bwd(g: torch.Tensor, x: torch.Tensor, K: int) -> bool:
    """
    Backward TMA loads the BF16 output gradient as contiguous channel tiles.
    """
    num_channels = x.shape[2]
    return is_tma_input(x, K) and g.dtype == torch.bfloat16 and g.stride(2) == 1 and g.stride(1) == num_channels and g.data_ptr() % 16 == 0


def get_tile_counters(device: torch.device, num_channel_tiles: int) -> torch.Tensor:
    """
    Zeroed per-channel-tile counters of the current stream; each launch leaves them zero again.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    counters = COUNTERS.get(key)
    if counters is None or counters.numel() < num_channel_tiles:
        counters = COUNTERS[key] = torch.zeros(max(num_channel_tiles, 1024), dtype=torch.int64, device=device)
    return counters


def get_direct_backward_splits(B: int, input_length: int, num_channel_tiles: int) -> tuple[int, int]:
    chunk_rows = BWD_WARPS * BLOCKS_PER_WARP * BWD_TOKENS
    num_chunks = (input_length + chunk_rows - 1) // chunk_rows
    num_splits = min(num_chunks, max(1, round(SM_COUNT * TARGET_CTAS / (B * num_channel_tiles))))
    num_iterations = (num_chunks + num_splits - 1) // num_splits
    return (num_chunks + num_iterations - 1) // num_iterations, num_iterations


def causal_conv1d_bwd_sm90(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute BF16 input gradients and reduce weight gradients in a fixed order.
    """
    B, input_length, num_channels = x.shape
    K, output_length = w.shape[-1], input_length - offset
    if K > MAX_TAPS:
        raise ValueError(f"causal_conv1d on SM90 supports kernel widths up to {MAX_TAPS}, got {K}")
    if B * output_length * num_channels == 0:
        return torch.zeros_like(x), torch.zeros_like(w)
    w_kernel = prepare_weights(w)
    exact_output = w.dtype == torch.float16 or (w.dtype not in DW_TYPES and w.is_floating_point())
    storage_dtype = torch.float64 if exact_output else w.dtype if w.dtype in DW_TYPES else torch.float32
    dw_type = DW_TYPES[storage_dtype]
    is_target_shape = (input_length, num_channels, K, offset) == (4096, 10240, 4, 0)
    use_certificate = w.dtype in (torch.bfloat16, torch.float32)
    use_generic_certificate = not (is_target_shape and B in (1, 2) and w.dtype == torch.bfloat16)
    num_channel_tiles = (num_channels + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
    use_pinned_tma = is_tma_bwd(g, x, K)
    if use_pinned_tma:
        num_splits, num_iterations = 1, (input_length + BWD_TMA_ROWS - 1) // BWD_TMA_ROWS
    else:
        num_splits, num_iterations = get_direct_backward_splits(B, input_length, num_channel_tiles)
    has_batched_tma_layout = B == 2 and is_target_shape and x.is_contiguous() and g.is_contiguous() and is_tma_bwd(g[:1], x[:1], K)
    use_batched_tma = has_batched_tma_layout and w.dtype == torch.bfloat16
    use_tma = use_batched_tma or use_pinned_tma or (has_batched_tma_layout and not use_certificate)
    if not use_tma and g.stride(2) != 1 and g.stride(1) != 1:
        g = g.contiguous()

    dx, dw = torch.empty_like(x), torch.empty_like(w_kernel, dtype=storage_dtype)
    if use_tma:
        num_sequence_splits = 8 if use_batched_tma else (4 if is_target_shape else 1)
        certify_single_sequence = use_certificate and use_pinned_tma
        partials, num_spans = launch_tma_backward(g, x, w_kernel, dx, dw, offset, dw_type, num_sequence_splits, use_batched_tma, certify_single_sequence)
        return dx, finalize_weight_gradient(g, x, w, dw, offset, exact_output, use_pinned_tma, num_splits, num_iterations, partials, num_spans,
                                            generic_certificate=certify_single_sequence and use_generic_certificate)

    partials = launch_direct_backward(g, x, w_kernel, dx, dw, offset, dw_type, num_channel_tiles, num_splits, num_iterations, use_certificate)
    if use_certificate:
        num_spans = 0 if use_generic_certificate else B * num_splits
        return dx, finalize_weight_gradient(g, x, w, dw, offset, exact_output, use_pinned_tma, num_splits, num_iterations, partials, num_spans, CHANNELS_PER_CTA, use_generic_certificate)
    return dx, finalize_weight_gradient(g, x, w, dw, offset, exact_output, use_pinned_tma, num_splits, num_iterations)


def launch_tma_backward(g, x, w_kernel, dx, dw, offset: int, dw_type: type, num_sequence_splits: int, use_batched_tma: bool, certify_single_sequence: bool):
    B, input_length, num_channels = x.shape
    K = w_kernel.shape[-1]
    num_spans = B * num_sequence_splits
    if use_batched_tma or certify_single_sequence:
        partials = torch.empty(num_spans * (num_channels * K + num_channels // TMA_CHANNELS), dtype=torch.float64, device=x.device)
    elif num_sequence_splits > 1:
        partials = torch.empty((B, num_sequence_splits, num_channels * K), dtype=torch.float64, device=x.device)
    else:
        partials = None

    key = ("conv_bwd_tma", num_channels, K, w_kernel.dtype == torch.float32, num_sequence_splits, dw_type, use_batched_tma, certify_single_sequence)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_tma_backward(*key[1:])
    if use_batched_tma:
        fn(g[0], x[0], g[1], x[1], w_kernel.view(-1), dx.view(-1, num_channels), partials, input_length, offset)
        return partials, num_spans
    if certify_single_sequence:
        fn(g[0], x[0], w_kernel.view(-1), dx.view(-1), partials, input_length, offset)
        return partials, num_sequence_splits
    if num_sequence_splits > 1:
        for sequence_idx in range(B):
            fn(g[sequence_idx], x[sequence_idx], w_kernel.view(-1), dx[sequence_idx].view(-1), partials[sequence_idx].view(-1), input_length, offset)
        merge_key = ("conv_bwd_merge_spans", num_spans, dw_type)
        merge = COMPILED.get(merge_key)
        if merge is None:
            merge = COMPILED[merge_key] = cute.compile(merge_dw_spans, make_fake_conv_vector(Float64), make_fake_conv_vector(dw_type), Int32(1), fake_stream(), num_spans, options="--enable-tvm-ffi")
        merge(partials.view(-1), dw.view(-1), num_channels * K)
    else:
        fn(g[0], x[0], w_kernel.view(-1), dx.view(-1), dw.view(-1), input_length, offset)
    return None, 0


def launch_direct_backward(g, x, w_kernel, dx, dw, offset: int, dw_type: type, num_channel_tiles: int, num_splits: int, num_iterations: int, use_certificate: bool):
    B, input_length, num_channels = x.shape
    K, output_length = w_kernel.shape[-1], input_length - offset
    is_grad_channels_first = g.stride(2) != 1
    grad_batch_stride, grad_major_stride = g.stride(0), g.stride(2) if is_grad_channels_first else g.stride(1)
    is_grad_aligned = g.data_ptr() % 16 == 0 and grad_batch_stride % 8 == 0 and grad_major_stride % 8 == 0
    use_vector_grad = is_grad_aligned and (output_length % 8 == 0 if is_grad_channels_first else num_channels % 8 == 0)
    grad_vector_shift = -offset % 8 if is_grad_channels_first and use_vector_grad else 0
    use_vector_input = num_channels % 8 == 0 and x.data_ptr() % 16 == 0
    entries_per_span = num_channels * K + (num_channel_tiles if use_certificate else 0)

    partials = torch.empty(B * num_splits * entries_per_span, dtype=torch.float64, device=x.device)
    counters = get_tile_counters(x.device, num_channel_tiles)

    key = ("conv_bwd", num_channels, K, w_kernel.dtype == torch.float32, use_vector_input, is_grad_channels_first, use_vector_grad, grad_vector_shift, dw_type, use_certificate)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_direct_backward(*key[1:])
    fn(g.as_strided((1,), (1,)), x.view(-1), w_kernel.view(-1), dx.view(-1), dw.view(-1), partials, counters,
       B, input_length, output_length, offset, grad_batch_stride, grad_major_stride, num_splits, num_iterations)
    return partials
