"""
SM90 depthwise causal convolution and helpers shared by its backward implementation.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, bf16_hi, bf16_lo, expect_slot_bytes, fma_f32, ld_b16, ld_relaxed, ld_v4, lds_v4, pack_bf16, release_slot, st_b16, st_v4, tma_load_rows, to_ir, u32_as_f32, wait_tma_slot

__all__ = ["BLOCKS_PER_WARP", "CHANNELS_PER_CTA", "GROUPS_PER_WARP", "MAX_TAPS", "SMEM_LIMIT", "TMA_CHANNELS", "TMA_CONSUMERS", "ThreadTile", "TmaRing", "TmaRingMemory", "causal_conv1d_fwd_sm90", "get_tma_descriptor_addr", "initialize_tma_ring", "is_tma_input", "load_channel_segment", "load_input_window", "load_weights", "make_fake_conv_vector", "make_fake_tma_rows", "prepare_weights", "unpack_bf16_element"]

GROUPS_PER_WARP, BLOCKS_PER_WARP = 4, 8  # channel groups of 8 and token blocks of TOKENS per warp
FWD_TOKENS, FWD_WARPS = 8, 4
CHANNELS_PER_CTA = GROUPS_PER_WARP * 8
TMA_CHANNELS = 64  # channels per CTA of the TMA kernels: one 128-byte box row
TMA_CONSUMERS = 128  # consumer threads per TMA group
TMA_MAX_TAPS = 4
MAX_TAPS = 8  # K of any input: the direct kernels keep K taps per channel in registers and K partials per channel in smem
FWD_TMA_ROWS, FWD_TMA_STAGES, FWD_TMA_GROUPS = 128, 8, 2
SMEM_LIMIT = 227 * 1024


@dsl_user_op
def get_tma_descriptor_addr(atom, *, loc=None, ip=None) -> Int64:
    """
    Generic address of a TMA atom's descriptor (a kernel parameter), for tma_load_rows.
    """
    exec_atom = cute_nvgpu.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    ptr_type = ir.Type.parse("!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>")
    return cute_nvgpu.get_tma_desc_addr(ptr_type, exec_atom, loc=loc, ip=ip).toint(loc=loc, ip=ip)


@dsl_user_op
def initialize_tma_ring(s_full_barriers: Int32, s_empty_barriers: Int32, num_slots: int, *, loc=None, ip=None) -> None:
    """
    The full barriers count the producer and TMA bytes. The empty barriers count consumers.
    Publish both sets to the async proxy before the first transfer.
    """
    for slot_idx in range(num_slots):
        asm(None, [to_ir(s_full_barriers + 8 * slot_idx, Int32, loc, ip)], "mbarrier.init.shared::cta.b64 [$0], 1;", "r", side_effects=True)
        asm(None, [to_ir(s_empty_barriers + 8 * slot_idx, Int32, loc, ip)], f"mbarrier.init.shared::cta.b64 [$0], {TMA_CONSUMERS};", "r", side_effects=True)
    asm(None, [], "fence.mbarrier_init.release.cluster;", "", side_effects=True)


def unpack_bf16_element(words: list | tuple, element_idx: int) -> Float32:
    """
    Return one of the two BF16 values packed into each word as FP32.
    """
    return bf16_hi(words[element_idx // 2]) if element_idx % 2 else bf16_lo(words[element_idx // 2])


def load_channel_segment(addr: Int64, row_ok: Int32, channel_start: Int32, num_channels: cutlass.Constexpr, use_vector: cutlass.Constexpr) -> list:
    """
    Load an eight-channel segment as packed BF16 words and zero out-of-range elements.
    """
    words = []
    if cutlass.const_expr(use_vector):
        words = list(ld_v4(addr, row_ok & Int32(channel_start < num_channels)))
    else:
        for word_idx in range(4):
            word_lo = ld_b16(addr + 4 * word_idx, row_ok & Int32(channel_start + 2 * word_idx < num_channels))
            word_hi = ld_b16(addr + 4 * word_idx + 2, row_ok & Int32(channel_start + 2 * word_idx + 1 < num_channels))
            words.append(word_lo | (word_hi << 16))
    return words


def load_input_window(x_base: Int64, row_start: Int32, input_length: Int32, channel_start: Int32, num_channels: int, num_window_rows: int, use_vector: bool) -> list:
    """
    Load the thread's causal input window with out-of-range rows zeroed.
    """
    rows = []
    for i in range(num_window_rows):
        row_idx = row_start + i
        row_ok = Int32(row_idx >= 0) & Int32(row_idx < input_length)
        rows.append(load_channel_segment(x_base + (Int64(row_idx) * num_channels + channel_start) * 2, row_ok, channel_start, num_channels, use_vector))
    return rows


def load_weights(w_base: Int64, channel_start: Int32, num_channels: cutlass.Constexpr, K: cutlass.Constexpr, is_weight_fp32: cutlass.Constexpr, num_thread_channels: int = 8) -> list:
    """
    Load weights in channel and tap order, zeroing channels outside the tensor.
    """
    weights = []
    for channel_idx in range(num_thread_channels):
        channel_ok = Int32(channel_start + channel_idx < num_channels)
        taps = []
        for k in range(K):
            element_addr = w_base + (Int64(channel_start + channel_idx) * K + k) * (4 if is_weight_fp32 else 2)
            if cutlass.const_expr(is_weight_fp32):
                taps.append(u32_as_f32(Uint32(ld_relaxed(element_addr, channel_ok))))
            else:
                taps.append(bf16_lo(ld_b16(element_addr, channel_ok)))
        weights.append(taps)
    return weights


class ThreadTile:
    """
    Map one thread to its channel group and token block within the CTA.
    """

    def __init__(self, tidx, by, tokens: int) -> None:
        lane, warp = tidx % 32, tidx // 32
        self.channel_start = by * CHANNELS_PER_CTA + (lane % GROUPS_PER_WARP) * 8
        block_idx = warp * BLOCKS_PER_WARP + lane // GROUPS_PER_WARP
        self.block_row_offset = block_idx * tokens


class DirectConvForward:
    """
    Each CTA writes a token chunk and a 32-channel tile of one sequence.
    """

    def __init__(self, num_channels: int, K: int, is_weight_fp32: bool, use_vector_input: bool, use_vector_output: bool) -> None:
        self.num_channels, self.K, self.is_weight_fp32 = num_channels, K, is_weight_fp32
        self.use_vector_input, self.use_vector_output = use_vector_input, use_vector_output
        self.chunk_tokens = FWD_WARPS * BLOCKS_PER_WARP * FWD_TOKENS

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, B: Int32, input_length: Int32, output_length: Int32, offset: Int32, stream) -> None:
        num_chunks = (output_length + self.chunk_tokens - 1) // self.chunk_tokens
        num_channel_tiles = (self.num_channels + CHANNELS_PER_CTA - 1) // CHANNELS_PER_CTA
        self.kernel(mX, mW, mY, input_length, output_length, offset).launch(grid=[num_chunks, num_channel_tiles, B], block=[FWD_WARPS * 32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, input_length: Int32, output_length: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        tile = ThreadTile(tidx, by, FWD_TOKENS)
        channel_start, block_row_offset = tile.channel_start, tile.block_row_offset
        output_start, x_base = bx * self.chunk_tokens + block_row_offset, mX.iterator.toint() + Int64(bz) * input_length * self.num_channels * 2
        y_base = mY.iterator.toint() + Int64(bz) * self.num_channels * output_length * 2

        weights = load_weights(mW.iterator.toint(), channel_start, self.num_channels, self.K, self.is_weight_fp32)
        rows = load_input_window(x_base, output_start + offset - (self.K - 1), input_length, channel_start, self.num_channels, FWD_TOKENS + self.K - 1, self.use_vector_input)
        out_words = self.convolve(rows, weights)
        self.store_channels_first(y_base, out_words, output_start, output_length, channel_start)

    def convolve(self, rows: list, weights: list) -> list:
        """
        Pack the rounded outputs by channel for channels-first stores.
        """
        out_words = []
        for channel_idx in range(8):
            outputs = []
            for j in range(FWD_TOKENS):
                acc_output = Float32(0.0)
                for k in range(self.K):
                    acc_output = fma_f32(weights[channel_idx][k], unpack_bf16_element(rows[j + k], channel_idx), acc_output)
                outputs.append(acc_output)
            out_words.append([pack_bf16(outputs[2 * i], outputs[2 * i + 1]) for i in range(FWD_TOKENS // 2)])
        return out_words

    def store_channels_first(self, y_base: Int64, out_words: list, output_start: Int32, output_length: Int32, channel_start: Int32) -> None:
        for channel_idx in range(8):
            channel_ok = Int32(channel_start + channel_idx < self.num_channels)
            channel_addr = y_base + (Int64(channel_start + channel_idx) * output_length + output_start) * 2
            if cutlass.const_expr(self.use_vector_output):
                for i in range(FWD_TOKENS // 8):
                    st_v4(channel_addr + 16 * i, tuple(out_words[channel_idx][4 * i : 4 * i + 4]), channel_ok & Int32(output_start + 8 * i < output_length))
            else:
                for j in range(FWD_TOKENS):
                    word = out_words[channel_idx][j // 2]
                    st_b16(channel_addr + 2 * j, (word >> 16) if j % 2 else word, channel_ok & Int32(output_start + j < output_length))


class TmaRingMemory:
    def __init__(self, s_slots: Int32, s_full_barriers: Int32, s_empty_barriers: Int32) -> None:
        self.s_slots, self.s_full_barriers, self.s_empty_barriers = s_slots, s_full_barriers, s_empty_barriers


class TmaRing:
    """
    Assign each slot to one consumer group because parity cannot distinguish phases two apart.
    The producer waits on the opposite parity so its first pass can proceed.
    """

    def __init__(self, num_slots: int, slot_bytes: int, num_groups: int) -> None:
        assert num_slots % num_groups == 0
        self.num_slots, self.slot_bytes = num_slots, slot_bytes

    def allocate(self, smem_allocator) -> TmaRingMemory:
        s_slots = Int32(smem_allocator.allocate(self.num_slots * self.slot_bytes, byte_alignment=128).toint())
        s_barriers = Int32(smem_allocator.allocate(2 * self.num_slots * 8, byte_alignment=8).toint())
        return TmaRingMemory(s_slots, s_barriers, s_barriers + 8 * self.num_slots)

    def get_slot_coordinates(self, stage_idx: Int32) -> tuple:
        return stage_idx % self.num_slots, (stage_idx // self.num_slots) % 2


class TmaConvForward:
    """
    Two consumer groups alternate stages so each warp can write contiguous channel segments.
    """

    def __init__(self, num_channels: int, K: int, is_weight_fp32: bool, sequence_splits: int) -> None:
        self.num_channels, self.K, self.is_weight_fp32 = num_channels, K, is_weight_fp32
        self.sequence_splits = sequence_splits
        self.stage_rows, self.tokens = FWD_TMA_ROWS, FWD_TMA_ROWS // 16
        self.box_rows = self.stage_rows + K - 1
        self.box_bytes = self.box_rows * TMA_CHANNELS * 2
        self.ring = TmaRing(FWD_TMA_STAGES, (self.box_bytes + 127) // 128 * 128, FWD_TMA_GROUPS)
        self.producer_warp = 0
        assert self.tokens == 8 and self.ring.num_slots * self.ring.slot_bytes + 1024 <= SMEM_LIMIT

    @cute.jit
    def __call__(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, output_length: Int32, offset: Int32, stream) -> None:
        box_layout = cute.make_layout((self.box_rows, TMA_CHANNELS), stride=(TMA_CHANNELS, 1))
        atom_x, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mX, box_layout, (self.box_rows, TMA_CHANNELS))
        self.kernel(atom_x, mW, mY, output_length, offset).launch(grid=[self.num_channels // TMA_CHANNELS, self.sequence_splits, 1], block=[32 + TMA_CONSUMERS * FWD_TMA_GROUPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, atom_x: cute.CopyAtom, mW: cute.Tensor, mY: cute.Tensor, output_length: Int32, offset: Int32) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        smem_allocator = cutlass.memory.SmemAllocator()
        ring_memory = self.ring.allocate(smem_allocator)
        if tidx == 0:
            initialize_tma_ring(ring_memory.s_full_barriers, ring_memory.s_empty_barriers, self.ring.num_slots)
        cute.arch.sync_threads()
        rows_per_stage_across_spans = self.stage_rows * self.sequence_splits
        num_stages = (output_length + rows_per_stage_across_spans - 1) // rows_per_stage_across_spans
        span_row_start = Int32(0)
        if cutlass.const_expr(self.sequence_splits > 1):
            span_row_start = by * num_stages * self.stage_rows
        tile_channel_start = bx * TMA_CHANNELS
        if warp == self.producer_warp:
            if tidx == 0:
                self.load_loop(atom_x, ring_memory, num_stages, tile_channel_start, offset + span_row_start)
        else:
            self.compute_loop(mW, mY, ring_memory, num_stages, tile_channel_start, output_length, span_row_start, tidx - 32)

    @cute.jit
    def load_loop(self, atom_x: cute.CopyAtom, ring_memory: TmaRingMemory, num_stages: Int32, tile_channel_start: Int32, offset: Int32) -> None:
        s_slots, s_full_barriers, s_empty_barriers = ring_memory.s_slots, ring_memory.s_full_barriers, ring_memory.s_empty_barriers
        cpasync.prefetch_descriptor(atom_x)
        input_descriptor_addr = get_tma_descriptor_addr(atom_x)
        for stage_idx in cutlass.range(num_stages, unroll=1):
            slot_idx, parity = self.ring.get_slot_coordinates(stage_idx)
            wait_tma_slot(s_empty_barriers + 8 * slot_idx, parity ^ 1)
            s_full_barrier = s_full_barriers + 8 * slot_idx
            expect_slot_bytes(s_full_barrier, self.box_bytes)
            tma_load_rows(s_slots + slot_idx * self.ring.slot_bytes, input_descriptor_addr, tile_channel_start, stage_idx * self.stage_rows + offset - (self.K - 1), s_full_barrier)

    @cute.jit
    def compute_loop(self, mW: cute.Tensor, mY: cute.Tensor, ring_memory: TmaRingMemory, num_stages: Int32, tile_channel_start: Int32, output_length: Int32, span_row_start: Int32, consumer_idx: Int32) -> None:
        consumer_group_idx, group_thread_idx = consumer_idx // TMA_CONSUMERS, consumer_idx % TMA_CONSUMERS
        lane, group_warp = group_thread_idx % 32, group_thread_idx // 32
        group_idx, block_idx = (group_warp % 2) * 4 + lane % 4, (group_warp // 2) * 8 + lane // 4
        channel_start = tile_channel_start + group_idx * 8
        weights = load_weights(mW.iterator.toint(), channel_start, self.num_channels, self.K, self.is_weight_fp32)
        local_row_start = block_idx * self.tokens
        for stage_idx in cutlass.range(consumer_group_idx, num_stages, FWD_TMA_GROUPS, unroll=1):
            self.compute_stage(mY, ring_memory, stage_idx, local_row_start, group_idx, weights, span_row_start, output_length, channel_start)

    @cute.jit
    def compute_stage(self, mY: cute.Tensor, ring_memory: TmaRingMemory, stage_idx: Int32, local_row_start: Int32, group_idx: Int32, weights: list, span_row_start: Int32, output_length: Int32, channel_start: Int32) -> None:
        s_slots, s_full_barriers, s_empty_barriers = ring_memory.s_slots, ring_memory.s_full_barriers, ring_memory.s_empty_barriers
        slot_idx, parity = self.ring.get_slot_coordinates(stage_idx)
        wait_tma_slot(s_full_barriers + 8 * slot_idx, parity)
        s_slot = s_slots + slot_idx * self.ring.slot_bytes
        rows = []
        for i in cutlass.range_constexpr(self.tokens + self.K - 1):
            rows.append(lds_v4(s_slot + ((local_row_start + i) * TMA_CHANNELS + group_idx * 8) * 2))
        release_slot(s_empty_barriers + 8 * slot_idx)
        self.store_y(mY.iterator.toint(), rows, weights, span_row_start + stage_idx * self.stage_rows + local_row_start, output_length, channel_start)

    def store_y(self, y_base: Int64, rows: list, weights: list, output_start: Int32, output_length: Int32, channel_start: Int32) -> None:
        """
        Convolve in tap order and write eight consecutive outputs per vector store.
        """
        for channel_idx in range(8):
            outputs = []
            for j in range(self.tokens):
                acc_output = Float32(0.0)
                for k in range(self.K):
                    acc_output = fma_f32(weights[channel_idx][k], unpack_bf16_element(rows[j + k], channel_idx), acc_output)
                outputs.append(acc_output)
            words = [pack_bf16(outputs[2 * i], outputs[2 * i + 1]) for i in range(self.tokens // 2)]
            st_v4(y_base + (Int64(channel_start + channel_idx) * output_length + output_start) * 2, tuple(words), Int32(output_start < output_length))


COMPILED: dict = {}


def make_fake_conv_vector(dtype: type) -> object:
    return fake(dtype, (cute.sym_int(),), (1,), dtype.width // 8)


def compile_direct_forward(num_channels: int, K: int, is_weight_fp32: bool, use_vector_input: bool, use_vector_output: bool) -> Callable:
    w = make_fake_conv_vector(Float32 if is_weight_fp32 else cutlass.BFloat16)
    return cute.compile(DirectConvForward(num_channels, K, is_weight_fp32, use_vector_input, use_vector_output), make_fake_conv_vector(cutlass.BFloat16), w, make_fake_conv_vector(cutlass.BFloat16), Int32(1), Int32(1), Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def make_fake_tma_rows(dtype: type, num_channels: int) -> object:
    """
    A (rows, C) row-major tensor with a dynamic row count, as the TMA descriptors read it.
    """
    return fake(dtype, (cute.sym_int(), num_channels), (num_channels, 1))


def compile_tma_forward(num_channels: int, K: int, is_weight_fp32: bool, sequence_splits: int) -> Callable:
    w = make_fake_conv_vector(Float32 if is_weight_fp32 else cutlass.BFloat16)
    return cute.compile(TmaConvForward(num_channels, K, is_weight_fp32, sequence_splits), make_fake_tma_rows(cutlass.BFloat16, num_channels), w, make_fake_conv_vector(cutlass.BFloat16), Int32(1), Int32(0), fake_stream(), options="--enable-tvm-ffi")


def is_tma_input(x: torch.Tensor, K: int) -> bool:
    """
    TMA descriptors require 16-byte alignment and a whole 64-channel tile.
    """
    B, _, num_channels = x.shape
    return B == 1 and num_channels % TMA_CHANNELS == 0 and 2 <= K <= TMA_MAX_TAPS and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0


def is_tma_fwd(x: torch.Tensor, K: int, output_length: int) -> bool:
    """
    Each channels-first vector store requires eight output tokens.
    """
    return is_tma_input(x, K) and output_length % 8 == 0


def prepare_weights(w: torch.Tensor) -> torch.Tensor:
    """
    Use contiguous BF16 or FP32 weights for the ordered FP32 convolution.
    """
    return w.contiguous() if w.dtype in (torch.bfloat16, torch.float32) else w.float()


def causal_conv1d_fwd_sm90(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Run depthwise causal convolution with ordered FP32 FMAs and one BF16 rounding.
    """
    B, input_length, num_channels = x.shape
    K, output_length = w.shape[-1], input_length - offset
    if K > MAX_TAPS:
        raise ValueError(f"causal_conv1d on SM90 supports kernel widths up to {MAX_TAPS}, got {K}")
    y = x.new_empty((B, num_channels, output_length)).transpose(1, 2)
    if y.numel() == 0:
        return y
    w = prepare_weights(w)
    if is_tma_fwd(x, K, output_length):
        sequence_splits = 4 if output_length == 4096 and offset == 0 and w.shape == (10240, 1, 4) and w.dtype == torch.bfloat16 else 1
        key = ("conv_fwd_tma", num_channels, K, w.dtype == torch.float32, sequence_splits)
        fn = COMPILED.get(key)
        if fn is None:
            fn = COMPILED[key] = compile_tma_forward(*key[1:])
        fn(x[0], w.view(-1), y.transpose(1, 2).view(-1), output_length, offset)
        return y
    use_vector_input, use_vector_output = num_channels % 8 == 0 and x.data_ptr() % 16 == 0, output_length % 8 == 0
    key = ("conv_fwd", num_channels, K, w.dtype == torch.float32, use_vector_input, use_vector_output)
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_direct_forward(*key[1:])
    fn(x.view(-1), w.view(-1), y.transpose(1, 2).view(-1), B, input_length, output_length, offset)
    return y
