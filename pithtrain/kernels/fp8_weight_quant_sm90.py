"""
SM90 blockwise FP8 weight quantization, transposition and parameter all-gather kernels.
"""

import fcntl
import hashlib
import os
import socket
import subprocess
import sys
import sysconfig
from typing import Optional, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import tvm_ffi
import tvm_ffi.libinfo
from cutlass import Int32, Int64, Uint8, Uint32, const_expr
from torch.utils import cpp_extension
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm_args_last, ld_v2, lds_b32, lds_v4, max_u16x2, max_u32, prmt_args_last, st_b32, st_b8, st_v2, st_v4, sts_b32_raw, sts_v2, to_ir, unpack
from pithtrain.kernels.fp8_e8m0_scale_sm90 import AMAX_MIN, FP8_MAX_RCP, compute_scale_rcp_bits, quantize_four_from_bits

__all__ = ["fp8_quant_weight_sm90", "fp8_quant_weight_grouped_sm90", "fp8_shard_amax_sm90", "fp8_shard_cast_sm90", "fp8_scales_sm90", "fp8_pack_transpose_sm90", "fp8_pack_gathered_transpose_sm90", "fp8_shard_amax_multi_sm90", "fp8_scales_multi_sm90"]

BLOCK = 128
# Below this many tiles the extra barrier of the shared-memory store path costs more than its
# wider stores save.
SMEM_MIN_TILES = 256
NUM_THREADS = 256
NUM_WARPS = NUM_THREADS // 32


@dsl_user_op
def ld_v4_ordered(addr: Int64, load_ok: Int32, *, loc=None, ip=None) -> list:
    """
    Load 16 bytes with predication and zero-fill. Side effects keep the load ahead of stores.
    """
    result = asm_args_last(llvm.StructType.get_literal([T.i32()] * 4),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r", [to_ir(addr, Int64, loc, ip), to_ir(load_ok, Int32, loc, ip)], side_effects=True,
    )
    return list(unpack(result, 4, Uint32, loc, ip))


@dsl_user_op
def ld_scalar_ordered(addr: Int64, load_ok: Int32, num_bytes: int, *, loc=None, ip=None) -> Uint32:
    """
    Load one element with zero-fill and keep the load ordered ahead of stores.
    """
    ptx_type = {1: "u8", 2: "u16", 4: "b32"}[num_bytes]
    result = asm_args_last(T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, 0;\n"
        f"@p ld.global.{ptx_type} $0, [$1];\n}}",
        "=r,l,r", [to_ir(addr, Int64, loc, ip), to_ir(load_ok, Int32, loc, ip)], side_effects=True,
    )
    return Uint32(result)


@dsl_user_op
def pack_bf16_from_fp32_bits(fp32_bits_lo: Uint32, fp32_bits_hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Round two fp32 bit patterns to bf16, with the first in the low half of the word.
    """
    return Uint32(asm_args_last(T.i32(), "{\n.reg .f32 a, b;\nmov.b32 a, $1;\nmov.b32 b, $2;\ncvt.rn.bf16x2.f32 $0, b, a;\n}", "=r,r,r", [to_ir(fp32_bits_lo, Uint32, loc, ip), to_ir(fp32_bits_hi, Uint32, loc, ip)]))


@dsl_user_op
def compute_scale_bits(amax_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    """
    Return the power-of-two E8M0 scale as an fp32 bit pattern.
    """
    scale_bits = asm_args_last(T.i32(),
        "{\n.reg .f32 a;\n.reg .b32 bits, m, e;\n"
        "mov.b32 a, $1;\n"
        f"max.f32 a, a, {AMAX_MIN};\n"
        f"mul.rn.f32 a, a, {FP8_MAX_RCP};\n"
        "mov.b32 bits, a;\n"
        "and.b32 m, bits, 0x007FFFFF;\n"
        "and.b32 e, bits, 0x7F800000;\n"
        "add.u32 m, m, 0x007FFFFF;\n"
        "shr.u32 m, m, 23;\n"
        "shl.b32 m, m, 23;\n"
        "add.u32 e, e, m;\n"
        "max.u32 $0, e, 0x00800000;\n}",  # max(scale, 2^-126) on non-negative bits
        "=r,r", [to_ir(amax_bits, Uint32, loc, ip)],
    )
    return Uint32(scale_bits)


@dsl_user_op
def ld_u64_nc(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(asm_args_last(T.i64(), "ld.global.nc.u64 $0, [$1];", "=l,l", [to_ir(addr, Int64, loc, ip)]))


@cute.jit
def thread_coords() -> tuple:
    """
    Assign each of the 256 threads an 8x8 sub-block of the 128x128 tile.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp, lane = tidx // 32, tidx % 32
    thread_row_offset, thread_col_offset = (warp // 2) * 32 + (lane // 8) * 8, (warp % 2) * 64 + (lane % 8) * 8
    return tidx, warp, lane, thread_row_offset, thread_col_offset


@cute.jit
def load_rows(x_base: Int64, block_start: tuple, num_rows: Int32, K: Int32, row_stride: Int64, input_bytes: cutlass.Constexpr[int], use_bf16_rounding: cutlass.Constexpr[bool], is_aligned: cutlass.Constexpr[bool]) -> list:
    """
    Read the thread's 8x8 block as packed words, with zero-fill outside the matrix.
    """
    row_start, col_start = block_start
    rows = []
    for i in cutlass.range_constexpr(8):
        row = row_start + i
        row_ok = Int32(row >= 0) & Int32(row < num_rows)
        row_addr = x_base + (Int64(row) * row_stride + Int64(col_start)) * input_bytes
        if const_expr(is_aligned):
            vector_ok = row_ok & Int32(col_start < K)
            if const_expr(input_bytes == 2):
                words = ld_v4_ordered(row_addr, vector_ok)
            elif const_expr(input_bytes == 4):
                words = ld_v4_ordered(row_addr, vector_ok) + ld_v4_ordered(row_addr + 16, vector_ok)
            else:
                # Side effects keep this packed FP8 load ahead of the row stores.
                result = asm_args_last(llvm.StructType.get_literal([T.i32()] * 2),
                    "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
                    "@p ld.global.v2.b32 {$0, $1}, [$2];\n}",
                    "=r,=r,l,r", [Int64(row_addr).ir_value(), Int32(vector_ok).ir_value()], side_effects=True,
                )
                words = list(unpack(result, 2, Uint32))
        else:
            elements = []
            for j in cutlass.range_constexpr(8):
                element_ok = row_ok & Int32(col_start + j < K)
                elements.append(ld_scalar_ordered(row_addr + j * input_bytes, element_ok, input_bytes))
            if const_expr(input_bytes == 2):
                words = [elements[2 * k] | (elements[2 * k + 1] << 16) for k in range(4)]
            elif const_expr(input_bytes == 4):
                words = elements
            else:
                words = [elements[4 * k] | (elements[4 * k + 1] << 8) | (elements[4 * k + 2] << 16) | (elements[4 * k + 3] << 24) for k in range(2)]
        if const_expr(input_bytes == 4 and use_bf16_rounding):
            words = [pack_bf16_from_fp32_bits(words[2 * k], words[2 * k + 1]) for k in range(4)]
        rows.append(words)
    return rows


@cute.jit
def reduce_thread_amax(rows: list, is_packed_bf16: cutlass.Constexpr[bool]) -> Uint32:
    """
    Return the maximum magnitude of the thread's sub-block as fp32 bits.
    """
    # Unsigned magnitude bits order like non-negative floats. NaNs would outrank infinity.
    acc_amax = Uint32(0)
    if const_expr(is_packed_bf16):
        for i in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(4):
                acc_amax = max_u16x2(acc_amax, rows[i][k] & Uint32(0x7FFF7FFF))
        acc_amax = max_u32(acc_amax & Uint32(0xFFFF), acc_amax >> 16) << 16
    else:
        for i in cutlass.range_constexpr(8):
            for k in cutlass.range_constexpr(8):
                acc_amax = max_u32(acc_amax, rows[i][k] & Uint32(0x7FFFFFFF))
    return acc_amax


@cute.jit
def reduce_block_amax(amax_bits: Uint32, warp: Int32, lane: Int32, s_amax: cute.Tensor) -> Uint32:
    """
    Broadcast the CTA-wide maximum to every thread.
    """
    amax_bits = Uint32(asm_args_last(T.i32(), "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r", [amax_bits.ir_value()], side_effects=True))

    if lane == 0:
        s_amax[warp] = amax_bits
    cute.arch.sync_threads()

    acc_amax = s_amax[0]
    for i in cutlass.range_constexpr(1, NUM_WARPS):
        acc_amax = max_u32(acc_amax, s_amax[i])
    return acc_amax


@cute.jit
def quantize_rows(rows: list, rcp_bits: Uint32, is_packed_bf16: cutlass.Constexpr[bool]) -> list:
    """
    Pack each row of eight E4M3 values into two words.
    """
    quantized_rows = []
    for i in cutlass.range_constexpr(8):
        fp32_bits = []
        if const_expr(is_packed_bf16):
            for k in cutlass.range_constexpr(4):
                word = rows[i][k]
                fp32_bits.append(word << 16)
                fp32_bits.append(word & Uint32(0xFFFF0000))
        else:
            fp32_bits = rows[i]
        quantized_rows.append([quantize_four_from_bits(fp32_bits[0], fp32_bits[1], fp32_bits[2], fp32_bits[3], rcp_bits), quantize_four_from_bits(fp32_bits[4], fp32_bits[5], fp32_bits[6], fp32_bits[7], rcp_bits)])
    return quantized_rows


@cute.jit
def transpose4(row0_word: Uint32, row1_word: Uint32, row2_word: Uint32, row3_word: Uint32) -> list:
    """
    Transpose a 4x4 byte block from one word per row to one word per column.
    """
    rows01_cols01, rows01_cols23 = prmt_args_last(row0_word, row1_word, 0x5140), prmt_args_last(row0_word, row1_word, 0x7362)
    rows23_cols01, rows23_cols23 = prmt_args_last(row2_word, row3_word, 0x5140), prmt_args_last(row2_word, row3_word, 0x7362)
    return [prmt_args_last(rows01_cols01, rows23_cols01, 0x5410), prmt_args_last(rows01_cols01, rows23_cols01, 0x7632), prmt_args_last(rows01_cols23, rows23_cols23, 0x5410), prmt_args_last(rows01_cols23, rows23_cols23, 0x7632)]


@cute.jit
def transpose8(rows: list) -> list:
    """
    Transpose eight rows of two E4M3 words into eight columns of two words.
    """
    top_left, bottom_left = transpose4(rows[0][0], rows[1][0], rows[2][0], rows[3][0]), transpose4(rows[4][0], rows[5][0], rows[6][0], rows[7][0])
    top_right, bottom_right = transpose4(rows[0][1], rows[1][1], rows[2][1], rows[3][1]), transpose4(rows[4][1], rows[5][1], rows[6][1], rows[7][1])
    return [[top_left[j], bottom_left[j]] for j in range(4)] + [[top_right[j], bottom_right[j]] for j in range(4)]


@cute.jit
def store_rows(out_base: Int64, quantized_rows: list, row_start: Int32, col_start: Int32, num_rows: Int32, K: Int32, is_aligned: cutlass.Constexpr[bool]) -> None:
    """
    Write eight rows of packed FP8 words directly to global memory.
    """
    for i in cutlass.range_constexpr(8):
        row = row_start + i
        row_ok = Int32(row >= 0) & Int32(row < num_rows)
        row_addr = out_base + Int64(row) * Int64(K) + Int64(col_start)
        if const_expr(is_aligned):
            st_v2(row_addr, quantized_rows[i][0], quantized_rows[i][1], row_ok & Int32(col_start < K))
        else:
            for j in cutlass.range_constexpr(8):
                byte = (quantized_rows[i][j // 4] >> (8 * (j % 4))) & Uint32(0xFF)
                st_b8(row_addr + j, byte, row_ok & Int32(col_start + j < K))


# A 128x128 byte tile staged in shared memory as 128 rows of eight 16-byte chunks. The chunk index
# is XOR-swizzled by s(row) = ((row >> 3) & 1) << 2 | ((row >> 3) & 7) >> 1, which makes both the
# row-block writes (8 bytes per lane, rows 8 apart) and the transposed-block writes (rows 8 apart
# the other way) hit the minimum two wavefronts; readback reads whole rows, one wavefront each.
TILE_BYTES = BLOCK * BLOCK


@cute.jit
def swizzle_tile_offset(row: Int32, chunk_idx: Int32) -> Int32:
    """
    Compute the byte offset of a 16-byte chunk in the staged tile.
    """
    row_block = (row >> 3) & 7
    return row * BLOCK + ((chunk_idx ^ (((row_block & 1) << 2) | (row_block >> 1))) << 4)


@cute.jit
def stage_output_block(s_tile: Int32, quantized_rows: list, tile_row_start: Int32, tile_col_start: Int32) -> None:
    """
    Write a thread's packed 8x8 block into the swizzled shared tile.
    """
    for i in cutlass.range_constexpr(8):
        sts_v2(s_tile + swizzle_tile_offset(tile_row_start + i, tile_col_start >> 4) + (tile_col_start & 8), quantized_rows[i][0], quantized_rows[i][1])


@cute.jit
def store_staged_tile(s_tile: Int32, out_tile_addr: Int64, num_rows: Int32, num_cols: Int32, row_stride: Int32, num_threads: cutlass.Constexpr[int] = NUM_THREADS, store_bytes: cutlass.Constexpr[int] = 16) -> None:
    """
    Drain the staged tile in contiguous vector or scalar stores per row.
    """
    tidx, _, _ = cute.arch.thread_idx()
    units_per_row = BLOCK // store_bytes
    for i in cutlass.range_constexpr(TILE_BYTES // store_bytes // num_threads):
        unit_idx = i * num_threads + tidx
        row_idx, col_unit_idx = unit_idx // units_per_row, unit_idx % units_per_row
        unit_ok = Int32(row_idx < num_rows) & Int32(col_unit_idx * store_bytes < num_cols)
        out_addr = out_tile_addr + Int64(row_idx) * Int64(row_stride) + Int64(col_unit_idx * store_bytes)
        if const_expr(store_bytes == 16):
            st_v4(out_addr, lds_v4(s_tile + swizzle_tile_offset(row_idx, col_unit_idx)), unit_ok)
        elif const_expr(store_bytes == 4):
            st_b32(out_addr, lds_b32(s_tile + swizzle_tile_offset(row_idx, col_unit_idx >> 2) + ((col_unit_idx & 3) << 2)), unit_ok)
        else:
            value = Uint32(asm_args_last(T.i32(), "ld.shared.u8 $0, [$1];", "=r,r", [Int32(s_tile + swizzle_tile_offset(row_idx, col_unit_idx >> 4) + (col_unit_idx & 15)).ir_value()], side_effects=True))
            st_b8(out_addr, value, unit_ok)


# Bound the number of element loads in flight to limit live registers.
STAGE_BATCH = 16


class TileCoordinates:
    def __init__(self, tile_row_start, tile_col_start, thread_row_offset, thread_col_offset):
        self.tile_row_start, self.tile_col_start = tile_row_start, tile_col_start
        self.thread_row_offset, self.thread_col_offset = thread_row_offset, thread_col_offset


@cute.jit
def stage_input_tile(x_base: Int64, s_tile: Int32, tile: TileCoordinates, N: Int32, K: Int32, row_stride: Int64, col_stride: Int64, input_bytes: cutlass.Constexpr[int]) -> None:
    """
    Stage the input tile in shared memory, with zero-fill outside the matrix.
    """
    tile_row_start, tile_col_start = tile.tile_row_start, tile.tile_col_start
    tidx, _, _ = cute.arch.thread_idx()
    # Consecutive lanes read consecutive columns for coalescing when the column stride is one.
    col, row_phase = tidx % BLOCK, tidx // BLOCK
    rows_per_pass = NUM_THREADS // BLOCK
    col_ok = Int32(tile_col_start + col < K)
    col_addr = x_base + Int64(tile_col_start + col) * col_stride * input_bytes
    for batch in cutlass.range_constexpr(BLOCK // rows_per_pass // STAGE_BATCH):
        values = []
        for i in cutlass.range_constexpr(STAGE_BATCH):
            row = (batch * STAGE_BATCH + i) * rows_per_pass + row_phase
            values.append(ld_scalar_ordered(col_addr + Int64(tile_row_start + row) * row_stride * input_bytes, col_ok & Int32(tile_row_start + row < N), input_bytes))
        for i in cutlass.range_constexpr(STAGE_BATCH):
            row = (batch * STAGE_BATCH + i) * rows_per_pass + row_phase
            s_element_addr = s_tile + (row * BLOCK + col) * input_bytes
            if const_expr(input_bytes == 4):
                sts_b32_raw(s_element_addr, values[i])
            else:
                asm_args_last(None, "st.shared.b16 [$0], $1;", "r,r", [Int32(s_element_addr).ir_value(), Uint32(values[i]).ir_value()], side_effects=True)


@cute.jit
def load_staged_rows(s_tile: Int32, thread_row_offset: Int32, thread_col_offset: Int32, input_bytes: cutlass.Constexpr[int]) -> list:
    """
    Read the thread's 8x8 block from the staged input in the same format as load_rows.
    """
    rows = []
    for i in cutlass.range_constexpr(8):
        row_addr = s_tile + ((thread_row_offset + i) * BLOCK + thread_col_offset) * input_bytes
        words = list(lds_v4(row_addr))
        if const_expr(input_bytes == 4):
            words = words + list(lds_v4(row_addr + 16))
        rows.append(words)
    return rows


class BlockwiseTranspose:
    """
    Quantize a 128x128 tile and its transpose from one load of the input.
    """

    def __init__(self, input_bytes: int, load_mode: str, store_mode: str, out_store_bytes: int, out_t_store_bytes: int, is_batched: bool) -> None:
        self.input_bytes, self.load_mode, self.store_mode, self.is_batched = input_bytes, load_mode, store_mode, is_batched
        self.out_store_bytes, self.out_t_store_bytes = out_store_bytes, out_t_store_bytes
        self.num_shared_words = max(TILE_BYTES * input_bytes if load_mode == "staged" else 0, 2 * TILE_BYTES if store_mode == "staged" else 0) // 4

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, group_stride: Int64, row_stride: Int64, col_stride: Int64, stream: cuda.CUstream):
        if const_expr(self.is_batched):
            G, N, K = mOut.shape[0], mOut.shape[1], mOut.shape[2]  # (group, row, column)
        else:
            G, N, K = 1, mOut.shape[0], mOut.shape[1]  # (row, column)
        num_row_tiles, num_col_tiles = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mX.iterator.toint(), mOut.iterator.toint(), mScale.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint(), Int32(N), Int32(K), group_stride, row_stride, col_stride, Int32(num_row_tiles), Int32(num_col_tiles)).launch(grid=[num_row_tiles, num_col_tiles, G], block=[NUM_THREADS, 1, 1], smem=self.num_shared_words * 4 + 128, stream=stream)

    @cute.kernel
    def kernel(self, x_base: Int64, out_base: Int64, scale_base: Int64, out_t_base: Int64, scale_t_base: Int64, N: Int32, K: Int32, group_stride: Int64, row_stride: Int64, col_stride: Int64, num_row_tiles: Int32, num_col_tiles: Int32):
        bx, by, bz = cute.arch.block_idx()
        tidx, warp, lane, thread_row_offset, thread_col_offset = thread_coords()
        shared_alloc = cutlass.memory.SmemAllocator()
        s_amax = shared_alloc.allocate_tensor(Uint32, cute.make_layout(NUM_WARPS), byte_alignment=16)  # (warp,)

        tile_row_start, tile_col_start = bx * BLOCK, by * BLOCK
        x_group_base = x_base + Int64(bz) * group_stride * self.input_bytes
        s_tile = Int32(0)
        if const_expr(self.num_shared_words > 0):
            s_tile_words = shared_alloc.allocate_tensor(Uint32, cute.make_layout(self.num_shared_words), byte_alignment=128)  # (word,)
            s_tile = s_tile_words.iterator.toint()
        tile = TileCoordinates(tile_row_start, tile_col_start, thread_row_offset, thread_col_offset)

        rows = self.load_tile(x_group_base, s_tile, tile, N, K, row_stride, col_stride)

        is_packed_bf16 = const_expr(self.input_bytes == 2)
        # The barrier inside also orders the staged input reads before the staged output writes.
        amax_bits = reduce_block_amax(reduce_thread_amax(rows, is_packed_bf16), warp, lane, s_amax)
        scale_bits = compute_scale_bits(amax_bits)
        quantized_rows = quantize_rows(rows, compute_scale_rcp_bits(scale_bits), is_packed_bf16)
        quantized_rows_t = transpose8(quantized_rows)

        self.store_tile_pair(out_base, out_t_base, bz, N, K, tile, s_tile, quantized_rows, quantized_rows_t)
        store_tile_scales(scale_base, scale_t_base, scale_bits, tidx, bx, by, bz, num_row_tiles, num_col_tiles)

    @cute.jit
    def load_tile(self, x_group_base: Int64, s_tile: Int32, tile: TileCoordinates, N: Int32, K: Int32, row_stride: Int64, col_stride: Int64) -> list:
        thread_row_offset, thread_col_offset = tile.thread_row_offset, tile.thread_col_offset
        if const_expr(self.load_mode == "staged"):
            stage_input_tile(x_group_base, s_tile, tile, N, K, row_stride, col_stride, self.input_bytes)
            cute.arch.sync_threads()
            rows = load_staged_rows(s_tile, thread_row_offset, thread_col_offset, self.input_bytes)
        else:
            tile_row_start, tile_col_start = tile.tile_row_start, tile.tile_col_start
            rows = load_rows(x_group_base, (tile_row_start + thread_row_offset, tile_col_start + thread_col_offset), N, K, row_stride, self.input_bytes, False, True)
        return rows

    @cute.jit
    def store_tile_pair(self, out_base: Int64, out_t_base: Int64, group_idx: Int32, N: Int32, K: Int32, tile: TileCoordinates, s_tile: Int32, quantized_rows: list, quantized_rows_t: list) -> None:
        tile_row_start, tile_col_start = tile.tile_row_start, tile.tile_col_start
        thread_row_offset, thread_col_offset = tile.thread_row_offset, tile.thread_col_offset
        num_group_elements = Int64(N) * Int64(K)
        out_group_base, out_t_group_base = out_base + Int64(group_idx) * num_group_elements, out_t_base + Int64(group_idx) * num_group_elements
        if const_expr(self.store_mode == "staged"):
            stage_output_block(s_tile, quantized_rows, thread_row_offset, thread_col_offset)
            stage_output_block(s_tile + TILE_BYTES, quantized_rows_t, thread_col_offset, thread_row_offset)
            cute.arch.sync_threads()
            out_tile_addr = out_group_base + Int64(tile_row_start) * Int64(K) + Int64(tile_col_start)
            store_staged_tile(s_tile, out_tile_addr, N - tile_row_start, K - tile_col_start, K, NUM_THREADS, self.out_store_bytes)
            out_t_tile_addr = out_t_group_base + Int64(tile_col_start) * Int64(N) + Int64(tile_row_start)
            store_staged_tile(s_tile + TILE_BYTES, out_t_tile_addr, K - tile_col_start, N - tile_row_start, N, NUM_THREADS, self.out_t_store_bytes)
        else:
            row_start, col_start = tile_row_start + thread_row_offset, tile_col_start + thread_col_offset
            store_rows(out_group_base, quantized_rows, row_start, col_start, N, K, True)
            store_rows(out_t_group_base, quantized_rows_t, col_start, row_start, K, N, True)


@cute.jit
def store_tile_scales(scale_base: Int64, scale_t_base: Int64, scale_bits: Uint32, tidx: Int32, bx: Int32, by: Int32, bz: Int32, num_row_tiles: Int32, num_col_tiles: Int32) -> None:
    if tidx == 0:
        num_tiles = Int64(num_row_tiles) * Int64(num_col_tiles)
        scale_idx = Int64(bz) * num_tiles + Int64(bx) * Int64(num_col_tiles) + Int64(by)
        scale_t_idx = Int64(bz) * num_tiles + Int64(by) * Int64(num_row_tiles) + Int64(bx)
        st_b32(scale_base + scale_idx * 4, scale_bits, Int32(1))
        st_b32(scale_t_base + scale_t_idx * 4, scale_bits, Int32(1))


COMPILED: dict = {}


def make_fake_row_major(dtype, num_dims: int, align: int) -> cute.Tensor:
    shape = tuple(cute.sym_int() for _ in range(num_dims))
    stride = tuple(cute.sym_int64() for _ in range(num_dims - 1)) + (1,)
    return fake(dtype, shape, stride, align)


def make_fake_strided(dtype, num_dims: int, align: int) -> cute.Tensor:
    return fake(dtype, tuple(cute.sym_int() for _ in range(num_dims)), tuple(cute.sym_int64() for _ in range(num_dims)), align)


def compile_blockwise(dtype: torch.dtype, load_mode: str, store_mode: str, out_store_bytes: int, out_t_store_bytes: int, is_batched: bool):
    key = ("blockwise_transpose", dtype, load_mode, store_mode, out_store_bytes, out_t_store_bytes, is_batched)
    fn = COMPILED.get(key)
    if fn is None:
        input_bytes, num_dims = torch.tensor([], dtype=dtype).element_size(), 3 if is_batched else 2
        fn = cute.compile(BlockwiseTranspose(input_bytes, load_mode, store_mode, out_store_bytes, out_t_store_bytes, is_batched), make_fake_strided(TORCH2CUTE[dtype], num_dims, input_bytes),
                          make_fake_row_major(cutlass.Float8E4M3FN, num_dims, 1), make_fake_row_major(cutlass.Float32, num_dims, 4), make_fake_row_major(cutlass.Float8E4M3FN, num_dims, 1), make_fake_row_major(cutlass.Float32, num_dims, 4),
                          Int64(0), Int64(0), Int64(0), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    return fn


def select_store_bytes(row_bytes: int) -> int:
    return 16 if row_bytes % 16 == 0 else 4 if row_bytes % 4 == 0 else 1


def resolve_blockwise_kernel(dtype: torch.dtype, shape: tuple, stride: tuple, address_mod_16: int) -> tuple:
    """
    Select the compiled launch, any output swap, and the effective input strides.
    """
    is_batched = len(shape) == 3
    G, N, K = shape[0] if is_batched else 1, shape[-2], shape[-1]
    if G == 0 or N == 0 or K == 0:
        return None, False, ()

    group_stride, row_stride, col_stride = (stride[0] if G > 1 else 0) if is_batched else 0, stride[-2], stride[-1]
    # A unit row stride lets the transposed input use the vector load path.
    is_swapped = N > 1 and K > 1 and col_stride != 1 and row_stride == 1
    if is_swapped:
        N, K, row_stride, col_stride = K, N, col_stride, row_stride
    # Singleton dimensions never move the address, so ignore their strides in the alignment checks.
    row_stride, col_stride = row_stride if N > 1 else 0, col_stride if K > 1 else 1
    input_bytes = dtype.itemsize
    use_vector_loads = col_stride == 1 and K % 8 == 0 and address_mod_16 % 16 == 0 and group_stride * input_bytes % 16 == 0 and row_stride * input_bytes % 16 == 0
    if N % 16 == 0 and K % 16 == 0 and G * ((N + BLOCK - 1) // BLOCK) * ((K + BLOCK - 1) // BLOCK) >= SMEM_MIN_TILES:
        store_mode, out_store_bytes, out_t_store_bytes = "staged", 16, 16
    elif N % 8 == 0 and K % 8 == 0:
        store_mode, out_store_bytes, out_t_store_bytes = "registers", 8, 8
    else:
        store_mode, out_store_bytes, out_t_store_bytes = "staged", select_store_bytes(K), select_store_bytes(N)
    return compile_blockwise(dtype, "vector" if use_vector_loads else "staged", store_mode, out_store_bytes, out_t_store_bytes, is_batched), is_swapped, (group_stride, row_stride, col_stride)


# Opaque C++ ops keep eager and traced calls on the same launch path without re-entering
# Python on a cache hit. The custom Autograd dispatch avoids the fallback on real tensors.
OPS_SOURCE = r"""
#include <Python.h>
// tvm-ffi first: its DLPack header (1.4) and ATen's (1.3) share an include guard, and tvm-ffi needs 1.4.
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace {

using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

struct Key {
  int64_t words[8];
  bool operator==(const Key& other) const { return std::memcmp(words, other.words, sizeof(words)) == 0; }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    size_t hash = 1469598103934665603ull;
    for (int64_t word : key.words) hash = (hash ^ static_cast<size_t>(word)) * 1099511628211ull;
    return hash;
  }
};

struct Entry {
  tvm::ffi::Function fn;
  bool swap = false, empty = true;
  int64_t strides[3] = {0, 0, 0};
};

constexpr size_t max_entries = 4096;
// Leaked on purpose: destroying them after the interpreter finalizes would crash.
auto& cache = *new std::unordered_map<Key, Entry, KeyHash>();
std::mutex cache_mutex;

// Sets tvm-ffi's environment stream for a device and restores the previous one, also on exceptions.
struct EnvStream {
  int32_t device;
  TVMFFIStreamHandle previous = nullptr;
  EnvStream(int32_t device, cudaStream_t stream) : device(device) { TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous); }
  ~EnvStream() { TVMFFIEnvSetStream(kDLCUDA, device, previous, nullptr); }
};

DLTensor as_dl(const at::Tensor& tensor) {
  DLTensor dl_tensor;
  dl_tensor.data = tensor.data_ptr();
  dl_tensor.device = DLDevice{kDLCUDA, static_cast<int32_t>(tensor.device().index())};
  dl_tensor.ndim = static_cast<int32_t>(tensor.dim());
  dl_tensor.dtype = at::getDLDataType(tensor);
  dl_tensor.shape = const_cast<int64_t*>(tensor.sizes().data());
  dl_tensor.strides = const_cast<int64_t*>(tensor.strides().data());
  dl_tensor.byte_offset = 0;
  return dl_tensor;
}

Entry lookup(const at::Tensor& x) {
  Key key;
  std::memset(key.words, 0, sizeof(key.words));
  int64_t ndim = x.dim();
  key.words[0] = static_cast<int64_t>(x.scalar_type()) * 16 + ndim;
  for (int64_t i = 0; i < ndim; ++i) key.words[1 + i] = x.size(i), key.words[4 + i] = x.stride(i);
  key.words[7] = reinterpret_cast<int64_t>(x.data_ptr()) & 15;
  std::lock_guard<std::mutex> guard(cache_mutex);
  auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  static tvm::ffi::Function& resolve = *new tvm::ffi::Function(tvm::ffi::Function::GetGlobalRequired("pithtrain.fp8_quant_weight_resolve_sm90"));
  tvm::ffi::Array<int64_t> sizes(x.sizes().begin(), x.sizes().end()), strides(x.strides().begin(), x.strides().end());
  // the dispatcher released the GIL, and the resolver is Python
  PyGILState_STATE gil = PyGILState_Ensure();
  auto result = resolve(static_cast<int64_t>(x.scalar_type()), sizes, strides, key.words[7]).cast<tvm::ffi::Array<tvm::ffi::Any>>();
  PyGILState_Release(gil);
  Entry entry;
  // Function is a nullable reference, so an empty input's None is told apart by its type index
  entry.empty = result[0].type_index() == tvm::ffi::TypeIndex::kTVMFFINone;
  if (!entry.empty) {
    entry.fn = result[0].cast<tvm::ffi::Function>();
    entry.swap = result[1].cast<bool>();
    for (int i = 0; i < 3; ++i) entry.strides[i] = result[2 + i].cast<int64_t>();
  }
  if (cache.size() >= max_entries) cache.clear();
  cache.emplace(key, entry);
  return entry;
}

template <int64_t ndim>
Outputs quant(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == ndim, "fp8_quant_weight on SM90 takes a ", ndim, "-D CUDA tensor, got ", x.dim(), "-D on ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat, "fp8_quant_weight on SM90 takes bf16 or fp32, got ", x.scalar_type());
  c10::cuda::CUDAGuard device_guard(x.device());
  int64_t G = ndim == 3 ? x.size(0) : 1, N = x.size(-2), K = x.size(-1), rows = (N + 127) / 128, cols = (K + 127) / 128;
  // the CUDA allocator directly, not at::empty through the dispatcher (host cost per call)
  auto alloc = [&](int64_t num_rows, int64_t num_cols, at::ScalarType dtype) {
    int64_t sizes[3] = {G, num_rows, num_cols};
    return at::Tensor(at::detail::empty_cuda(at::IntArrayRef(sizes + (3 - ndim), ndim), dtype, x.device(), std::nullopt));
  };
  at::Tensor out = alloc(N, K, at::kFloat8_e4m3fn), scale = alloc(rows, cols, at::kFloat), out_t = alloc(K, N, at::kFloat8_e4m3fn), scale_t = alloc(cols, rows, at::kFloat);
  Entry entry = lookup(x);
  if (entry.empty) return {out, scale, out_t, scale_t};
  DLTensor d_x = as_dl(x), d_out = as_dl(out), d_scale = as_dl(scale), d_out_t = as_dl(out_t), d_scale_t = as_dl(scale_t);
  if (entry.swap) std::swap(d_out, d_out_t), std::swap(d_scale, d_scale_t);
  int32_t device = static_cast<int32_t>(x.device().index());
  EnvStream env(device, c10::cuda::getCurrentCUDAStream(device).stream());
  entry.fn(&d_x, &d_out, &d_scale, &d_out_t, &d_scale_t, entry.strides[0], entry.strides[1], entry.strides[2]);
  return {out, scale, out_t, scale_t};
}

template <int64_t ndim, const char* name>
Outputs quant_autograd(const at::Tensor& x) {
  constexpr c10::DispatchKeySet traced({c10::DispatchKey::Python, c10::DispatchKey::PythonTLSSnapshot, c10::DispatchKey::Functionalize});
  if (!x.key_set().has_any(traced) && !c10::impl::tls_local_dispatch_key_set().included_.has_any(traced)) return quant<ndim>(x);
  static auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, "").typed<Outputs(const at::Tensor&)>();
  at::AutoDispatchBelowADInplaceOrView guard;
  return op.call(x);
}

constexpr char quant_name[] = "pithtrain::fp8_quant_weight_sm90";
constexpr char quant_grouped_name[] = "pithtrain::fp8_quant_weight_grouped_sm90";

}  // namespace

TORCH_LIBRARY_FRAGMENT(pithtrain, m) {
  m.def("fp8_quant_weight_sm90(Tensor x) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("fp8_quant_weight_grouped_sm90(Tensor x) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(pithtrain, CUDA, m) {
  m.impl("fp8_quant_weight_sm90", &quant<2>);
  m.impl("fp8_quant_weight_grouped_sm90", &quant<3>);
}

TORCH_LIBRARY_IMPL(pithtrain, Autograd, m) {
  m.impl("fp8_quant_weight_sm90", &quant_autograd<2, quant_name>);
  m.impl("fp8_quant_weight_grouped_sm90", &quant_autograd<3, quant_grouped_name>);
}
"""


@tvm_ffi.register_global_func("pithtrain.fp8_quant_weight_resolve_sm90")
def resolve_weight_launch(scalar_type: int, sizes: Sequence[int], strides: Sequence[int], address_mod_16: int) -> list:
    """
    Adapt the blockwise launch and effective strides to the C++ caller's descriptor.
    """
    dtype = {15: torch.bfloat16, 6: torch.float32}[scalar_type]
    fn, is_swapped, input_strides = resolve_blockwise_kernel(dtype, tuple(sizes), tuple(strides), address_mod_16)
    return [None, False, 0, 0, 0] if fn is None else [fn._tvm_ffi_function, is_swapped, *input_strides]


def build_ops() -> str:
    """
    Build and cache the shared library implementing the two opaque weight-quantization ops.
    """
    digest = hashlib.sha256("\n".join([OPS_SOURCE, torch.__version__, tvm_ffi.__version__, sys.version]).encode()).hexdigest()[:16]
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    root = os.path.join(cache_home, "pithtrain", "fp8_weight_quant_ops_sm90", digest)
    library = os.path.join(root, "ops.so")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(library):
            # Publish atomically because the file lock may fail to exclude builders on other nodes.
            unique = f"{socket.gethostname()}.{os.getpid()}"
            source, partial = os.path.join(root, f"ops.{unique}.cpp"), f"{library}.{unique}.tmp"
            with open(source, "w") as source_file:
                source_file.write(OPS_SOURCE)
            includes = cpp_extension.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"], tvm_ffi.libinfo.find_include_path(), tvm_ffi.libinfo.find_dlpack_include_path()]
            tvm_ffi_dir = os.path.dirname(tvm_ffi.libinfo.find_libtvm_ffi())
            command = ["c++", "-O2", "-std=c++20", "-shared", "-fPIC", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", source, "-o", partial]
            command += [f"-I{path}" for path in includes] + [f"-L{path}" for path in cpp_extension.library_paths(device_type="cuda")] + [f"-L{tvm_ffi_dir}", f"-Wl,-rpath,{tvm_ffi_dir}", "-lc10", "-lc10_cuda", "-ltorch_cpu", "-ltorch_cuda", "-ltorch", "-ltvm_ffi"]
            subprocess.run(command, check=True)
            os.replace(partial, library)
            os.remove(source)
    return library


def make_fake_weight_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate the two fake FP8 matrices and their corresponding block scale grids.
    """
    *batch_shape, N, K = x.shape
    num_row_tiles, num_col_tiles = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
    return x.new_empty((*batch_shape, N, K), dtype=torch.float8_e4m3fn), x.new_empty((*batch_shape, num_row_tiles, num_col_tiles), dtype=torch.float32), x.new_empty((*batch_shape, K, N), dtype=torch.float8_e4m3fn), x.new_empty((*batch_shape, num_col_tiles, num_row_tiles), dtype=torch.float32)


torch.ops.load_library(build_ops())
torch.library.register_fake("pithtrain::fp8_quant_weight_sm90", make_fake_weight_outputs)
torch.library.register_fake("pithtrain::fp8_quant_weight_grouped_sm90", make_fake_weight_outputs)
# The dispatcher objects let torch.compile see the opaque ops directly.
fp8_quant_weight_sm90 = torch.ops.pithtrain.fp8_quant_weight_sm90.default
fp8_quant_weight_grouped_sm90 = torch.ops.pithtrain.fp8_quant_weight_grouped_sm90.default


class ShardTileCoordinates:
    def __init__(self, local_rows, K, tile_row_idx, tile_col_idx, group_idx, row_offset):
        self.local_rows, self.K = local_rows, K
        self.tile_row_idx, self.tile_col_idx = tile_row_idx, tile_col_idx
        self.group_idx, self.row_offset = group_idx, row_offset


@cute.jit
def process_shard_tile(shard_base: Int64, amax_addr: Int64, out_base: Int64, tile: ShardTileCoordinates, is_cast: cutlass.Constexpr[bool], mode: cutlass.Constexpr[str]) -> None:
    if const_expr(mode == "row"):
        process_shard_tile_by_row(shard_base, amax_addr, out_base, tile, is_cast)
        return
    local_rows, K = tile.local_rows, tile.K
    tile_row_idx, tile_col_idx = tile.tile_row_idx, tile.tile_col_idx
    group_idx, row_offset = tile.group_idx, tile.row_offset
    tidx, warp, lane, thread_row_offset, thread_col_offset = thread_coords()
    row_start, col_start = tile_row_idx * BLOCK + thread_row_offset - row_offset, tile_col_idx * BLOCK + thread_col_offset
    group_start = Int64(group_idx) * Int64(local_rows) * Int64(K)
    rows = load_rows(shard_base + group_start * 4, (row_start, col_start), local_rows, K, Int64(K), 4, True, False)
    if const_expr(is_cast):
        scale_bits = compute_scale_bits(ld_scalar_ordered(amax_addr, Int32(1), 4))
        quantized_rows = quantize_rows(rows, compute_scale_rcp_bits(scale_bits), True)
        store_rows(out_base + group_start, quantized_rows, row_start, col_start, local_rows, K, False)
    else:
        shared_alloc = cutlass.memory.SmemAllocator()
        s_amax = shared_alloc.allocate_tensor(Uint32, cute.make_layout(NUM_WARPS), byte_alignment=16)  # (warp,)
        amax_bits = reduce_block_amax(reduce_thread_amax(rows, True), warp, lane, s_amax)
        if tidx == 0:
            st_b32(amax_addr, amax_bits, Int32(1))


class ShardRowSlice:
    def __init__(self, group_start, row_start, warp, col_start, K):
        self.group_start, self.row_start = group_start, row_start
        self.warp, self.col_start, self.K = warp, col_start, K


@cute.jit
def process_shard_tile_by_row(shard_base: Int64, amax_addr: Int64, out_base: Int64, tile: ShardTileCoordinates, is_cast: cutlass.Constexpr[bool]) -> None:
    """
    Use all eight warps even when a shard contains only a few rows of the tile.
    """
    local_rows, K = tile.local_rows, tile.K
    tile_row_idx, tile_col_idx = tile.tile_row_idx, tile.tile_col_idx
    group_idx, row_offset = tile.group_idx, tile.row_offset
    tidx, _, _ = cute.arch.thread_idx()
    warp, lane = tidx // 32, tidx % 32
    col_start = tile_col_idx * BLOCK + lane * 4
    group_start = Int64(group_idx) * Int64(local_rows) * Int64(K)
    tile_row_start = tile_row_idx * BLOCK - row_offset
    if const_expr(is_cast):
        rcp_bits = compute_scale_rcp_bits(compute_scale_bits(ld_scalar_ordered(amax_addr, Int32(1), 4)))
        rows_per_warp = BLOCK // NUM_WARPS // 2
        # Store the first half of the tile before loading the second to limit live registers.
        for batch in cutlass.range_constexpr(2):
            batch_start = tile_row_start + batch * (BLOCK // 2)
            shard_rows = ShardRowSlice(group_start, batch_start, warp, col_start, K)
            words, rows_ok = load_shard_rows(shard_base, shard_rows, local_rows, rows_per_warp)
            store_cast_shard_rows(out_base, rcp_bits, words, rows_ok, shard_rows, rows_per_warp)
    else:
        shard_rows = ShardRowSlice(group_start, tile_row_start, warp, col_start, K)
        words, _ = load_shard_rows(shard_base, shard_rows, local_rows, BLOCK // NUM_WARPS)
        amax_bits = reduce_shard_rows_amax(words, warp, lane)
        if tidx == 0:
            st_b32(amax_addr, amax_bits, Int32(1))


@cute.jit
def load_shard_rows(shard_base: Int64, shard_rows: ShardRowSlice, local_rows: Int32, rows_per_warp: cutlass.Constexpr[int]) -> tuple:
    """
    Load the warp's rows as rounded BF16 pairs and return their store predicates.
    """
    group_start, row_start = shard_rows.group_start, shard_rows.row_start
    warp, col_start, K = shard_rows.warp, shard_rows.col_start, shard_rows.K
    col_ok = Int32(col_start < K)
    words, rows_ok = [], []
    for i in cutlass.range_constexpr(rows_per_warp):
        row = row_start + i * NUM_WARPS + warp
        row_ok = Int32(row >= 0) & Int32(row < local_rows) & col_ok
        fp32_bits = ld_v4_ordered(shard_base + (group_start + Int64(row) * Int64(K) + Int64(col_start)) * 4, row_ok)
        words.append([pack_bf16_from_fp32_bits(fp32_bits[0], fp32_bits[1]), pack_bf16_from_fp32_bits(fp32_bits[2], fp32_bits[3])])
        rows_ok.append(row_ok)
    return words, rows_ok


@cute.jit
def store_cast_shard_rows(out_base: Int64, rcp_bits: Uint32, words: list, rows_ok: list, shard_rows: ShardRowSlice, rows_per_warp: cutlass.Constexpr[int]) -> None:
    group_start, row_start = shard_rows.group_start, shard_rows.row_start
    warp, col_start, K = shard_rows.warp, shard_rows.col_start, shard_rows.K
    for i in cutlass.range_constexpr(rows_per_warp):
        row = row_start + i * NUM_WARPS + warp
        word_lo, word_hi = words[i]
        quantized_word = quantize_four_from_bits(word_lo << 16, word_lo & Uint32(0xFFFF0000), word_hi << 16, word_hi & Uint32(0xFFFF0000), rcp_bits)
        st_b32(out_base + group_start + Int64(row) * Int64(K) + Int64(col_start), quantized_word, rows_ok[i])


@cute.jit
def reduce_shard_rows_amax(words: list, warp: Int32, lane: Int32) -> Uint32:
    """
    Reduce the shard tile's BF16 magnitudes to a CTA-wide maximum, as fp32 bits.
    """
    acc_amax = Uint32(0)
    for i in cutlass.range_constexpr(BLOCK // NUM_WARPS):
        acc_amax = max_u16x2(acc_amax, words[i][0] & Uint32(0x7FFF7FFF))
        acc_amax = max_u16x2(acc_amax, words[i][1] & Uint32(0x7FFF7FFF))
    acc_amax = max_u32(acc_amax & Uint32(0xFFFF), acc_amax >> 16) << 16
    shared_alloc = cutlass.memory.SmemAllocator()
    s_amax = shared_alloc.allocate_tensor(Uint32, cute.make_layout(NUM_WARPS), byte_alignment=16)  # (warp,)
    return reduce_block_amax(acc_amax, warp, lane, s_amax)


class ShardTile:
    def __init__(self, is_cast: bool, mode: str) -> None:
        self.is_cast, self.mode = is_cast, mode

    @cute.jit
    def __call__(self, mShard: cute.Tensor, mAmax: cute.Tensor, mOut: cute.Tensor, row_offset: Int32, group_offset: Int32, stream: cuda.CUstream):
        num_local_groups, local_rows, K = mShard.shape[0], mShard.shape[1], mShard.shape[2]  # (local group, local row, column)
        first_tile_row_idx, end_tile_row_idx = row_offset // BLOCK, (row_offset + local_rows + BLOCK - 1) // BLOCK
        num_col_tiles = (K + BLOCK - 1) // BLOCK
        self.kernel(mShard.iterator.toint(), mAmax.iterator.toint(), mOut.iterator.toint(), Int32(local_rows), Int32(K), Int32(first_tile_row_idx), Int32(mAmax.shape[1]), Int32(mAmax.shape[2]), row_offset, group_offset).launch(grid=[end_tile_row_idx - first_tile_row_idx, num_col_tiles, num_local_groups], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, shard_base: Int64, amax_base: Int64, out_base: Int64, local_rows: Int32, K: Int32, first_tile_row_idx: Int32, amax_rows: Int32, amax_cols: Int32, row_offset: Int32, group_offset: Int32):
        bx, by, bz = cute.arch.block_idx()
        tile_row_idx = first_tile_row_idx + bx
        amax_idx = (Int64(group_offset + bz) * Int64(amax_rows) + Int64(tile_row_idx)) * Int64(amax_cols) + Int64(by)
        process_shard_tile(shard_base, amax_base + amax_idx * 4, out_base, ShardTileCoordinates(local_rows, K, tile_row_idx, by, bz, row_offset), self.is_cast, self.mode)


def launch_shard_tiles(shard: torch.Tensor, amax: torch.Tensor, out: Optional[torch.Tensor], row_offset: int, group_offset: int) -> None:
    if shard.numel() == 0:
        return
    K = shard.shape[2]
    is_cast = out is not None
    if K % 4 == 0 and shard.data_ptr() % 16 == 0 and (not is_cast or out.data_ptr() % 4 == 0):
        mode = "row"
    else:
        mode = "scalar"
    key = ("shard", is_cast, mode, out.dtype if is_cast else None)
    fn = COMPILED.get(key)
    if fn is None:
        fn = cute.compile(ShardTile(is_cast, mode), make_fake_row_major(cutlass.Float32, 3, 4), make_fake_row_major(cutlass.Float32, 3, 4), make_fake_row_major(TORCH2CUTE[out.dtype] if is_cast else cutlass.Float32, 3, 1),
                          Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(shard, amax, out if is_cast else shard, row_offset, group_offset)


def fp8_shard_amax_sm90(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """
    Write the per-tile amax of an fp32 shard after rounding to BF16.
    """
    launch_shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm90(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """
    Cast a BF16-rounded shard to E4M3 using the reduced per-tile amax.
    """
    launch_shard_tiles(shard, amax, out, row_offset, group_offset)


class Scales:
    @cute.jit
    def __call__(self, mAmax: cute.Tensor, mScale: cute.Tensor, mScaleT: cute.Tensor, stream: cuda.CUstream):
        G, amax_rows, amax_cols = mAmax.shape[0], mAmax.shape[1], mAmax.shape[2]  # (group, tile row, tile column)
        self.kernel(mAmax.iterator.toint(), mScale.iterator.toint(), mScaleT.iterator.toint(), Int32(amax_rows), Int32(amax_cols)).launch(grid=[(amax_rows * amax_cols + NUM_THREADS - 1) // NUM_THREADS, G, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, amax_base: Int64, scale_base: Int64, scale_t_base: Int64, amax_rows: Int32, amax_cols: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, group_idx, _ = cute.arch.block_idx()
        store_scale(amax_base, scale_base, scale_t_base, Int64(group_idx), bx * NUM_THREADS + tidx, amax_rows, amax_cols)


@cute.jit
def store_scale(amax_base: Int64, scale_base: Int64, scale_t_base: Int64, group_idx: Int64, tile_idx: Int32, amax_rows: Int32, amax_cols: Int32):
    num_tiles = amax_rows * amax_cols
    tile_ok = Int32(tile_idx < num_tiles)
    group_start = group_idx * Int64(num_tiles)
    scale_bits = compute_scale_bits(ld_scalar_ordered(amax_base + (group_start + Int64(tile_idx)) * 4, tile_ok, 4))
    tile_row = tile_idx // amax_cols
    tile_col = tile_idx - tile_row * amax_cols
    st_b32(scale_base + (group_start + Int64(tile_idx)) * 4, scale_bits, tile_ok)
    st_b32(scale_t_base + (group_start + Int64(tile_col) * Int64(amax_rows) + Int64(tile_row)) * 4, scale_bits, tile_ok)


def fp8_scales_sm90(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """
    Write scales and transposed scales in place from the reduced amax grid.
    """
    key = ("scales",)
    fn = COMPILED.get(key)
    if fn is None:
        fake_grids = [make_fake_row_major(cutlass.Float32, 3, 4) for _ in range(3)]
        fn = cute.compile(Scales(), *fake_grids, fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(amax, scale, scale_t)


PACK_THREADS = 128
PACK_ROW_TILES = 8


class PackTranspose:
    def __init__(self, mode: str, load_bytes: int = 16) -> None:
        self.mode, self.load_bytes = mode, load_bytes
        self.num_threads = PACK_THREADS if mode in ("wide", "wide_banded", "gathered") else NUM_THREADS

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        G, N, K = mX.shape[0], mX.shape[1], mX.shape[2]
        shard_rows, shard_stride = Int32(N), Int64(mX.stride[0])
        num_row_tiles, num_col_tiles = (N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        if const_expr(self.mode == "gathered"):
            grid = [num_row_tiles, num_col_tiles, G]
            G, N = 1, G * N
        elif const_expr(self.mode == "wide_banded"):
            grid = [num_row_tiles * num_col_tiles, 1, G]
        else:
            grid = [num_row_tiles, num_col_tiles, G]
        self.kernel(mX.iterator.toint(), mOut.iterator.toint(), Int32(N), Int32(K), Int32(G), shard_rows, shard_stride).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x_base: Int64, out_base: Int64, N: Int32, K: Int32, G: Int32, shard_rows: Int32, shard_stride: Int64):
        if const_expr(self.mode in ("wide", "wide_banded", "gathered")):
            self.pack_wide_tile(x_base, out_base, N, K, G, shard_rows, shard_stride)
        else:
            self.pack_narrow_tile(x_base, out_base, N, K, G)

    @cute.jit
    def pack_wide_tile(self, x_base: Int64, out_base: Int64, N: Int32, K: Int32, G: Int32, shard_rows: Int32, shard_stride: Int64):
        """
        Assign each thread an 8x16 block and use coalesced 16-byte transposed stores.
        """
        bx, by, bz = cute.arch.block_idx()
        if const_expr(self.mode == "wide_banded"):
            num_col_tiles = K // BLOCK
            column_band = bx // PACK_ROW_TILES
            row_band = column_band // num_col_tiles
            bx, by = row_band * PACK_ROW_TILES + bx % PACK_ROW_TILES, column_band - row_band * num_col_tiles
        if const_expr(self.mode == "gathered"):
            x_base = x_base + Int64(bz) * (shard_stride - Int64(shard_rows) * Int64(K))
            bx, bz = bx + bz * (shard_rows // BLOCK), 0
        tidx, _, _ = cute.arch.thread_idx()
        warp, lane = tidx // 32, tidx % 32
        thread_row_offset, thread_col_offset = warp * 32 + (lane // 8) * 8, (lane % 8) * 16
        tile_row_start, tile_col_start = bx * BLOCK, by * BLOCK
        num_group_elements = Int64(N) * Int64(K)
        group_start = Int64(bz) * num_group_elements
        col_ok = Int32(tile_col_start + thread_col_offset < K)
        tile = TileCoordinates(tile_row_start, tile_col_start, thread_row_offset, thread_col_offset)

        words = self.load_wide_rows(x_base, group_start, tile, N, K, col_ok)
        self.store_wide_rows(out_base, group_start, tile, N, K, col_ok, words)
        self.store_wide_transpose(out_base, G, num_group_elements, group_start, tile, N, K, words)

    @cute.jit
    def load_wide_rows(self, x_base: Int64, group_start: Int64, tile: TileCoordinates, N: Int32, K: Int32, col_ok: Int32) -> list:
        tile_row_start, tile_col_start = tile.tile_row_start, tile.tile_col_start
        thread_row_offset, thread_col_offset = tile.thread_row_offset, tile.thread_col_offset
        words = []
        for i in cutlass.range_constexpr(8):
            row = tile_row_start + thread_row_offset + i
            row_addr = x_base + group_start + Int64(row) * Int64(K) + Int64(tile_col_start + thread_col_offset)
            load_ok = Int32(row < N) & col_ok
            if const_expr(self.load_bytes == 16):
                packed = ld_v4_ordered(row_addr, load_ok)
            elif const_expr(self.load_bytes == 8):
                lo, hi = ld_v2(row_addr, load_ok), ld_v2(row_addr + 8, load_ok)
                packed = [lo[0], lo[1], hi[0], hi[1]]
            else:
                packed = [ld_scalar_ordered(row_addr + word_idx * 4, load_ok, 4) for word_idx in range(4)]
            words.append(packed)
        return words

    @cute.jit
    def store_wide_rows(self, out_base: Int64, group_start: Int64, tile: TileCoordinates, N: Int32, K: Int32, col_ok: Int32, words: list) -> None:
        tile_row_start, tile_col_start = tile.tile_row_start, tile.tile_col_start
        thread_row_offset, thread_col_offset = tile.thread_row_offset, tile.thread_col_offset
        for i in cutlass.range_constexpr(8):
            row = tile_row_start + thread_row_offset + i
            row_addr = out_base + group_start + Int64(row) * Int64(K) + Int64(tile_col_start + thread_col_offset)
            st_v4(row_addr, words[i], Int32(row < N) & col_ok)

    @cute.jit
    def store_wide_transpose(self, out_base: Int64, G: Int32, num_group_elements: Int64, group_start: Int64, tile: TileCoordinates, N: Int32, K: Int32, words: list) -> None:
        tile_row_start, tile_col_start = tile.tile_row_start, tile.tile_col_start
        thread_row_offset, thread_col_offset = tile.thread_row_offset, tile.thread_col_offset
        words_lo, words_hi = transpose8([[words[i][0], words[i][1]] for i in range(8)]), transpose8([[words[i][2], words[i][3]] for i in range(8)])
        shared_alloc = cutlass.memory.SmemAllocator()
        s_tile_words = shared_alloc.allocate_tensor(Uint32, cute.make_layout(TILE_BYTES // 4), byte_alignment=128)  # (swizzled word,)
        s_tile = s_tile_words.iterator.toint()
        stage_output_block(s_tile, words_lo, thread_col_offset, thread_row_offset)
        stage_output_block(s_tile, words_hi, thread_col_offset + 8, thread_row_offset)
        cute.arch.sync_threads()
        out_t_tile_addr = out_base + Int64(G) * num_group_elements + group_start + Int64(tile_col_start) * Int64(N) + Int64(tile_row_start)
        store_staged_tile(s_tile, out_t_tile_addr, K - tile_col_start, N - tile_row_start, N, PACK_THREADS)

    @cute.jit
    def pack_narrow_tile(self, x_base: Int64, out_base: Int64, N: Int32, K: Int32, G: Int32):
        bx, by, bz = cute.arch.block_idx()
        _, _, _, thread_row_offset, thread_col_offset = thread_coords()
        row_start, col_start = bx * BLOCK + thread_row_offset, by * BLOCK + thread_col_offset
        num_group_elements = Int64(N) * Int64(K)
        group_start = Int64(bz) * num_group_elements
        is_aligned = const_expr(self.mode != "scalar")
        words = load_rows(x_base + group_start, (row_start, col_start), N, K, Int64(K), 1, False, is_aligned)
        words_t = transpose8(words)
        out_t_base = out_base + Int64(G) * num_group_elements + group_start
        store_rows(out_base + group_start, words, row_start, col_start, N, K, is_aligned)
        store_rows(out_t_base, words_t, col_start, row_start, K, N, is_aligned)


def fp8_pack_transpose_sm90(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    Write the FP8 weight and its transpose into successive regions of out.
    """
    _, N, K = x.shape
    if x.numel() == 0:
        return
    if N % 16 == 0 and K % 16 == 0 and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0:
        # Visit nearby columns before moving on to the next complete band of tile rows.
        mode = "wide_banded" if N % (BLOCK * PACK_ROW_TILES) == 0 and K % BLOCK == 0 else "wide"
    elif N % 8 == 0 and K % 8 == 0 and x.data_ptr() % 8 == 0 and out.data_ptr() % 8 == 0:
        mode = "narrow_aligned"
    else:
        mode = "scalar"
    key = ("pack", mode, x.dtype, out.dtype)
    fn = COMPILED.get(key)
    if fn is None:
        fn = cute.compile(PackTranspose(mode), make_fake_row_major(TORCH2CUTE[x.dtype], 3, 1), make_fake_row_major(TORCH2CUTE[out.dtype], 1, 1), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(x, out)


def fp8_pack_gathered_transpose_sm90(shards: torch.Tensor, out: torch.Tensor) -> None:
    """
    Pack the rank-major byte shards directly into their concatenation and transpose.
    """
    alignment = shards.data_ptr() | shards.stride(0)
    load_bytes = min(16, alignment & -alignment)
    key = ("pack", "gathered", load_bytes)
    fn = COMPILED.get(key)
    if fn is None:
        num_ranks, shard_rows, K = cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=16)
        data = fake(Uint8, (num_ranks, shard_rows, K), (cute.sym_int64(divisibility=load_bytes), K, 1), load_bytes)
        fn = cute.compile(PackTranspose("gathered", load_bytes), data, make_fake_row_major(Uint8, 1, 16), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(shards, out)


DESCRIPTOR_WORDS = 8  # int64 words per descriptor row


class ShardAmaxMulti:
    """
    Look up each CTA's shard in a table of two descriptor rows per shard.
    """

    @cute.jit
    def __call__(self, mTable: cute.Tensor, mShardForCta: cute.Tensor, num_ctas: Int32, stream: cuda.CUstream):
        table_base = mTable.iterator.toint()  # (shard descriptor row, field)
        self.kernel(table_base, mShardForCta.iterator.toint()).launch(grid=[num_ctas, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table_base: Int64, shard_for_cta_base: Int64):
        bx, _, _ = cute.arch.block_idx()
        shard_base, amax_addr, tile = load_shard_descriptor(table_base, shard_for_cta_base, bx)
        process_shard_tile_by_row(shard_base, amax_addr, amax_addr, tile, False)


def load_shard_descriptor(table_base: Int64, shard_for_cta_base: Int64, bx: Int32) -> tuple:
    shard_idx = Int32(ld_scalar_ordered(shard_for_cta_base + Int64(bx) * 4, Int32(1), 4))
    descriptor_addr = table_base + Int64(shard_idx) * 2 * DESCRIPTOR_WORDS * 8
    shard_base, amax_base = ld_u64_nc(descriptor_addr), ld_u64_nc(descriptor_addr + 8)
    local_rows, K = Int32(ld_u64_nc(descriptor_addr + 16)), Int32(ld_u64_nc(descriptor_addr + 24))
    row_offset, first_tile_row_idx = Int32(ld_u64_nc(descriptor_addr + 32)), Int32(ld_u64_nc(descriptor_addr + 40))
    amax_group_start, first_cta_idx = ld_u64_nc(descriptor_addr + 48), Int32(ld_u64_nc(descriptor_addr + 56))
    amax_rows, amax_cols = Int32(ld_u64_nc(descriptor_addr + 64)), Int32(ld_u64_nc(descriptor_addr + 72))
    num_col_tiles, num_row_tiles = Int32(ld_u64_nc(descriptor_addr + 80)), Int32(ld_u64_nc(descriptor_addr + 88))
    local_cta_idx = bx - first_cta_idx
    tile_col_idx, row_group_idx = local_cta_idx % num_col_tiles, local_cta_idx // num_col_tiles
    local_tile_row, group_idx = row_group_idx % num_row_tiles, row_group_idx // num_row_tiles
    tile_row_idx = first_tile_row_idx + local_tile_row
    amax_idx = amax_group_start + (Int64(group_idx) * Int64(amax_rows) + Int64(tile_row_idx)) * Int64(amax_cols) + Int64(tile_col_idx)
    return shard_base, amax_base + amax_idx * 4, ShardTileCoordinates(local_rows, K, tile_row_idx, tile_col_idx, group_idx, row_offset)


class FP8ShardAmaxGroup:
    """
    Capture shard and amax addresses for repeated launches while their storage stays live.
    """

    def __init__(self, shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> None:
        descriptors, shard_for_cta, self.fallback = [], [], []
        for shard, amax, row_offset, group_offset in zip(shards, amaxes, row_offsets, group_offsets):
            num_local_groups, local_rows, K = shard.shape
            if shard.numel() == 0:
                continue
            if K % 4 != 0 or shard.data_ptr() % 16 != 0:
                self.fallback.append((shard, amax, row_offset, group_offset))
                continue
            _, amax_rows, amax_cols = amax.shape
            first_tile_row_idx = row_offset // BLOCK
            num_row_tiles = (row_offset + local_rows + BLOCK - 1) // BLOCK - first_tile_row_idx
            num_col_tiles, shard_idx = (K + BLOCK - 1) // BLOCK, len(descriptors) // 2
            first_cta_idx = len(shard_for_cta)
            descriptors.append([shard.data_ptr(), amax.data_ptr(), local_rows, K, row_offset, first_tile_row_idx, group_offset * amax_rows * amax_cols, first_cta_idx])
            descriptors.append([amax_rows, amax_cols, num_col_tiles, num_row_tiles, 0, 0, 0, 0])
            shard_for_cta.extend([shard_idx] * (num_row_tiles * num_col_tiles * num_local_groups))
        self.num_ctas = len(shard_for_cta)
        device = shards[0].device
        self.table = torch.tensor(descriptors or [[0] * DESCRIPTOR_WORDS], dtype=torch.int64, device=device)
        self.shard_for_cta = torch.tensor(shard_for_cta, dtype=torch.int32, device=device)
        key = ("amax_multi",)
        fn = COMPILED.get(key)
        if fn is None:
            fn = cute.compile(ShardAmaxMulti(), make_fake_row_major(cutlass.Int64, 2, 8), make_fake_row_major(Int32, 1, 4), Int32(0), fake_stream(), options="--enable-tvm-ffi")
            COMPILED[key] = fn
        self.fn = fn

    def __call__(self) -> None:
        if self.num_ctas:
            self.fn(self.table, self.shard_for_cta, self.num_ctas)
        for shard, amax, row_offset, group_offset in self.fallback:
            fp8_shard_amax_sm90(shard, amax, row_offset, group_offset)


class ScalesMulti:
    """
    Look up each CTA's grid in a table of one descriptor row per scale grid.
    """

    @cute.jit
    def __call__(self, mTable: cute.Tensor, mGridForCta: cute.Tensor, num_ctas: Int32, stream: cuda.CUstream):
        table_base = mTable.iterator.toint()  # (scale grid, field)
        self.kernel(table_base, mGridForCta.iterator.toint()).launch(grid=[num_ctas, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table_base: Int64, grid_for_cta_base: Int64):
        bx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        amax_base, scale_base, scale_t_base, group_idx, group_cta_idx, amax_rows, amax_cols = load_scale_descriptor(table_base, grid_for_cta_base, bx)
        store_scale(amax_base, scale_base, scale_t_base, Int64(group_idx), group_cta_idx * NUM_THREADS + tidx, amax_rows, amax_cols)


@cute.jit
def load_scale_descriptor(table_base: Int64, grid_for_cta_base: Int64, bx: Int32) -> tuple:
    grid_idx = Int32(ld_scalar_ordered(grid_for_cta_base + Int64(bx) * 4, Int32(1), 4))
    descriptor_addr = table_base + Int64(grid_idx) * DESCRIPTOR_WORDS * 8
    amax_base, scale_base = ld_u64_nc(descriptor_addr), ld_u64_nc(descriptor_addr + 8)
    scale_t_base = ld_u64_nc(descriptor_addr + 16)
    amax_rows, amax_cols = Int32(ld_u64_nc(descriptor_addr + 24)), Int32(ld_u64_nc(descriptor_addr + 32))
    num_ctas_per_group, first_cta_idx = Int32(ld_u64_nc(descriptor_addr + 40)), Int32(ld_u64_nc(descriptor_addr + 56))
    local_cta_idx = bx - first_cta_idx
    group_idx = local_cta_idx // num_ctas_per_group
    group_cta_idx = local_cta_idx - group_idx * num_ctas_per_group
    return amax_base, scale_base, scale_t_base, group_idx, group_cta_idx, amax_rows, amax_cols


class FP8ScalesGroup:
    """
    Capture amax and scale addresses for repeated launches while their storage stays live.
    """

    def __init__(self, amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> None:
        grid_table, grid_for_cta = [], []
        for amax, scale, scale_t in zip(amaxes, scales, scales_t):
            G, amax_rows, amax_cols = amax.shape
            num_ctas_per_group = (amax_rows * amax_cols + NUM_THREADS - 1) // NUM_THREADS
            grid_idx, first_cta_idx = len(grid_table), len(grid_for_cta)
            grid_table.append([amax.data_ptr(), scale.data_ptr(), scale_t.data_ptr(), amax_rows, amax_cols, num_ctas_per_group, 0, first_cta_idx])
            grid_for_cta.extend([grid_idx] * (G * num_ctas_per_group))
        self.num_ctas = len(grid_for_cta)
        device = amaxes[0].device
        self.table, self.grid_for_cta = torch.tensor(grid_table, dtype=torch.int64, device=device), torch.tensor(grid_for_cta, dtype=torch.int32, device=device)
        key = ("scales_multi",)
        fn = COMPILED.get(key)
        if fn is None:
            fn = cute.compile(ScalesMulti(), make_fake_row_major(cutlass.Int64, 2, 8), make_fake_row_major(Int32, 1, 4), Int32(0), fake_stream(), options="--enable-tvm-ffi")
            COMPILED[key] = fn
        self.fn = fn

    def __call__(self) -> None:
        if self.num_ctas:
            self.fn(self.table, self.grid_for_cta, self.num_ctas)


def fp8_shard_amax_multi_sm90(shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> FP8ShardAmaxGroup:
    """
    Prepare an in-place amax launch for a fixed list of shards.
    """
    return FP8ShardAmaxGroup(shards, amaxes, row_offsets, group_offsets)


def fp8_scales_multi_sm90(amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> FP8ScalesGroup:
    """
    Prepare an in-place scale and transpose launch for a fixed list of grids.
    """
    return FP8ScalesGroup(amaxes, scales, scales_t)
