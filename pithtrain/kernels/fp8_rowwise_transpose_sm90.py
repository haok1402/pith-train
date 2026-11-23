"""
Rowwise FP8 quantization of an input and its transpose on SM90.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, asm_args_last, bf16_hi, bf16_lo, cp_async16, pack_bf16, lds_b32, lds_v4, max2, prmt, st_b32, st_b8, st_v2, st_v4, sts_b32_raw, sts_v2, unpack
from pithtrain.kernels.fp8_e8m0_scale_sm90 import compute_scale_rcp_bits, quantize_four_from_bits, reduce_bf16x2_amax4
from pithtrain.kernels.fp8_scale_layout import mn_major_scale

from pithtrain.kernels.silu_mul_sm90 import SIGMOID_PTX

__all__ = ["fp8_quant_grad_sm90", "fp8_quant_delta_grad_sm90"]

TILE = 128
THREADS = 256

# A nonzero mantissa rounds the exponent up to the next power of two.
SCALE_PTX = """{
.reg .f32 a, s;
.reg .b32 b, m;
.reg .pred p;
mov.b32 a, $1;
max.f32 a, a, 0f38D1B717;
mul.rn.f32 s, a, 0f3B124925;
mov.b32 b, s;
and.b32 m, b, 0x007FFFFF;
and.b32 b, b, 0x7F800000;
setp.ne.b32 p, m, 0;
@p add.s32 b, b, 0x00800000;
max.s32 b, b, 0x00800000;
mov.b32 $0, b;
}"""

# Shifting each BF16 half into FP32 is exact, as is multiplication by a power of two.
QUANT_ROW_PTX = """{
.reg .f32 a0, a1, a2, a3, a4, a5, a6, a7, r;
.reg .b16 h0, h1, h2, h3;
mov.b32 r, $6;
shl.b32 $0, $2, 16; mov.b32 a0, $0; and.b32 $0, $2, 0xFFFF0000; mov.b32 a1, $0;
shl.b32 $0, $3, 16; mov.b32 a2, $0; and.b32 $0, $3, 0xFFFF0000; mov.b32 a3, $0;
shl.b32 $0, $4, 16; mov.b32 a4, $0; and.b32 $0, $4, 0xFFFF0000; mov.b32 a5, $0;
shl.b32 $0, $5, 16; mov.b32 a6, $0; and.b32 $0, $5, 0xFFFF0000; mov.b32 a7, $0;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
mul.rn.f32 a4, a4, r; mul.rn.f32 a5, a5, r; mul.rn.f32 a6, a6, r; mul.rn.f32 a7, a7, r;
cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;
cvt.rn.satfinite.e4m3x2.f32 h2, a5, a4;
cvt.rn.satfinite.e4m3x2.f32 h3, a7, a6;
mov.b32 $0, {h0, h1};
mov.b32 $1, {h2, h3};
}"""

# EXTRACT selects the BF16 half belonging to this column from each of four rows.
QUANT_COL_PTX = """{
.reg .f32 a0, a1, a2, a3, r;
.reg .b16 h0, h1;
mov.b32 r, $5;
EXTRACT $0, $1; mov.b32 a0, $0;
EXTRACT $0, $2; mov.b32 a1, $0;
EXTRACT $0, $3; mov.b32 a2, $0;
EXTRACT $0, $4; mov.b32 a3, $0;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;
mov.b32 $0, {h0, h1};
}"""
QUANT_COL_EXTRACT = ("shl.b32 {}, {}, 16;", "and.b32 {}, {}, 0xFFFF0000;")

QUANT_ROW_F16_PTX = """{
.reg .f32 a0, a1, a2, a3, a4, a5, a6, a7, r;
.reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;
mov.b32 r, $6;
mov.b32 {l0, h0}, $2; mov.b32 {l1, h1}, $3; mov.b32 {l2, h2}, $4; mov.b32 {l3, h3}, $5;
cvt.f32.f16 a0, l0; cvt.f32.f16 a1, h0; cvt.f32.f16 a2, l1; cvt.f32.f16 a3, h1;
cvt.f32.f16 a4, l2; cvt.f32.f16 a5, h2; cvt.f32.f16 a6, l3; cvt.f32.f16 a7, h3;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
mul.rn.f32 a4, a4, r; mul.rn.f32 a5, a5, r; mul.rn.f32 a6, a6, r; mul.rn.f32 a7, a7, r;
cvt.rn.satfinite.e4m3x2.f32 l0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 l1, a3, a2;
cvt.rn.satfinite.e4m3x2.f32 l2, a5, a4;
cvt.rn.satfinite.e4m3x2.f32 l3, a7, a6;
mov.b32 $0, {l0, l1};
mov.b32 $1, {l2, l3};
}"""

QUANT_ROW_F32_PTX = """{
.reg .f32 a0, a1, a2, a3, a4, a5, a6, a7, r;
.reg .b16 q0, q1, q2, q3;
mov.b32 r, $10;
mov.b32 a0, $2; mov.b32 a1, $3; mov.b32 a2, $4; mov.b32 a3, $5;
mov.b32 a4, $6; mov.b32 a5, $7; mov.b32 a6, $8; mov.b32 a7, $9;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
mul.rn.f32 a4, a4, r; mul.rn.f32 a5, a5, r; mul.rn.f32 a6, a6, r; mul.rn.f32 a7, a7, r;
cvt.rn.satfinite.e4m3x2.f32 q0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 q1, a3, a2;
cvt.rn.satfinite.e4m3x2.f32 q2, a5, a4;
cvt.rn.satfinite.e4m3x2.f32 q3, a7, a6;
mov.b32 $0, {q0, q1};
mov.b32 $1, {q2, q3};
}"""

# HALF is l (low fp16 of each word) or h (high)
QUANT_COL_F16_PTX = """{
.reg .f32 a0, a1, a2, a3, r;
.reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;
mov.b32 r, $5;
mov.b32 {l0, h0}, $1; mov.b32 {l1, h1}, $2; mov.b32 {l2, h2}, $3; mov.b32 {l3, h3}, $4;
cvt.f32.f16 a0, HALF0; cvt.f32.f16 a1, HALF1; cvt.f32.f16 a2, HALF2; cvt.f32.f16 a3, HALF3;
mul.rn.f32 a0, a0, r; mul.rn.f32 a1, a1, r; mul.rn.f32 a2, a2, r; mul.rn.f32 a3, a3, r;
cvt.rn.satfinite.e4m3x2.f32 l0, a1, a0;
cvt.rn.satfinite.e4m3x2.f32 l1, a3, a2;
mov.b32 $0, {l0, l1};
}"""

F16_TO_F32_PTX = """{
.reg .b16 l, h;
.reg .f32 f;
mov.b32 {l, h}, $1;
cvt.f32.f16 f, HALF;
mov.b32 $0, f;
}"""

LD_V4_NC_PTX = """{
.reg .pred p;
setp.ne.b32 p, $5, 0;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];
}"""

LD_V4_NC_FULL_PTX = "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];"

LD_B32_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
mov.b32 $0, 0;
@p ld.global.nc.b32 $0, [$1];
}"""

LD_U16_PTX = """{
.reg .pred p;
.reg .b16 h;
setp.ne.b32 p, $2, 0;
mov.b16 h, 0;
@p ld.global.nc.b16 h, [$1];
cvt.u32.u16 $0, h;
}"""

# Select the widest legal store at runtime so unaligned rows can avoid byte stores when possible.
ST8_ANY_PTX = """{
.reg .pred p, p8, p4, p2, p1;
.reg .b32 al, t;
setp.ne.b32 p, $3, 0;
cvt.u32.u64 al, $0;
and.b32 t, al, 7;
setp.eq.and.b32 p8, t, 0, p;
setp.eq.and.b32 p4, t, 4, p;
and.b32 t, al, 3;
setp.eq.and.b32 p2, t, 2, p;
and.b32 t, al, 1;
setp.eq.and.b32 p1, t, 1, p;
@p8 st.global.v2.b32 [$0], {$1, $2};
@p4 st.global.b32 [$0], $1;
@p4 st.global.b32 [$0+4], $2;
@p2 st.global.b16 [$0], $1;
@p2 st.global.b16 [$0+4], $2;
@p1 st.global.b8 [$0], $1;
@p1 st.global.b8 [$0+4], $2;
shr.b32 t, $1, 8;
@p1 st.global.b8 [$0+1], t;
shr.b32 t, $1, 16;
@p1 st.global.b8 [$0+2], t;
@p2 st.global.b16 [$0+2], t;
shr.b32 t, $1, 24;
@p1 st.global.b8 [$0+3], t;
shr.b32 t, $2, 8;
@p1 st.global.b8 [$0+5], t;
shr.b32 t, $2, 16;
@p1 st.global.b8 [$0+6], t;
@p2 st.global.b16 [$0+6], t;
shr.b32 t, $2, 24;
@p1 st.global.b8 [$0+7], t;
}"""


def emit_u32(ptx: str, constraints: str, operands: list) -> Uint32:
    return Uint32(asm_args_last(T.i32(), ptx, constraints, operands))


def emit_words(ptx: str, constraints: str, operands: list, num_words: int) -> list:
    packed_type = llvm.StructType.get_literal([T.i32()] * num_words)
    return list(unpack(asm_args_last(packed_type, ptx, constraints, operands), num_words, Uint32))


@dsl_user_op
def e8m0_scale_bits(amax_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    return emit_u32(SCALE_PTX, "=r,r", [amax_bits.ir_value(loc=loc, ip=ip)])


@dsl_user_op
def load_v4_nc_pred(x_addr: Int64, load_ok: Int32, *, loc=None, ip=None) -> list:
    return emit_words(LD_V4_NC_PTX, "=r,=r,=r,=r,l,r", [x_addr.ir_value(loc=loc, ip=ip), load_ok.ir_value(loc=loc, ip=ip)], 4)


def quantize_column(words: list, rcp_bits: Uint32, half_idx: int, input_format: str) -> Uint32:
    """
    Pack four FP8 bytes from one column, selecting a half of each packed 16-bit input.
    """
    if input_format == "f32":
        return quantize_four_from_bits(words[0], words[1], words[2], words[3], rcp_bits)
    if input_format == "f16":
        ptx = QUANT_COL_F16_PTX
        for j in range(4):
            ptx = ptx.replace(f"HALF{j}", f"{'lh'[half_idx]}{j}")
        return emit_u32(ptx, "=r,r,r,r,r,r", [word.ir_value() for word in words] + [rcp_bits.ir_value()])
    extract = QUANT_COL_EXTRACT[half_idx].format("$0", "{}")
    ptx = QUANT_COL_PTX
    for j in range(4):
        ptx = ptx.replace(f"EXTRACT $0, ${j + 1};", extract.format(f"${j + 1}"))
    return emit_u32(ptx, "=&r,r,r,r,r,r", [word.ir_value() for word in words] + [rcp_bits.ir_value()])


def quantize_row(words: list, rcp_bits: Uint32, input_format: str) -> list:
    """
    Pack eight FP8 bytes from consecutive columns of one row.
    """
    if input_format == "f32":
        return emit_words(QUANT_ROW_F32_PTX, "=r,=r,r,r,r,r,r,r,r,r,r", [word.ir_value() for word in words] + [rcp_bits.ir_value()], 2)
    if input_format == "f16":
        return emit_words(QUANT_ROW_F16_PTX, "=r,=r,r,r,r,r,r", [word.ir_value() for word in words] + [rcp_bits.ir_value()], 2)
    return emit_words(QUANT_ROW_PTX, "=&r,=&r,r,r,r,r,r", [word.ir_value() for word in words] + [rcp_bits.ir_value()], 2)


def max_magnitude_words(first: Uint32, second: Uint32, input_format: str) -> Uint32:
    """
    Take the NaN-ignoring maximum of packed magnitudes in the input format.
    """
    if input_format == "bf16":
        return max2(first, second)
    return emit_u32("max.f16x2 $0, $1, $2;" if input_format == "f16" else "max.f32 $0, $1, $2;", "=r,r,r", [first.ir_value(), second.ir_value()])


def widen_word_bits(word: Uint32, half_idx: int, input_format: str) -> Uint32:
    """
    Return the FP32 bits of one element selected from a packed word.
    """
    if input_format == "f32":
        return word
    if input_format == "f16":
        return emit_u32(F16_TO_F32_PTX.replace("HALF", "lh"[half_idx]), "=r,r", [word.ir_value()])
    return word << 16 if half_idx == 0 else word & Uint32(0xFFFF0000)


@dsl_user_op
def store_eight_bytes(out_addr: Int64, word_lo: Uint32, word_hi: Uint32, store_ok: Int32, *, loc=None, ip=None) -> None:
    asm_args_last(None, ST8_ANY_PTX, "l,r,r,r", [out_addr.ir_value(loc=loc, ip=ip), word_lo.ir_value(loc=loc, ip=ip), word_hi.ir_value(loc=loc, ip=ip), store_ok.ir_value(loc=loc, ip=ip)], side_effects=True)


def make_predicate(condition) -> Int32:
    return Int32(cutlass.select_(condition, Int32(1), Int32(0)))


class SubBlock:
    """
    Coordinates and bounds masks for one thread's 8 by 8 block.
    """

    def __init__(self, tidx: Int32, by: Int32, bx: Int32, M: Int32, N: Int32, use_full_tiles: bool) -> None:
        self.warp, self.lane = tidx // 32, tidx % 32
        self.col_group_idx = self.lane % 16
        self.row_group_idx = self.warp * 2 + self.lane // 16
        self.by, self.bx = by, bx
        self.row_start, self.col_start = by * TILE + self.row_group_idx * 8, bx * TILE + self.col_group_idx * 8
        self.use_full_tiles = use_full_tiles
        if use_full_tiles:
            self.rows_ok = self.cols_ok = [Int32(1)] * 8
            self.full_cols_ok = Int32(1)
        else:
            self.rows_ok = [make_predicate(self.row_start + i < M) for i in range(8)]
            self.cols_ok = [make_predicate(self.col_start + j < N) for j in range(8)]
            self.full_cols_ok = make_predicate(self.col_start + 8 <= N)


def load_rows(x_base: Int64, row_stride: Int64, sub_block: SubBlock, input_format: str) -> list:
    """
    Load aligned 16-byte groups into 32-bit words, zeroing groups outside the input.
    """
    row_start, col_start, rows_ok, cols_ok = sub_block.row_start, sub_block.col_start, sub_block.rows_ok, sub_block.cols_ok
    use_full_tiles = sub_block.use_full_tiles
    rows = []
    for i in range(8):
        elem_idx = Int64(row_start + i) * row_stride + Int64(col_start)
        if use_full_tiles:
            rows.append(emit_words(LD_V4_NC_FULL_PTX, "=r,=r,=r,=r,l", [(x_base + elem_idx * 2).ir_value()], 4))
        elif input_format == "f32":
            words = []
            for half_idx in range(2):
                load_ok = rows_ok[i] & cols_ok[4 * half_idx]
                byte_offset = cutlass.select_(load_ok != 0, elem_idx * 4 + 16 * half_idx, Int64(0))
                words += load_v4_nc_pred(x_base + byte_offset, load_ok)
            rows.append(words)
        else:
            load_ok = rows_ok[i] & cols_ok[0]
            byte_offset = cutlass.select_(load_ok != 0, elem_idx * 2, Int64(0))
            rows.append(load_v4_nc_pred(x_base + byte_offset, load_ok))
    return rows


def stage_input_tile(x_base: Int64, row_stride: Int64, col_stride: Int64, sub_block: SubBlock, input_format: str, s_raw: Int32, M: Int32, N: Int32) -> None:
    lane, warp, by, bx = sub_block.lane, sub_block.warp, sub_block.by, sub_block.bx
    elem_bytes = 4 if input_format == "f32" else 2
    tile_row_start, tile_col_start = by * TILE, bx * TILE
    col_offsets, cols_ok = [], []
    # Each warp gathers 64 neighboring columns per pass for coalescing when columns are contiguous.
    for col_idx in range(4):
        x_col_idx = tile_col_start + 64 * (col_idx // 2) + 2 * lane + col_idx % 2
        col_offsets.append(Int64(x_col_idx) * col_stride)
        cols_ok.append(make_predicate(x_col_idx < N))

    for warp_row_idx in range(16):
        tile_row_idx = warp * 16 + warp_row_idx
        row_offset = Int64(tile_row_start + tile_row_idx) * row_stride
        row_ok = make_predicate(tile_row_start + tile_row_idx < M)
        for pass_idx in range(2):
            words = []
            for half_idx in range(2):
                load_ok = row_ok & cols_ok[2 * pass_idx + half_idx]
                byte_offset = cutlass.select_(load_ok != 0, (row_offset + col_offsets[2 * pass_idx + half_idx]) * elem_bytes, Int64(0))
                words.append(emit_u32(LD_B32_PTX if input_format == "f32" else LD_U16_PTX, "=r,l,r", [(x_base + byte_offset).ir_value(), load_ok.ir_value()]))
            s_addr = s_raw + (tile_row_idx * TILE + 64 * pass_idx + 2 * lane) * elem_bytes
            if input_format == "f32":
                sts_v2(s_addr, words[0], words[1])
            else:
                sts_b32_raw(s_addr, words[0] | (words[1] << 16))
    cute.arch.barrier()


def load_staged_rows(s_raw: Int32, sub_block: SubBlock, input_format: str) -> list:
    row_group_idx, col_group_idx = sub_block.row_group_idx, sub_block.col_group_idx
    elem_bytes = 4 if input_format == "f32" else 2
    rows = []
    for i in range(8):
        s_addr = s_raw + ((row_group_idx * 8 + i) * TILE + col_group_idx * 8) * elem_bytes
        rows.append(list(lds_v4(s_addr)) + (list(lds_v4(s_addr + 16)) if input_format == "f32" else []))
    return rows


def get_magnitude_bits(rows: list, num_words: int, input_format: str) -> list:
    abs_mask = Uint32(0x7FFFFFFF) if input_format == "f32" else Uint32(0x7FFF7FFF)
    return [[rows[i][k] & abs_mask for k in range(num_words)] for i in range(8)]


def share_col_amax(row_magnitude_bits: list, sub_block: SubBlock, s_col_amax: Int32, input_format: str) -> None:
    """
    Share each warp's partial column maxima for the cross-warp reduction.
    """
    warp, lane, col_group_idx = sub_block.warp, sub_block.lane, sub_block.col_group_idx
    num_words = len(row_magnitude_bits[0])
    col_amax = []
    for word_idx in range(num_words):
        acc_amax = row_magnitude_bits[0][word_idx]
        for row_idx in range(1, 8):
            acc_amax = max_magnitude_words(acc_amax, row_magnitude_bits[row_idx][word_idx], input_format)
        peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(acc_amax), offset=16))
        col_amax.append(max_magnitude_words(acc_amax, peer_amax, input_format))

    for quad_idx in range(num_words // 4):
        s_addr = s_col_amax + (warp * 16 * num_words + col_group_idx * num_words + 4 * quad_idx) * 4
        operands = [s_addr.ir_value()] + [amax.ir_value() for amax in col_amax[4 * quad_idx : 4 * quad_idx + 4]]
        operands += [make_predicate(lane < 16).ir_value()]
        asm_args_last(None, "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n}", "r,r,r,r,r,r", operands, side_effects=True)


def reduce_row_amax(row_magnitude_bits: list, input_format: str) -> list:
    """
    Reduce over 16 lanes per row and return the FP32 bits of each row maximum.
    """
    if input_format == "f32":
        row_amax = []
        for i in range(8):
            acc_amax = row_magnitude_bits[i][0]
            for k in range(1, 8):
                acc_amax = max_magnitude_words(acc_amax, row_magnitude_bits[i][k], input_format)
            row_amax.append(acc_amax)
        for step in range(4):
            for i in range(8):
                peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(row_amax[i]), offset=1 << step))
                row_amax[i] = max_magnitude_words(row_amax[i], peer_amax, input_format)
        return row_amax
    row_amax = [max_magnitude_words(max_magnitude_words(row_magnitude_bits[i][0], row_magnitude_bits[i][1], input_format), max_magnitude_words(row_magnitude_bits[i][2], row_magnitude_bits[i][3], input_format), input_format) for i in range(8)]
    row_pair_amax = []
    for pair_idx in range(4):
        amax_lo, amax_hi = prmt(row_amax[2 * pair_idx], row_amax[2 * pair_idx + 1], 0x5410), prmt(row_amax[2 * pair_idx], row_amax[2 * pair_idx + 1], 0x7632)
        row_pair_amax.append(max_magnitude_words(amax_lo, amax_hi, input_format))
    for step in range(4):
        for pair_idx in range(4):
            peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(row_pair_amax[pair_idx]), offset=1 << step))
            row_pair_amax[pair_idx] = max_magnitude_words(row_pair_amax[pair_idx], peer_amax, input_format)
    return [widen_word_bits(row_pair_amax[i // 2], i % 2, input_format) for i in range(8)]


def store_row_scales(row_amax: list, sub_block: SubBlock, scale_base: Int64, num_rows: Int64) -> list:
    rows_ok, col_group_idx, bx, row_start = sub_block.rows_ok, sub_block.col_group_idx, sub_block.bx, sub_block.row_start
    row_scale_bits = [e8m0_scale_bits(amax) for amax in row_amax]
    num_padded_rows = (num_rows + 3) // 4 * 4
    for i in range(8):
        store_ok = rows_ok[i] & make_predicate(col_group_idx == i)
        st_b32(scale_base + (Int64(bx) * num_padded_rows + Int64(row_start + i)) * 4, row_scale_bits[i], store_ok)
    return row_scale_bits


def quantize_store_rows(rows: list, row_scale_bits: list, sub_block: SubBlock, out_base: Int64, num_cols: Int64, input_format: str, is_out_aligned: bool) -> None:
    row_start, col_start = sub_block.row_start, sub_block.col_start
    rows_ok, cols_ok, full_cols_ok = sub_block.rows_ok, sub_block.cols_ok, sub_block.full_cols_ok
    # Subtract the scale bits from 0x7F000000 for an exact power-of-two reciprocal.
    for i in range(8):
        quant_lo, quant_hi = quantize_row(rows[i], compute_scale_rcp_bits(row_scale_bits[i]), input_format)
        elem_idx = Int64(row_start + i) * num_cols + Int64(col_start)
        if const_expr(is_out_aligned):
            st_v2(out_base + elem_idx, quant_lo, quant_hi, rows_ok[i] & cols_ok[0])
        else:
            store_eight_bytes(out_base + elem_idx, quant_lo, quant_hi, rows_ok[i] & full_cols_ok)
            partial_ok = rows_ok[i] & (full_cols_ok ^ 1)
            for byte_idx in range(7):
                word = quant_lo if byte_idx < 4 else quant_hi
                st_b8(out_base + elem_idx + byte_idx, word >> (8 * (byte_idx % 4)), partial_ok & cols_ok[byte_idx])


def reduce_col_amax(sub_block: SubBlock, s_col_amax: Int32, input_format: str) -> list:
    """
    Return the FP32 bits of the column maxima across all warps.
    """
    col_group_idx = sub_block.col_group_idx
    num_words = 8 if input_format == "f32" else 4
    warp_amax = [sum((list(lds_v4(s_col_amax + (warp * 16 * num_words + col_group_idx * num_words + 4 * quad) * 4)) for quad in range(num_words // 4)), []) for warp in range(8)]
    col_amax = []
    for word_idx in range(num_words):
        acc_amax = warp_amax[0][word_idx]
        for warp in range(1, 8):
            acc_amax = max_magnitude_words(acc_amax, warp_amax[warp][word_idx], input_format)
        col_amax.append(acc_amax)
    return col_amax if input_format == "f32" else [widen_word_bits(col_amax[j // 2], j % 2, input_format) for j in range(8)]


def store_col_scales(col_amax: list, sub_block: SubBlock, scale_t_base: Int64, num_cols: Int64) -> list:
    cols_ok, row_group_idx, by, col_start = sub_block.cols_ok, sub_block.row_group_idx, sub_block.by, sub_block.col_start
    col_scale_bits = [e8m0_scale_bits(amax) for amax in col_amax]
    num_padded_cols = (num_cols + 3) // 4 * 4
    for j in range(8):
        store_ok = cols_ok[j] & make_predicate(row_group_idx == j)
        st_b32(scale_t_base + (Int64(by) * num_padded_cols + Int64(col_start + j)) * 4, col_scale_bits[j], store_ok)
    return col_scale_bits


def quantize_stage_transpose(rows: list, col_scale_bits: list, sub_block: SubBlock, s_tile: Int32, input_format: str) -> None:
    """
    Quantize the sub-block's columns into the shared transposed tile.
    """
    col_group_idx, row_group_idx = sub_block.col_group_idx, sub_block.row_group_idx
    # XOR the chunk index with the column group so the warp's lanes hit distinct banks.
    for col_idx in range(8):
        word_idx, half_idx = (col_idx, 0) if input_format == "f32" else (col_idx // 2, col_idx % 2)
        rcp_bits = compute_scale_rcp_bits(col_scale_bits[col_idx])
        quant_lo = quantize_column([rows[i][word_idx] for i in range(4)], rcp_bits, half_idx, input_format)
        quant_hi = quantize_column([rows[i][word_idx] for i in range(4, 8)], rcp_bits, half_idx, input_format)
        sts_v2(s_tile + (col_group_idx * 8 + col_idx) * TILE + (row_group_idx ^ col_group_idx) * 8, quant_lo, quant_hi)


def store_transpose(sub_block: SubBlock, s_tile: Int32, out_t_base: Int64, num_rows: Int64, num_cols: Int64, transpose_alignment: int) -> None:
    """
    Write four 128-byte transposed rows per warp and pass.
    """
    lane, warp, by, bx = sub_block.lane, sub_block.warp, sub_block.by, sub_block.bx
    use_full_tiles = sub_block.use_full_tiles
    chunk_idx = lane % 8
    x_row_start = Int64(by * TILE + chunk_idx * 16)
    for pass_idx in range(4):
        tile_row_idx = pass_idx * 32 + warp * 4 + lane // 8
        swizzle = (tile_row_idx // 8) % 16
        words = lds_v4(s_tile + tile_row_idx * TILE + (((2 * chunk_idx) ^ swizzle) & 14) * 8)
        # An odd swizzle exchanges the two halves of the aligned 16-byte load.
        is_odd_swizzle = (swizzle % 2) != 0
        words_lo = [cutlass.select_(is_odd_swizzle, words[2], words[0]), cutlass.select_(is_odd_swizzle, words[3], words[1])]
        words_hi = [cutlass.select_(is_odd_swizzle, words[0], words[2]), cutlass.select_(is_odd_swizzle, words[1], words[3])]
        x_col_idx = bx * TILE + tile_row_idx
        col_ok = Int32(1) if use_full_tiles else make_predicate(x_col_idx < num_cols)
        out_addr = out_t_base + Int64(x_col_idx) * num_rows + x_row_start
        if const_expr(transpose_alignment == 16):
            store_ok = col_ok if use_full_tiles else col_ok & make_predicate(x_row_start < num_rows)
            st_v4(out_addr, words_lo + words_hi, store_ok)
        else:
            for half_idx, (word_lo, word_hi) in enumerate((words_lo, words_hi)):
                half_row_start = x_row_start + 8 * half_idx
                full_store_ok = col_ok & make_predicate(half_row_start + 8 <= num_rows)
                if const_expr(transpose_alignment == 8):
                    st_v2(out_addr + 8 * half_idx, word_lo, word_hi, full_store_ok)
                else:
                    store_eight_bytes(out_addr + 8 * half_idx, word_lo, word_hi, full_store_ok)
                    partial_ok = col_ok & make_predicate(half_row_start < num_rows) & (full_store_ok ^ 1)
                    for byte_idx in range(7):
                        word = word_lo if byte_idx < 4 else word_hi
                        st_b8(out_addr + 8 * half_idx + byte_idx, word >> (8 * (byte_idx % 4)), partial_ok & make_predicate(half_row_start + byte_idx < num_rows))


@dsl_user_op
def delta_gate_backward(grad: Float32, value: Float32, inv_rms: Float32, weight: Float32, gate: Float32, *, loc=None, ip=None) -> Float32:
    """The compiled SiLU derivative, rounded to BF16 by the caller."""
    ptx = (
        "{\n.reg .f32 t, s, d;\n"
        + SIGMOID_PTX.replace("$G", "$5")
        + "mul.rn.f32 d, $2, $3;\n"
        + "mul.rn.f32 d, d, $4;\n"
        + "mul.rn.f32 d, $1, d;\n"
        + "mul.rn.f32 d, d, s;\n"
        + "sub.rn.f32 t, 0f3F800000, s;\n"
        + "fma.rn.f32 t, t, $5, 0f3F800000;\n"
        + "mul.rn.f32 $0, d, t;\n}"
    )
    operands = [val.ir_value(loc=loc, ip=ip) for val in (grad, value, inv_rms, weight, gate)]
    return Float32(asm(T.f32(), operands, ptx, "=f,f,f,f,f,f"))


def load_delta_gate_grad(mGrad, mValue, mGate, mWeight, mInvRms, sub_block, N):
    """Produce the BF16-rounded gate gradient directly in the quantizer's registers."""
    rows = []
    x_base = mValue.iterator.toint()
    gate_base = mGate.iterator.toint()
    grad_base = mGrad.iterator.toint()
    head_column = sub_block.col_start % TILE
    for i in range(8):
        row = sub_block.row_start + i
        inverse_rms = mInvRms[row * (N // TILE) + sub_block.bx]
        offset = (Int64(row) * Int64(N) + Int64(sub_block.col_start)) * 2
        words = [
            emit_words(LD_V4_NC_FULL_PTX, "=r,=r,=r,=r,l", [(base + offset).ir_value()], 4)
            for base in (grad_base, x_base, gate_base)
        ]
        grad_words, value_words, gate_words = words
        row_words = []
        for j in range(4):
            weight_lo = mWeight[head_column + 2 * j]
            weight_hi = mWeight[head_column + 2 * j + 1]
            grad_lo = delta_gate_backward(
                bf16_lo(grad_words[j]), bf16_lo(value_words[j]), inverse_rms, weight_lo, bf16_lo(gate_words[j]),
            )
            grad_hi = delta_gate_backward(
                bf16_hi(grad_words[j]), bf16_hi(value_words[j]), inverse_rms, weight_hi, bf16_hi(gate_words[j]),
            )
            row_words.append(pack_bf16(grad_lo, grad_hi))
        rows.append(row_words)
    return rows


@cute.kernel
def rowwise_transpose_kernel(mX: cute.Tensor, auxiliaries: tuple, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor,
    input_format: cutlass.Constexpr[str], is_out_aligned: cutlass.Constexpr[bool], transpose_alignment: cutlass.Constexpr[int], use_gather: cutlass.Constexpr[bool], use_full_tiles: cutlass.Constexpr[bool], fuse_delta_gate: cutlass.Constexpr[bool]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    sub_block = SubBlock(tidx, by, bx, M, N, use_full_tiles)
    smem = cutlass.memory.SmemAllocator()
    num_words = 8 if const_expr(input_format == "f32") else 4
    s_col_amax = smem.allocate_array(Uint32, 8 * 16 * num_words, byte_alignment=16).toint()
    num_rows, num_cols = Int64(M), Int64(N)

    if const_expr(use_gather):
        # The column-maxima barrier lets the transpose reuse the input tile safely.
        s_tile = smem.allocate_array(cutlass.Uint8, TILE * TILE * (4 if const_expr(input_format == "f32") else 2), byte_alignment=128).toint()
        stage_input_tile(mX.iterator.toint(), Int64(mX.stride[0]), Int64(mX.stride[1]), sub_block, input_format, s_tile, M, N)
        rows = load_staged_rows(s_tile, sub_block, input_format)
    else:
        s_tile = smem.allocate_array(cutlass.Uint8, TILE * TILE, byte_alignment=128).toint()
        if const_expr(fuse_delta_gate):
            mValue, mGate, mWeight, mInvRms = auxiliaries
            rows = load_delta_gate_grad(mX, mValue, mGate, mWeight, mInvRms, sub_block, N)
        else:
            rows = load_rows(mX.iterator.toint(), Int64(mX.stride[0]), sub_block, input_format)

    out_base, scale_base, out_t_base, scale_t_base = mOut.iterator.toint(), mScale.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint()
    row_magnitude_bits = get_magnitude_bits(rows, num_words, input_format)
    share_col_amax(row_magnitude_bits, sub_block, s_col_amax, input_format)
    # Column reductions read the partial maxima from every warp.
    cute.arch.barrier()

    row_amax = reduce_row_amax(row_magnitude_bits, input_format)
    row_scale_bits = store_row_scales(row_amax, sub_block, scale_base, num_rows)
    quantize_store_rows(rows, row_scale_bits, sub_block, out_base, num_cols, input_format, is_out_aligned)

    col_amax = reduce_col_amax(sub_block, s_col_amax, input_format)
    col_scale_bits = store_col_scales(col_amax, sub_block, scale_t_base, num_cols)
    quantize_stage_transpose(rows, col_scale_bits, sub_block, s_tile, input_format)
    cute.arch.barrier()
    store_transpose(sub_block, s_tile, out_t_base, num_rows, num_cols, transpose_alignment)


def get_staged_chunk_addr(s_input: Int32, row_idx: Int32, chunk_idx: Int32) -> Int32:
    """
    Address one of the sixteen 16-byte chunks swizzled across four-row groups.
    """
    return s_input + row_idx * (TILE * 2) + (chunk_idx ^ ((row_idx // 4) % 8)) * 16


class StagedBf16Transpose:
    def __init__(self) -> None:
        # Row warps occupy the first four slots. The remaining warps handle columns.
        self.num_row_warps = 4

    def load_input_tile(self, x_base: Int64, row_stride: Int64, s_input: Int32, tidx: Int32, origin: tuple) -> None:
        row_start, col_start = origin
        for segment_idx in range(8):
            row_idx, chunk_idx = tidx // 16 + segment_idx * 16, tidx % 16
            x_addr = x_base + (Int64(row_start + row_idx) * row_stride + Int64(col_start + chunk_idx * 8)) * 2
            cp_async16(get_staged_chunk_addr(s_input, row_idx, chunk_idx), x_addr, Int32(16))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

    @cute.jit
    def row_quantize_loop(self, s_input: Int32, lane: Int32, warp: Int32, origin: tuple, shape: tuple, out_base: Int64, scale_base: Int64) -> None:
        lane_in_row = lane % 16
        for step in cutlass.range(16, unroll=2):
            tile_row_idx = warp * 2 + lane // 16 + step * 8
            self.quantize_row_step(s_input, tile_row_idx, lane_in_row, origin, shape, out_base, scale_base)

    @cute.jit
    def quantize_row_step(self, s_input: Int32, tile_row_idx: Int32, lane_in_row: Int32, origin: tuple, shape: tuple, out_base: Int64, scale_base: Int64) -> None:
        row_start, col_start = origin
        M, N = shape
        words = list(lds_v4(get_staged_chunk_addr(s_input, tile_row_idx, lane_in_row)))
        magnitudes = [word & Uint32(0x7FFF7FFF) for word in words]
        amax = reduce_bf16x2_amax4(magnitudes[0], magnitudes[1], magnitudes[2], magnitudes[3])
        amax = max2(amax, (amax << 16) | (amax >> 16))
        for stage in cutlass.range_constexpr(4):
            peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(amax), offset=1 << stage))
            amax = max2(amax, peer_amax)
        scale_bits = e8m0_scale_bits(widen_word_bits(amax, 0, "bf16"))
        row_idx = Int64(row_start + tile_row_idx)
        scale_idx = Int64(col_start // TILE) * M + row_idx
        st_b32(scale_base + scale_idx * 4, scale_bits, make_predicate(lane_in_row == 0))
        rcp_bits = compute_scale_rcp_bits(scale_bits)
        quant_lo, quant_hi = quantize_row(words, rcp_bits, "bf16")
        st_v2(out_base + row_idx * N + Int64(col_start + lane_in_row * 8), quant_lo, quant_hi, Int32(1))

    @cute.jit
    def column_quantize_loop(self, s_input: Int32, lane: Int32, warp: Int32, origin: tuple, shape: tuple, out_t_base: Int64, scale_t_base: Int64) -> None:
        tile_row_start = (lane % 8) * 4
        pair_in_chunk = lane // 8
        for step in cutlass.range(4, unroll=1):
            chunk_idx = warp + step * 4
            self.quantize_column_step(s_input, (lane, tile_row_start, chunk_idx, pair_in_chunk), origin, shape, out_t_base, scale_t_base)

    @cute.jit
    def quantize_column_step(self, s_input: Int32, coords: tuple, origin: tuple, shape: tuple, out_t_base: Int64, scale_t_base: Int64) -> None:
        row_start, col_start = origin
        M, N = shape
        lane, tile_row_start, chunk_idx, pair_in_chunk = coords
        # The swizzle gives the eight lanes of a column pair distinct four-bank groups.
        # The four pairs in a chunk use all four banks in each group.
        words = [lds_b32(get_staged_chunk_addr(s_input, tile_row_start + i % 4 + (i // 4) * 32, chunk_idx) + pair_in_chunk * 4) for i in range(16)]
        amax = words[0] & Uint32(0x7FFF7FFF)
        for i in cutlass.range_constexpr(1, 16):
            amax = max2(amax, words[i] & Uint32(0x7FFF7FFF))
        for stage in cutlass.range_constexpr(3):
            peer_amax = Uint32(cute.arch.shuffle_sync_bfly(Int32(amax), offset=1 << stage))
            amax = max2(amax, peer_amax)
        for half_idx in cutlass.range_constexpr(2):
            col_idx = Int64(col_start + chunk_idx * 8 + pair_in_chunk * 2 + half_idx)
            scale_bits = e8m0_scale_bits(widen_word_bits(amax, half_idx, "bf16"))
            scale_idx = Int64(row_start // TILE) * N + col_idx
            st_b32(scale_t_base + scale_idx * 4, scale_bits, make_predicate(lane % 8 == 0))
            rcp_bits = compute_scale_rcp_bits(scale_bits)
            out_addr = out_t_base + col_idx * M + Int64(row_start + tile_row_start)
            for slab_idx in cutlass.range_constexpr(4):
                quant = quantize_column(words[4 * slab_idx : 4 * slab_idx + 4], rcp_bits, half_idx, "bf16")
                st_b32(out_addr + slab_idx * 32, quant, Int32(1))

    @cute.kernel
    def staged_bf16_kernel(self, mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        M, N = Int64(mX.shape[0]), Int64(mX.shape[1])
        row_start, col_start = by * TILE, bx * TILE
        smem = cutlass.memory.SmemAllocator()
        s_input = smem.allocate_array(cutlass.Uint8, TILE * TILE * 2, byte_alignment=128).toint()

        origin = (row_start, col_start)
        self.load_input_tile(mX.iterator.toint(), Int64(mX.stride[0]), s_input, tidx, origin)

        warp, lane = tidx // 32, tidx % 32
        if warp < self.num_row_warps:
            self.row_quantize_loop(s_input, lane, warp, origin, (M, N), mOut.iterator.toint(), mScale.iterator.toint())
        else:
            self.column_quantize_loop(s_input, lane, warp - self.num_row_warps, origin, (M, N), mOutT.iterator.toint(), mScaleT.iterator.toint())


@cute.jit
def launch_fp8_quant_grad(mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, stream,
    input_format: cutlass.Constexpr[str], is_out_aligned: cutlass.Constexpr[bool], transpose_alignment: cutlass.Constexpr[int], use_gather: cutlass.Constexpr[bool], use_full_tiles: cutlass.Constexpr[bool], use_staging: cutlass.Constexpr[bool]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    grid = [(N + (TILE - 1)) // TILE, (M + (TILE - 1)) // TILE, 1]
    if const_expr(use_staging):
        StagedBf16Transpose().staged_bf16_kernel(mX, mOut, mScale, mOutT, mScaleT).launch(grid=grid, block=[THREADS, 1, 1], stream=stream)
    else:
        rowwise_transpose_kernel(mX, (), mOut, mScale, mOutT, mScaleT, input_format, is_out_aligned, transpose_alignment, use_gather, use_full_tiles, False).launch(grid=grid, block=[THREADS, 1, 1], stream=stream)


INPUT_FORMATS = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
COMPILED_KERNELS: dict[tuple[str, str, bool, int, bool, bool, bool], Callable] = {}


def compile_kernel(input_format: str, is_out_aligned: bool, transpose_alignment: int, use_gather: bool, use_full_tiles: bool, use_staging: bool = False) -> Callable:
    M, N, num_scale_cols, num_scale_t_cols = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    dtype = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16, "f32": cutlass.Float32}[input_format]
    x = fake(dtype, (M, N), (cute.sym_int64(), cute.sym_int64() if use_gather else 1), dtype.width // 8 if use_gather else 16)
    out, scale = fake(cutlass.Float8E4M3FN, (M, N), (cute.sym_int64(), 1)), fake(cutlass.Float32, (M, num_scale_cols), (1, cute.sym_int64()))
    out_t, scale_t = fake(cutlass.Float8E4M3FN, (N, M), (cute.sym_int64(), 1)), fake(cutlass.Float32, (N, num_scale_t_cols), (1, cute.sym_int64()))
    return cute.compile(launch_fp8_quant_grad, x, out, scale, out_t, scale_t, fake_stream(), input_format, is_out_aligned, transpose_alignment, use_gather, use_full_tiles, use_staging, options="--enable-tvm-ffi")


def launch_fp8_quant_grad_view(x: torch.Tensor, input_format: str, M: int, N: int, row_stride: int, col_stride: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    elems_per_16_bytes = 4 if input_format == "f32" else 8
    use_gather = col_stride != 1 or x.data_ptr() % 16 != 0 or row_stride % elems_per_16_bytes != 0 or N % elems_per_16_bytes != 0
    # The aligned BF16 load path and complete tiles make every vector access in-bounds.
    use_full_tiles = input_format == "bf16" and not use_gather and M % TILE == 0 and N % TILE == 0
    # Staging has enough work to amortize its CTA rendezvous on large grids.
    use_staging = use_full_tiles and M >= 4096 and N >= 10240 and (M // TILE) * (N // TILE) >= 4352

    device = x.device
    out, out_t = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    scale, scale_t = mn_major_scale(M, (N + 127) >> 7, device), mn_major_scale(N, (M + 127) >> 7, device)

    # A row width divisible by eight keeps the FP8 vector stores aligned.
    key = ("staged_bf16" if use_staging else "rowwise_transpose", input_format, N % 8 == 0, 16 if M % 16 == 0 else 8 if M % 8 == 0 else 1, use_gather, use_full_tiles, use_staging)
    fn = COMPILED_KERNELS.get(key)
    if fn is None:
        fn = COMPILED_KERNELS[key] = compile_kernel(*key[1:])
    fn(x, out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


def fp8_quant_grad_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run fp8_quant_grad on SM90 with MN-major scales padded to a multiple of four rows.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    input_format = INPUT_FORMATS.get(x.dtype)
    if input_format is None:
        raise TypeError(f"fp8_quant_grad takes bfloat16, float16 or float32 input, got {x.dtype}")
    (M, N), (row_stride, col_stride) = x.shape, x.stride()
    if M == 0 or N == 0:
        device = x.device
        return torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(M, (N + 127) >> 7, device), torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(N, (M + 127) >> 7, device)
    if M == 1 or N == 1:
        # A size-one dimension uses one address, so its stride can be normalized for dispatch.
        row_stride, col_stride = row_stride if M > 1 else N, col_stride if N > 1 else 1
        x = x.as_strided((M, N), (row_stride, col_stride))
    if col_stride == 1 or row_stride != 1:
        return launch_fp8_quant_grad_view(x, input_format, M, N, row_stride, col_stride)
    # Swap outputs so a unit row stride can use the contiguous-column path.
    out_t, scale_t, out, scale = launch_fp8_quant_grad_view(x.T, input_format, N, M, col_stride, 1)
    return out, scale, out_t, scale_t


@cute.jit
def launch_delta_grad(mGrad, mValue, mGate, mWeight, mInvRms, mOut, mScale, mOutT, mScaleT, stream):
    M, N = mGrad.shape
    rowwise_transpose_kernel(
        mGrad, (mValue, mGate, mWeight, mInvRms), mOut, mScale, mOutT, mScaleT,
        "bf16", True, 16, False, True, True,
    ).launch(grid=[N // TILE, M // TILE, 1], block=[THREADS, 1, 1], stream=stream)


DELTA_GRAD_KERNEL = None


def fp8_quant_delta_grad_sm90(grad, value, gate, weight, inv_rms):
    global DELTA_GRAD_KERNEL
    if DELTA_GRAD_KERNEL is None:
        M, N, scale_rows, scale_cols = (cute.sym_int() for _ in range(4))
        x = fake(cutlass.BFloat16, (M, N), (N, 1), 16)
        weight_fake = fake(Float32, (TILE,), (1,), 16)
        inv_rms_fake = fake(Float32, (cute.sym_int(),), (1,), 16)
        out = fake(cutlass.Float8E4M3FN, (M, N), (N, 1), 16)
        out_t = fake(cutlass.Float8E4M3FN, (N, M), (M, 1), 16)
        scale = fake(Float32, (M, scale_cols), (1, cute.sym_int64()))
        scale_t = fake(Float32, (N, scale_rows), (1, cute.sym_int64()))
        DELTA_GRAD_KERNEL = cute.compile(
            launch_delta_grad, x, x, x, weight_fake, inv_rms_fake, out, scale, out_t, scale_t,
            fake_stream(), options="--enable-tvm-ffi",
        )
    M, N = grad.shape
    out = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=grad.device)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=grad.device)
    scale = mn_major_scale(M, N // TILE, grad.device)
    scale_t = mn_major_scale(N, M // TILE, grad.device)
    DELTA_GRAD_KERNEL(grad, value, gate, weight, inv_rms, out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t
