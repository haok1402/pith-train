"""
CuTe-DSL (SM90) rowwise FP8 quantization of x and of x.T from one read of x, bit-identical to the
Triton kernel in fp8_rowwise_transpose_sm100.

One CTA (256 threads) per 128x128 tile; each thread owns an 8x8 block. The transposed tile goes
through bank-swizzled smem so every warp writes whole 128-byte rows of x.T: storing it straight from
registers (16 rows per warp instruction) was much slower.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from pithtrain.kernels.cute_compile_sm90 import fake_stream
from pithtrain.kernels.fp8_rowwise_transpose_sm100 import fp8cast_rowwise_transpose_sm100
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, to_mn_major

__all__ = ["fp8cast_rowwise_transpose_sm90", "supports_rowwise_transpose"]

TILE, THREADS = 128, 256

# fp32 bit patterns of the constants compute_fp8_scale uses (Triton materializes the Python
# floats as fp32): 1e-4 -> 0x38D1B717, 1/448 -> 0x3B124925.
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

# Rowwise quantize: 4 bf16x2 words (8 consecutive columns) times one reciprocal scale -> 8 fp8
# bytes in two words. bf16 -> f32 is a 16-bit shift; the multiply by a power of 2 is exact and
# cvt.rn.satfinite matches Triton's .to(tl.float8e4nv).
QROW_PTX = """{
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

# Column quantize: one half (EXTRACT = low or high bf16) of 4 words (4 consecutive rows of one
# column) -> 4 fp8 bytes in one word.
QCOL_PTX = """{
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
QCOL_EXTRACT = ("shl.b32 {}, {}, 16;", "and.b32 {}, {}, 0xFFFF0000;")

LD_V4_PTX = """{
.reg .pred p;
setp.ne.b32 p, $5, 0;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];
}"""

LD_U16_PTX = """{
.reg .pred p;
.reg .b16 h;
setp.ne.b32 p, $2, 0;
mov.b16 h, 0;
@p ld.global.nc.b16 h, [$1];
cvt.u32.u16 $0, h;
}"""

ST_V2_PTX = """{
.reg .pred p;
setp.ne.b32 p, $3, 0;
@p st.global.v2.b32 [$0], {$1, $2};
}"""

ST_B32_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
@p st.global.b32 [$0], $1;
}"""

ST_B8_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
@p st.global.b8 [$0], $1;
}"""

# 8 contiguous bf16 at an address that is only 2-byte aligned: the widest access the runtime
# alignment allows (16/8/4/2 bytes). The four paths are mutually exclusive predicates.
LD8_ANY_PTX = """{
.reg .pred p, p16, p8, p4, p2;
.reg .b32 al, t;
.reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
setp.ne.b32 p, $5, 0;
cvt.u32.u64 al, $4;
and.b32 t, al, 15;
setp.eq.and.b32 p16, t, 0, p;
setp.eq.and.b32 p8, t, 8, p;
and.b32 t, al, 7;
setp.eq.and.b32 p4, t, 4, p;
and.b32 t, al, 3;
setp.eq.and.b32 p2, t, 2, p;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p16 ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];
@p8 ld.global.nc.v2.b32 {$0, $1}, [$4];
@p8 ld.global.nc.v2.b32 {$2, $3}, [$4+8];
@p4 ld.global.nc.b32 $0, [$4];
@p4 ld.global.nc.b32 $1, [$4+4];
@p4 ld.global.nc.b32 $2, [$4+8];
@p4 ld.global.nc.b32 $3, [$4+12];
@p2 ld.global.nc.b16 h0, [$4];
@p2 ld.global.nc.b16 h1, [$4+2];
@p2 ld.global.nc.b16 h2, [$4+4];
@p2 ld.global.nc.b16 h3, [$4+6];
@p2 ld.global.nc.b16 h4, [$4+8];
@p2 ld.global.nc.b16 h5, [$4+10];
@p2 ld.global.nc.b16 h6, [$4+12];
@p2 ld.global.nc.b16 h7, [$4+14];
@p2 mov.b32 $0, {h0, h1};
@p2 mov.b32 $1, {h2, h3};
@p2 mov.b32 $2, {h4, h5};
@p2 mov.b32 $3, {h6, h7};
}"""

# 8 fp8 bytes (w0 = bytes 0-3, w1 = bytes 4-7) to an address of any alignment: the widest store
# the runtime alignment allows (8/4/2/1 bytes), mutually exclusive predicates.
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


def ir(v):
    return v.ir_value()


def asm_u32(ptx: str, constraints: str, args: list) -> Uint32:
    return Uint32(llvm.inline_asm(T.i32(), args, ptx, constraints, has_side_effects=False, is_align_stack=False))


def asm_store(ptx: str, constraints: str, args: list) -> None:
    llvm.inline_asm(None, args, ptx, constraints, has_side_effects=True, is_align_stack=False)


def bmax(a: Uint32, b: Uint32) -> Uint32:
    return asm_u32("max.bf16x2 $0, $1, $2;", "=r,r,r", [ir(a), ir(b)])


def prmt(a: Uint32, b: Uint32, sel: int) -> Uint32:
    return asm_u32(f"prmt.b32 $0, $1, $2, {sel};", "=r,r,r", [ir(a), ir(b)])


def scale_bits(amax_f32_bits: Uint32) -> Uint32:
    return asm_u32(SCALE_PTX, "=r,r", [ir(amax_f32_bits)])


def qrow(words: list, rcp: Uint32) -> tuple:
    res = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * 2), [ir(w) for w in words] + [ir(rcp)], QROW_PTX, "=&r,=&r,r,r,r,r,r", has_side_effects=False, is_align_stack=False)
    return tuple(Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(2))


def qcol(words: list, rcp: Uint32, half: int) -> Uint32:
    ext = QCOL_EXTRACT[half].format("$0", "{}")
    ptx = QCOL_PTX
    for j in range(4):
        ptx = ptx.replace(f"EXTRACT $0, ${j + 1};", ext.format(f"${j + 1}"))
    return asm_u32(ptx, "=&r,r,r,r,r,r", [ir(w) for w in words] + [ir(rcp)])


def ld_v4(addr: Int64, pred: Int32) -> list:
    res = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * 4), [ir(addr), ir(pred)], LD_V4_PTX, "=r,=r,=r,=r,l,r", has_side_effects=False, is_align_stack=False)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


def ld_u16(addr: Int64, pred: Int32) -> Uint32:
    return asm_u32(LD_U16_PTX, "=r,l,r", [ir(addr), ir(pred)])


def st_v2(addr: Int64, w0: Uint32, w1: Uint32, pred: Int32) -> None:
    asm_store(ST_V2_PTX, "l,r,r,r", [ir(addr), ir(w0), ir(w1), ir(pred)])


def st_b32(addr: Int64, w: Uint32, pred: Int32) -> None:
    asm_store(ST_B32_PTX, "l,r,r", [ir(addr), ir(w), ir(pred)])


def st_b8(addr: Int64, w: Uint32, pred: Int32) -> None:
    asm_store(ST_B8_PTX, "l,r,r", [ir(addr), ir(w), ir(pred)])


def ld8_any(addr: Int64, pred: Int32) -> list:
    res = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * 4), [ir(addr), ir(pred)], LD8_ANY_PTX, "=r,=r,=r,=r,l,r", has_side_effects=False, is_align_stack=False)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


def st8_any(addr: Int64, w0: Uint32, w1: Uint32, pred: Int32) -> None:
    asm_store(ST8_ANY_PTX, "l,r,r,r", [ir(addr), ir(w0), ir(w1), ir(pred)])


def pred_i32(cond) -> Int32:
    return Int32(cutlass.select_(cond, Int32(1), Int32(0)))


def ld_shared_v4(addr: Int32) -> list:
    res = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * 4), [ir(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


def st_shared_v4(addr: Int32, w: list, pred: Int32) -> None:
    asm_store("{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n}", "r,r,r,r,r,r", [ir(addr)] + [ir(x) for x in w] + [ir(pred)])


def st_shared_v2(addr: Int32, w0: Uint32, w1: Uint32) -> None:
    asm_store("st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", [ir(addr), ir(w0), ir(w1)])


def st_v4(addr: Int64, w: list, pred: Int32) -> None:
    asm_store("{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}", "l,r,r,r,r,r", [ir(addr)] + [ir(x) for x in w] + [ir(pred)])


class Thread:
    """
    Trace-time per-thread constants.
    """

    def __init__(self, tidx: Int32) -> None:
        self.warp = tidx // 32
        self.lane = tidx % 32
        self.cg = self.lane % 16  # 8-column group within the tile
        self.rg = self.warp * 2 + self.lane // 16  # 8-row group within the tile


class Tile:
    """
    Trace-time coordinates and bounds predicates of one 128x128 tile.
    """

    def __init__(self, th: Thread, tm: Int32, tn: Int32, M: Int32, N: Int32) -> None:
        self.tm = tm
        self.tn = tn
        self.row0 = tm * TILE + th.rg * 8
        self.col0 = tn * TILE + th.cg * 8
        self.rowv = [pred_i32(self.row0 + i < M) for i in range(8)]
        self.colv = [pred_i32(self.col0 + c < N) for c in range(8)]
        self.full_rows = pred_i32(self.row0 + 8 <= M)
        self.full_cols = pred_i32(self.col0 + 8 <= N)


def load_global(x_base: Int64, N64: Int64, tl: Tile, vec_n: bool) -> list:
    """
    The thread's 8x8 block as bf16x2 words: v[i][k] = columns (2k, 2k+1) of row i.
    """
    v = []
    if vec_n:
        for i in range(8):
            pred = tl.rowv[i] & tl.colv[0]
            off = cutlass.select_(pred != 0, (Int64(tl.row0 + i) * N64 + Int64(tl.col0)) * 2, Int64(0))
            v.append(ld_v4(x_base + off, pred))
    else:
        # Rows of x are not 16-byte aligned: full 8-column groups take the widest access the
        # runtime alignment allows; the partial group at the N edge loads element by element.
        for i in range(8):
            full, part = tl.rowv[i] & tl.full_cols, tl.rowv[i] & (tl.full_cols ^ 1)
            row_off = Int64(tl.row0 + i) * N64 + Int64(tl.col0)
            words = ld8_any(x_base + cutlass.select_(full != 0, row_off * 2, Int64(0)), full)
            for k in range(4):
                halves = []
                for h in range(2):
                    pred = part & tl.colv[2 * k + h]
                    off = cutlass.select_(pred != 0, (row_off + (2 * k + h)) * 2, Int64(0))
                    halves.append(ld_u16(x_base + off, pred))
                words[k] = words[k] | halves[0] | (halves[1] << 16)
            v.append(words)
    return v


def quantize_tile(v: list, th: Thread, tl: Tile, sred: Int32, stile: Int32, bases: tuple, M64: Int64, N64: Int64, vec_n: bool, m_align: int) -> None:
    """
    Amax, scales and both FP8 outputs of one tile from the thread's 8x8 block v; sred is 2 KB of
    smem for the cross-warp column amax, stile 16 KB for the transposed tile.
    """
    tok_base, stok_base, t_base, st_base = bases
    a = [[v[i][k] & Uint32(0x7FFF7FFF) for k in range(4)] for i in range(8)]

    # column amax: own 8 rows, the other half-warp, then the 8 warps via smem
    cm = []
    for k in range(4):
        c = a[0][k]
        for i in range(1, 8):
            c = bmax(c, a[i][k])
        other = Uint32(cute.arch.shuffle_sync_bfly(Int32(c), offset=16))
        cm.append(bmax(c, other))
    st_shared_v4(sred + (th.warp * 64 + th.cg * 4) * 4, cm, pred_i32(th.lane < 16))
    cute.arch.barrier()

    # Row amax (two rows packed per word) over the 16 lanes sharing the rows; independent of the
    # smem round trip above, so it overlaps the barrier wait.
    rm = [bmax(bmax(a[i][0], a[i][1]), bmax(a[i][2], a[i][3])) for i in range(8)]
    rpk = []
    for j in range(4):
        lo, hi = prmt(rm[2 * j], rm[2 * j + 1], 0x5410), prmt(rm[2 * j], rm[2 * j + 1], 0x7632)
        rpk.append(bmax(lo, hi))
    for s in range(4):
        for j in range(4):
            other = Uint32(cute.arch.shuffle_sync_bfly(Int32(rpk[j]), offset=1 << s))
            rpk[j] = bmax(rpk[j], other)

    tok_scale = []
    for j in range(4):
        tok_scale.append(scale_bits(rpk[j] << 16))
        tok_scale.append(scale_bits(rpk[j] & Uint32(0xFFFF0000)))
    # MN-major scales: scale[r, j] at j * align(rows, 4) + r.
    m_pad, n_pad = (M64 + 3) // 4 * 4, (N64 + 3) // 4 * 4
    for i in range(8):
        pred = tl.rowv[i] & pred_i32(th.cg == i)
        off = (Int64(tl.tn) * m_pad + Int64(tl.row0 + i)) * 4
        st_b32(stok_base + off, tok_scale[i], pred)
    for i in range(8):
        w0, w1 = qrow(v[i], Uint32(0x7F000000) - tok_scale[i])
        row_off = Int64(tl.row0 + i) * N64 + Int64(tl.col0)
        if const_expr(vec_n):
            st_v2(tok_base + row_off, w0, w1, tl.rowv[i] & tl.colv[0])
        else:
            st8_any(tok_base + row_off, w0, w1, tl.rowv[i] & tl.full_cols)
            part = tl.rowv[i] & (tl.full_cols ^ 1)
            for c in range(7):
                w = w0 if c < 4 else w1
                st_b8(tok_base + row_off + c, w >> (8 * (c % 4)), part & tl.colv[c])

    parts = [ld_shared_v4(sred + (w * 64 + th.cg * 4) * 4) for w in range(8)]
    for k in range(4):
        c = parts[0][k]
        for w in range(1, 8):
            c = bmax(c, parts[w][k])
        cm[k] = c
    t_scale = []
    for k in range(4):
        t_scale.append(scale_bits(cm[k] << 16))
        t_scale.append(scale_bits(cm[k] & Uint32(0xFFFF0000)))
    for c in range(8):
        pred = tl.colv[c] & pred_i32(th.rg == c)
        off = (Int64(tl.tm) * n_pad + Int64(tl.col0 + c)) * 4
        st_b32(st_base + off, t_scale[c], pred)
    # Stage the tile of x.T in smem (row = one column of x, 128 B). The 8-byte chunk index is
    # swizzled by the column group so the 32 lanes of a warp hit distinct banks.
    for c in range(8):
        k = c // 2
        rcp = Uint32(0x7F000000) - t_scale[c]
        w0, w1 = qcol([v[i][k] for i in range(4)], rcp, c % 2), qcol([v[i][k] for i in range(4, 8)], rcp, c % 2)
        st_shared_v2(stile + (th.cg * 8 + c) * TILE + (th.rg ^ th.cg) * 8, w0, w1)
    cute.arch.barrier()
    # Each warp writes 4 contiguous 128-byte rows of out_t per pass (16 B per lane).
    ch = th.lane % 8
    start = Int64(tl.tm * TILE + ch * 16)
    for k in range(4):
        r = k * 32 + th.warp * 4 + th.lane // 8
        s = (r // 8) % 16
        w = ld_shared_v4(stile + r * TILE + (((2 * ch) ^ s) & 14) * 8)
        odd = (s % 2) != 0
        lo, hi = [cutlass.select_(odd, w[2], w[0]), cutlass.select_(odd, w[3], w[1])], [cutlass.select_(odd, w[0], w[2]), cutlass.select_(odd, w[1], w[3])]
        col = tl.tn * TILE + r
        col_ok = pred_i32(col < N64)
        addr = t_base + Int64(col) * M64 + start
        if const_expr(m_align == 16):
            st_v4(addr, lo + hi, col_ok & pred_i32(start < M64))
        else:
            for h, (a0, a1) in enumerate((lo, hi)):
                hs = start + 8 * h
                full = col_ok & pred_i32(hs + 8 <= M64)
                if const_expr(m_align == 8):
                    st_v2(addr + 8 * h, a0, a1, full)
                else:
                    st8_any(addr + 8 * h, a0, a1, full)
                    part = col_ok & pred_i32(hs < M64) & (full ^ 1)
                    for i in range(7):
                        wd = a0 if i < 4 else a1
                        pred = part & pred_i32(hs + i < M64)
                        st_b8(addr + 8 * h + i, wd >> (8 * (i % 4)), pred)


@cute.kernel
def rowwise_transpose_kernel(mX: cute.Tensor, mOutTok: cute.Tensor, mScaleTok: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, vec_n: cutlass.Constexpr[bool], m_align: cutlass.Constexpr[int]) -> None:
    """
    One 128x128 tile per CTA.
    """
    M, N = mX.shape[0], mX.shape[1]
    tidx, _, _ = cute.arch.thread_idx()
    tn, tm, _ = cute.arch.block_idx()
    th = Thread(tidx)
    tl = Tile(th, tm, tn, M, N)
    smem = cutlass.memory.SmemAllocator()
    sred = smem.allocate_array(Uint32, 8 * 16 * 4, byte_alignment=16).toint()
    stile = smem.allocate_array(cutlass.Uint8, TILE * TILE, byte_alignment=128).toint()
    v = load_global(mX.iterator.toint(), Int64(N), tl, vec_n)
    bases = (mOutTok.iterator.toint(), mScaleTok.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint())
    quantize_tile(v, th, tl, sred, stile, bases, Int64(M), Int64(N), vec_n, m_align)


@cute.jit
def launch(mX: cute.Tensor, mOutTok: cute.Tensor, mScaleTok: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, stream, vec_n: cutlass.Constexpr[bool], m_align: cutlass.Constexpr[int]) -> None:
    M, N = mX.shape[0], mX.shape[1]
    rowwise_transpose_kernel(mX, mOutTok, mScaleTok, mOutT, mScaleT, vec_n, m_align).launch(grid=[(N + (TILE - 1)) // TILE, (M + (TILE - 1)) // TILE, 1], block=[THREADS, 1, 1], stream=stream)


COMPILED: dict[tuple[bool, int], Callable] = {}


def compile_kernel(vec_n: bool, m_align: int) -> Callable:
    m, n, sn, sm = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()

    def fake(dtype, shape, align):
        return cute.runtime.make_fake_tensor(dtype, shape, stride=(cute.sym_int64(), 1), assumed_align=align)

    def mn_major(rows, cols):
        return cute.runtime.make_fake_tensor(cutlass.Float32, (rows, cols), stride=(1, cute.sym_int64()), assumed_align=16)

    return cute.compile(launch, fake(cutlass.BFloat16, (m, n), 16 if vec_n else 2), fake(cutlass.Float8E4M3FN, (m, n), 16), mn_major(m, sn), fake(cutlass.Float8E4M3FN, (n, m), 16), mn_major(n, sm), fake_stream(), vec_n, m_align, options="--enable-tvm-ffi")


def get_compiled(key: tuple[bool, int]) -> Callable:
    fn = COMPILED.get(key)
    if fn is None:
        fn = COMPILED[key] = compile_kernel(*key)
    return fn


def supports_rowwise_transpose(x: torch.Tensor) -> bool:
    """
    Whether the CuTe kernel covers x: non-empty, contiguous, bf16.
    """
    M, N = x.shape
    return x.dtype is torch.bfloat16 and x.is_contiguous() and bool(M) and bool(N)


def fp8cast_rowwise_transpose_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fp8cast_rowwise_transpose with both scales MN-major; inputs supports_rowwise_transpose rejects
    go to the Triton kernel.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    if not supports_rowwise_transpose(x):
        out_tok, scale_tok, out_t, scale_t = fp8cast_rowwise_transpose_sm100(x)
        return out_tok, to_mn_major(scale_tok), out_t, to_mn_major(scale_t)
    dev = x.device
    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_tok = mn_major_scale(M, (N + 127) >> 7, dev)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = mn_major_scale(N, (M + 127) >> 7, dev)
    key = (N % 8 == 0 and x.data_ptr() % 16 == 0, 16 if M % 16 == 0 else 8 if M % 8 == 0 else 1)
    fn = COMPILED.get(key) or get_compiled(key)
    fn(x, out_tok, scale_tok, out_t, scale_t)
    return out_tok, scale_tok, out_t, scale_t
