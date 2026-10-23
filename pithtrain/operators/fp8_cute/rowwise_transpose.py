"""CuTe-DSL port of ``fp8cast_rowwise_transpose`` (rowwise FP8 of x and of x.T, one read of x).

Bit-identical to the Triton kernel in ``pithtrain/operators/deepgemm_quantize.py`` on SM90: E8M0
(power-of-2) scales from ``_compute_fp8_scale``'s software branch and an RN-satfinite e4m3 cast.

One CTA (256 threads) per 128x128 tile of x. Thread ``(warp, lane)`` owns an 8x8 block:
columns ``8 * (lane & 15) ..`` and rows ``8 * (2 * warp + (lane >> 4)) ..`` of the tile, loaded
with 16-byte loads. Rows are reduced with packed bf16x2 max + four butterfly shuffles over the 16
lanes sharing a row; columns with bf16x2 max over the thread's 8 rows, one shuffle and an 8-warp
combine through 2 KB of smem. The rowwise output is stored from registers (8 B per lane, two full
rows per warp). The transposed tile goes through 16 KB of bank-swizzled smem so that every warp
writes whole 128-byte rows of x.T: storing it straight from registers (8 B per lane, 16 rows per
warp instruction) cost about a third of the kernel time at (4096, 4096).

Unaligned shapes (N % 8, M % 16, misaligned base) use runtime-alignment-dispatched accesses plus
per-element edge handling; non-bf16, non-contiguous inputs and non-SM90 devices use Triton.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from pithtrain.operators.fp8_cute.scale_layout import mn_major_scale, to_mn_major

__all__ = ["fp8cast_rowwise_transpose"]

_TILE = 128
_THREADS = 256

# fp32 bit patterns of the constants _compute_fp8_scale uses (Triton materializes the Python
# floats as fp32): 1e-4 -> 0x38D1B717, 1/448 -> 0x3B124925.
_SCALE_PTX = """{
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
# cvt.rn.satfinite matches Triton's `.to(tl.float8e4nv)`.
_QROW_PTX = """{
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
# column) times that column's reciprocal scale -> 4 fp8 bytes in one word.
_QCOL_PTX = """{
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
_QCOL_EXTRACT = ("shl.b32 {}, {}, 16;", "and.b32 {}, {}, 0xFFFF0000;")

_LD_V4_PTX = """{
.reg .pred p;
setp.ne.b32 p, $5, 0;
mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;
@p ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];
}"""

_LD_U16_PTX = """{
.reg .pred p;
.reg .b16 h;
setp.ne.b32 p, $2, 0;
mov.b16 h, 0;
@p ld.global.nc.b16 h, [$1];
cvt.u32.u16 $0, h;
}"""

_ST_V2_PTX = """{
.reg .pred p;
setp.ne.b32 p, $3, 0;
@p st.global.v2.b32 [$0], {$1, $2};
}"""

_ST_B32_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
@p st.global.b32 [$0], $1;
}"""

_ST_B8_PTX = """{
.reg .pred p;
setp.ne.b32 p, $2, 0;
@p st.global.b8 [$0], $1;
}"""

# 8 contiguous bf16 at an address that is only 2-byte aligned: the widest access the runtime
# alignment allows (16/8/4/2 bytes). The four paths are mutually exclusive predicates.
_LD8_ANY_PTX = """{
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
_ST8_ANY_PTX = """{
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


def _ir(v):  # noqa: ANN001, ANN202 - DSL value -> MLIR value
    return v.ir_value()


def _asm_u32(ptx: str, constraints: str, args: list) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(), args, ptx, constraints, has_side_effects=False, is_align_stack=False
        )
    )


def _asm_store(ptx: str, constraints: str, args: list) -> None:
    llvm.inline_asm(None, args, ptx, constraints, has_side_effects=True, is_align_stack=False)


def _bmax(a: Uint32, b: Uint32) -> Uint32:
    return _asm_u32("max.bf16x2 $0, $1, $2;", "=r,r,r", [_ir(a), _ir(b)])


def _prmt(a: Uint32, b: Uint32, sel: int) -> Uint32:
    return _asm_u32(f"prmt.b32 $0, $1, $2, {sel};", "=r,r,r", [_ir(a), _ir(b)])


def _scale_bits(amax_f32_bits: Uint32) -> Uint32:
    return _asm_u32(_SCALE_PTX, "=r,r", [_ir(amax_f32_bits)])


def _qrow(words: list, rcp: Uint32) -> tuple:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_ir(w) for w in words] + [_ir(rcp)],
        _QROW_PTX,
        "=&r,=&r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(2))


def _qcol(words: list, rcp: Uint32, half: int) -> Uint32:
    ext = _QCOL_EXTRACT[half].format("$0", "{}")
    ptx = _QCOL_PTX
    for j in range(4):
        ptx = ptx.replace(f"EXTRACT $0, ${j + 1};", ext.format(f"${j + 1}"))
    return _asm_u32(ptx, "=&r,r,r,r,r,r", [_ir(w) for w in words] + [_ir(rcp)])


def _ld_v4(addr: Int64, pred: Int32) -> list:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(addr), _ir(pred)],
        _LD_V4_PTX,
        "=r,=r,=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


def _ld_u16(addr: Int64, pred: Int32) -> Uint32:
    return _asm_u32(_LD_U16_PTX, "=r,l,r", [_ir(addr), _ir(pred)])


def _st_v2(addr: Int64, w0: Uint32, w1: Uint32, pred: Int32) -> None:
    _asm_store(_ST_V2_PTX, "l,r,r,r", [_ir(addr), _ir(w0), _ir(w1), _ir(pred)])


def _st_b32(addr: Int64, w: Uint32, pred: Int32) -> None:
    _asm_store(_ST_B32_PTX, "l,r,r", [_ir(addr), _ir(w), _ir(pred)])


def _st_b8(addr: Int64, w: Uint32, pred: Int32) -> None:
    _asm_store(_ST_B8_PTX, "l,r,r", [_ir(addr), _ir(w), _ir(pred)])


def _ld8_any(addr: Int64, pred: Int32) -> list:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(addr), _ir(pred)],
        _LD8_ANY_PTX,
        "=r,=r,=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


def _st8_any(addr: Int64, w0: Uint32, w1: Uint32, pred: Int32) -> None:
    _asm_store(_ST8_ANY_PTX, "l,r,r,r", [_ir(addr), _ir(w0), _ir(w1), _ir(pred)])


def _b(cond) -> Int32:  # noqa: ANN001 - DSL Boolean -> Int32 predicate for inline asm
    return Int32(cutlass.select_(cond, Int32(1), Int32(0)))


def _ld_shared_v4(addr: Int32) -> list:
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_ir(addr)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4)]


def _st_shared_v4(addr: Int32, w: list, pred: Int32) -> None:
    _asm_store(
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "r,r,r,r,r,r",
        [_ir(addr)] + [_ir(x) for x in w] + [_ir(pred)],
    )


def _st_shared_v2(addr: Int32, w0: Uint32, w1: Uint32) -> None:
    _asm_store("st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", [_ir(addr), _ir(w0), _ir(w1)])


def _st_v4(addr: Int64, w: list, pred: Int32) -> None:
    _asm_store(
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        [_ir(addr)] + [_ir(x) for x in w] + [_ir(pred)],
    )


class _Thread:
    """Trace-time bundle of the per-thread constants shared by both kernels."""

    def __init__(self, tidx: Int32) -> None:
        self.warp = tidx // 32
        self.lane = tidx % 32
        self.cg = self.lane % 16  # 8-column group within the tile
        self.rg = self.warp * 2 + self.lane // 16  # 8-row group within the tile


class _Tile:
    """Trace-time bundle of one 128x128 tile's coordinates and bounds predicates."""

    def __init__(self, th: _Thread, tm: Int32, tn: Int32, M: Int32, N: Int32) -> None:
        self.tm = tm
        self.tn = tn
        self.row0 = tm * _TILE + th.rg * 8
        self.col0 = tn * _TILE + th.cg * 8
        self.rowv = [_b(self.row0 + i < M) for i in range(8)]
        self.colv = [_b(self.col0 + c < N) for c in range(8)]
        self.full_rows = _b(self.row0 + 8 <= M)
        self.full_cols = _b(self.col0 + 8 <= N)


def _load_global(x_base: Int64, N64: Int64, tl: _Tile, vec_n: bool) -> list:
    """The thread's 8x8 block as bf16x2 words: v[i][k] = columns (2k, 2k+1) of row i."""
    v = []
    if vec_n:
        for i in range(8):
            pred = tl.rowv[i] & tl.colv[0]
            off = cutlass.select_(
                pred != 0, (Int64(tl.row0 + i) * N64 + Int64(tl.col0)) * 2, Int64(0)
            )
            v.append(_ld_v4(x_base + off, pred))
    else:
        # Rows of x are not 16-byte aligned: full 8-column groups take the widest access the
        # runtime alignment allows; the partial group at the N edge loads element by element.
        for i in range(8):
            full = tl.rowv[i] & tl.full_cols
            part = tl.rowv[i] & (tl.full_cols ^ 1)
            row_off = Int64(tl.row0 + i) * N64 + Int64(tl.col0)
            words = _ld8_any(x_base + cutlass.select_(full != 0, row_off * 2, Int64(0)), full)
            for k in range(4):
                halves = []
                for h in range(2):
                    pred = part & tl.colv[2 * k + h]
                    off = cutlass.select_(pred != 0, (row_off + (2 * k + h)) * 2, Int64(0))
                    halves.append(_ld_u16(x_base + off, pred))
                words[k] = words[k] | halves[0] | (halves[1] << 16)
            v.append(words)
    return v


def _quantize_tile(
    v: list,
    th: _Thread,
    tl: _Tile,
    sred: Int32,
    stile: Int32,
    bases: tuple,
    M64: Int64,
    N64: Int64,
    grid_m: Int32,
    grid_n: Int32,
    vec_n: bool,
    m_align: int,
) -> None:
    """Amax, scales and both FP8 outputs of one tile from the thread's 8x8 block ``v``.

    ``sred`` is the 32-bit shared address of a 2 KB scratch for the cross-warp column amax.
    """
    tok_base, stok_base, t_base, st_base = bases
    a = [[v[i][k] & Uint32(0x7FFF7FFF) for k in range(4)] for i in range(8)]

    # ---- column amax: own 8 rows, the other half-warp, then the 8 warps via smem ----
    cm = []
    for k in range(4):
        c = a[0][k]
        for i in range(1, 8):
            c = _bmax(c, a[i][k])
        other = Uint32(cute.arch.shuffle_sync_bfly(Int32(c), offset=16))
        cm.append(_bmax(c, other))
    _st_shared_v4(sred + (th.warp * 64 + th.cg * 4) * 4, cm, _b(th.lane < 16))
    cute.arch.barrier()

    # ---- row amax (two rows packed per word), reduced over the 16 lanes sharing the rows;
    # independent of the smem round trip above, so it overlaps the barrier wait ----
    rm = [_bmax(_bmax(a[i][0], a[i][1]), _bmax(a[i][2], a[i][3])) for i in range(8)]
    rpk = []
    for j in range(4):
        lo = _prmt(rm[2 * j], rm[2 * j + 1], 0x5410)
        hi = _prmt(rm[2 * j], rm[2 * j + 1], 0x7632)
        rpk.append(_bmax(lo, hi))
    for s in range(4):
        for j in range(4):
            other = Uint32(cute.arch.shuffle_sync_bfly(Int32(rpk[j]), offset=1 << s))
            rpk[j] = _bmax(rpk[j], other)

    # ---- rowwise scales and output: out_tok[row, col0:col0+8] ----
    tok_scale = []
    for j in range(4):
        tok_scale.append(_scale_bits(rpk[j] << 16))
        tok_scale.append(_scale_bits(rpk[j] & Uint32(0xFFFF0000)))
    # MN-major scales (fp8_cute/scale_layout.py): scale[r, j] at j * align(rows, 4) + r, so
    # lanes cg = 0..7 store 8 consecutive rows (32 B) instead of 8 words sf_k apart.
    m_pad = (M64 + 3) // 4 * 4
    n_pad = (N64 + 3) // 4 * 4
    for i in range(8):
        pred = tl.rowv[i] & _b(th.cg == i)
        off = (Int64(tl.tn) * m_pad + Int64(tl.row0 + i)) * 4
        _st_b32(stok_base + off, tok_scale[i], pred)
    for i in range(8):
        w0, w1 = _qrow(v[i], Uint32(0x7F000000) - tok_scale[i])
        row_off = Int64(tl.row0 + i) * N64 + Int64(tl.col0)
        if const_expr(vec_n):
            _st_v2(tok_base + row_off, w0, w1, tl.rowv[i] & tl.colv[0])
        else:
            _st8_any(tok_base + row_off, w0, w1, tl.rowv[i] & tl.full_cols)
            part = tl.rowv[i] & (tl.full_cols ^ 1)
            for c in range(7):
                w = w0 if c < 4 else w1
                _st_b8(tok_base + row_off + c, w >> (8 * (c % 4)), part & tl.colv[c])

    # ---- column scales and transposed output: out_t[col, row0:row0+8] ----
    parts = [_ld_shared_v4(sred + (w * 64 + th.cg * 4) * 4) for w in range(8)]
    for k in range(4):
        c = parts[0][k]
        for w in range(1, 8):
            c = _bmax(c, parts[w][k])
        cm[k] = c
    t_scale = []
    for k in range(4):
        t_scale.append(_scale_bits(cm[k] << 16))
        t_scale.append(_scale_bits(cm[k] & Uint32(0xFFFF0000)))
    for c in range(8):
        pred = tl.colv[c] & _b(th.rg == c)
        off = (Int64(tl.tm) * n_pad + Int64(tl.col0 + c)) * 4
        _st_b32(st_base + off, t_scale[c], pred)
    # Stage the tile of x.T in smem (row = one column of x, 128 B). The 8-byte chunk index is
    # swizzled by the column group so the 32 lanes of a warp hit distinct banks.
    for c in range(8):
        k = c // 2
        rcp = Uint32(0x7F000000) - t_scale[c]
        w0 = _qcol([v[i][k] for i in range(4)], rcp, c % 2)
        w1 = _qcol([v[i][k] for i in range(4, 8)], rcp, c % 2)
        _st_shared_v2(stile + (th.cg * 8 + c) * _TILE + (th.rg ^ th.cg) * 8, w0, w1)
    cute.arch.barrier()
    # Each warp then writes 4 contiguous 128-byte rows of out_t per pass (16 B per lane).
    ch = th.lane % 8
    start = Int64(tl.tm * _TILE + ch * 16)
    for k in range(4):
        r = k * 32 + th.warp * 4 + th.lane // 8
        s = (r // 8) % 16
        w = _ld_shared_v4(stile + r * _TILE + (((2 * ch) ^ s) & 14) * 8)
        odd = (s % 2) != 0
        lo = [cutlass.select_(odd, w[2], w[0]), cutlass.select_(odd, w[3], w[1])]
        hi = [cutlass.select_(odd, w[0], w[2]), cutlass.select_(odd, w[1], w[3])]
        col = tl.tn * _TILE + r
        col_ok = _b(col < N64)
        addr = t_base + Int64(col) * M64 + start
        if const_expr(m_align == 16):
            _st_v4(addr, lo + hi, col_ok & _b(start < M64))
        else:
            for h, (a0, a1) in enumerate((lo, hi)):
                hs = start + 8 * h
                full = col_ok & _b(hs + 8 <= M64)
                if const_expr(m_align == 8):
                    _st_v2(addr + 8 * h, a0, a1, full)
                else:
                    _st8_any(addr + 8 * h, a0, a1, full)
                    part = col_ok & _b(hs < M64) & (full ^ 1)
                    for i in range(7):
                        wd = a0 if i < 4 else a1
                        pred = part & _b(hs + i < M64)
                        _st_b8(addr + 8 * h + i, wd >> (8 * (i % 4)), pred)


@cute.kernel
def _rowwise_transpose_kernel(
    mX: cute.Tensor,
    mOutTok: cute.Tensor,
    mScaleTok: cute.Tensor,
    mOutT: cute.Tensor,
    mScaleT: cute.Tensor,
    vec_n: cutlass.Constexpr[bool],
    m_align: cutlass.Constexpr[int],
) -> None:
    """One 128x128 tile per CTA."""
    M = mX.shape[0]
    N = mX.shape[1]
    tidx, _, _ = cute.arch.thread_idx()
    tn, tm, _ = cute.arch.block_idx()
    grid_n, grid_m, _ = cute.arch.grid_dim()
    th = _Thread(tidx)
    tl = _Tile(th, tm, tn, M, N)
    smem = cutlass.memory.SmemAllocator()
    sred = smem.allocate_array(Uint32, 8 * 16 * 4, byte_alignment=16).toint()
    stile = smem.allocate_array(cutlass.Uint8, _TILE * _TILE, byte_alignment=128).toint()
    v = _load_global(mX.iterator.toint(), Int64(N), tl, vec_n)
    bases = (
        mOutTok.iterator.toint(),
        mScaleTok.iterator.toint(),
        mOutT.iterator.toint(),
        mScaleT.iterator.toint(),
    )
    _quantize_tile(
        v, th, tl, sred, stile, bases, Int64(M), Int64(N), grid_m, grid_n, vec_n, m_align
    )


@cute.jit
def _launch(
    mX: cute.Tensor,
    mOutTok: cute.Tensor,
    mScaleTok: cute.Tensor,
    mOutT: cute.Tensor,
    mScaleT: cute.Tensor,
    stream,  # noqa: ANN001 - cuda stream (tvm-ffi env stream)
    vec_n: cutlass.Constexpr[bool],
    m_align: cutlass.Constexpr[int],
) -> None:
    M = mX.shape[0]
    N = mX.shape[1]
    _rowwise_transpose_kernel(mX, mOutTok, mScaleTok, mOutT, mScaleT, vec_n, m_align).launch(
        grid=[(N + (_TILE - 1)) // _TILE, (M + (_TILE - 1)) // _TILE, 1],
        block=[_THREADS, 1, 1],
        stream=stream,
    )


# The bit-exactness contract is against the SM90 (software E8M0) branch of the Triton kernel.
_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
_COMPILED: dict[tuple[bool, int], Callable] = {}


def _compile(vec_n: bool, m_align: int) -> Callable:
    m = cute.sym_int()
    n = cute.sym_int()
    sn = cute.sym_int()
    sm = cute.sym_int()

    def fake(dtype, shape, align):  # noqa: ANN001, ANN202
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride=(cute.sym_int64(), 1), assumed_align=align
        )

    def mn_major(rows, cols):  # noqa: ANN001, ANN202 - fp8_cute.scale_layout (1, align(rows, 4))
        return cute.runtime.make_fake_tensor(
            cutlass.Float32, (rows, cols), stride=(1, cute.sym_int64()), assumed_align=16
        )

    return cute.compile(
        _launch,
        fake(cutlass.BFloat16, (m, n), 16 if vec_n else 2),
        fake(cutlass.Float8E4M3FN, (m, n), 16),
        mn_major(m, sn),
        fake(cutlass.Float8E4M3FN, (n, m), 16),
        mn_major(n, sm),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        vec_n,
        m_align,
        options="--enable-tvm-ffi",
    )


def _get_compiled(key: tuple[bool, int]) -> Callable:
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = _compile(*key)
    return fn


def fp8cast_rowwise_transpose(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise FP8 quantization of x *and* x.T (single read of *x*).

    Drop-in for ``deepgemm_quantize.fp8cast_rowwise_transpose``. Non-bf16 or non-contiguous
    inputs are forwarded to the Triton kernel.

    Returns (tok_fp8 (M, N), tok_scale (M, ceil(N/128)) f32, t_fp8 (N, M),
    t_scale (N, ceil(M/128)) f32). Both scales are MN-major and TMA-aligned
    (``fp8_cute.scale_layout``), the layout DeepGEMM's SM90 GEMMs read without a transpose.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, N = x.shape
    if not _SM90 or x.dtype is not torch.bfloat16 or not x.is_contiguous() or not M or not N:
        from pithtrain.operators.deepgemm_quantize import _fp8cast_rowwise_transpose_triton as ref

        out_tok, scale_tok, out_t, scale_t = ref(x)
        return out_tok, to_mn_major(scale_tok), out_t, to_mn_major(scale_t)
    dev = x.device
    out_tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_tok = mn_major_scale(M, (N + 127) >> 7, dev)
    out_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = mn_major_scale(N, (M + 127) >> 7, dev)
    key = (N % 8 == 0 and x.data_ptr() % 16 == 0, 16 if M % 16 == 0 else 8 if M % 8 == 0 else 1)
    fn = _COMPILED.get(key) or _get_compiled(key)
    fn(x, out_tok, scale_tok, out_t, scale_t)
    return out_tok, scale_tok, out_t, scale_t
