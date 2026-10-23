"""
CuTe-DSL port of :func:`pithtrain.operators.deepgemm_quantize.fp8cast_rowwise_kmajor` (SM90).

One read of a BF16 ``(M, N)`` tensor produces

* 1x128 rowwise FP8 + float32 E8M0-valued scales ``(M, ceil(N/128))`` for the m-grouped
  forward/dgrad GEMM, and
* 128x1 per-channel FP8 written in DeepGEMM's K-major per-group flat layout (group ``g`` occupies
  ``[start_g * N, end_g * N)`` as an ``(N, K_g)`` row-major block) + transposed scales
  ``(N, M // 128)`` for the k-grouped wgrad GEMM.

FP8 bytes and scale values are bit-identical to the Triton kernel. Both scales are returned in the
MN-major layout DeepGEMM's SM90 GEMMs read without a ``transpose_fp32`` (strides
``(1, align(rows, 4))``, see :mod:`pithtrain.operators.fp8_cute.scale_layout`).

Each CTA (256 threads) quantizes one 128 x 128 tile; a thread owns an 8 x 8 sub-block (eight
16-byte row vectors), so the per-channel FP8 bytes of eight consecutive rows of one column are
produced directly in K-major order (no transpose). Every row / column scale is computed by one
lane and broadcast with a shuffle, and each element is widened to fp32 once for both outputs.
The group lookup (``searchsorted`` over ``grouped_mm_offs``) is done in-kernel with a warp ballot,
so the wrapper is a single launch.
"""

import struct
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.operators import deepgemm_quantize as _dq
from pithtrain.operators.fp8_cute.scale_layout import mn_major_scale, mn_pad, to_mn_major

__all__ = ["fp8cast_rowwise_kmajor"]

_THREADS = 256
_TILE = 128


def _f32_hex(v: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08X}"


_AMAX_MIN = _f32_hex(1e-4)  # tl.maximum(amax, 1e-4) as fp32
_FP8_MAX_RCP = _f32_hex(1.0 / 448.0)  # FP8_MAX_RCP as fp32


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------


def _asm(res_type, args, ptx: str, cons: str, side_effects: bool = False):
    return llvm.inline_asm(
        res_type,
        args,
        ptx,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_v4(addr: Int64, pred: Int32, *, loc=None, ip=None) -> tuple:
    """16-byte global load; zeros where ``pred == 0``.

    A plain ``ld.global`` (L1-allocating): measured 8% faster for this kernel than the
    ``ld.global.nc.L1::no_allocate`` read-only path (workspace/kda-tasks/round2/expert_quant).
    """
    st = llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()])
    r = _asm(
        st,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
    )
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_b32(addr: Int64, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    """Predicated 4-byte global load returning ``other`` where ``pred == 0``."""
    return Int32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(pred).ir_value(loc=loc, ip=ip),
                Int32(other).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, $3;\n\t"
            "@p ld.global.nc.b32 $0, [$1];\n\t}",
            "=r,l,r,r",
        )
    )


@dsl_user_op
def _st_v2(addr: Int64, a: Uint32, b: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t@p st.global.v2.b32 [$0], {$1, $2};\n\t}",
        "l,r,r,r",
        side_effects=True,
    )


@dsl_user_op
def _st_b32(addr: Int64, a: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}",
        "l,r,r",
        side_effects=True,
    )


@dsl_user_op
def _lds_b32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            side_effects=True,
        )
    )


@dsl_user_op
def _sts_b32(addr: Int32, a: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.b32 [$0], $1;\n\t}",
        "r,r,r",
        side_effects=True,
    )


@dsl_user_op
def _max_u16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Packed max of two halves; equals the bf16 max for non-negative (abs) values."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.u16x2 $0, $1, $2;",
            "=r,r,r",
        )
    )


@dsl_user_op
def _prmt(a: Uint32, b: Uint32, sel: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, $2, 0x{sel:04X};",
            "=r,r,r",
        )
    )


@dsl_user_op
def _selp(pred: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """``pred ? a : b``."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Int32(pred).ir_value(loc=loc, ip=ip),
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\tselp.b32 $0, $2, $3, p;\n\t}",
            "=r,r,r,r",
        )
    )


@dsl_user_op
def _scale(amax2: Uint32, hi: Int32, *, loc=None, ip=None) -> tuple:
    """E8M0 scale of the bf16 amax in the low (``hi == 0``) or high half of ``amax2``.

    Mirrors ``_compute_fp8_scale`` (SM90 branch) bit for bit and returns the fp32 bit patterns
    ``(scale, reciprocal)``.
    """
    st = llvm.StructType.get_literal([T.i32(), T.i32()])
    r = _asm(
        st,
        [Uint32(amax2).ir_value(loc=loc, ip=ip), Int32(hi).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 a, b, m, s;\n\t.reg .f32 f;\n\t.reg .pred p, q;\n\t"
        "setp.ne.b32 q, $3, 0;\n\t"
        "and.b32 a, $2, 0xFFFF0000;\n\t"
        "shl.b32 b, $2, 16;\n\t"
        "selp.b32 a, a, b, q;\n\t"
        f"max.u32 a, a, {_AMAX_MIN};\n\t"  # amax >= 0, so integer max == float max
        "mov.b32 f, a;\n\t"
        f"mul.rn.f32 f, f, 0f{_FP8_MAX_RCP[2:]};\n\t"
        "mov.b32 a, f;\n\t"
        "and.b32 m, a, 0x007FFFFF;\n\t"
        "and.b32 s, a, 0x7F800000;\n\t"
        "setp.ne.b32 p, m, 0;\n\t"
        "@p add.u32 s, s, 0x00800000;\n\t"
        "max.u32 s, s, 0x00800000;\n\t"  # max(scale, 2^-126)
        "mov.b32 $0, s;\n\t"
        "sub.u32 $1, 0x7F000000, s;\n\t}",
        "=r,=r,r,r",
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), r, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), r, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _qblock(words: list, rr: list, cr: list, *, loc=None, ip=None) -> tuple:
    """Quantize a 4 x 4 BF16 block for both outputs, widening each element to fp32 once.

    ``words[r * 2 + q]`` holds row ``r``, columns ``2q, 2q + 1``. Returns four rowwise words (row
    ``r``, columns 0..3, times ``rr[r]``) and four K-major words (column ``c``, rows 0..3, times
    ``cr[c]``); byte ``k`` of a word is its element ``k``. ``cvt.rn.satfinite.e4m3x2.f32`` is what
    Triton's ``.to(tl.float8e4nv)`` lowers to on SM90.
    """
    L = ["{", ".reg .b32 x<16>;", ".reg .f32 y<4>;", ".reg .b16 l, h;"]
    for r in range(4):  # x[r * 4 + c] = element (r, c) as fp32
        for q in range(2):
            w = f"${8 + r * 2 + q}"
            L.append(f"shl.b32 x{r * 4 + 2 * q}, {w}, 16;")
            L.append(f"and.b32 x{r * 4 + 2 * q + 1}, {w}, 0xFFFF0000;")
    for r in range(4):  # rowwise
        for c in range(4):
            L.append(f"mul.rn.f32 y{c}, x{r * 4 + c}, ${16 + r};")
        L += [
            "cvt.rn.satfinite.e4m3x2.f32 l, y1, y0;",
            "cvt.rn.satfinite.e4m3x2.f32 h, y3, y2;",
            f"mov.b32 ${r}, {{l, h}};",
        ]
    for c in range(4):  # K-major
        for r in range(4):
            L.append(f"mul.rn.f32 y{r}, x{r * 4 + c}, ${20 + c};")
        L += [
            "cvt.rn.satfinite.e4m3x2.f32 l, y1, y0;",
            "cvt.rn.satfinite.e4m3x2.f32 h, y3, y2;",
            f"mov.b32 ${4 + c}, {{l, h}};",
        ]
    L.append("}")
    st = llvm.StructType.get_literal([T.i32()] * 8)
    args = [Uint32(x).ir_value(loc=loc, ip=ip) for x in list(words) + list(rr) + list(cr)]
    res = _asm(st, args, "\n\t".join(L), ",".join(["=r"] * 8 + ["r"] * 16))
    out = [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(8)]
    return out[:4], out[4:]


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class _RowwiseKmajor:
    """One 128 x 128 tile per CTA; 256 threads, each owning an 8 x 8 sub-block.

    Lane ``l`` of warp ``w`` owns column vector ``cv = l % 8 + 8 * (w % 2)`` (8 columns) and row
    block ``rb = l // 8 + 4 * (w // 2)`` (8 rows). The grid walks column blocks fastest, so
    co-resident CTAs cover whole rows of the input and outputs (and long runs of each K-major
    column), which keeps DRAM access local.
    """

    RED_WORDS = 64 * 6
    """Shared-memory words _tile reduces through; subclasses (scatter_quant) size theirs by it."""

    def __init__(self, n: int, num_groups: int) -> None:
        self.n = n
        self.nb = (n + _TILE - 1) // _TILE
        self.n_pad = mn_pad(n)  # column stride of the MN-major per-channel scale
        self.num_groups = num_groups

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mOffs: cute.Tensor,
        mTok: cute.Tensor,
        mTokScale: cute.Tensor,
        mKm: cute.Tensor,
        mKmScale: cute.Tensor,
        stream,
    ) -> None:
        mb = mX.shape[0] // _TILE
        self.kernel(mX, mOffs, mTok, mTokScale, mKm, mKmScale).launch(
            grid=[self.nb, mb, 1], block=[_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mOffs: cute.Tensor,
        mTok: cute.Tensor,
        mTokScale: cute.Tensor,
        mKm: cute.Tensor,
        mKmScale: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid_col, pid_row, _ = cute.arch.block_idx()
        gp = (
            mX.iterator.toint(),
            mOffs.iterator.toint(),
            mTok.iterator.toint(),
            mTokScale.iterator.toint(),
            mKm.iterator.toint(),
            mKmScale.iterator.toint(),
        )
        smem = cutlass.memory.SmemAllocator()
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)

        v = []
        for i in cutlass.range_constexpr(8):
            addr, ok = self._src(gp[0], pid_row, pid_col, tidx, i)
            v.append(_ld_v4(addr, ok))
        self._tile(v, pid_row, pid_col, tidx, mX.shape[0], gp, s_red.iterator.toint())

    # -- trace-time helpers (no dynamic control flow) --------------------------------------------

    def _coords(self, tidx):
        lane = tidx % 32
        warp = tidx // 32
        cv = lane % 8 + 8 * (warp % 2)
        rb = lane // 8 + 4 * (warp // 2)
        return lane, warp, cv, rb

    def _src(self, x_base, pid_row, pid_col, tidx, i):
        """Global address of row ``i`` of this thread's sub-block, and whether it is in bounds."""
        _, _, cv, rb = self._coords(tidx)
        col0 = pid_col * _TILE + cv * 8
        row = Int64(pid_row) * _TILE + rb * 8 + i
        return x_base + (row * self.n + col0) * 2, Int32(col0 < self.n)

    def _tile(self, v, pid_row, pid_col, tidx, M, gp, red_base) -> None:
        x_base, offs_base, tok_base, tok_scale_base, km_base, km_scale_base = gp
        N, G = self.n, self.num_groups
        lane, warp, cv, rb = self._coords(tidx)
        cl = lane % 8  # column vector within the warp
        rl = lane // 8  # row block within the warp

        row_start = Int64(pid_row) * _TILE
        col0 = pid_col * _TILE + cv * 8  # first of this thread's 8 columns
        col_ok = Int32(col0 < N)
        row0 = row_start + rb * 8  # first of this thread's 8 rows

        # ---- group lookup: gid = min(searchsorted(offs, row_start, right), G - 1) ----
        cnt = Int32(0)
        for c in range((G + 31) // 32):
            idx = lane + c * 32
            o = _ld_b32(offs_base + Int64(idx) * 4, Int32(idx < G), Int32(0x7FFFFFFF))
            cnt = cnt + cute.arch.popc(cute.arch.vote_ballot_sync(Int64(o) <= row_start))
        gid = cutlass.min(cnt, G - 1)
        g_end = Int64(_ld_b32(offs_base + Int64(gid) * 4, Int32(1), Int32(0)))
        g_start = Int64(_ld_b32(offs_base + Int64(gid - 1) * 4, Int32(gid > 0), Int32(0)))
        k_g = g_end - g_start

        # ---- amax ----
        a = [[v[i][k] & Uint32(0x7FFF7FFF) for k in range(4)] for i in range(8)]
        # column amax over this thread's 8 rows: cp[k] packs columns (2k, 2k + 1)
        cp = []
        for k in range(4):
            m = a[0][k]
            for i in range(1, 8):
                m = _max_u16x2(m, a[i][k])
            cp.append(m)
        # row amax over this thread's 8 columns: rp[p] packs rows (2p, 2p + 1)
        rp = []
        for p in range(4):
            r0, r1 = (
                _max_u16x2(_max_u16x2(a[i][0], a[i][1]), _max_u16x2(a[i][2], a[i][3]))
                for i in (2 * p, 2 * p + 1)
            )
            rp.append(_max_u16x2(_prmt(r0, r1, 0x5410), _prmt(r0, r1, 0x7632)))

        # in-warp butterflies: lane bits 0-2 run along cv, bits 3-4 along rb
        for off in (1, 2, 4):
            for p in range(4):
                rp[p] = _max_u16x2(rp[p], cute.arch.shuffle_sync_bfly(rp[p], offset=off))
        for off in (8, 16):
            for k in range(4):
                cp[k] = _max_u16x2(cp[k], cute.arch.shuffle_sync_bfly(cp[k], offset=off))

        # cross-warp: 2 warps along cv (rows), 4 along rb (columns), through shared memory
        row_red = red_base
        col_red = red_base + 64 * 2 * 4
        for p in range(4):
            _sts_b32(row_red + ((rb * 4 + p) * 2 + warp % 2) * 4, rp[p], Int32(cl == 0))
        for k in range(4):
            _sts_b32(col_red + ((cv * 4 + k) * 4 + warp // 2) * 4, cp[k], Int32(rl == 0))
        cute.arch.sync_threads()
        for p in range(4):
            base = row_red + (rb * 4 + p) * 2 * 4
            rp[p] = _max_u16x2(_lds_b32(base), _lds_b32(base + 4))
        for k in range(4):
            base = col_red + (cv * 4 + k) * 4 * 4
            m = _lds_b32(base)
            for w in range(1, 4):
                m = _max_u16x2(m, _lds_b32(base + w * 4))
            cp[k] = m

        # ---- scales: one lane per (warp, row) / (warp, column), broadcast by shuffle ----
        # this lane: row rb * 8 + cl (half cl % 2 of rp[cl // 2]), columns rl and rl + 4
        rsel = _selp(
            Int32(cl < 4),
            _selp(Int32(cl < 2), rp[0], rp[1]),
            _selp(Int32(cl < 6), rp[2], rp[3]),
        )
        r_sc, r_rc = _scale(rsel, Int32(cl % 2))
        ca_sc, ca_rc = _scale(_selp(Int32(rl < 2), cp[0], cp[1]), Int32(rl % 2))
        cb_sc, cb_rc = _scale(_selp(Int32(rl < 2), cp[2], cp[3]), Int32(rl % 2))
        row_rcp = [
            Uint32(cute.arch.shuffle_sync(Int32(r_rc), offset=(lane & 24) | i)) for i in range(8)
        ]
        col_rcp = [
            Uint32(
                cute.arch.shuffle_sync(Int32(ca_rc if j < 4 else cb_rc), offset=(j % 4) * 8 + cl)
            )
            for j in range(8)
        ]

        # MN-major scales: tok_scale[r, j] at j * M + r, km_scale[c, b] at b * align(N, 4) + c
        _st_b32(tok_scale_base + (Int64(pid_col) * M + row0 + cl) * 4, r_sc, Int32(warp % 2 == 0))
        for c, sc in ((col0 + rl, ca_sc), (col0 + rl + 4, cb_sc)):
            addr = km_scale_base + (Int64(pid_row) * self.n_pad + c) * 4
            _st_b32(addr, sc, Int32(warp // 2 == 0) & Int32(c < N))

        # ---- FP8: rowwise (M, N) and K-major (group block (N, K_g) at g_start * N) ----
        km_row = km_base + g_start * N + (row0 - g_start)
        tok = [[None, None] for _ in range(8)]
        km = [[None, None] for _ in range(8)]
        for m in range(2):
            for h in range(2):
                words = [v[4 * m + r][2 * h + q] for r in range(4) for q in range(2)]
                t4, k4 = _qblock(words, row_rcp[4 * m : 4 * m + 4], col_rcp[4 * h : 4 * h + 4])
                for r in range(4):
                    tok[4 * m + r][h] = t4[r]
                for c in range(4):
                    km[4 * h + c][m] = k4[c]
            for r in range(4):
                i = 4 * m + r
                _st_v2(tok_base + (row0 + i) * N + col0, tok[i][0], tok[i][1], col_ok)
        for j in range(8):
            _st_v2(km_row + Int64(col0 + j) * k_g, km[j][0], km[j][1], col_ok)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

_COMPILED: dict = {}


def _compile(n: int, num_groups: int) -> Callable:
    fake = cute.runtime.make_fake_tensor
    m = cute.sym_int()
    mn = cute.sym_int()
    mb = cute.sym_int()
    nb = (n + _TILE - 1) // _TILE
    x = fake(cutlass.BFloat16, (m, n), stride=(n, 1), assumed_align=16)
    offs = fake(Int32, (num_groups,), stride=(1,), assumed_align=4)
    tok = fake(cutlass.Float8E4M3FN, (m, n), stride=(n, 1), assumed_align=16)
    tok_scale = fake(cutlass.Float32, (m, nb), stride=(1, cute.sym_int64()), assumed_align=16)
    km = fake(cutlass.Float8E4M3FN, (mn,), stride=(1,), assumed_align=16)
    km_scale = fake(cutlass.Float32, (n, mb), stride=(1, mn_pad(n)), assumed_align=16)
    return cute.compile(
        _RowwiseKmajor(n, num_groups),
        x,
        offs,
        tok,
        tok_scale,
        km,
        km_scale,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _empty_outputs(
    x: torch.Tensor, M: int, N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = x.device
    return (
        torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev),
        mn_major_scale(M, (N + _TILE - 1) // _TILE, dev),
        torch.empty(M * N, dtype=torch.float8_e4m3fn, device=dev),
        mn_major_scale(N, M // _TILE, dev),
    )


def fp8cast_rowwise_kmajor(
    x: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise + K-major colwise FP8 quantization (single read of *x*).

    Drop-in for :func:`pithtrain.operators.deepgemm_quantize.fp8cast_rowwise_kmajor`: same
    arguments, output shapes, dtypes, FP8 bytes and scale values; both scales come back MN-major
    (strides ``(1, align(rows, 4))``) so DeepGEMM consumes them without a transpose.
    ``grouped_mm_offs`` holds cumulative row counts (no leading zero) whose entries are multiples
    of 128 and whose last entry is ``M``. Inputs the CuTe kernel does not cover (non-BF16,
    non-contiguous, ``N % 8 != 0``, unaligned, no groups) take the Triton kernel, with its scales
    converted to the same layout.
    """
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (
        f"Expected 2D tensor with M%128==0, got shape {x.shape}"
    )
    M, N = x.shape
    if M == 0:
        return _empty_outputs(x, M, N)
    G = grouped_mm_offs.shape[0]
    if (
        x.dtype != torch.bfloat16
        or N % 8 != 0
        or not x.is_contiguous()
        or x.data_ptr() % 16 != 0
        or grouped_mm_offs.dtype != torch.int32
        or G == 0
        or not grouped_mm_offs.is_contiguous()
    ):
        tok, s_tok, km, s_km = _dq._fp8cast_rowwise_kmajor_triton(x, grouped_mm_offs)
        return tok, to_mn_major(s_tok), km, to_mn_major(s_km)

    out = _empty_outputs(x, M, N)
    if N == 0:
        return out
    key = (N, G)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = _compile(N, G)
    fn(x, grouped_mm_offs, *out)
    return out
