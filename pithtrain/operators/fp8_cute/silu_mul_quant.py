"""
SwiGLU fused with the Hopper FP8 quantization of its consumers, in CuTe-DSL.

The routed experts compute ``h = silu(g) * u`` and hand ``h`` straight to ``down_proj``, which
quantizes it with :func:`~pithtrain.operators.deepgemm_quantize.fp8cast_rowwise_kmajor`. In
backward, ``dh`` turns into ``dg`` and ``du``, which ``gate_proj`` / ``up_proj`` quantize the same
way. Nothing else reads the BF16 ``h``, ``dg`` or ``du``, so the kernels here never write them:

- :func:`silu_mul_quant_fwd` reads ``g`` and ``u`` and writes the four outputs of
  ``fp8cast_rowwise_kmajor(silu_mul(g, u), grouped_mm_offs)``.
- :func:`silu_mul_bwd_quant` reads ``dh``, ``g`` and ``u`` and writes the four outputs of
  ``fp8cast_rowwise_kmajor`` for each of ``dg`` and ``du``.

Both are bit-identical to the unfused Triton sequence (``pithtrain/operators/silu_mul.py`` then
``fp8cast_rowwise_kmajor``): the fp32 SwiGLU arithmetic reproduces the Triton kernels' PTX (see
``_SIGMOID_PTX`` for the one place a shorter, exhaustively verified sequence stands in), each
result is rounded to BF16 exactly as the Triton kernel stores it, and the BF16 value is then
quantized with the SM90 E8M0 scale of ``_compute_fp8_scale``. As from the CuTe
``fp8cast_rowwise_kmajor``, both scales come back MN-major (strides ``(1, align(rows, 4))``, see
:mod:`pithtrain.operators.fp8_cute.scale_layout`), which DeepGEMM reads without a transpose and
which makes every scale store coalesced.

The kernel is persistent (one 512-thread CTA per SM) over 128 x 128 tiles, so a tile's 128 rows
lie inside one expert group exactly as in the Triton kernel. Inputs are staged into shared memory
with ``cp.async`` two tiles ahead. Rowwise amax is a 16-lane butterfly on packed ``max.bf16x2`` of
the BF16 magnitudes; colwise amax reduces 4 rows in registers, then across the warp and the 16
warps through shared memory. The K-major colwise FP8 tile is transposed through a swizzled 16 KB
shared buffer so every column's 128 bytes leave as one coalesced 128-byte burst.

Only ``N % 8 == 0`` with at most 1024 groups runs fused; anything else falls back to the unfused
Triton sequence, which is the reference and therefore trivially bit-identical.
"""

from __future__ import annotations

import struct

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.operators.deepgemm_quantize import fp8cast_rowwise_kmajor
from pithtrain.operators.fp8_cute.scale_layout import mn_major_scale, mn_pad
from pithtrain.operators.silu_mul import silu_mul

__all__ = ["silu_mul_quant_fwd", "silu_mul_bwd_quant"]

_BLOCK = 128  # rows per quantization group and columns per CTA tile
_THREADS = 512
_VEC = 8  # BF16 columns per thread (one 16-byte load)


def _f32_hex(x: float) -> str:
    """PTX hex literal of the float32 nearest to ``x``."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


_FP8_MAX_RCP = _f32_hex(1.0 / 448.0)
_AMAX_MIN = _f32_hex(1e-4)


# ---------------------------------------------------------------------------
# Inline-PTX helpers
# ---------------------------------------------------------------------------


def _asm(res_type, ptx: str, cons: str, args: list, side_effects: bool = False):
    return llvm.inline_asm(
        res_type, args, ptx, cons, has_side_effects=side_effects, is_align_stack=False
    )


def _ir(x, ty, loc=None, ip=None):
    return ty(x).ir_value(loc=loc, ip=ip)


def _unpack(res, n: int, ty, loc=None, ip=None) -> tuple:
    mlir_ty = T.f32() if ty is Float32 else T.i32()
    return tuple(ty(llvm.extractvalue(mlir_ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _ld_b32(addr: Int64, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    """Predicated 4-byte global load; ``other`` when ``pred`` is 0."""
    return Int32(
        _asm(
            T.i32(),
            "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, $3;\n@p ld.global.b32 $0, [$1];\n}",
            "=r,l,r,r",
            [_ir(addr, Int64, loc, ip), _ir(pred, Int32, loc, ip), _ir(other, Int32, loc, ip)],
        )
    )


@dsl_user_op
def _st_v2(addr: Int64, a: Uint32, b: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    _asm(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\n@p st.global.v2.b32 [$1], {$2, $3};\n"
        "mov.b32 $0, 0;\n}",
        "=r,l,r,r,r",
        [_ir(addr, Int64, loc, ip)]
        + [_ir(v, Uint32, loc, ip) for v in (a, b)]
        + [_ir(pred, Int32, loc, ip)],
        side_effects=True,
    )


@dsl_user_op
def _st_v4(addr: Int64, v: tuple, pred: Int32, *, loc=None, ip=None) -> None:
    _asm(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $6, 0;\n@p st.global.v4.b32 [$1], {$2, $3, $4, $5};\n"
        "mov.b32 $0, 0;\n}",
        "=r,l,r,r,r,r,r",
        [_ir(addr, Int64, loc, ip)]
        + [_ir(x, Uint32, loc, ip) for x in v]
        + [_ir(pred, Int32, loc, ip)],
        side_effects=True,
    )


@dsl_user_op
def _st_f32(addr: Int64, v: Float32, pred: Int32, *, loc=None, ip=None) -> None:
    _asm(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\n@p st.global.f32 [$1], $2;\nmov.b32 $0, 0;\n}",
        "=r,l,f,r",
        [_ir(addr, Int64, loc, ip), _ir(v, Float32, loc, ip), _ir(pred, Int32, loc, ip)],
        side_effects=True,
    )


@dsl_user_op
def _sts_b32(addr: Int32, v: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        T.i32(),
        "st.shared.b32 [$1], $2;\nmov.b32 $0, 0;",
        "=r,r,r",
        [_ir(addr, Int32, loc, ip), _ir(v, Uint32, loc, ip)],
        side_effects=True,
    )


@dsl_user_op
def _sts_v4(addr: Int32, v: tuple, *, loc=None, ip=None) -> None:
    _asm(
        T.i32(),
        "st.shared.v4.b32 [$1], {$2, $3, $4, $5};\nmov.b32 $0, 0;",
        "=r,r,r,r,r,r",
        [_ir(addr, Int32, loc, ip)] + [_ir(x, Uint32, loc, ip) for x in v],
        side_effects=True,
    )


@dsl_user_op
def _lds_v4(addr: Int32, *, loc=None, ip=None) -> tuple:
    ty = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = _asm(
        ty,
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        [_ir(addr, Int32, loc, ip)],
        side_effects=True,  # must stay behind the barrier
    )
    return _unpack(res, 4, Uint32, loc, ip)


@dsl_user_op
def _cp_async16(dst: Int32, src: Int64, src_size: Int32, *, loc=None, ip=None) -> None:
    """16-byte global->shared ``cp.async.cg``; zero-fills when ``src_size`` is 0."""
    _asm(
        T.i32(),
        "cp.async.cg.shared.global [$1], [$2], 16, $3;\nmov.b32 $0, 0;",
        "=r,r,l,r",
        [_ir(dst, Int32, loc, ip), _ir(src, Int64, loc, ip), _ir(src_size, Int32, loc, ip)],
        side_effects=True,
    )


@dsl_user_op
def _lds_b32(addr: Int32, pred: Int32, other: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, $3;\n@p ld.shared.b32 $0, [$1];\n}",
            "=r,r,r,r",
            [_ir(addr, Int32, loc, ip), _ir(pred, Int32, loc, ip), _ir(other, Int32, loc, ip)],
            side_effects=True,
        )
    )


@dsl_user_op
def _selp(pred: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """``pred ? a : b``."""
    return Uint32(
        _asm(
            T.i32(),
            "{\n.reg .pred p;\nsetp.ne.b32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            [_ir(pred, Int32, loc, ip), _ir(a, Uint32, loc, ip), _ir(b, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def _bf16_sel(pred_hi: Int32, x: Uint32, *, loc=None, ip=None) -> Float32:
    """fp32 value of the high (``pred_hi``) or low BF16 half of ``x``."""
    return Float32(
        _asm(
            T.f32(),
            "{\n.reg .pred p;\n.reg .b32 a, b;\nsetp.ne.b32 p, $1, 0;\n"
            "and.b32 a, $2, 0xFFFF0000;\nshl.b32 b, $2, 16;\nselp.b32 $0, a, b, p;\n}",
            "=f,r,r",
            [_ir(pred_hi, Int32, loc, ip), _ir(x, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def _f32_as_u32(x: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), "mov.b32 $0, $1;", "=r,f", [_ir(x, Float32, loc, ip)]))


@dsl_user_op
def _u32_as_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), "mov.b32 $0, $1;", "=f,r", [_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def _bf16_lo(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), "shl.b32 $0, $1, 16;", "=f,r", [_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def _bf16_hi(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), "and.b32 $0, $1, 0xFFFF0000;", "=f,r", [_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def _pack_bf16(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Round two fp32 to BF16 (RN, as Triton's ``.to(tl.bfloat16)``) and pack ``{lo, hi}``."""
    return Uint32(
        _asm(
            T.i32(),
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            [_ir(hi, Float32, loc, ip), _ir(lo, Float32, loc, ip)],
        )
    )


@dsl_user_op
def _abs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Magnitude bits of two packed BF16."""
    return Uint32(_asm(T.i32(), "and.b32 $0, $1, 0x7FFF7FFF;", "=r,r", [_ir(x, Uint32, loc, ip)]))


@dsl_user_op
def _max2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Lane-wise max of two packed BF16 magnitudes (NaN-ignoring, like Triton's ``max.f32``)."""
    return Uint32(
        _asm(
            T.i32(),
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            [_ir(a, Uint32, loc, ip), _ir(b, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def _prmt(a: Uint32, b: Uint32, sel: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            f"prmt.b32 $0, $1, $2, {sel:#x};",
            "=r,r,r",
            [_ir(a, Uint32, loc, ip), _ir(b, Uint32, loc, ip)],
        )
    )


@dsl_user_op
def _scale(amax: Float32, *, loc=None, ip=None) -> tuple:
    """SM90 branch of ``_compute_fp8_scale``: (scale, reciprocal), both exact powers of 2."""
    ty = ir.Type.parse("!llvm.struct<(f32, f32)>")
    res = _asm(
        ty,
        "{\n.reg .f32 a, s;\n.reg .b32 b, m, e;\n.reg .pred p;\n"
        f"max.f32 a, $2, {_AMAX_MIN};\n"
        f"mul.rn.f32 s, a, {_FP8_MAX_RCP};\n"
        "mov.b32 b, s;\n"
        "and.b32 m, b, 0x007FFFFF;\n"
        "and.b32 e, b, 0x7F800000;\n"
        "setp.ne.b32 p, m, 0;\n"
        "@p add.s32 e, e, 0x00800000;\n"
        "mov.b32 s, e;\n"
        "max.f32 s, s, 0f00800000;\n"
        "mov.b32 $0, s;\n"
        "mov.b32 e, s;\n"
        "sub.s32 e, 0x7F000000, e;\n"
        "mov.b32 $1, e;\n}",
        "=f,=f,f",
        [_ir(amax, Float32, loc, ip)],
    )
    return _unpack(res, 2, Float32, loc, ip)


@dsl_user_op
def _quant4(
    x0: Float32, x1: Float32, x2: Float32, x3: Float32, r: tuple, *, loc=None, ip=None
) -> Uint32:
    """Four ``cvt.rn.satfinite.e4m3`` of ``x_i * r_i`` packed little-endian into one word."""
    args = [_ir(v, Float32, loc, ip) for v in (x0, x1, x2, x3)]
    args += [_ir(v, Float32, loc, ip) for v in r]
    return Uint32(
        _asm(
            T.i32(),
            "{\n.reg .f32 y0, y1, y2, y3;\n.reg .b16 lo, hi;\n"
            "mul.rn.f32 y0, $1, $5;\nmul.rn.f32 y1, $2, $6;\n"
            "mul.rn.f32 y2, $3, $7;\nmul.rn.f32 y3, $4, $8;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, y1, y0;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, y3, y2;\n"
            "mov.b32 $0, {lo, hi};\n}",
            "=r,f,f,f,f,f,f,f,f",
            args,
        )
    )


# --- SwiGLU math: the exact PTX sequence of the Triton silu_mul kernels -----------------------

# sigmoid(x) = 1 / (1 + exp(-x)). Triton 3.7 emits, for fp32 (see
# workspace/kda-tasks/silu_mul_quant/runs/{fwd,bwd}.ptx):
#     sub.f32 t, 0, x;  mul.f32 t, t, log2e;  ex2.approx.f32 e, t;  add.f32 d, e, 1;
#     div.full.f32 s, 1, d
# which SASS expands to ~12 instructions (denormal rescaling around MUFU.EX2, range scaling
# around MUFU.RCP). This 5-instruction sequence gives the same bits for every BF16 gate value,
# checked exhaustively by workspace/kda-tasks/silu_mul_quant/scripts/verify_sigmoid.py:
# - ``x * -log2e`` == ``(0 - x) * log2e``: negation is exact; ex2(+-0) = 1; NaN is canonical.
# - ex2 non-ftz differs from ftz only when e is subnormal, and then 1 + e == 1 either way.
# - div.full(1, d) == 0.25 * rcp(0.25 * d) for every fp32 d >= 1 (also checked exhaustively),
#   and 0.25 * (1 + e) == fma(e, 0.25, 0.25) because scaling by 0.25 is exact here.
# ``.rn`` pins every mul/add so ptxas cannot contract differently from the reference.
_SIGMOID_PTX = (
    "mul.rn.f32 t, $G, 0fBFB8AA3B;\n"
    "ex2.approx.ftz.f32 t, t;\n"
    "fma.rn.f32 t, t, 0f3E800000, 0f3E800000;\n"
    "rcp.approx.ftz.f32 s, t;\n"
    "mul.rn.f32 s, s, 0f3E800000;\n"
)


@dsl_user_op
def _swiglu_fwd(g: Float32, u: Float32, *, loc=None, ip=None) -> Float32:
    """``silu(g) * u`` in fp32: ``(sigmoid(g) * g) * u``, as ``_silu_mul_fwd_kernel``."""
    return Float32(
        _asm(
            T.f32(),
            "{\n.reg .f32 t, s;\n"
            + _SIGMOID_PTX.replace("$G", "$1")
            + "mul.rn.f32 s, s, $1;\nmul.rn.f32 $0, s, $2;\n}",
            "=f,f,f",
            [_ir(g, Float32, loc, ip), _ir(u, Float32, loc, ip)],
        )
    )


@dsl_user_op
def _swiglu_bwd(dh: Float32, g: Float32, u: Float32, *, loc=None, ip=None) -> tuple:
    """``(grad_gate, grad_up)`` in fp32, as ``_silu_mul_bwd_kernel`` after LLVM contraction:

    grad_up   = dh * (s * g)
    grad_gate = ((dh * u) * s) * fma(-s, g, 1 + g)      with s = sigmoid(g)
    """
    ty = ir.Type.parse("!llvm.struct<(f32, f32)>")
    res = _asm(
        ty,
        "{\n.reg .f32 t, s, a, b;\n"
        + _SIGMOID_PTX.replace("$G", "$3")
        + "mul.rn.f32 a, s, $3;\n"  # silu
        "mul.rn.f32 $1, a, $2;\n"  # grad_up
        "mul.rn.f32 b, $2, $4;\n"  # dh * u
        "mul.rn.f32 b, b, s;\n"  # * sigmoid
        "add.rn.f32 a, $3, 0f3F800000;\n"  # 1 + g
        "neg.f32 t, s;\n"
        "fma.rn.f32 a, t, $3, a;\n"  # 1 + g - s * g, unrounded product
        "mul.rn.f32 $0, b, a;\n}",
        "=f,=f,f,f,f",
        [_ir(v, Float32, loc, ip) for v in (dh, g, u)],
    )
    return _unpack(res, 2, Float32, loc, ip)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


_G_CAP = 1024  # expert groups the shared-memory offset table holds
_STAGES = 2  # cp.async tiles in flight per CTA (3 measured no faster; bwd cannot fit 3)


class _SiluMulQuant:
    """Persistent kernel, one compiled per (N, backward).

    Each CTA (512 threads, one per SM) walks 128 x 128 tiles ``bid, bid + grid, ...``. Inputs are
    staged into shared memory with ``cp.async`` ``stages`` tiles ahead, so HBM reads continue
    through the reductions and barriers of the current tile. Every thread reads back only the
    16-byte chunks it copied itself, so the staging needs no barrier, only ``cp.async.wait_group``.

    Thread ``t``: ``cv = t % 16`` owns columns ``cv*8 .. cv*8+7``, ``rg = t // 16`` owns rows
    ``rg*4 .. rg*4+3`` of the tile.
    """

    def __init__(self, N: int, backward: bool, num_ctas: int):
        self.N = N
        self.backward = backward
        self.NB = (N + _BLOCK - 1) // _BLOCK
        self.n_in = 3 if backward else 2
        self.n_out = 2 if backward else 1
        self.stages = _STAGES
        self.num_ctas = num_ctas
        self.stage_bytes = self.n_in * 4 * _THREADS * 16

    @cute.jit
    def __call__(
        self,
        mDh: cute.Tensor,
        mG: cute.Tensor,
        mU: cute.Tensor,
        mOffs: cute.Tensor,
        mTok0: cute.Tensor,
        mTokS0: cute.Tensor,
        mKm0: cute.Tensor,
        mKmS0: cute.Tensor,
        mTok1: cute.Tensor,
        mTokS1: cute.Tensor,
        mKm1: cute.Tensor,
        mKmS1: cute.Tensor,
        stream,
    ):
        total = (mG.shape[0] // _BLOCK) * self.NB
        self.kernel(
            mDh, mG, mU, mOffs, mTok0, mTokS0, mKm0, mKmS0, mTok1, mTokS1, mKm1, mKmS1
        ).launch(
            grid=[cutlass.min(total, self.num_ctas), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    def _tile(self, t: Int32) -> tuple:
        """Tile ``t`` -> (row block, column block); column blocks of a row block are adjacent,
        which measured faster than the transposed order (better read locality)."""
        return t // self.NB, t % self.NB

    def _issue(self, t: Int32, stage_addr: Int32, total: Int32, tid: Int32, in_base: list):
        """cp.async tile ``t`` (zero-filled when out of range) into a stage; traced inline."""
        N = self.N
        ok = Int32(t < total)
        rb, cb = self._tile(t)
        r = rb * _BLOCK + (tid // 16) * 4
        c = cb * _BLOCK + (tid % 16) * _VEC
        if cutlass.const_expr(N % _BLOCK != 0):
            ok = ok & Int32(c < N)
        off = (Int64(r) * N + c) * 2 * Int64(ok)
        for x in range(self.n_in):
            for k in range(4):
                _cp_async16(
                    stage_addr + ((x * 4 + k) * _THREADS + tid) * 16,
                    in_base[x] + off + k * N * 2,
                    ok * 16,
                )
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        mDh: cute.Tensor,
        mG: cute.Tensor,
        mU: cute.Tensor,
        mOffs: cute.Tensor,
        mTok0: cute.Tensor,
        mTokS0: cute.Tensor,
        mKm0: cute.Tensor,
        mKmS0: cute.Tensor,
        mTok1: cute.Tensor,
        mTokS1: cute.Tensor,
        mKm1: cute.Tensor,
        mKmS1: cute.Tensor,
    ):
        N = self.N
        NB = self.NB
        S = self.stages
        even_n = cutlass.const_expr(N % _BLOCK == 0)
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = tid % 32
        warp = tid // 32
        cv = tid % 16
        rg = tid // 16
        M = mG.shape[0]
        nrb = M // _BLOCK
        total = nrb * NB
        G = mOffs.shape[0]

        smem = cutlass.memory.SmemAllocator()
        s_in = smem.allocate(S * self.stage_bytes, byte_alignment=128).toint()
        s_tr = smem.allocate(_BLOCK * _BLOCK, byte_alignment=128).toint()
        s_cm = smem.allocate((_THREADS // 32) * 16 * 16, byte_alignment=16).toint()
        s_rcp = smem.allocate(_BLOCK * 4, byte_alignment=16).toint()
        s_offs = smem.allocate(_G_CAP * 4, byte_alignment=16).toint()

        ins = [mG, mU]
        if cutlass.const_expr(self.backward):
            ins = [mDh, mG, mU]
        in_base = [t.iterator.toint() for t in ins]

        for s in cutlass.range_constexpr(S):
            self._issue(bid + s * grid, s_in + s * self.stage_bytes, total, tid, in_base)

        # Expert-group offsets into shared memory (searched once per tile below).
        for gc in cutlass.range((G + _THREADS - 1) // _THREADS, unroll=1):
            j = gc * _THREADS + tid
            if j < G:
                v = _ld_b32(mOffs.iterator.toint() + Int64(j) * 4, Int32(1), Int32(0))
                _sts_b32(s_offs + j * 4, Uint32(v))
        cute.arch.sync_threads()

        outs = [(mTok0, mTokS0, mKm0, mKmS0), (mTok1, mTokS1, mKm1, mKmS1)]
        n_it = (total - bid + grid - 1) // grid
        for it in cutlass.range(n_it, unroll=1):
            t = bid + it * grid
            rb, cb = self._tile(t)
            row0 = rb * _BLOCK
            col0 = cb * _BLOCK
            col = col0 + cv * _VEC
            col_ok = Int32(1)
            if cutlass.const_expr(not even_n):
                col_ok = Int32(col < N)
            stage_addr = s_in + (it % S) * self.stage_bytes

            cute.arch.cp_async_wait_group(S - 1)
            raw = [[None] * 4 for _ in range(self.n_in)]
            for x in cutlass.range_constexpr(self.n_in):
                for k in cutlass.range_constexpr(4):
                    raw[x][k] = _lds_v4(stage_addr + ((x * 4 + k) * _THREADS + tid) * 16)

            # SwiGLU in fp32, rounded to BF16 and packed in pairs: h[o][k][m].
            h = [[[None] * 4 for _ in range(4)] for _ in range(self.n_out)]
            for k in cutlass.range_constexpr(4):
                for m in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self.backward):
                        d, g, u = raw[0][k][m], raw[1][k][m], raw[2][k][m]
                        dg_lo, du_lo = _swiglu_bwd(_bf16_lo(d), _bf16_lo(g), _bf16_lo(u))
                        dg_hi, du_hi = _swiglu_bwd(_bf16_hi(d), _bf16_hi(g), _bf16_hi(u))
                        h[0][k][m] = _pack_bf16(dg_lo, dg_hi)
                        h[1][k][m] = _pack_bf16(du_lo, du_hi)
                    else:
                        g, u = raw[0][k][m], raw[1][k][m]
                        lo = _swiglu_fwd(_bf16_lo(g), _bf16_lo(u))
                        hi = _swiglu_fwd(_bf16_hi(g), _bf16_hi(u))
                        h[0][k][m] = _pack_bf16(lo, hi)

            # The stage is consumed: refill it with the tile S iterations ahead.
            self._issue(t + S * grid, stage_addr, total, tid, in_base)

            # Expert group of this row block: searchsorted(offs, row0, right=True), clamped.
            cnt = Int32(0)
            for gc in cutlass.range((G + 31) // 32, unroll=1):
                j = gc * 32 + lane
                v = _lds_b32(s_offs + j * 4, Int32(j < G), Int32(0x7FFFFFFF))
                cnt += cute.arch.popc(cute.arch.vote_ballot_sync(v <= row0))
            gid = cutlass.min(cnt, G - 1)
            ge = _lds_b32(s_offs + gid * 4, Int32(1), Int32(0))
            gs = _lds_b32(s_offs + (gid - 1) * 4, Int32(gid > 0), Int32(0))
            k_g = Int64(ge - gs)

            for o in cutlass.range_constexpr(self.n_out):
                mTok, mTokS, mKm, mKmS = outs[o]
                ho = h[o]
                a = [[_abs2(ho[k][m]) for m in range(4)] for k in range(4)]

                # -- colwise amax: 4 rows in registers, the other row group in the warp, then
                #    one partial per warp into shared memory --
                cm = [None] * 4
                for m in cutlass.range_constexpr(4):
                    x = _max2(_max2(a[0][m], a[1][m]), _max2(a[2][m], a[3][m]))
                    y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=16))
                    cm[m] = _max2(x, y)
                if lane < 16:
                    _sts_v4(s_cm + (warp * 16 + cv) * 16, tuple(cm))

                # -- rowwise amax: rows (0,1) and (2,3) packed per word, 16-lane butterfly --
                c4 = [_max2(_max2(a[k][0], a[k][1]), _max2(a[k][2], a[k][3])) for k in range(4)]
                rw = [None] * 2
                for p in cutlass.range_constexpr(2):
                    x0, x1 = c4[2 * p], c4[2 * p + 1]
                    x = _max2(_prmt(x0, x1, 0x5410), _prmt(x0, x1, 0x7632))
                    for sh in cutlass.range_constexpr(4):
                        y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=1 << sh))
                        x = _max2(x, y)
                    rw[p] = x

                # -- rowwise scale: lane cv < 4 of each row group owns row rg*4 + cv --
                r_amax = _bf16_sel(Int32(cv % 2), _selp(Int32(cv < 2), rw[0], rw[1]))
                r_sc, r_rc = _scale(r_amax)
                _st_f32(
                    mTokS.iterator.toint() + (Int64(cb) * M + row0 + rg * 4 + cv) * 4,
                    r_sc,
                    Int32(cv < 4),
                )
                for k in cutlass.range_constexpr(4):
                    rc = cute.arch.shuffle_sync(r_rc, offset=(lane & 16) | k)
                    rr = (rc, rc, rc, rc)
                    hv = ho[k]
                    q0 = _quant4(
                        _bf16_lo(hv[0]), _bf16_hi(hv[0]), _bf16_lo(hv[1]), _bf16_hi(hv[1]), rr
                    )
                    q1 = _quant4(
                        _bf16_lo(hv[2]), _bf16_hi(hv[2]), _bf16_lo(hv[3]), _bf16_hi(hv[3]), rr
                    )
                    r = row0 + rg * 4 + k
                    _st_v2(mTok.iterator.toint() + Int64(r) * N + col, q0, q1, col_ok)

                cute.arch.sync_threads()

                # -- colwise scale: thread c < 128 owns column c of the tile --
                if tid < _BLOCK:
                    cw = Uint32(0)
                    for w in cutlass.range_constexpr(_THREADS // 32):
                        v = _lds_b32(
                            s_cm + (w * 16 + tid // 8) * 16 + ((tid % 8) // 2) * 4,
                            Int32(1),
                            Int32(0),
                        )
                        cw = _max2(cw, Uint32(v))
                    c_sc, c_rc = _scale(_bf16_sel(Int32(tid % 2), cw))
                    c = col0 + tid
                    c_ok = Int32(1)
                    if cutlass.const_expr(not even_n):
                        c_ok = Int32(c < N)
                    _st_f32(mKmS.iterator.toint() + (Int64(rb) * mn_pad(N) + c) * 4, c_sc, c_ok)
                    _sts_b32(s_rcp + tid * 4, Uint32(_f32_as_u32(c_rc)))

                cute.arch.sync_threads()

                # -- colwise FP8 into the transposed, swizzled tile (16 B chunks ^ cv%8) --
                rv = _lds_v4(s_rcp + cv * 32) + _lds_v4(s_rcp + cv * 32 + 16)
                sw = (cv & 7) << 4
                for j in cutlass.range_constexpr(8):
                    m = j // 2
                    f = _bf16_lo if j % 2 == 0 else _bf16_hi
                    rc = _u32_as_f32(rv[j])
                    q = _quant4(f(ho[0][m]), f(ho[1][m]), f(ho[2][m]), f(ho[3][m]), (rc,) * 4)
                    cl = cv * _VEC + j
                    _sts_b32(s_tr + cl * _BLOCK + (((rg // 4) << 4) ^ sw) + (rg % 4) * 4, q)

                cute.arch.sync_threads()

                # -- K-major store: 8 threads per column, 128 contiguous bytes per column --
                km_base = mKm.iterator.toint() + Int64(gs) * N + Int64(row0 - gs)
                for p in cutlass.range_constexpr(_BLOCK * _BLOCK // 16 // _THREADS):
                    idx = p * _THREADS + tid
                    cl = idx // 8
                    qq = idx % 8
                    v = _lds_v4(s_tr + cl * _BLOCK + ((qq ^ ((cl // 8) & 7)) << 4))
                    c = col0 + cl
                    c_ok = Int32(1)
                    if cutlass.const_expr(not even_n):
                        c_ok = Int32(c < N)
                    _st_v4(km_base + Int64(c) * k_g + qq * 16, v, c_ok)

        cute.arch.cp_async_wait_group(0)


_compiled: dict = {}


def _fake(dtype: type, shape: tuple, stride: tuple, align: int = 16) -> object:
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)


def _get_kernel(N: int, backward: bool, device: torch.device) -> object:
    """Compiled kernel for (N, direction, device); M and the group count stay dynamic."""
    key = (N, backward, device.index)
    fn = _compiled.get(key)
    if fn is None:
        m = cute.sym_int()
        nb = (N + _BLOCK - 1) // _BLOCK
        bf = _fake(cutlass.BFloat16, (m, N), (N, 1))
        fp8 = cutlass.Float8E4M3FN
        outs = (
            _fake(fp8, (m, N), (N, 1)),
            _fake(Float32, (m, nb), (1, cute.sym_int64()), 16),
            _fake(fp8, (cute.sym_int(),), (1,)),
            _fake(Float32, (N, cute.sym_int()), (1, mn_pad(N)), 16),
        )
        offs = _fake(Int32, (cute.sym_int(),), (1,), 4)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        fn = cute.compile(
            _SiluMulQuant(N, backward, num_sms),
            bf if backward else None,
            bf,
            bf,
            offs,
            *outs,
            *(outs if backward else (None,) * 4),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compiled[key] = fn
    return fn


def _alloc(
    M: int, N: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four outputs of ``fp8cast_rowwise_kmajor``, each its own allocation (they have
    different lifetimes: the rowwise pair feeds the next GEMM, the K-major pair waits for wgrad)."""
    nb = (N + _BLOCK - 1) // _BLOCK
    return (
        torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device),
        mn_major_scale(M, nb, device),
        torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device),
        mn_major_scale(N, M // _BLOCK, device),
    )


def _check(xs: list[torch.Tensor], grouped_mm_offs: torch.Tensor) -> None:
    for x in xs:
        assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.is_contiguous()
        assert x.shape == xs[0].shape
    assert xs[0].shape[0] % _BLOCK == 0, f"M must be a multiple of 128, got {xs[0].shape}"
    assert grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous()


def _fusable(N: int, grouped_mm_offs: torch.Tensor) -> bool:
    """Shapes the fused kernel handles; others take the (bit-identical) unfused path."""
    return N % _VEC == 0 and 0 < grouped_mm_offs.shape[0] <= _G_CAP


def silu_mul_quant_fwd(
    g: torch.Tensor,
    u: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(silu_mul(g, u), grouped_mm_offs)`` without materializing h.

    Args:
        g, u: BF16 (M, N) gate / up projections, M a multiple of 128.
        grouped_mm_offs: int32 (num_groups,) cumulative row counts, multiples of 128,
            last entry == M.

    Returns:
        (tok_fp8, tok_scale, kmajor_fp8, kmajor_scale), bit-identical to the unfused sequence;
        both scales MN-major.
    """
    _check([g, u], grouped_mm_offs)
    M, N = g.shape
    if M != 0 and N != 0 and not _fusable(N, grouped_mm_offs):
        return fp8cast_rowwise_kmajor(silu_mul(g, u), grouped_mm_offs)
    out = _alloc(M, N, g.device)
    if M == 0 or N == 0:
        return out
    _get_kernel(N, False, g.device)(None, g, u, grouped_mm_offs, *out, None, None, None, None)
    return out


def silu_mul_bwd_quant(
    dh: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """``fp8cast_rowwise_kmajor`` of the SwiGLU gradients without materializing them.

    Args:
        dh: BF16 (M, N) gradient of ``silu(g) * u``.
        g, u: BF16 (M, N) saved gate / up projections.
        grouped_mm_offs: as in :func:`silu_mul_quant_fwd`.

    Returns:
        (dg_quant, du_quant), each the 4-tuple ``fp8cast_rowwise_kmajor`` returns for the BF16
        ``grad_gate`` / ``grad_up`` of ``silu_mul``'s backward, bit for bit.
    """
    _check([dh, g, u], grouped_mm_offs)
    M, N = g.shape
    if M == 0 or N == 0:
        return _alloc(M, N, g.device), _alloc(M, N, g.device)
    if not _fusable(N, grouped_mm_offs):
        dg, du = _silu_mul_bwd_reference(dh, g, u)
        return fp8cast_rowwise_kmajor(dg, grouped_mm_offs), fp8cast_rowwise_kmajor(
            du, grouped_mm_offs
        )
    dg_q = _alloc(M, N, g.device)
    du_q = _alloc(M, N, g.device)
    _get_kernel(N, True, g.device)(dh, g, u, grouped_mm_offs, *dg_q, *du_q)
    return dg_q, du_q


def _silu_mul_bwd_reference(
    dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 (grad_gate, grad_up) from the Triton silu_mul backward kernel."""
    g = g.detach().requires_grad_()
    u = u.detach().requires_grad_()
    silu_mul(g, u).backward(dh)
    return g.grad, u.grad
