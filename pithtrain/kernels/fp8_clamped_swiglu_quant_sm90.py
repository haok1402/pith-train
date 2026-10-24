"""
gpt-oss's clamped SwiGLU fused with the Hopper FP8 quantization of its consumers, in CuTe-DSL.

The gpt-oss routed experts compute ``gate_up = x @ W_gu.T + b_gu[group]`` (interleaved: gate at
even, up at odd columns), ``h = clamped_swiglu(gate_up)`` and hand ``h`` to ``down_proj``, which
quantizes it with :func:`~pithtrain.kernels.fp8_rowwise_kmajor.fp8cast_rowwise_kmajor`. In
backward, ``dh`` turns into ``d(gate_up)``, which the gate_up GEMMs quantize the same way and the
bias gradient sums per expert. Nothing else reads the BF16 biased ``gate_up``, ``h`` or
``d(gate_up)``, so the kernels here never write them:

- :func:`clamped_swiglu_quant_fwd_sm90` (and its rowwise / K-major halves) reads the raw
  (bias-free) gate_up GEMM output and the bias and writes the four outputs of
  ``fp8cast_rowwise_kmajor(clamped_swiglu(indexed_bias_add(gate_up, bias, ...)), offs)``.
- :func:`clamped_swiglu_bwd_quant_sm90` reads ``dh``, the raw gate_up and the bias, and writes the
  four outputs of ``fp8cast_rowwise_kmajor(d(gate_up), offs)`` plus the bias gradient.

The quantized outputs are bit-identical to the unfused sequence (``indexed_bias_add``, the Triton
kernels of ``pithtrain/kernels/clamped_swiglu.py``, then ``fp8cast_rowwise_kmajor``): the bias
add is one fp32 add rounded to BF16 (what eager ``x + bias[group_ids]`` does), the fp32 clamped
SwiGLU replays the Triton kernels' PTX (workspace/kda-tasks/round3/gptoss/profile/{fwd,bwd}.ptx,
including the ``fma`` LLVM contracts in backward), and each result is rounded to BF16 exactly as
the Triton kernel stores it before the SM90 E8M0 quantization. Both scales come back MN-major as
from the CuTe ``fp8cast_rowwise_kmajor``.

The bias gradient is the per-expert column sum of the BF16 ``d(gate_up)``. The kernel writes the
fp32 column sums of every 128-row block, and a small Triton kernel (``kernels.segment_sum.block_segment_sum``) adds the blocks of each
expert in row order. The reference (``indexed_bias_add``'s segment sum) has the same per-128-row-block
structure but a different order inside a block, so the bias gradient matches it to fp32
reassociation, not bit for bit.

The kernel is the persistent design of :mod:`pithtrain.kernels.fp8_silu_mul_quant_sm90` (one
512-thread CTA per SM over 128 x 128 output tiles, ``cp.async`` staging two tiles ahead, packed
``max.bf16x2`` amax, swizzled shared-memory transpose for the K-major store). A thread owns four
rows and eight output columns; in forward those are sixteen interleaved input columns (two 16-byte
chunks per row), in backward eight interleaved gradient columns, i.e. four ``dh`` columns (8 bytes)
and eight gate_up columns (16 bytes) per row.

Only ``N % 8 == 0`` with 1 to 1024 groups and a BF16 bias runs fused; anything else falls back to
the unfused sequence (``biased_gate_up`` of the ``fp8_clamped_swiglu_quant_sm100`` sibling, the
Triton ``clamped_swiglu`` kernels, ``block_segment_sum`` and the SM90 ``fp8cast_rowwise_kmajor_sm90``),
which is the reference and therefore trivially bit-identical.
"""

from __future__ import annotations

import struct

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.arch import sm_count
from pithtrain.kernels.clamped_swiglu import clamped_swiglu_fwd
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import (
    abs2,
    asm_args_last,
    bf16_hi,
    bf16_lo,
    bf16_sel,
    cp_async8,
    cp_async16,
    f32_as_u32,
    ld_b32_coherent,
    lds_b32_pred,
    lds_v2,
    lds_v4,
    max2,
    pack_bf16,
    prmt_args_last,
    selp_args_last,
    st_f32,
    st_v2_args_last,
    st_v4_args_last,
    sts_b32_nopred,
    sts_v4,
    to_ir,
    u32_as_f32,
    unpack,
)
from pithtrain.kernels.fp8_clamped_swiglu_quant_sm100 import (
    biased_gate_up,
    clamped_swiglu_bwd_biased,
)
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale_f32, quant4_e4m3
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.segment_sum import block_segment_sum

__all__ = [
    "ClampedSwiGLUQuant",
    "clamped_swiglu_bwd_quant_sm90",
    "clamped_swiglu_quant_fusable",
    "clamped_swiglu_quant_fwd_sm90",
    "clamped_swiglu_quant_kmajor_sm90",
    "clamped_swiglu_quant_rowwise_sm90",
]

_BLOCK = 128  # rows per quantization group and output columns per CTA tile
_THREADS = 512
_VEC = 8  # output columns per thread
_G_CAP = 1024  # expert groups the shared-memory offset table holds
_STAGES = {False: 3, True: 2}  # cp.async tiles in flight per CTA, by direction (backward):
# forward 3 measured ~2% faster than 2 (and 1), backward flat (scripts/bench_stages.py)


def _f32_hex(x: float) -> str:
    """PTX hex literal of the float32 nearest to ``x``."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


def _f32(x: float) -> float:
    """``x`` rounded to float32, as Triton receives a Python float scalar argument."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------
# Inline-PTX helpers not in kernels/cute_ptx_sm90.py
# ---------------------------------------------------------------------------


@dsl_user_op
def _ld_v4(addr: Int64, pred: Int32, *, loc=None, ip=None) -> tuple:
    """16-byte global load (L1-allocating: the bias row is reused by every row block of its
    expert); zeros where ``pred == 0``."""
    ty = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = asm_args_last(
        ty,
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        [to_ir(addr, Int64, loc, ip), to_ir(pred, Int32, loc, ip)],
    )
    return unpack(res, 4, Uint32, loc, ip)


@dsl_user_op
def _add_bias2(w: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Two packed BF16 plus two packed BF16, each an fp32 add rounded to BF16 (RN)."""
    return Uint32(
        asm_args_last(
            T.i32(),
            "{\n.reg .f32 a0, a1, b0, b1;\n"
            "shl.b32 a0, $1, 16;\nand.b32 a1, $1, 0xFFFF0000;\n"
            "shl.b32 b0, $2, 16;\nand.b32 b1, $2, 0xFFFF0000;\n"
            "add.rn.f32 a0, a0, b0;\nadd.rn.f32 a1, a1, b1;\n"
            "cvt.rn.bf16x2.f32 $0, a1, a0;\n}",
            "=r,r,r",
            [to_ir(w, Uint32, loc, ip), to_ir(b, Uint32, loc, ip)],
        )
    )


# --- clamped SwiGLU math: the exact PTX sequence of the Triton clamped_swiglu kernels ---------
#
# Triton 3.7 emits (profile/fwd.ptx, profile/bwd.ptx), with L = limit, a = alpha:
#   gc = min.f32(g, L);  uc = min.f32(max.f32(u, 0 - L), L)
#   sigmoid(a * gc):  t = fma(-a, gc, 0);  t = t * log2e;  e = ex2.approx.f32 t;  d = e + 1;
#                     s = div.full.f32(1, d)
# fma(-a, gc, 0) == -(a * gc) rounded, so ``t * log2e`` == ``(a * gc) * -log2e`` (negation is
# exact). The remaining sigmoid steps are the 5-instruction sequence of silu_mul_quant's
# _SIGMOID_PTX, verified there for every fp32 d >= 1 (div.full) and every ex2 input.
#   forward:  out = (uc + 1) * (gc * s)
#   backward: grad_uc = (gc * s) * do
#             grad_gc = ((uc + 1) * do * s) * fma(a * gc, 1 - s, 1)      (LLVM contraction)
#             grad_g  = g <= L ? grad_gc : 0;   grad_u = -L <= u <= L ? grad_uc : 0
def _sigmoid_ptx(x: str) -> str:
    """PTX computing ``s = sigmoid(-(-x))`` of the fp32 register ``x`` (uses ``t``, ``s``)."""
    return (
        f"mul.rn.f32 t, {x}, 0fBFB8AA3B;\n"
        "ex2.approx.ftz.f32 t, t;\n"
        "fma.rn.f32 t, t, 0f3E800000, 0f3E800000;\n"
        "rcp.approx.ftz.f32 s, t;\n"
        "mul.rn.f32 s, s, 0f3E800000;\n"
    )


def _clamp_ptx(alpha: float, limit: float) -> str:
    """gc, uc, ag = alpha * gc and s = sigmoid(ag) from g = $G, u = $U."""
    L, nL, A = _f32_hex(limit), _f32_hex(-limit), _f32_hex(alpha)
    return (
        f"min.f32 gc, $G, {L};\n"
        f"max.f32 uc, $U, {nL};\n"
        f"min.f32 uc, uc, {L};\n"
        f"mul.rn.f32 ag, gc, {A};\n" + _sigmoid_ptx("ag")
    )


def _make_fwd_op(alpha: float, limit: float):
    body = _clamp_ptx(alpha, limit).replace("$G", "$1").replace("$U", "$2")

    @dsl_user_op
    def op(g: Float32, u: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(
            asm_args_last(
                T.f32(),
                "{\n.reg .f32 gc, uc, ag, t, s;\n" + body + "mul.rn.f32 s, gc, s;\n"  # glu
                "add.rn.f32 uc, uc, 0f3F800000;\n"
                "mul.rn.f32 $0, uc, s;\n}",
                "=f,f,f",
                [to_ir(g, Float32, loc, ip), to_ir(u, Float32, loc, ip)],
            )
        )

    return op


def _make_bwd_op(alpha: float, limit: float):
    body = _clamp_ptx(alpha, limit).replace("$G", "$3").replace("$U", "$4")
    L, nL = _f32_hex(limit), _f32_hex(-limit)

    @dsl_user_op
    def op(do: Float32, g: Float32, u: Float32, *, loc=None, ip=None) -> tuple:
        ty = ir.Type.parse("!llvm.struct<(f32, f32)>")
        res = asm_args_last(
            ty,
            "{\n.reg .f32 gc, uc, ag, t, s, a, b;\n.reg .pred pg, pu, pl;\n"
            + body
            + "add.rn.f32 uc, uc, 0f3F800000;\n"
            "mul.rn.f32 a, uc, $2;\n"  # (uc + 1) * do
            "mul.rn.f32 b, gc, s;\n"  # glu
            "mul.rn.f32 b, b, $2;\n"  # grad_uc
            "mul.rn.f32 a, a, s;\n"
            "sub.rn.f32 t, 0f3F800000, s;\n"
            "fma.rn.f32 t, ag, t, 0f3F800000;\n"
            "mul.rn.f32 a, a, t;\n"  # grad_gc
            f"setp.le.f32 pg, $3, {L};\n"
            f"setp.le.f32 pu, $4, {L};\n"
            f"setp.ge.f32 pl, $4, {nL};\n"
            "and.pred pu, pu, pl;\n"
            "selp.f32 $0, a, 0f00000000, pg;\n"
            "selp.f32 $1, b, 0f00000000, pu;\n}",
            "=f,=f,f,f,f",
            [to_ir(v, Float32, loc, ip) for v in (do, g, u)],
        )
        return unpack(res, 2, Float32, loc, ip)

    return op


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class ClampedSwiGLUQuant:
    """Persistent kernel, one compiled per (N, backward, alpha, limit).

    ``N`` is the SwiGLU output width (gate_up has 2N columns). The quantized output is ``h``
    (M, N) in forward and ``d(gate_up)`` (M, 2N) in backward; tiles are 128 x 128 of it.
    Thread ``t``: ``cv = t % 16`` owns output columns ``cv*8 .. cv*8+7``, ``rg = t // 16`` owns
    rows ``rg*4 .. rg*4+3`` of the tile.

    Stage layout (bytes, per stage): forward, two 16-byte slots per (row, thread) holding the
    interleaved gate_up columns of output columns ``+0..3`` and ``+4..7``; backward, an 8-byte
    ``dh`` slot then a 16-byte gate_up slot per (row, thread).
    """

    def __init__(
        self,
        N: int,
        backward: bool,
        alpha: float,
        limit: float,
        num_ctas: int,
        KP: int,
        rowwise: bool = True,
        kmajor: bool = True,
    ):
        self.N = N
        self.backward = backward
        self.NO = 2 * N if backward else N
        self.KP = KP  # channels of the K-major output (>= NO; the rest is left unwritten)
        self.rowwise = rowwise  # False: only the K-major pair (backward recompute of h)
        self.kmajor = kmajor  # False: only the rowwise pair (forward when h is recomputed)
        assert rowwise or kmajor
        assert kmajor or not backward
        self.NB = (self.NO + _BLOCK - 1) // _BLOCK
        self.stages = _STAGES[backward]
        self.num_ctas = num_ctas
        self.stage_bytes = 4 * _THREADS * (24 if backward else 32)
        self.fwd_op = _make_fwd_op(alpha, limit)
        self.bwd_op = _make_bwd_op(alpha, limit)

    @cute.jit
    def __call__(
        self,
        mDh: cute.Tensor,
        mGU: cute.Tensor,
        mBias: cute.Tensor,
        mOffs: cute.Tensor,
        mTok: cute.Tensor,
        mTokS: cute.Tensor,
        mKm: cute.Tensor,
        mKmS: cute.Tensor,
        mPart: cute.Tensor,
        stream,
    ):
        total = (mGU.shape[0] // _BLOCK) * self.NB
        self.kernel(mDh, mGU, mBias, mOffs, mTok, mTokS, mKm, mKmS, mPart).launch(
            grid=[cutlass.min(total, self.num_ctas), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    def _tile(self, t: Int32) -> tuple:
        """Tile ``t`` -> (row block, column block); column blocks of a row block adjacent."""
        return t // self.NB, t % self.NB

    def _issue(self, t: Int32, stage_addr: Int32, total: Int32, tid: Int32, dh_base, gu_base):
        """cp.async tile ``t`` (zero-filled when out of range) into a stage; traced inline."""
        N, NO = self.N, self.NO
        ok = Int32(t < total)
        rb, cb = self._tile(t)
        r = rb * _BLOCK + (tid // 16) * 4
        c = cb * _BLOCK + (tid % 16) * _VEC  # first output column
        if cutlass.const_expr(NO % _BLOCK != 0):
            ok = ok & Int32(c < NO)
        if cutlass.const_expr(self.backward):
            # dh columns c/2 .. c/2+3 (8 B); gate_up columns c .. c+7 (16 B).
            dh_off = (Int64(r) * N + c // 2) * 2 * Int64(ok)
            gu_off = (Int64(r) * (2 * N) + c) * 2 * Int64(ok)
            for k in range(4):
                cp_async8(
                    stage_addr + (k * _THREADS + tid) * 8, dh_base + dh_off + k * N * 2, ok * 8
                )
            for k in range(4):
                cp_async16(
                    stage_addr + 4 * _THREADS * 8 + (k * _THREADS + tid) * 16,
                    gu_base + gu_off + k * (2 * N) * 2,
                    ok * 16,
                )
        else:
            # gate_up columns 2c .. 2c+15 as two 16-byte chunks.
            gu_off = (Int64(r) * (2 * N) + 2 * c) * 2 * Int64(ok)
            for q in range(2):
                for k in range(4):
                    cp_async16(
                        stage_addr + ((q * 4 + k) * _THREADS + tid) * 16,
                        gu_base + gu_off + k * (2 * N) * 2 + q * 16,
                        ok * 16,
                    )
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        mDh: cute.Tensor,
        mGU: cute.Tensor,
        mBias: cute.Tensor,
        mOffs: cute.Tensor,
        mTok: cute.Tensor,
        mTokS: cute.Tensor,
        mKm: cute.Tensor,
        mKmS: cute.Tensor,
        mPart: cute.Tensor,
    ):
        N, NO, NB = self.N, self.NO, self.NB
        S = self.stages
        even_n = cutlass.const_expr(NO % _BLOCK == 0)
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = tid % 32
        warp = tid // 32
        cv = tid % 16
        rg = tid // 16
        M = mGU.shape[0]
        nrb = M // _BLOCK
        total = nrb * NB
        G = mOffs.shape[0]

        smem = cutlass.memory.SmemAllocator()
        s_in = smem.allocate(S * self.stage_bytes, byte_alignment=128).toint()
        s_tr = smem.allocate(_BLOCK * _BLOCK, byte_alignment=128).toint()
        s_cm = smem.allocate((_THREADS // 32) * 16 * 16, byte_alignment=16).toint()
        s_rcp = smem.allocate(_BLOCK * 4, byte_alignment=16).toint()
        s_offs = smem.allocate(_G_CAP * 4, byte_alignment=16).toint()
        s_ps = s_cm
        if cutlass.const_expr(self.backward):
            s_ps = smem.allocate((_THREADS // 32) * _BLOCK * 4, byte_alignment=16).toint()

        gu_base = mGU.iterator.toint()
        dh_base = gu_base
        if cutlass.const_expr(self.backward):
            dh_base = mDh.iterator.toint()
        bias_base = mBias.iterator.toint()

        for s in cutlass.range_constexpr(S):
            self._issue(bid + s * grid, s_in + s * self.stage_bytes, total, tid, dh_base, gu_base)

        # Expert-group offsets into shared memory (searched once per tile below).
        for gc in cutlass.range((G + _THREADS - 1) // _THREADS, unroll=1):
            j = gc * _THREADS + tid
            if j < G:
                v = ld_b32_coherent(mOffs.iterator.toint() + Int64(j) * 4, Int32(1), Int32(0))
                sts_b32_nopred(s_offs + j * 4, Uint32(v))
        cute.arch.sync_threads()

        n_it = (total - bid + grid - 1) // grid
        for it in cutlass.range(n_it, unroll=1):
            t = bid + it * grid
            rb, cb = self._tile(t)
            row0 = rb * _BLOCK
            col0 = cb * _BLOCK
            col = col0 + cv * _VEC
            col_ok = Int32(1)
            if cutlass.const_expr(not even_n):
                col_ok = Int32(col < NO)
            stage_addr = s_in + (it % S) * self.stage_bytes

            # Expert group of this row block: searchsorted(offs, row0, right=True), clamped.
            cnt = Int32(0)
            for gc in cutlass.range((G + 31) // 32, unroll=1):
                j = gc * 32 + lane
                v = lds_b32_pred(s_offs + j * 4, Int32(j < G), Int32(0x7FFFFFFF))
                cnt += cute.arch.popc(cute.arch.vote_ballot_sync(v <= row0))
            gid = cutlass.min(cnt, G - 1)
            ge = lds_b32_pred(s_offs + gid * 4, Int32(1), Int32(0))
            gs = lds_b32_pred(s_offs + (gid - 1) * 4, Int32(gid > 0), Int32(0))
            k_g = Int64(ge - gs)

            # Bias of this expert for the thread's interleaved gate_up columns.
            bias_row = bias_base + Int64(gid) * (2 * N) * 2
            if cutlass.const_expr(self.backward):
                bw = _ld_v4(bias_row + Int64(col) * 2, col_ok)
            else:
                bw = _ld_v4(bias_row + Int64(col) * 4, col_ok) + _ld_v4(
                    bias_row + Int64(col) * 4 + 16, col_ok
                )

            cute.arch.cp_async_wait_group(S - 1)
            # h[k][m]: output columns 2m, 2m+1 of row k, two BF16 packed in one word.
            h = [[None] * 4 for _ in range(4)]
            if cutlass.const_expr(self.backward):
                for k in cutlass.range_constexpr(4):
                    d = lds_v2(stage_addr + (k * _THREADS + tid) * 8)
                    gw = lds_v4(stage_addr + 4 * _THREADS * 8 + (k * _THREADS + tid) * 16)
                    for m in cutlass.range_constexpr(4):
                        w = _add_bias2(gw[m], bw[m])
                        f = bf16_lo if m % 2 == 0 else bf16_hi
                        dg, du = self.bwd_op(f(d[m // 2]), bf16_lo(w), bf16_hi(w))
                        h[k][m] = pack_bf16(dg, du)
            else:
                for k in cutlass.range_constexpr(4):
                    a = lds_v4(stage_addr + (k * _THREADS + tid) * 16)
                    b = lds_v4(stage_addr + ((4 + k) * _THREADS + tid) * 16)
                    ws = a + b
                    o = [None] * 8
                    for j in cutlass.range_constexpr(8):
                        w = _add_bias2(ws[j], bw[j])
                        o[j] = self.fwd_op(bf16_lo(w), bf16_hi(w))
                    for m in cutlass.range_constexpr(4):
                        h[k][m] = pack_bf16(o[2 * m], o[2 * m + 1])

            # The stage is consumed: refill it with the tile S iterations ahead.
            self._issue(t + S * grid, stage_addr, total, tid, dh_base, gu_base)

            a = [[abs2(h[k][m]) for m in range(4)] for k in range(4)]

            if cutlass.const_expr(self.kmajor):
                # -- colwise amax: 4 rows in registers, the other row group in the warp, then one
                #    partial per warp into shared memory --
                cm = [None] * 4
                for m in cutlass.range_constexpr(4):
                    x = max2(max2(a[0][m], a[1][m]), max2(a[2][m], a[3][m]))
                    y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=16))
                    cm[m] = max2(x, y)
                if lane < 16:
                    sts_v4(s_cm + (warp * 16 + cv) * 16, tuple(cm))

            if cutlass.const_expr(self.backward):
                # -- bias-gradient column sums of the BF16 gradient, same staging as cm --
                ps = [None] * 8
                for m in cutlass.range_constexpr(4):
                    lo = ((bf16_lo(h[0][m]) + bf16_lo(h[1][m])) + bf16_lo(h[2][m])) + bf16_lo(
                        h[3][m]
                    )
                    hi = ((bf16_hi(h[0][m]) + bf16_hi(h[1][m])) + bf16_hi(h[2][m])) + bf16_hi(
                        h[3][m]
                    )
                    lo = lo + cute.arch.shuffle_sync_bfly(lo, offset=16)
                    hi = hi + cute.arch.shuffle_sync_bfly(hi, offset=16)
                    ps[2 * m] = f32_as_u32(lo)
                    ps[2 * m + 1] = f32_as_u32(hi)
                if lane < 16:
                    base = s_ps + (warp * _BLOCK + cv * _VEC) * 4
                    sts_v4(base, tuple(ps[0:4]))
                    sts_v4(base + 16, tuple(ps[4:8]))

            if cutlass.const_expr(self.rowwise):
                # -- rowwise amax: rows (0,1) and (2,3) packed per word, 16-lane butterfly --
                c4 = [max2(max2(a[k][0], a[k][1]), max2(a[k][2], a[k][3])) for k in range(4)]
                rw = [None] * 2
                for p in cutlass.range_constexpr(2):
                    x0, x1 = c4[2 * p], c4[2 * p + 1]
                    x = max2(prmt_args_last(x0, x1, 0x5410), prmt_args_last(x0, x1, 0x7632))
                    for sh in cutlass.range_constexpr(4):
                        y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=1 << sh))
                        x = max2(x, y)
                    rw[p] = x

                # -- rowwise scale: lane cv < 4 of each row group owns row rg*4 + cv --
                r_amax = bf16_sel(Int32(cv % 2), selp_args_last(Int32(cv < 2), rw[0], rw[1]))
                r_sc, r_rc = e8m0_scale_f32(r_amax)
                st_f32(
                    mTokS.iterator.toint() + (Int64(cb) * M + row0 + rg * 4 + cv) * 4,
                    r_sc,
                    Int32(cv < 4),
                )
                for k in cutlass.range_constexpr(4):
                    rc = cute.arch.shuffle_sync(r_rc, offset=(lane & 16) | k)
                    rr = (rc, rc, rc, rc)
                    hv = h[k]
                    q0 = quant4_e4m3(
                        bf16_lo(hv[0]), bf16_hi(hv[0]), bf16_lo(hv[1]), bf16_hi(hv[1]), rr
                    )
                    q1 = quant4_e4m3(
                        bf16_lo(hv[2]), bf16_hi(hv[2]), bf16_lo(hv[3]), bf16_hi(hv[3]), rr
                    )
                    r = row0 + rg * 4 + k
                    st_v2_args_last(mTok.iterator.toint() + Int64(r) * NO + col, q0, q1, col_ok)

            if cutlass.const_expr(self.kmajor):
                cute.arch.sync_threads()

                # -- colwise scale (and bias partial): thread c < 128 owns column c of the tile --
                if tid < _BLOCK:
                    cw = Uint32(0)
                    for wi in cutlass.range_constexpr(_THREADS // 32):
                        v = lds_b32_pred(
                            s_cm + (wi * 16 + tid // 8) * 16 + ((tid % 8) // 2) * 4,
                            Int32(1),
                            Int32(0),
                        )
                        cw = max2(cw, Uint32(v))
                    c_sc, c_rc = e8m0_scale_f32(bf16_sel(Int32(tid % 2), cw))
                    c = col0 + tid
                    c_ok = Int32(1)
                    if cutlass.const_expr(not even_n):
                        c_ok = Int32(c < NO)
                    st_f32(
                        mKmS.iterator.toint() + (Int64(rb) * mn_pad(self.KP) + c) * 4, c_sc, c_ok
                    )
                    sts_b32_nopred(s_rcp + tid * 4, Uint32(f32_as_u32(c_rc)))
                    if cutlass.const_expr(self.backward):
                        acc = Float32(0.0)
                        for wi in cutlass.range_constexpr(_THREADS // 32):
                            v = lds_b32_pred(s_ps + (wi * _BLOCK + tid) * 4, Int32(1), Int32(0))
                            acc = acc + u32_as_f32(Uint32(v))
                        st_f32(mPart.iterator.toint() + (Int64(rb) * NO + c) * 4, acc, c_ok)

                cute.arch.sync_threads()

                # -- colwise FP8 into the transposed, swizzled tile (16 B chunks ^ cv%8) --
                rv = lds_v4(s_rcp + cv * 32) + lds_v4(s_rcp + cv * 32 + 16)
                sw = (cv & 7) << 4
                for j in cutlass.range_constexpr(8):
                    m = j // 2
                    f = bf16_lo if j % 2 == 0 else bf16_hi
                    rc = u32_as_f32(rv[j])
                    q = quant4_e4m3(f(h[0][m]), f(h[1][m]), f(h[2][m]), f(h[3][m]), (rc,) * 4)
                    cl = cv * _VEC + j
                    sts_b32_nopred(s_tr + cl * _BLOCK + (((rg // 4) << 4) ^ sw) + (rg % 4) * 4, q)

                cute.arch.sync_threads()

                # -- K-major store: 8 threads per column, 128 contiguous bytes per column --
                km_base = mKm.iterator.toint() + Int64(gs) * self.KP + Int64(row0 - gs)
                for p in cutlass.range_constexpr(_BLOCK * _BLOCK // 16 // _THREADS):
                    idx = p * _THREADS + tid
                    cl = idx // 8
                    qq = idx % 8
                    v = lds_v4(s_tr + cl * _BLOCK + ((qq ^ ((cl // 8) & 7)) << 4))
                    c = col0 + cl
                    c_ok = Int32(1)
                    if cutlass.const_expr(not even_n):
                        c_ok = Int32(c < NO)
                    st_v4_args_last(km_base + Int64(c) * k_g + qq * 16, v, c_ok)

        cute.arch.cp_async_wait_group(0)


_compiled: dict = {}


def _get_kernel(
    N: int,
    backward: bool,
    alpha: float,
    limit: float,
    device: torch.device,
    KP: int,
    rowwise: bool = True,
    kmajor: bool = True,
):
    """Compiled kernel for (N, direction, alpha, limit, device); M and the group count stay
    dynamic."""
    key = (N, backward, alpha, limit, device.index, _STAGES[backward], KP, rowwise, kmajor)
    fn = _compiled.get(key)
    if fn is None:
        m = cute.sym_int()
        NO = 2 * N if backward else N
        nb = (NO + _BLOCK - 1) // _BLOCK
        bf = cutlass.BFloat16
        fp8 = cutlass.Float8E4M3FN
        gu = fake(bf, (m, 2 * N), (2 * N, 1))
        dh = fake(bf, (m, N), (N, 1)) if backward else None
        bias = fake(bf, (cute.sym_int(), 2 * N), (2 * N, 1))
        outs = (
            fake(fp8, (m, NO), (NO, 1)),
            fake(Float32, (m, nb), (1, cute.sym_int64()), 16),
            fake(fp8, (cute.sym_int(),), (1,)),
            fake(Float32, (KP, cute.sym_int()), (1, mn_pad(KP)), 16),
        )
        if not rowwise:
            outs = (None, None) + outs[2:]
        if not kmajor:
            outs = outs[:2] + (None, None)
        part = fake(Float32, (cute.sym_int(), NO), (NO, 1)) if backward else None
        offs = fake(Int32, (cute.sym_int(),), (1,), 4)
        num_sms = sm_count(device)
        fn = cute.compile(
            ClampedSwiGLUQuant(N, backward, alpha, limit, num_sms, KP, rowwise, kmajor),
            dh,
            gu,
            bias,
            offs,
            *outs,
            part,
            fake_stream(),
            options="--enable-tvm-ffi",
        )
        _compiled[key] = fn
    return fn


def _alloc(
    M: int, N: int, device: torch.device, KP: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four outputs of ``fp8cast_rowwise_kmajor`` for an (M, N) input, the K-major pair with
    KP >= N channels."""
    nb = (N + _BLOCK - 1) // _BLOCK
    KP = N if KP is None else KP
    return (
        torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device),
        mn_major_scale(M, nb, device),
        torch.empty(M * KP, dtype=torch.float8_e4m3fn, device=device),
        mn_major_scale(KP, M // _BLOCK, device),
    )


def clamped_swiglu_quant_fusable(
    gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> bool:
    """Inputs the fused kernel handles; others take the (bit-identical) unfused path."""
    N = gate_up.shape[1] // 2
    return (
        gate_up.dtype == torch.bfloat16
        and gate_up.is_contiguous()
        and gate_up.data_ptr() % 16 == 0
        and bias.dtype == torch.bfloat16
        and bias.is_contiguous()
        and bias.data_ptr() % 16 == 0
        and N % _VEC == 0
        and grouped_mm_offs.dtype == torch.int32
        and grouped_mm_offs.is_contiguous()
        and 0 < grouped_mm_offs.shape[0] <= _G_CAP
    )


def clamped_swiglu_quant_fwd_sm90(
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
    kmajor_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(clamped_swiglu(gate_up + bias[group], alpha, limit), offs)``
    without materializing the biased gate_up or h.

    Args:
        gate_up: BF16 (M, 2N) raw gate_up GEMM output, gate at even / up at odd columns, M a
            multiple of 128.
        bias: (G, 2N) per-expert gate_up bias, same interleaving.
        grouped_mm_offs: int32 (G,) cumulative row counts, multiples of 128, last entry == M.
        alpha, limit: clamped SwiGLU constants (rounded to fp32, as Triton takes them).
        kmajor_channels: if set (>= N, fused path only), the K-major pair is laid out for that
            many channels: group g's block is (kmajor_channels, k_g) and the scale is
            (kmajor_channels, M / 128); channels >= N are left unwritten. Lets the caller give
            DeepGEMM's k-grouped wgrad an N it does not deadlock on (see operators/fp8_experts.py).

    Returns:
        (tok_fp8, tok_scale, kmajor_fp8, kmajor_scale) of h (M, N), bit-identical to the unfused
        sequence (on the first N channels); both scales MN-major.
    """
    assert gate_up.ndim == 2 and gate_up.shape[1] % 2 == 0 and gate_up.shape[0] % _BLOCK == 0
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0:
        return _alloc(M, N, gate_up.device)
    if not clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs):
        h = clamped_swiglu_fwd(biased_gate_up(gate_up, bias, grouped_mm_offs), alpha, limit)
        return fp8cast_rowwise_kmajor_sm90(h, grouped_mm_offs)
    KP = N if kmajor_channels is None else kmajor_channels
    assert KP >= N
    out = _alloc(M, N, gate_up.device, KP)
    fn = _get_kernel(N, False, _f32(alpha), _f32(limit), gate_up.device, KP)
    fn(None, gate_up, bias, grouped_mm_offs, *out, None)
    return out


def clamped_swiglu_quant_rowwise_sm90(
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only the rowwise pair (tok_fp8, tok_scale) of :func:`clamped_swiglu_quant_fwd_sm90`, the same
    bytes, for a forward whose K-major copy of h is recomputed in backward
    (:func:`clamped_swiglu_quant_kmajor_sm90`)."""
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0 or not clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs):
        return clamped_swiglu_quant_fwd_sm90(gate_up, bias, grouped_mm_offs, alpha, limit)[:2]
    tok = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=gate_up.device)
    tok_scale = mn_major_scale(M, (N + _BLOCK - 1) // _BLOCK, gate_up.device)
    fn = _get_kernel(N, False, _f32(alpha), _f32(limit), gate_up.device, N, kmajor=False)
    fn(None, gate_up, bias, grouped_mm_offs, tok, tok_scale, None, None, None)
    return tok, tok_scale


def clamped_swiglu_quant_kmajor_sm90(
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
    kmajor_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only the K-major pair (kmajor_fp8, kmajor_scale) of :func:`clamped_swiglu_quant_fwd_sm90`, the
    same bytes, without writing the rowwise pair: for recomputing h's wgrad operand in backward
    instead of keeping it from forward."""
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0 or not clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs):
        return clamped_swiglu_quant_fwd_sm90(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)[2:]  # fmt: skip
    KP = N if kmajor_channels is None else kmajor_channels
    assert KP >= N
    km = torch.empty(M * KP, dtype=torch.float8_e4m3fn, device=gate_up.device)
    km_scale = mn_major_scale(KP, M // _BLOCK, gate_up.device)
    fn = _get_kernel(N, False, _f32(alpha), _f32(limit), gate_up.device, KP, rowwise=False)
    fn(None, gate_up, bias, grouped_mm_offs, None, None, km, km_scale, None)
    return km, km_scale


def clamped_swiglu_bwd_quant_sm90(
    dh: torch.Tensor,
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
    kmajor_channels: int | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """``fp8cast_rowwise_kmajor`` of the clamped SwiGLU input gradient, plus the bias gradient,
    without materializing the gradient.

    Args:
        dh: BF16 (M, N) gradient of the clamped SwiGLU output.
        gate_up, bias, grouped_mm_offs, alpha, limit: as in :func:`clamped_swiglu_quant_fwd_sm90`.

        kmajor_channels: as in :func:`clamped_swiglu_quant_fwd_sm90`, for the 2N gradient channels.

    Returns:
        (dgu_quant, bias_grad): the 4-tuple ``fp8cast_rowwise_kmajor`` returns for the BF16
        interleaved (M, 2N) gradient, bit for bit, and the (G, 2N) bias gradient in bias.dtype
        (per-expert sum of that BF16 gradient, accumulated in fp32).
    """
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    assert dh.shape == (M, N), (dh.shape, gate_up.shape)
    G = grouped_mm_offs.shape[0]
    if M == 0 or N == 0:
        return _alloc(M, 2 * N, gate_up.device), bias.new_zeros(bias.shape)
    if not (clamped_swiglu_quant_fusable(gate_up, bias, grouped_mm_offs) and dh.dtype == torch.bfloat16 and dh.is_contiguous()):  # fmt: skip
        dgu = clamped_swiglu_bwd_biased(dh, gate_up, bias, grouped_mm_offs, alpha, limit)
        partials = dgu.float().view(M // _BLOCK, _BLOCK, 2 * N).sum(1)
        return fp8cast_rowwise_kmajor_sm90(dgu, grouped_mm_offs), block_segment_sum(
            partials, grouped_mm_offs, bias.dtype
        )
    KP = 2 * N if kmajor_channels is None else kmajor_channels
    assert KP >= 2 * N
    out = _alloc(M, 2 * N, gate_up.device, KP)
    partials = torch.empty((M // _BLOCK, 2 * N), dtype=torch.float32, device=gate_up.device)
    fn = _get_kernel(N, True, _f32(alpha), _f32(limit), gate_up.device, KP)
    fn(dh, gate_up, bias, grouped_mm_offs, *out, partials)
    assert G > 0
    return out, block_segment_sum(partials, grouped_mm_offs, bias.dtype)
