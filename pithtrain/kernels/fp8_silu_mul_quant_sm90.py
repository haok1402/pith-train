"""
SM90 fp8_silu_mul_quant: CuTe-DSL kernels computing silu(g) * u (or its two gradients) and writing
only its rowwise + K-major FP8 quantization, never the bf16 intermediate. Bit-identical to
silu_mul_fwd / silu_mul_bwd followed by fp8cast_rowwise_kmajor; both scales MN-major. The
SiluMulQuantGU variant reads g and u as the halves of a fused (M, 2N) gate_up output and writes the
backward as one (M, 2N) operand. Shapes the fused kernels do not take (see silu_mul_quant_fusable,
silu_mul_quant_gu_fusable) run the unfused sequence.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.arch import sm_count
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import abs2, asm_args_last, bf16_hi, bf16_lo, bf16_sel, cp_async16, f32_as_u32, ld_b32_coherent, lds_b32_pred, lds_v4, max2, pack_bf16, prmt_args_last, selp_args_last, st_f32, st_v2_args_last, st_v4_args_last, sts_b32_nopred, sts_v4, to_ir, u32_as_f32, unpack
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale_f32, quant4_e4m3
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.silu_mul import silu_mul_bwd, silu_mul_fwd

__all__ = ["SiluMulQuant", "SiluMulQuantGU", "silu_mul_bwd_quant_gu_sm90", "silu_mul_bwd_quant_sm90", "silu_mul_quant_fusable", "silu_mul_quant_fwd_sm90", "silu_mul_quant_gu_fusable", "silu_mul_quant_gu_fwd_sm90"]

BLOCK = 128  # rows per quantization group and columns per CTA tile
THREADS = 512
VEC = 8  # BF16 columns per thread (one 16-byte load)


# sigmoid(x) = 1 / (1 + exp(-x)). Triton 3.7 emits, for fp32:
#     sub.f32 t, 0, x;  mul.f32 t, t, log2e;  ex2.approx.f32 e, t;  add.f32 d, e, 1;
#     div.full.f32 s, 1, d
# This 5-instruction sequence gives the same bits for every bf16 gate value (checked exhaustively):
# - x * -log2e == (0 - x) * log2e: negation is exact; ex2(+-0) = 1; NaN is canonical.
# - ex2 non-ftz differs from ftz only when e is subnormal, and then 1 + e == 1 either way.
# - div.full(1, d) == 0.25 * rcp(0.25 * d) for every fp32 d >= 1 (also checked exhaustively),
#   and 0.25 * (1 + e) == fma(e, 0.25, 0.25) because scaling by 0.25 is exact here.
# .rn pins every mul/add so ptxas cannot contract differently from the Triton kernels.
SIGMOID_PTX = (
    "mul.rn.f32 t, $G, 0fBFB8AA3B;\n"
    "ex2.approx.ftz.f32 t, t;\n"
    "fma.rn.f32 t, t, 0f3E800000, 0f3E800000;\n"
    "rcp.approx.ftz.f32 s, t;\n"
    "mul.rn.f32 s, s, 0f3E800000;\n"
)


@dsl_user_op
def swiglu_fwd(g: Float32, u: Float32, *, loc=None, ip=None) -> Float32:
    """silu(g) * u in fp32 as (sigmoid(g) * g) * u, the Triton forward's exact op order."""
    return Float32(asm_args_last(T.f32(), "{\n.reg .f32 t, s;\n" + SIGMOID_PTX.replace("$G", "$1") + "mul.rn.f32 s, s, $1;\nmul.rn.f32 $0, s, $2;\n}", "=f,f,f", [to_ir(g, Float32, loc, ip), to_ir(u, Float32, loc, ip)]))


@dsl_user_op
def swiglu_bwd(dh: Float32, g: Float32, u: Float32, *, loc=None, ip=None) -> tuple:
    """(grad_gate, grad_up) in fp32, the Triton backward's ops after LLVM contraction:

    grad_up   = dh * (s * g)
    grad_gate = ((dh * u) * s) * fma(-s, g, 1 + g)      with s = sigmoid(g)
    """
    ty = ir.Type.parse("!llvm.struct<(f32, f32)>")
    res = asm_args_last(
        ty,
        "{\n.reg .f32 t, s, a, b;\n"
        + SIGMOID_PTX.replace("$G", "$3")
        + "mul.rn.f32 a, s, $3;\n"  # silu
        "mul.rn.f32 $1, a, $2;\n"  # grad_up
        "mul.rn.f32 b, $2, $4;\n"  # dh * u
        "mul.rn.f32 b, b, s;\n"  # * sigmoid
        "add.rn.f32 a, $3, 0f3F800000;\n"  # 1 + g
        "neg.f32 t, s;\n"
        "fma.rn.f32 a, t, $3, a;\n"  # 1 + g - s * g, unrounded product
        "mul.rn.f32 $0, b, a;\n}",
        "=f,=f,f,f,f",
        [to_ir(v, Float32, loc, ip) for v in (dh, g, u)],
    )
    return unpack(res, 2, Float32, loc, ip)


G_CAP = 1024  # expert groups the shared-memory offset table holds
_STAGES = 2  # cp.async tiles in flight per CTA (backward cannot fit 3 in shared memory)


class SiluMulQuant:
    """Persistent kernel (one 512-thread CTA per SM over 128 x 128 tiles), compiled per (N, backward).

    Inputs are staged into shared memory with cp.async, stages tiles ahead. Every thread reads back
    only the 16-byte chunks it copied itself, so the staging needs no barrier, only
    cp.async.wait_group. Thread t: cv = t % 16 owns columns cv*8 .. cv*8+7, rg = t // 16 owns rows
    rg*4 .. rg*4+3 of the tile. A tile's 128 rows lie inside one expert group.
    """

    def __init__(self, N: int, backward: bool, num_ctas: int):
        self.N = N
        self.backward = backward
        self.NB = (N + BLOCK - 1) // BLOCK
        self.n_in = 3 if backward else 2
        self.n_out = 2 if backward else 1
        self.stages = _STAGES
        self.num_ctas = num_ctas
        self.stage_bytes = self.n_in * 4 * THREADS * 16

    @cute.jit
    def __call__(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok0: cute.Tensor, mTokS0: cute.Tensor, mKm0: cute.Tensor, mKmS0: cute.Tensor, mTok1: cute.Tensor, mTokS1: cute.Tensor, mKm1: cute.Tensor, mKmS1: cute.Tensor, stream):
        total = (mG.shape[0] // BLOCK) * self.NB
        self.kernel(mDh, mG, mU, mOffs, mTok0, mTokS0, mKm0, mKmS0, mTok1, mTokS1, mKm1, mKmS1).launch(grid=[cutlass.min(total, self.num_ctas), 1, 1], block=[THREADS, 1, 1], stream=stream)

    def _tile(self, t: Int32) -> tuple:
        """Tile t -> (row block, column block); column blocks of a row block are adjacent for read locality."""
        return t // self.NB, t % self.NB

    def _issue(self, t: Int32, stage_addr: Int32, total: Int32, tid: Int32, in_base: list):
        """cp.async tile t (zero-filled when out of range) into a stage."""
        N = self.N
        ok = Int32(t < total)
        rb, cb = self._tile(t)
        r = rb * BLOCK + (tid // 16) * 4
        c = cb * BLOCK + (tid % 16) * VEC
        if cutlass.const_expr(N % BLOCK != 0):
            ok = ok & Int32(c < N)
        off = (Int64(r) * N + c) * 2 * Int64(ok)
        for x in range(self.n_in):
            for k in range(4):
                cp_async16(stage_addr + ((x * 4 + k) * THREADS + tid) * 16, in_base[x] + off + k * N * 2, ok * 16)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok0: cute.Tensor, mTokS0: cute.Tensor, mKm0: cute.Tensor, mKmS0: cute.Tensor, mTok1: cute.Tensor, mTokS1: cute.Tensor, mKm1: cute.Tensor, mKmS1: cute.Tensor):
        N = self.N
        NB = self.NB
        S = self.stages
        even_n = cutlass.const_expr(N % BLOCK == 0)
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = tid % 32
        warp = tid // 32
        cv = tid % 16
        rg = tid // 16
        M = mG.shape[0]
        nrb = M // BLOCK
        total = nrb * NB
        G = mOffs.shape[0]

        smem = cutlass.memory.SmemAllocator()
        s_in = smem.allocate(S * self.stage_bytes, byte_alignment=128).toint()
        s_tr = smem.allocate(BLOCK * BLOCK, byte_alignment=128).toint()
        s_cm = smem.allocate((THREADS // 32) * 16 * 16, byte_alignment=16).toint()
        s_rcp = smem.allocate(BLOCK * 4, byte_alignment=16).toint()
        s_offs = smem.allocate(G_CAP * 4, byte_alignment=16).toint()

        ins = [mG, mU]
        if cutlass.const_expr(self.backward):
            ins = [mDh, mG, mU]
        in_base = [t.iterator.toint() for t in ins]

        for s in cutlass.range_constexpr(S):
            self._issue(bid + s * grid, s_in + s * self.stage_bytes, total, tid, in_base)

        # Expert-group offsets into shared memory, searched once per tile below.
        for gc in cutlass.range((G + THREADS - 1) // THREADS, unroll=1):
            j = gc * THREADS + tid
            if j < G:
                v = ld_b32_coherent(mOffs.iterator.toint() + Int64(j) * 4, Int32(1), Int32(0))
                sts_b32_nopred(s_offs + j * 4, Uint32(v))
        cute.arch.sync_threads()

        outs = [(mTok0, mTokS0, mKm0, mKmS0), (mTok1, mTokS1, mKm1, mKmS1)]
        n_it = (total - bid + grid - 1) // grid
        for it in cutlass.range(n_it, unroll=1):
            t = bid + it * grid
            rb, cb = self._tile(t)
            row0 = rb * BLOCK
            col0 = cb * BLOCK
            col = col0 + cv * VEC
            col_ok = Int32(1)
            if cutlass.const_expr(not even_n):
                col_ok = Int32(col < N)
            stage_addr = s_in + (it % S) * self.stage_bytes

            cute.arch.cp_async_wait_group(S - 1)
            raw = [[None] * 4 for _ in range(self.n_in)]
            for x in cutlass.range_constexpr(self.n_in):
                for k in cutlass.range_constexpr(4):
                    raw[x][k] = lds_v4(stage_addr + ((x * 4 + k) * THREADS + tid) * 16)

            # SwiGLU in fp32, rounded to bf16 and packed in pairs: h[o][k][m].
            h = [[[None] * 4 for _ in range(4)] for _ in range(self.n_out)]
            for k in cutlass.range_constexpr(4):
                for m in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self.backward):
                        d, g, u = raw[0][k][m], raw[1][k][m], raw[2][k][m]
                        dg_lo, du_lo = swiglu_bwd(bf16_lo(d), bf16_lo(g), bf16_lo(u))
                        dg_hi, du_hi = swiglu_bwd(bf16_hi(d), bf16_hi(g), bf16_hi(u))
                        h[0][k][m] = pack_bf16(dg_lo, dg_hi)
                        h[1][k][m] = pack_bf16(du_lo, du_hi)
                    else:
                        g, u = raw[0][k][m], raw[1][k][m]
                        lo = swiglu_fwd(bf16_lo(g), bf16_lo(u))
                        hi = swiglu_fwd(bf16_hi(g), bf16_hi(u))
                        h[0][k][m] = pack_bf16(lo, hi)

            # The stage is consumed: refill it with the tile S iterations ahead.
            self._issue(t + S * grid, stage_addr, total, tid, in_base)

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

            for o in cutlass.range_constexpr(self.n_out):
                mTok, mTokS, mKm, mKmS = outs[o]
                ho = h[o]
                a = [[abs2(ho[k][m]) for m in range(4)] for k in range(4)]

                # -- colwise amax: 4 rows in registers, the other row group in the warp, then
                #    one partial per warp into shared memory --
                cm = [None] * 4
                for m in cutlass.range_constexpr(4):
                    x = max2(max2(a[0][m], a[1][m]), max2(a[2][m], a[3][m]))
                    y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=16))
                    cm[m] = max2(x, y)
                if lane < 16:
                    sts_v4(s_cm + (warp * 16 + cv) * 16, tuple(cm))

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
                st_f32(mTokS.iterator.toint() + (Int64(cb) * M + row0 + rg * 4 + cv) * 4, r_sc, Int32(cv < 4))
                for k in cutlass.range_constexpr(4):
                    rc = cute.arch.shuffle_sync(r_rc, offset=(lane & 16) | k)
                    rr = (rc, rc, rc, rc)
                    hv = ho[k]
                    q0 = quant4_e4m3(bf16_lo(hv[0]), bf16_hi(hv[0]), bf16_lo(hv[1]), bf16_hi(hv[1]), rr)
                    q1 = quant4_e4m3(bf16_lo(hv[2]), bf16_hi(hv[2]), bf16_lo(hv[3]), bf16_hi(hv[3]), rr)
                    r = row0 + rg * 4 + k
                    st_v2_args_last(mTok.iterator.toint() + Int64(r) * N + col, q0, q1, col_ok)

                cute.arch.sync_threads()

                # -- colwise scale: thread c < 128 owns column c of the tile --
                if tid < BLOCK:
                    cw = Uint32(0)
                    for w in cutlass.range_constexpr(THREADS // 32):
                        v = lds_b32_pred(s_cm + (w * 16 + tid // 8) * 16 + ((tid % 8) // 2) * 4, Int32(1), Int32(0))
                        cw = max2(cw, Uint32(v))
                    c_sc, c_rc = e8m0_scale_f32(bf16_sel(Int32(tid % 2), cw))
                    c = col0 + tid
                    c_ok = Int32(1)
                    if cutlass.const_expr(not even_n):
                        c_ok = Int32(c < N)
                    st_f32(mKmS.iterator.toint() + (Int64(rb) * mn_pad(N) + c) * 4, c_sc, c_ok)
                    sts_b32_nopred(s_rcp + tid * 4, Uint32(f32_as_u32(c_rc)))

                cute.arch.sync_threads()

                # -- colwise FP8 into the transposed, swizzled tile (16 B chunks ^ cv%8) --
                rv = lds_v4(s_rcp + cv * 32) + lds_v4(s_rcp + cv * 32 + 16)
                sw = (cv & 7) << 4
                for j in cutlass.range_constexpr(8):
                    m = j // 2
                    f = bf16_lo if j % 2 == 0 else bf16_hi
                    rc = u32_as_f32(rv[j])
                    q = quant4_e4m3(f(ho[0][m]), f(ho[1][m]), f(ho[2][m]), f(ho[3][m]), (rc,) * 4)
                    cl = cv * VEC + j
                    sts_b32_nopred(s_tr + cl * BLOCK + (((rg // 4) << 4) ^ sw) + (rg % 4) * 4, q)

                cute.arch.sync_threads()

                # -- K-major store: 8 threads per column, 128 contiguous bytes per column --
                km_base = mKm.iterator.toint() + Int64(gs) * N + Int64(row0 - gs)
                for p in cutlass.range_constexpr(BLOCK * BLOCK // 16 // THREADS):
                    idx = p * THREADS + tid
                    cl = idx // 8
                    qq = idx % 8
                    v = lds_v4(s_tr + cl * BLOCK + ((qq ^ ((cl // 8) & 7)) << 4))
                    c = col0 + cl
                    c_ok = Int32(1)
                    if cutlass.const_expr(not even_n):
                        c_ok = Int32(c < N)
                    st_v4_args_last(km_base + Int64(c) * k_g + qq * 16, v, c_ok)

        cute.arch.cp_async_wait_group(0)


class SiluMulQuantGU(SiluMulQuant):
    """SiluMulQuant reading g / u as the column halves of one (M, 2N) buffer and, in backward,
    writing dg and du as the column halves of one quantized (M, 2N) operand: output o (0 = dg or h,
    1 = du) lands at channel offset o * N of an NC = n_out * N channel layout. Requires N % 128 == 0
    (every tile inside one half).
    """

    def __init__(self, N: int, backward: bool, num_ctas: int):
        super().__init__(N, backward, num_ctas)
        assert N % BLOCK == 0
        self.in_ld = 2 * N
        self.NC = self.n_out * N

    @cute.jit
    def __call__(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokS: cute.Tensor, mKm: cute.Tensor, mKmS: cute.Tensor, stream):
        total = (mG.shape[0] // BLOCK) * self.NB
        self.kernel(mDh, mG, mU, mOffs, mTok, mTokS, mKm, mKmS).launch(grid=[cutlass.min(total, self.num_ctas), 1, 1], block=[THREADS, 1, 1], stream=stream)

    def _issue(self, t: Int32, stage_addr: Int32, total: Int32, tid: Int32, in_base: list):
        """cp.async tile t (zero-filled when out of range); dh has row stride N, g / u 2N."""
        N = self.N
        ok = Int32(t < total)
        rb, cb = self._tile(t)
        r = rb * BLOCK + (tid // 16) * 4
        c = cb * BLOCK + (tid % 16) * VEC
        for x in range(self.n_in):
            ld = N if (self.backward and x == 0) else self.in_ld
            off = (Int64(r) * ld + c) * 2 * Int64(ok)
            for k in range(4):
                cp_async16(stage_addr + ((x * 4 + k) * THREADS + tid) * 16, in_base[x] + off + k * ld * 2, ok * 16)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(self, mDh: cute.Tensor, mG: cute.Tensor, mU: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokS: cute.Tensor, mKm: cute.Tensor, mKmS: cute.Tensor):
        N = self.N
        NB = self.NB
        NC = self.NC
        S = self.stages
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = tid % 32
        warp = tid // 32
        cv = tid % 16
        rg = tid // 16
        M = mG.shape[0]
        nrb = M // BLOCK
        total = nrb * NB
        G = mOffs.shape[0]

        smem = cutlass.memory.SmemAllocator()
        s_in = smem.allocate(S * self.stage_bytes, byte_alignment=128).toint()
        s_tr = smem.allocate(BLOCK * BLOCK, byte_alignment=128).toint()
        s_cm = smem.allocate((THREADS // 32) * 16 * 16, byte_alignment=16).toint()
        s_rcp = smem.allocate(BLOCK * 4, byte_alignment=16).toint()
        s_offs = smem.allocate(G_CAP * 4, byte_alignment=16).toint()

        ins = [mG, mU]
        if cutlass.const_expr(self.backward):
            ins = [mDh, mG, mU]
        in_base = [t.iterator.toint() for t in ins]
        tok_base = mTok.iterator.toint()
        toks_base = mTokS.iterator.toint()
        km_root = mKm.iterator.toint()
        kms_base = mKmS.iterator.toint()

        for s in cutlass.range_constexpr(S):
            self._issue(bid + s * grid, s_in + s * self.stage_bytes, total, tid, in_base)

        for gc in cutlass.range((G + THREADS - 1) // THREADS, unroll=1):
            j = gc * THREADS + tid
            if j < G:
                v = ld_b32_coherent(mOffs.iterator.toint() + Int64(j) * 4, Int32(1), Int32(0))
                sts_b32_nopred(s_offs + j * 4, Uint32(v))
        cute.arch.sync_threads()

        n_it = (total - bid + grid - 1) // grid
        for it in cutlass.range(n_it, unroll=1):
            t = bid + it * grid
            rb, cb = self._tile(t)
            row0 = rb * BLOCK
            col0 = cb * BLOCK
            col = col0 + cv * VEC
            stage_addr = s_in + (it % S) * self.stage_bytes

            cute.arch.cp_async_wait_group(S - 1)
            raw = [[None] * 4 for _ in range(self.n_in)]
            for x in cutlass.range_constexpr(self.n_in):
                for k in cutlass.range_constexpr(4):
                    raw[x][k] = lds_v4(stage_addr + ((x * 4 + k) * THREADS + tid) * 16)

            h = [[[None] * 4 for _ in range(4)] for _ in range(self.n_out)]
            for k in cutlass.range_constexpr(4):
                for m in cutlass.range_constexpr(4):
                    if cutlass.const_expr(self.backward):
                        d, g, u = raw[0][k][m], raw[1][k][m], raw[2][k][m]
                        dg_lo, du_lo = swiglu_bwd(bf16_lo(d), bf16_lo(g), bf16_lo(u))
                        dg_hi, du_hi = swiglu_bwd(bf16_hi(d), bf16_hi(g), bf16_hi(u))
                        h[0][k][m] = pack_bf16(dg_lo, dg_hi)
                        h[1][k][m] = pack_bf16(du_lo, du_hi)
                    else:
                        g, u = raw[0][k][m], raw[1][k][m]
                        lo = swiglu_fwd(bf16_lo(g), bf16_lo(u))
                        hi = swiglu_fwd(bf16_hi(g), bf16_hi(u))
                        h[0][k][m] = pack_bf16(lo, hi)

            self._issue(t + S * grid, stage_addr, total, tid, in_base)

            cnt = Int32(0)
            for gc in cutlass.range((G + 31) // 32, unroll=1):
                j = gc * 32 + lane
                v = lds_b32_pred(s_offs + j * 4, Int32(j < G), Int32(0x7FFFFFFF))
                cnt += cute.arch.popc(cute.arch.vote_ballot_sync(v <= row0))
            gid = cutlass.min(cnt, G - 1)
            ge = lds_b32_pred(s_offs + gid * 4, Int32(1), Int32(0))
            gs = lds_b32_pred(s_offs + (gid - 1) * 4, Int32(gid > 0), Int32(0))
            k_g = Int64(ge - gs)
            km_base = km_root + Int64(gs) * NC + Int64(row0 - gs)

            for o in cutlass.range_constexpr(self.n_out):
                ch = o * N  # channel offset of this output in the NC-channel layout
                ho = h[o]
                a = [[abs2(ho[k][m]) for m in range(4)] for k in range(4)]

                cm = [None] * 4
                for m in cutlass.range_constexpr(4):
                    x = max2(max2(a[0][m], a[1][m]), max2(a[2][m], a[3][m]))
                    y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=16))
                    cm[m] = max2(x, y)
                if lane < 16:
                    sts_v4(s_cm + (warp * 16 + cv) * 16, tuple(cm))

                c4 = [max2(max2(a[k][0], a[k][1]), max2(a[k][2], a[k][3])) for k in range(4)]
                rw = [None] * 2
                for p in cutlass.range_constexpr(2):
                    x0, x1 = c4[2 * p], c4[2 * p + 1]
                    x = max2(prmt_args_last(x0, x1, 0x5410), prmt_args_last(x0, x1, 0x7632))
                    for sh in cutlass.range_constexpr(4):
                        y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=1 << sh))
                        x = max2(x, y)
                    rw[p] = x

                r_amax = bf16_sel(Int32(cv % 2), selp_args_last(Int32(cv < 2), rw[0], rw[1]))
                r_sc, r_rc = e8m0_scale_f32(r_amax)
                st_f32(toks_base + (Int64(cb + o * NB) * M + row0 + rg * 4 + cv) * 4, r_sc, Int32(cv < 4))
                for k in cutlass.range_constexpr(4):
                    rc = cute.arch.shuffle_sync(r_rc, offset=(lane & 16) | k)
                    rr = (rc, rc, rc, rc)
                    hv = ho[k]
                    q0 = quant4_e4m3(bf16_lo(hv[0]), bf16_hi(hv[0]), bf16_lo(hv[1]), bf16_hi(hv[1]), rr)
                    q1 = quant4_e4m3(bf16_lo(hv[2]), bf16_hi(hv[2]), bf16_lo(hv[3]), bf16_hi(hv[3]), rr)
                    r = row0 + rg * 4 + k
                    st_v2_args_last(tok_base + Int64(r) * NC + ch + col, q0, q1, Int32(1))

                cute.arch.sync_threads()

                if tid < BLOCK:
                    cw = Uint32(0)
                    for w in cutlass.range_constexpr(THREADS // 32):
                        v = lds_b32_pred(s_cm + (w * 16 + tid // 8) * 16 + ((tid % 8) // 2) * 4, Int32(1), Int32(0))
                        cw = max2(cw, Uint32(v))
                    c_sc, c_rc = e8m0_scale_f32(bf16_sel(Int32(tid % 2), cw))
                    c = col0 + tid
                    st_f32(kms_base + (Int64(rb) * mn_pad(NC) + ch + c) * 4, c_sc, Int32(1))
                    sts_b32_nopred(s_rcp + tid * 4, Uint32(f32_as_u32(c_rc)))

                cute.arch.sync_threads()

                rv = lds_v4(s_rcp + cv * 32) + lds_v4(s_rcp + cv * 32 + 16)
                sw = (cv & 7) << 4
                for j in cutlass.range_constexpr(8):
                    m = j // 2
                    f = bf16_lo if j % 2 == 0 else bf16_hi
                    rc = u32_as_f32(rv[j])
                    q = quant4_e4m3(f(ho[0][m]), f(ho[1][m]), f(ho[2][m]), f(ho[3][m]), (rc,) * 4)
                    cl = cv * VEC + j
                    sts_b32_nopred(s_tr + cl * BLOCK + (((rg // 4) << 4) ^ sw) + (rg % 4) * 4, q)

                cute.arch.sync_threads()

                for p in cutlass.range_constexpr(BLOCK * BLOCK // 16 // THREADS):
                    idx = p * THREADS + tid
                    cl = idx // 8
                    qq = idx % 8
                    v = lds_v4(s_tr + cl * BLOCK + ((qq ^ ((cl // 8) & 7)) << 4))
                    c = col0 + cl
                    st_v4_args_last(km_base + Int64(ch + c) * k_g + qq * 16, v, Int32(1))

        cute.arch.cp_async_wait_group(0)


_compiled: dict = {}
_compiled_gu: dict = {}


def _get_kernel(N: int, backward: bool, device: torch.device) -> object:
    """Compiled kernel for (N, direction, device); M and the group count stay dynamic."""
    key = (N, backward, device.index)
    fn = _compiled.get(key)
    if fn is None:
        m = cute.sym_int()
        nb = (N + BLOCK - 1) // BLOCK
        bf = fake(cutlass.BFloat16, (m, N), (N, 1))
        fp8 = cutlass.Float8E4M3FN
        outs = (fake(fp8, (m, N), (N, 1)), fake(Float32, (m, nb), (1, cute.sym_int64()), 16), fake(fp8, (cute.sym_int(),), (1,)), fake(Float32, (N, cute.sym_int()), (1, mn_pad(N)), 16))
        offs = fake(Int32, (cute.sym_int(),), (1,), 4)
        num_sms = sm_count(device)
        fn = cute.compile(SiluMulQuant(N, backward, num_sms), bf if backward else None, bf, bf, offs, *outs, *(outs if backward else (None,) * 4), fake_stream(), options="--enable-tvm-ffi")
        _compiled[key] = fn
    return fn


def _get_kernel_gu(N: int, backward: bool, device: torch.device) -> object:
    """Compiled gate_up kernel for (N, direction, device); M and the group count stay dynamic."""
    key = (N, backward, device.index)
    fn = _compiled_gu.get(key)
    if fn is None:
        m = cute.sym_int()
        nc = 2 * N if backward else N
        half = fake(cutlass.BFloat16, (m, N), (2 * N, 1))
        fp8 = cutlass.Float8E4M3FN
        outs = (fake(fp8, (m, nc), (nc, 1)), fake(cutlass.Float32, (m, nc // BLOCK), (1, cute.sym_int64()), 16), fake(fp8, (cute.sym_int(),), (1,)), fake(cutlass.Float32, (nc, cute.sym_int()), (1, mn_pad(nc)), 16))
        offs = fake(Int32, (cute.sym_int(),), (1,), 4)
        num_sms = sm_count(device)
        fn = cute.compile(SiluMulQuantGU(N, backward, num_sms), fake(cutlass.BFloat16, (m, N), (N, 1)) if backward else None, half, half, offs, *outs, fake_stream(), options="--enable-tvm-ffi")
        _compiled_gu[key] = fn
    return fn


def alloc_rowwise_kmajor_outputs(M: int, N: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four outputs of fp8cast_rowwise_kmajor of an (M, N) input, both scales MN-major, each its
    own allocation (different lifetimes: the rowwise pair feeds the next GEMM, the K-major pair waits
    for wgrad)."""
    nb = (N + BLOCK - 1) // BLOCK
    return (torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device), mn_major_scale(M, nb, device), torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device), mn_major_scale(N, M // BLOCK, device))


def silu_mul_quant_fusable(N: int, grouped_mm_offs: torch.Tensor) -> bool:
    """Shapes the fused kernel handles; others take the bit-identical unfused path."""
    return N % VEC == 0 and 0 < grouped_mm_offs.shape[0] <= G_CAP


def silu_mul_quant_gu_fusable(N: int, M: int, grouped_mm_offs: torch.Tensor) -> bool:
    """Shapes the fused gate_up kernel handles; others take the bit-identical unfused path."""
    return N % BLOCK == 0 and M % BLOCK == 0 and 0 < grouped_mm_offs.shape[0] <= G_CAP


def silu_mul_quant_fwd_sm90(g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor(silu_mul_fwd(g, u), grouped_mm_offs) without writing h."""
    M, N = g.shape
    if M != 0 and N != 0 and not silu_mul_quant_fusable(N, grouped_mm_offs):
        return fp8cast_rowwise_kmajor_sm90(silu_mul_fwd(g, u), grouped_mm_offs)
    out = alloc_rowwise_kmajor_outputs(M, N, g.device)
    if M == 0 or N == 0:
        return out
    _get_kernel(N, False, g.device)(None, g, u, grouped_mm_offs, *out, None, None, None, None)
    return out


def silu_mul_bwd_quant_sm90(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """fp8cast_rowwise_kmajor of each gradient of silu_mul_bwd(dh, g, u) without writing them."""
    M, N = g.shape
    if M == 0 or N == 0:
        return (alloc_rowwise_kmajor_outputs(M, N, g.device), alloc_rowwise_kmajor_outputs(M, N, g.device))
    if not silu_mul_quant_fusable(N, grouped_mm_offs):
        dg, du = silu_mul_bwd(dh, g, u)
        return fp8cast_rowwise_kmajor_sm90(dg, grouped_mm_offs), fp8cast_rowwise_kmajor_sm90(du, grouped_mm_offs)
    dg_q = alloc_rowwise_kmajor_outputs(M, N, g.device)
    du_q = alloc_rowwise_kmajor_outputs(M, N, g.device)
    _get_kernel(N, True, g.device)(dh, g, u, grouped_mm_offs, *dg_q, *du_q)
    return dg_q, du_q


def silu_mul_quant_gu_fwd_sm90(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor(silu_mul_fwd(gu[:, :N], gu[:, N:]), offs) for gu (M, 2N)."""
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return alloc_rowwise_kmajor_outputs(0, N, gu.device)
    if not silu_mul_quant_gu_fusable(N, M, grouped_mm_offs):
        g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
        return fp8cast_rowwise_kmajor_sm90(silu_mul_fwd(g, u), grouped_mm_offs)
    out = alloc_rowwise_kmajor_outputs(M, N, gu.device)
    _get_kernel_gu(N, False, gu.device)(None, gu[:, :N], gu[:, N:], grouped_mm_offs, *out)
    return out


def silu_mul_bwd_quant_gu_sm90(dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor(cat([dg, du], 1), offs) of the SwiGLU gradients of gu (M, 2N)."""
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return alloc_rowwise_kmajor_outputs(0, 2 * N, gu.device)
    if not silu_mul_quant_gu_fusable(N, M, grouped_mm_offs):
        g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
        dg, du = silu_mul_bwd(dh, g, u)
        return fp8cast_rowwise_kmajor_sm90(torch.cat([dg, du], 1), grouped_mm_offs)
    out = alloc_rowwise_kmajor_outputs(M, 2 * N, gu.device)
    _get_kernel_gu(N, True, gu.device)(dh, gu[:, :N], gu[:, N:], grouped_mm_offs, *out)
    return out
