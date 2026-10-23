"""
FP8 routed experts with a fused gate_up projection (Qwen3.5), Hopper.

Qwen3.5 stores the routed gate and up projections as one ``(G, 2I, H)`` weight, so its FP8 expert
path is one m-grouped GEMM into ``gu = [g | u]`` (M, 2I), SwiGLU, ``down_proj``; in backward one
dgrad GEMM with K = 2I over ``[dg | du]`` and one k-grouped wgrad over 2I channels. This module
fuses SwiGLU into the quantization of its neighbours as ``FP8SwiGLUExpertsFunc`` does for separate
gate/up weights, keeping that op sequence:

- forward: :func:`silu_mul_quant_gu_fwd` reads g and u straight out of ``gu`` (row stride 2I) and
  writes ``fp8cast_rowwise_kmajor(silu_mul(g, u))``, so neither the ``.contiguous()`` copies of
  g / u nor the bf16 h are written.
- backward: :func:`silu_mul_bwd_quant_gu` writes the four outputs of
  ``fp8cast_rowwise_kmajor(cat([dg, du], 1))`` directly (rowwise ``(M, 2I)`` with its MN-major
  scale, and the K-major 2I-channel buffer with its scale), which is exactly the operand layout of
  the fused weight's dgrad and wgrad. No bf16 dg, du or dgu is written.

Both kernels are ``fp8_cute.silu_mul_quant._SiluMulQuant`` with an input row stride and an output
channel offset; every value and every byte position is the same as the unfused sequence, so the
result is bit-identical to FP8GroupedLinear(gate_up) -> silu_mul -> FP8GroupedLinear(down).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.operators.deepgemm_quantize import fp8cast_rowwise_kmajor
from pithtrain.operators.fp8_cute.scale_layout import mn_major_scale, mn_pad
from pithtrain.operators.fp8_cute.silu_mul_quant import (
    _BLOCK,
    _G_CAP,
    _THREADS,
    _VEC,
    _abs2,
    _bf16_hi,
    _bf16_lo,
    _bf16_sel,
    _cp_async16,
    _f32_as_u32,
    _fake,
    _ld_b32,
    _lds_b32,
    _lds_v4,
    _max2,
    _pack_bf16,
    _prmt,
    _quant4,
    _scale,
    _selp,
    _silu_mul_bwd_reference,
    _SiluMulQuant,
    _st_f32,
    _st_v2,
    _st_v4,
    _sts_b32,
    _sts_v4,
    _swiglu_bwd,
    _swiglu_fwd,
    _u32_as_f32,
)
from pithtrain.operators.grouped_linear import (
    _fp8_grouped_gemm,
    _fp8_grouped_wgrad,
    quantize_grouped_input,
)
from pithtrain.operators.silu_mul import silu_mul

__all__ = ["fp8_swiglu_gate_up_experts", "silu_mul_quant_gu_fwd", "silu_mul_bwd_quant_gu"]


class _SiluMulQuantGU(_SiluMulQuant):
    """``_SiluMulQuant`` reading g / u as column halves of one ``(M, 2N)`` buffer and, in
    backward, writing dg and du as the column halves of one quantized ``(M, 2N)`` operand.

    Output ``o`` (0 = dg or h, 1 = du) lands at channel offset ``o * N`` of a layout with
    ``NC = n_out * N`` channels: the rowwise FP8 at column ``o * N + c`` (row stride NC), its scale
    at K block ``o * NB + cb``, the K-major FP8 at channel ``o * N + c`` of each group's
    ``(NC, k_g)`` block, and its scale at row ``o * N + c`` of the ``(NC, M / 128)`` MN-major grid.
    Requires ``N % 128 == 0`` (every tile inside one half).
    """

    def __init__(self, N: int, backward: bool, num_ctas: int):
        super().__init__(N, backward, num_ctas)
        assert N % _BLOCK == 0
        self.in_ld = 2 * N
        self.NC = self.n_out * N

    @cute.jit
    def __call__(
        self,
        mDh: cute.Tensor,
        mG: cute.Tensor,
        mU: cute.Tensor,
        mOffs: cute.Tensor,
        mTok: cute.Tensor,
        mTokS: cute.Tensor,
        mKm: cute.Tensor,
        mKmS: cute.Tensor,
        stream,
    ):
        total = (mG.shape[0] // _BLOCK) * self.NB
        self.kernel(mDh, mG, mU, mOffs, mTok, mTokS, mKm, mKmS).launch(
            grid=[cutlass.min(total, self.num_ctas), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    def _issue(self, t: Int32, stage_addr: Int32, total: Int32, tid: Int32, in_base: list):
        """cp.async tile ``t`` (zero-filled when out of range); dh has row stride N, g / u 2N."""
        N = self.N
        ok = Int32(t < total)
        rb, cb = self._tile(t)
        r = rb * _BLOCK + (tid // 16) * 4
        c = cb * _BLOCK + (tid % 16) * _VEC
        for x in range(self.n_in):
            ld = N if (self.backward and x == 0) else self.in_ld
            off = (Int64(r) * ld + c) * 2 * Int64(ok)
            for k in range(4):
                _cp_async16(
                    stage_addr + ((x * 4 + k) * _THREADS + tid) * 16,
                    in_base[x] + off + k * ld * 2,
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
        mTok: cute.Tensor,
        mTokS: cute.Tensor,
        mKm: cute.Tensor,
        mKmS: cute.Tensor,
    ):
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
        tok_base = mTok.iterator.toint()
        toks_base = mTokS.iterator.toint()
        km_root = mKm.iterator.toint()
        kms_base = mKmS.iterator.toint()

        for s in cutlass.range_constexpr(S):
            self._issue(bid + s * grid, s_in + s * self.stage_bytes, total, tid, in_base)

        for gc in cutlass.range((G + _THREADS - 1) // _THREADS, unroll=1):
            j = gc * _THREADS + tid
            if j < G:
                v = _ld_b32(mOffs.iterator.toint() + Int64(j) * 4, Int32(1), Int32(0))
                _sts_b32(s_offs + j * 4, Uint32(v))
        cute.arch.sync_threads()

        n_it = (total - bid + grid - 1) // grid
        for it in cutlass.range(n_it, unroll=1):
            t = bid + it * grid
            rb, cb = self._tile(t)
            row0 = rb * _BLOCK
            col0 = cb * _BLOCK
            col = col0 + cv * _VEC
            stage_addr = s_in + (it % S) * self.stage_bytes

            cute.arch.cp_async_wait_group(S - 1)
            raw = [[None] * 4 for _ in range(self.n_in)]
            for x in cutlass.range_constexpr(self.n_in):
                for k in cutlass.range_constexpr(4):
                    raw[x][k] = _lds_v4(stage_addr + ((x * 4 + k) * _THREADS + tid) * 16)

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

            self._issue(t + S * grid, stage_addr, total, tid, in_base)

            cnt = Int32(0)
            for gc in cutlass.range((G + 31) // 32, unroll=1):
                j = gc * 32 + lane
                v = _lds_b32(s_offs + j * 4, Int32(j < G), Int32(0x7FFFFFFF))
                cnt += cute.arch.popc(cute.arch.vote_ballot_sync(v <= row0))
            gid = cutlass.min(cnt, G - 1)
            ge = _lds_b32(s_offs + gid * 4, Int32(1), Int32(0))
            gs = _lds_b32(s_offs + (gid - 1) * 4, Int32(gid > 0), Int32(0))
            k_g = Int64(ge - gs)
            km_base = km_root + Int64(gs) * NC + Int64(row0 - gs)

            for o in cutlass.range_constexpr(self.n_out):
                ch = o * N  # channel offset of this output in the NC-channel layout
                ho = h[o]
                a = [[_abs2(ho[k][m]) for m in range(4)] for k in range(4)]

                cm = [None] * 4
                for m in cutlass.range_constexpr(4):
                    x = _max2(_max2(a[0][m], a[1][m]), _max2(a[2][m], a[3][m]))
                    y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=16))
                    cm[m] = _max2(x, y)
                if lane < 16:
                    _sts_v4(s_cm + (warp * 16 + cv) * 16, tuple(cm))

                c4 = [_max2(_max2(a[k][0], a[k][1]), _max2(a[k][2], a[k][3])) for k in range(4)]
                rw = [None] * 2
                for p in cutlass.range_constexpr(2):
                    x0, x1 = c4[2 * p], c4[2 * p + 1]
                    x = _max2(_prmt(x0, x1, 0x5410), _prmt(x0, x1, 0x7632))
                    for sh in cutlass.range_constexpr(4):
                        y = Uint32(cute.arch.shuffle_sync_bfly(Int32(x), offset=1 << sh))
                        x = _max2(x, y)
                    rw[p] = x

                r_amax = _bf16_sel(Int32(cv % 2), _selp(Int32(cv < 2), rw[0], rw[1]))
                r_sc, r_rc = _scale(r_amax)
                _st_f32(
                    toks_base + (Int64(cb + o * NB) * M + row0 + rg * 4 + cv) * 4,
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
                    _st_v2(tok_base + Int64(r) * NC + ch + col, q0, q1, Int32(1))

                cute.arch.sync_threads()

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
                    _st_f32(kms_base + (Int64(rb) * mn_pad(NC) + ch + c) * 4, c_sc, Int32(1))
                    _sts_b32(s_rcp + tid * 4, Uint32(_f32_as_u32(c_rc)))

                cute.arch.sync_threads()

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

                for p in cutlass.range_constexpr(_BLOCK * _BLOCK // 16 // _THREADS):
                    idx = p * _THREADS + tid
                    cl = idx // 8
                    qq = idx % 8
                    v = _lds_v4(s_tr + cl * _BLOCK + ((qq ^ ((cl // 8) & 7)) << 4))
                    c = col0 + cl
                    _st_v4(km_base + Int64(ch + c) * k_g + qq * 16, v, Int32(1))

        cute.arch.cp_async_wait_group(0)


_compiled: dict = {}


def _get_kernel(N: int, backward: bool, device: torch.device) -> object:
    """Compiled kernel for (N, direction, device); M and the group count stay dynamic."""
    key = (N, backward, device.index)
    fn = _compiled.get(key)
    if fn is None:
        m = cute.sym_int()
        nc = 2 * N if backward else N
        half = _fake(cutlass.BFloat16, (m, N), (2 * N, 1))
        fp8 = cutlass.Float8E4M3FN
        outs = (
            _fake(fp8, (m, nc), (nc, 1)),
            _fake(cutlass.Float32, (m, nc // _BLOCK), (1, cute.sym_int64()), 16),
            _fake(fp8, (cute.sym_int(),), (1,)),
            _fake(cutlass.Float32, (nc, cute.sym_int()), (1, mn_pad(nc)), 16),
        )
        offs = _fake(Int32, (cute.sym_int(),), (1,), 4)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        fn = cute.compile(
            _SiluMulQuantGU(N, backward, num_sms),
            _fake(cutlass.BFloat16, (m, N), (N, 1)) if backward else None,
            half,
            half,
            offs,
            *outs,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compiled[key] = fn
    return fn


def _alloc(M: int, NC: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The four outputs of ``fp8cast_rowwise_kmajor`` of an (M, NC) input."""
    return (
        torch.empty((M, NC), dtype=torch.float8_e4m3fn, device=device),
        mn_major_scale(M, NC // _BLOCK, device),
        torch.empty(M * NC, dtype=torch.float8_e4m3fn, device=device),
        mn_major_scale(NC, M // _BLOCK, device),
    )


def _fusable(N: int, M: int, grouped_mm_offs: torch.Tensor) -> bool:
    return N % _BLOCK == 0 and M % _BLOCK == 0 and 0 < grouped_mm_offs.shape[0] <= _G_CAP


def _check(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> None:
    assert gu.dtype == torch.bfloat16 and gu.ndim == 2 and gu.is_contiguous()
    assert gu.shape[1] % 2 == 0
    assert grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous()


def silu_mul_quant_gu_fwd(
    gu: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(silu_mul(gu[:, :N], gu[:, N:]), offs)`` for ``gu`` (M, 2N)."""
    _check(gu, grouped_mm_offs)
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return _alloc(0, N, gu.device)
    if not _fusable(N, M, grouped_mm_offs):
        g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
        return fp8cast_rowwise_kmajor(silu_mul(g, u), grouped_mm_offs)
    out = _alloc(M, N, gu.device)
    _get_kernel(N, False, gu.device)(None, gu[:, :N], gu[:, N:], grouped_mm_offs, *out)
    return out


def silu_mul_bwd_quant_gu(
    dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(cat([dg, du], 1), offs)`` of the SwiGLU gradients, ``gu`` (M, 2N)
    the saved gate_up output and ``dh`` (M, N) the gradient of ``silu(g) * u``."""
    _check(gu, grouped_mm_offs)
    assert dh.dtype == torch.bfloat16 and dh.is_contiguous()
    M, N = gu.shape[0], gu.shape[1] // 2
    assert dh.shape == (M, N)
    if M == 0:
        return _alloc(0, 2 * N, gu.device)
    if not _fusable(N, M, grouped_mm_offs):
        g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
        dg, du = _silu_mul_bwd_reference(dh, g, u)
        return fp8cast_rowwise_kmajor(torch.cat([dg, du], 1), grouped_mm_offs)
    out = _alloc(M, 2 * N, gu.device)
    _get_kernel(N, True, gu.device)(dh, gu[:, :N], gu[:, N:], grouped_mm_offs, *out)
    return out


class FP8SwiGLUGateUpExpertsFunc(torch.autograd.Function):
    """
    gate_up_proj -> SwiGLU -> down_proj for FP8 routed experts with a fused gate_up weight
    (G, 2I, H), on Hopper, as one autograd node. The GEMMs are those of the unfused path (one
    gate_up GEMM, one down GEMM; in backward one dgrad and one wgrad each); SwiGLU lives inside the
    quantization kernels above.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list,
        ks_tensor: torch.Tensor,
        group_indices: torch.Tensor,
        q_gate_up: tuple[torch.Tensor, ...],
        q_down: tuple[torch.Tensor, ...],
        quantized_input: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if quantized_input is None:
            quantized_input = quantize_grouped_input(x, grouped_mm_offs)
        x_fp8, x_scale, x_ch, x_ch_scale = quantized_input
        inter2, hidden = w_gate_up.shape[1], w_down.shape[1]
        gu = _fp8_grouped_gemm((x_fp8, x_scale), q_gate_up[:2], inter2, grouped_mm_offs, group_indices)  # fmt: skip
        h_fp8, h_scale, _, _ = silu_mul_quant_gu_fwd(gu, grouped_mm_offs)
        y = _fp8_grouped_gemm((h_fp8, h_scale), q_down[:2], hidden, grouped_mm_offs, group_indices)
        # The K-major h for down_proj's wgrad is a pure function of gu, which is saved anyway:
        # backward quantizes it again (bit-identical) instead of keeping it alive meanwhile.
        ctx.save_for_backward(
            gu, x_ch, x_ch_scale, grouped_mm_offs, ks_tensor,
            q_gate_up[2], q_gate_up[3], q_down[2], q_down[3],
        )  # fmt: skip
        ctx.weights = (w_gate_up, w_down)
        ctx.ks, ctx.group_indices = ks, group_indices
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gu, x_ch, x_ch_scale, offs, ks_tensor, wgu_t, sgu_t, wd_t, sd_t = ctx.saved_tensors
        w_gate_up, w_down = ctx.weights
        ks, gi = ctx.ks, ctx.group_indices
        inter, hidden = w_down.shape[2], w_down.shape[1]

        dy_fp8, dy_scale, dy_ch, dy_ch_scale = fp8cast_rowwise_kmajor(dy, offs)
        dh = _fp8_grouped_gemm((dy_fp8, dy_scale), (wd_t, sd_t), inter, offs, gi)
        _, _, h_ch, h_ch_scale = silu_mul_quant_gu_fwd(gu, offs)
        _fp8_grouped_wgrad((dy_ch, dy_ch_scale), (h_ch, h_ch_scale), ks, ks_tensor, w_down, tuple(w_down.shape))  # fmt: skip
        dguq = silu_mul_bwd_quant_gu(dh, gu, offs)
        del dh
        dx = _fp8_grouped_gemm(dguq[:2], (wgu_t, sgu_t), hidden, offs, gi)
        _fp8_grouped_wgrad(dguq[2:], (x_ch, x_ch_scale), ks, ks_tensor, w_gate_up, tuple(w_gate_up.shape))  # fmt: skip
        return dx, None, None, None, None, None, None, None, None, None


def fp8_swiglu_gate_up_experts(
    gate_up_proj: torch.nn.Module,
    down_proj: torch.nn.Module,
    x: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    ks: list,
    ks_tensor: torch.Tensor,
    group_indices: torch.Tensor,
    quantized_input: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """
    down_proj(silu(g) * u) with [g | u] = gate_up_proj(x), both FP8GroupedLinear, through
    FP8SwiGLUGateUpExpertsFunc. With quantized_input (from fp8_scatter_quant) x is only a gradient
    carrier and is never read.
    """
    return FP8SwiGLUGateUpExpertsFunc.apply(
        x, gate_up_proj.weight, down_proj.weight, grouped_mm_offs, ks, ks_tensor, group_indices,
        gate_up_proj._get_quantized_weight(), down_proj._get_quantized_weight(), quantized_input,
    )  # fmt: skip
