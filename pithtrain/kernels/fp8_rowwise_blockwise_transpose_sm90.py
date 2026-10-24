"""
CuTe-DSL (SM90) port of ``fp8cast_rowwise_blockwise_transpose_sm100`` (SM90 E8M0 path).

One read of ``x (M, K)`` produces
  * the 1x128 rowwise FP8 copy ``tok_fp8 (M, K)`` with scales ``tok_scale (M, ceil(K/128))``
    (MN-major, see ``kernels/fp8_scale_layout.py``),
  * the transposed copy ``blk_t_fp8 (K, M)`` quantized with one scale per 128x128 block,
    ``blk_t_scale (ceil(K/128), ceil(M/128))``,
bit-identical to the Triton kernel in ``pithtrain/kernels/fp8_rowwise_blockwise_transpose_sm100.py``.
Every input is covered (a non-unit inner stride is made contiguous here), so there is no fallback.

Kernel layout: one CTA of ``1024 / R`` threads per 128x128 tile, ``R`` = 8 (large grids) or 4
(small grids). Lane ``l`` of warp ``w`` owns the R rows ``w*4R + (l//8)*R + [0, R)`` and the two
8-column chunks ``(l%8)*8 + [0, 8)`` and ``64 + (l%8)*8 + [0, 8)``: an R x 16 register block kept
as packed input-dtype words. Row amax is a 3-step butterfly over the 8 lanes sharing a row; tile
amax adds two more shuffles and one shared-memory exchange (the only barrier). The transposed
store needs no shared-memory transpose: each thread already holds R consecutive rows of every
column it owns and writes them as one R-byte vector; with R = 8 the 4 lanes of a column chunk
cover one full 32-byte sector.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake_stream
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale, exact_rcp
from pithtrain.kernels.fp8_scale_layout import mn_major_scale

__all__ = ["fp8cast_rowwise_blockwise_transpose_sm90"]

_BLOCK = 128

_SMALL_GRID_TILES = 200


# ---------------------------------------------------------------------------
# Device helpers (pure inline PTX, so LLVM may CSE / schedule them freely)
# ---------------------------------------------------------------------------


def _asm_u32(args: list, asm: str, constraints: str, loc=None, ip=None) -> Uint32:
    res = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip) for a in args],
        asm,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(res)


# The register block is kept as packed 32-bit words of the input dtype ("bf16", "f16": two
# elements per word, low half first; "f32": one). Every helper below consumes packed words and
# widens to fp32 only inside its own asm block, so LLVM cannot hoist the widening and keep an
# fp32 copy of the block live: that copy doubled the register footprint (166 regs/thread,
# 3 CTAs/SM) and forcing fewer registers made ptxas spill.


@dsl_user_op
def _absmax_acc(acc: Uint32, w: Uint32, kind: str, *, loc=None, ip=None) -> Uint32:
    """acc = max(acc, |w|) lane-wise in the input dtype (maxNum semantics, as tl.max)."""
    op = {"bf16": "max.bf16x2", "f16": "max.f16x2", "f32": "max.f32"}[kind]
    mask = "0x7FFFFFFF" if kind == "f32" else "0x7FFF7FFF"
    asm = f"{{\n\t.reg .b32 t;\n\tand.b32 t, $2, {mask};\n\t{op} $0, $1, t;\n\t}}"
    return _asm_u32([Uint32(acc), Uint32(w)], asm, "=r,r,r", loc, ip)


@dsl_user_op
def _amax_f32(acc: Uint32, kind: str, *, loc=None, ip=None) -> Float32:
    """Reduce a packed |max| word to one fp32 value (exact widening)."""
    if kind == "f32":
        body = "mov.b32 $0, $1;"
    else:
        ty = {"bf16": "bf16", "f16": "f16"}[kind]
        body = (
            ".reg .b16 l, h, m;\n\tmov.b32 {l, h}, $1;\n\t"
            f"max.{ty} m, l, h;\n\tcvt.f32.{ty} $0, m;"
        )
    res = llvm.inline_asm(
        T.f32(),
        [Uint32(acc).ir_value(loc=loc, ip=ip)],
        "{\n\t" + body + "\n\t}",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(res)


@dsl_user_op
def _q4(words: list, sel: list, kind: str, rcp: Float32, *, loc=None, ip=None) -> Uint32:
    """Quantize 4 elements to 4 packed e4m3 bytes (element 0 in the lowest byte).

    ``sel[k] = (j, half)`` picks element k from ``words[j]`` (``half`` 0 = low 16 bits). Each is
    widened to fp32, multiplied by ``rcp`` (an exact power of 2) and converted with
    ``cvt.rn.satfinite.e4m3x2.f32`` (RNE, saturate to +-448) -- the instruction Triton emits for
    ``.to(tl.float8e4nv)`` on SM90. The first source of ``e4m3x2`` lands in the upper byte.
    """
    n = len(words)
    rcp_op = f"${n + 1}"
    lines = [".reg .b32 f0, f1, f2, f3;", ".reg .b16 l, h, q0, q1;"]
    for k, (j, half) in enumerate(sel):
        w = f"${j + 1}"
        if kind == "bf16":
            if half == 0:
                lines.append(f"shl.b32 f{k}, {w}, 16;")
            else:
                lines.append(f"and.b32 f{k}, {w}, 0xFFFF0000;")
        elif kind == "f16":
            lines.append(f"mov.b32 {{l, h}}, {w};")
            lines.append(f"cvt.f32.f16 f{k}, {'l' if half == 0 else 'h'};")
        else:
            lines.append(f"mov.b32 f{k}, {w};")
        lines.append(f"mul.rn.f32 f{k}, f{k}, {rcp_op};")
    lines += [
        "cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0;",
        "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2;",
        "mov.b32 $0, {q0, q1};",
    ]
    asm = "{\n\t" + "\n\t".join(lines) + "\n\t}"
    constraints = "=r," + ",".join(["r"] * n) + ",f"
    args = [Uint32(w) for w in words] + [Float32(rcp)]
    return _asm_u32(args, asm, constraints, loc, ip)


def _gptr(base: cute.Pointer, offset: Int64, esize: int, dtype: type, align: int) -> cute.Pointer:
    """Global pointer ``offset`` elements (of ``esize`` bytes) past ``base``, re-typed to
    ``dtype`` and asserted ``align``-byte aligned (the host picked widths that make it true)."""
    addr = base.toint() + offset * esize
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class RowwiseBlockwiseTranspose:
    """``vec_k`` / ``vec_m``: element counts every global vector access along K (x and tok rows)
    / along M (blk_t rows) may use; chosen on the host so no vector straddles the edge."""

    def __init__(self, dtype: type, vec_k: int, vec_m: int, rows: int = 8) -> None:
        self.dtype = dtype
        self.rows = rows
        self.threads = 1024 // rows
        self.kind = {cutlass.BFloat16: "bf16", cutlass.Float16: "f16", Float32: "f32"}[dtype]
        self.vec_k = vec_k
        self.vec_m = vec_m

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mQT: cute.Tensor,
        mST: cute.Tensor,
        stream,
    ) -> None:
        M = mX.shape[0]
        K = mX.shape[1]
        grid_m = (M + _BLOCK - 1) // _BLOCK
        grid_k = (K + _BLOCK - 1) // _BLOCK
        self.kernel(mX, mQ, mS, mQT, mST).launch(
            grid=[grid_m, grid_k, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mQT: cute.Tensor,
        mST: cute.Tensor,
    ) -> None:
        VK = const_expr(self.vec_k)
        VM = const_expr(self.vec_m)
        R = const_expr(self.rows)
        NW = const_expr(32 // self.rows)
        esize = const_expr(self.dtype.width // 8)
        EPW = const_expr(32 // self.dtype.width)
        KIND = const_expr(self.kind)

        tidx, _, _ = cute.arch.thread_idx()
        bm, bk, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        rg = lane // 8
        cl = lane % 8

        M = Int32(mX.shape[0])
        K = Int32(mX.shape[1])
        sx = Int64(mX.stride[0])
        # mS is the MN-major tok_scale (M, nkb), strides (1, align(M, 4)): scale[r, bk] at
        # bk * s_ld + r, the layout DeepGEMM's SM90 GEMMs read as is.
        s_ld = Int64(mS.stride[1])
        nmb = Int32(mST.shape[1])

        row0 = bm * _BLOCK + warp * (4 * R) + rg * R
        col_base = bk * _BLOCK + cl * 8

        # ---- load the 8x16 register block, zero-filled outside (M, K) ----
        # Element (i, h, e) -- row row0 + i, column col_base + h * 64 + e -- sits at n = i*16+h*8+e.
        xr = cute.make_rmem_tensor(cute.make_layout(16 * R), self.dtype)
        xr.fill(0.0)
        for i in cutlass.range_constexpr(R):
            r = row0 + i
            for h in cutlass.range_constexpr(2):
                for p in cutlass.range_constexpr(8 // VK):
                    c = col_base + h * 64 + p * VK
                    if r < M:
                        if c < K:
                            off = Int64(r) * sx + Int64(c)
                            src = cute.make_tensor(
                                _gptr(mX.iterator, off, esize, self.dtype, VK * esize),
                                cute.make_layout(VK),
                            )
                            dst = cute.make_tensor(
                                xr.iterator + (i * 16 + h * 8 + p * VK), cute.make_layout(VK)
                            )
                            cute.autovec_copy(src, dst)
        xw = cute.recast_tensor(xr, Uint32)

        # ---- rowwise: amax over the 128 columns of each row ----
        row_amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for i in cutlass.range_constexpr(R):
            acc = Uint32(0)
            for n in cutlass.range_constexpr(i * 16, i * 16 + 16, EPW):
                acc = _absmax_acc(acc, xw[n // EPW], KIND)
            m = _amax_f32(acc, KIND)
            for off in cutlass.range_constexpr(3):
                m = cute.arch.fmax(m, cute.arch.shuffle_sync_bfly(m, offset=1 << off))
            row_amax[i] = m

        qb = cute.make_rmem_tensor(cute.make_layout(8), Uint8)
        qw = cute.recast_tensor(qb, Uint32)
        for i in cutlass.range_constexpr(R):
            r = row0 + i
            scale = e8m0_scale(row_amax[i])
            rcp = exact_rcp(scale)
            if r < M:
                if cl == i:
                    stile = cute.make_tensor(
                        _gptr(mS.iterator, Int64(bk) * s_ld + Int64(r), 4, Float32, 4),
                        cute.make_layout(1),
                    )
                    stile[0] = scale
                for h in cutlass.range_constexpr(2):
                    for g in cutlass.range_constexpr(2):
                        elems = [i * 16 + h * 8 + 4 * g + k for k in range(4)]
                        qw[g] = self._quant4(xw, elems, rcp)
                    for p in cutlass.range_constexpr(8 // VK):
                        c = col_base + h * 64 + p * VK
                        if c < K:
                            dst = cute.make_tensor(
                                _gptr(mQ.iterator, Int64(r) * Int64(K) + Int64(c), 1, Uint8, VK),
                                cute.make_layout(VK),
                            )
                            src = cute.make_tensor(qb.iterator + p * VK, cute.make_layout(VK))
                            cute.autovec_copy(src, dst)

        # ---- blockwise: amax of the whole 128x128 tile ----
        bmax = row_amax[0]
        for i in cutlass.range_constexpr(1, R):
            bmax = cute.arch.fmax(bmax, row_amax[i])
        bmax = cute.arch.fmax(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=8))
        bmax = cute.arch.fmax(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=16))

        smem = cutlass.memory.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout(NW), byte_alignment=16)
        if lane == 0:
            red[warp] = bmax
        cute.arch.barrier()
        bmax = red[0]
        for w in cutlass.range_constexpr(1, NW):
            bmax = cute.arch.fmax(bmax, red[w])
        bscale = e8m0_scale(bmax)
        brcp = exact_rcp(bscale)
        if tidx == 0:
            st = cute.make_tensor(
                _gptr(mST.iterator, Int64(bk) * Int64(nmb) + Int64(bm), 4, Float32, 4),
                cute.make_layout(1),
            )
            st[0] = bscale

        # ---- transposed store: column c of x becomes row c of blk_t ----
        # (threads whose rows all lie past M have nothing to store)
        if row0 < M:
            for h in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(8):
                    c = col_base + h * 64 + e
                    if c < K:
                        for g in cutlass.range_constexpr(R // 4):
                            qw[g] = self._quant4(
                                xw, [(4 * g + k) * 16 + h * 8 + e for k in range(4)], brcp
                            )
                        for p in cutlass.range_constexpr(R // VM):
                            r = row0 + p * VM
                            if r < M:
                                off = Int64(c) * Int64(M) + Int64(r)
                                dst = cute.make_tensor(
                                    _gptr(mQT.iterator, off, 1, Uint8, VM), cute.make_layout(VM)
                                )
                                src = cute.make_tensor(qb.iterator + p * VM, cute.make_layout(VM))
                                cute.autovec_copy(src, dst)

    def _quant4(self, xw: cute.Tensor, elems: list, rcp: Float32) -> Uint32:
        """Quantize the 4 register-block elements at flat indices ``elems``."""
        epw = 32 // self.dtype.width
        words: list = []
        sel: list = []
        for n in elems:
            w = n // epw
            if w not in words:
                words.append(w)
            sel.append((words.index(w), n % epw))
        return _q4([xw[w] for w in words], sel, self.kind, rcp)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

_COMPILED: dict[tuple, Callable] = {}


def _vec(n: int, cap: int) -> int:
    v = cap
    while n % v:
        v //= 2
    return v


def _compile(dtype: type, vec_k: int, vec_m: int, rows: int) -> Callable:
    m_sym, k_sym, kb_sym, mb_sym = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    esize = dtype.width // 8
    fake = cute.runtime.make_fake_tensor
    x = fake(
        dtype,
        (m_sym, k_sym),
        stride=(cute.sym_int64(divisibility=vec_k), 1),
        assumed_align=vec_k * esize,
    )
    q = fake(cutlass.Float8E4M3FN, (m_sym, k_sym), stride=(k_sym, 1), assumed_align=16)
    s = fake(Float32, (m_sym, kb_sym), stride=(1, cute.sym_int64()), assumed_align=16)
    qt = fake(cutlass.Float8E4M3FN, (k_sym, m_sym), stride=(m_sym, 1), assumed_align=16)
    st = fake(Float32, (kb_sym, mb_sym), stride=(mb_sym, 1), assumed_align=16)
    return cute.compile(
        RowwiseBlockwiseTranspose(dtype, vec_k, vec_m, rows),
        x,
        q,
        s,
        qt,
        st,
        fake_stream(),
        options="--enable-tvm-ffi",
    )


_DISPATCH: dict[tuple, tuple] = {}


def _resolve(dtype: torch.dtype, M: int, K: int, sx: int, ptr: int) -> tuple:
    """(compiled kernel or None for an empty input, ceil(K/128), ceil(M/128))."""
    nkb = (K + _BLOCK - 1) // _BLOCK
    nmb = (M + _BLOCK - 1) // _BLOCK
    if M == 0 or K == 0:
        return None, nkb, nmb
    esize = dtype.itemsize
    vec_k = _vec(K, min(16 // esize, 8))
    vec_k = _vec(sx, vec_k)
    vec_k = _vec(ptr // esize, vec_k)
    # 128 threads x 8 rows wins once the grid fills the GPU (fewer, fuller stores); a small grid
    # is latency-bound, so split each tile over 256 threads x 4 rows (H200 measurements in
    # workspace/kda-tasks/rowwise_blockwise_transpose/benchmark.csv).
    rows = 4 if nmb * nkb <= _SMALL_GRID_TILES else 8
    vec_m = _vec(M, rows)
    ckey = (dtype, vec_k, vec_m, rows)
    fn = _COMPILED.get(ckey)
    if fn is None:
        fn = _COMPILED[ckey] = _compile(TORCH2CUTE[dtype], vec_k, vec_m, rows)
    if len(_DISPATCH) > 4096:
        _DISPATCH.clear()
    return fn, nkb, nmb


def fp8cast_rowwise_blockwise_transpose_sm90(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise FP8 quant of *x* and blockwise FP8 quant of *x*.T.

    Drop-in replacement for ``fp8cast_rowwise_blockwise_transpose_sm100``.

    Args:
        x: 2D tensor (M, K) in BF16/FP16/FP32.

    Returns:
        (tok_fp8, tok_scale, blk_t_fp8, blk_t_scale):
            tok_fp8     -- float8_e4m3fn (M, K),
            tok_scale   -- float32 (M, ceil(K/128)), MN-major and TMA-aligned
                           (``kernels.fp8_scale_layout``: strides (1, align(M, 4))),
            blk_t_fp8   -- float8_e4m3fn (K, M),
            blk_t_scale -- float32 (ceil(K/128), ceil(M/128)).
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    sx, sk = x.stride()
    if sk != 1:
        # as_strided pins the unit inner stride that torch leaves arbitrary when K == 1.
        x = x.contiguous().as_strided((M, K), (K, 1))
        sx = K
    dtype = x.dtype
    # Host cost matters at ~770 calls/step: resolve everything per (dtype, shape, row stride,
    # pointer alignment) once, so the steady state is one dict lookup + 4 allocs + the call.
    key = (dtype, M, K, sx, x.data_ptr() & 15)
    entry = _DISPATCH.get(key)
    if entry is None:
        entry = _DISPATCH[key] = _resolve(dtype, M, K, sx, x.data_ptr())
    fn, nkb, nmb = entry
    dev = x.device
    out_tok = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    scale_tok = mn_major_scale(M, nkb, dev)
    out_blk_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_blk_t = torch.empty((nkb, nmb), dtype=torch.float32, device=dev)
    if fn is not None:
        fn(x, out_tok, scale_tok, out_blk_t, scale_blk_t)
    return out_tok, scale_tok, out_blk_t, scale_blk_t
