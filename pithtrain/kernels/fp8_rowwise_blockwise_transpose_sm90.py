"""
CuTe-DSL (SM90) rowwise FP8 quantization of x and 128x128 blockwise FP8 quantization of x.T from one
read of x, bit-identical to the Triton kernel in fp8_rowwise_blockwise_transpose_sm100. Covers every
input, so there is no fallback.

One CTA of 1024 / R threads per 128x128 tile, R = 8 (large grids) or 4 (small grids); each thread
owns R consecutive rows of 16 columns, so the transposed store needs no smem transpose: it writes
each column's R rows as one R-byte vector.
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

BLOCK = 128

SMALL_GRID_TILES = 200


def asm_u32(args: list, asm: str, constraints: str, loc=None, ip=None) -> Uint32:
    res = llvm.inline_asm(T.i32(), [a.ir_value(loc=loc, ip=ip) for a in args], asm, constraints, has_side_effects=False, is_align_stack=False, loc=loc, ip=ip)
    return Uint32(res)


# The register block is kept as packed 32-bit words of the input dtype (bf16 / f16: two elements
# per word, low half first; f32: one). Every helper widens to fp32 only inside its own asm block,
# so LLVM cannot hoist the widening and keep an fp32 copy of the block live: that copy doubled the
# register footprint, and forcing fewer registers made ptxas spill.


@dsl_user_op
def absmax_acc(acc: Uint32, w: Uint32, kind: str, *, loc=None, ip=None) -> Uint32:
    """acc = max(acc, |w|) lane-wise in the input dtype (maxNum semantics, as tl.max)."""
    op = {"bf16": "max.bf16x2", "f16": "max.f16x2", "f32": "max.f32"}[kind]
    mask = "0x7FFFFFFF" if kind == "f32" else "0x7FFF7FFF"
    asm = f"{{\n\t.reg .b32 t;\n\tand.b32 t, $2, {mask};\n\t{op} $0, $1, t;\n\t}}"
    return asm_u32([Uint32(acc), Uint32(w)], asm, "=r,r,r", loc, ip)


@dsl_user_op
def amax_f32(acc: Uint32, kind: str, *, loc=None, ip=None) -> Float32:
    """Reduce a packed |max| word to one fp32 value (exact widening)."""
    if kind == "f32":
        body = "mov.b32 $0, $1;"
    else:
        ty = {"bf16": "bf16", "f16": "f16"}[kind]
        body = (
            ".reg .b16 l, h, m;\n\tmov.b32 {l, h}, $1;\n\t"
            f"max.{ty} m, l, h;\n\tcvt.f32.{ty} $0, m;"
        )
    res = llvm.inline_asm(T.f32(), [Uint32(acc).ir_value(loc=loc, ip=ip)], "{\n\t" + body + "\n\t}", "=f,r", has_side_effects=False, is_align_stack=False, loc=loc, ip=ip)
    return Float32(res)


@dsl_user_op
def q4(words: list, sel: list, kind: str, rcp: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Quantize 4 elements to 4 packed e4m3 bytes (element 0 in the lowest byte). sel[k] = (j, half)
    picks element k from words[j] (half 0 = low 16 bits); rcp is an exact power of 2. The cvt is the
    one Triton emits for .to(tl.float8e4nv) on SM90; its first source lands in the upper byte.
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
    lines += ["cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0;", "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2;", "mov.b32 $0, {q0, q1};"]
    asm, constraints, args = "{\n\t" + "\n\t".join(lines) + "\n\t}", "=r," + ",".join(["r"] * n) + ",f", [Uint32(w) for w in words] + [Float32(rcp)]
    return asm_u32(args, asm, constraints, loc, ip)


def gptr(base: cute.Pointer, offset: Int64, esize: int, dtype: type, align: int) -> cute.Pointer:
    """
    Global pointer offset elements (of esize bytes) past base, as dtype,
    asserted align-byte aligned.
    """
    addr = base.toint() + offset * esize
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


class RowwiseBlockwiseTranspose:
    """
    vec_k / vec_m: elements per global vector access along K (x and tok rows) / along M (blk_t
    rows), chosen on the host so no vector straddles the edge.
    """

    def __init__(self, dtype: type, vec_k: int, vec_m: int, rows: int = 8) -> None:
        self.dtype = dtype
        self.rows = rows
        self.threads = 1024 // rows
        self.kind = {cutlass.BFloat16: "bf16", cutlass.Float16: "f16", Float32: "f32"}[dtype]
        self.vec_k = vec_k
        self.vec_m = vec_m

    @cute.jit
    def __call__(self, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor, mQT: cute.Tensor, mST: cute.Tensor, stream) -> None:
        M, K = mX.shape[0], mX.shape[1]
        grid_m, grid_k = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mX, mQ, mS, mQT, mST).launch(grid=[grid_m, grid_k, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor, mQT: cute.Tensor, mST: cute.Tensor) -> None:
        VK, VM, R, NW = const_expr(self.vec_k), const_expr(self.vec_m), const_expr(self.rows), const_expr(32 // self.rows)
        esize, EPW, KIND = const_expr(self.dtype.width // 8), const_expr(32 // self.dtype.width), const_expr(self.kind)
        tidx, _, _ = cute.arch.thread_idx()
        bm, bk, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        rg = lane // 8
        cl = lane % 8

        M = Int32(mX.shape[0])
        K = Int32(mX.shape[1])
        sx = Int64(mX.stride[0])
        # mS is the MN-major tok_scale: scale[r, bk] at bk * s_ld + r.
        s_ld = Int64(mS.stride[1])
        nmb = Int32(mST.shape[1])

        row0, col_base = bm * BLOCK + warp * (4 * R) + rg * R, bk * BLOCK + cl * 8

        # Register block zero-filled outside (M, K): element (row row0 + i, column col_base + h * 64 + e) at n = i * 16 + h * 8 + e.
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
                            src = cute.make_tensor(gptr(mX.iterator, off, esize, self.dtype, VK * esize), cute.make_layout(VK))
                            dst = cute.make_tensor(xr.iterator + (i * 16 + h * 8 + p * VK), cute.make_layout(VK))
                            cute.autovec_copy(src, dst)
        xw = cute.recast_tensor(xr, Uint32)

        row_amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for i in cutlass.range_constexpr(R):
            acc = Uint32(0)
            for n in cutlass.range_constexpr(i * 16, i * 16 + 16, EPW):
                acc = absmax_acc(acc, xw[n // EPW], KIND)
            m = amax_f32(acc, KIND)
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
                    stile = cute.make_tensor(gptr(mS.iterator, Int64(bk) * s_ld + Int64(r), 4, Float32, 4), cute.make_layout(1))
                    stile[0] = scale
                for h in cutlass.range_constexpr(2):
                    for g in cutlass.range_constexpr(2):
                        elems = [i * 16 + h * 8 + 4 * g + k for k in range(4)]
                        qw[g] = self.quant4(xw, elems, rcp)
                    for p in cutlass.range_constexpr(8 // VK):
                        c = col_base + h * 64 + p * VK
                        if c < K:
                            dst = cute.make_tensor(gptr(mQ.iterator, Int64(r) * Int64(K) + Int64(c), 1, Uint8, VK), cute.make_layout(VK))
                            src = cute.make_tensor(qb.iterator + p * VK, cute.make_layout(VK))
                            cute.autovec_copy(src, dst)

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
            st = cute.make_tensor(gptr(mST.iterator, Int64(bk) * Int64(nmb) + Int64(bm), 4, Float32, 4), cute.make_layout(1))
            st[0] = bscale

        if row0 < M:
            for h in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(8):
                    c = col_base + h * 64 + e
                    if c < K:
                        for g in cutlass.range_constexpr(R // 4):
                            qw[g] = self.quant4(xw, [(4 * g + k) * 16 + h * 8 + e for k in range(4)], brcp)
                        for p in cutlass.range_constexpr(R // VM):
                            r = row0 + p * VM
                            if r < M:
                                off = Int64(c) * Int64(M) + Int64(r)
                                dst = cute.make_tensor(gptr(mQT.iterator, off, 1, Uint8, VM), cute.make_layout(VM))
                                src = cute.make_tensor(qb.iterator + p * VM, cute.make_layout(VM))
                                cute.autovec_copy(src, dst)

    def quant4(self, xw: cute.Tensor, elems: list, rcp: Float32) -> Uint32:
        """Quantize the 4 register-block elements at flat indices elems."""
        epw = 32 // self.dtype.width
        words: list = []
        sel: list = []
        for n in elems:
            w = n // epw
            if w not in words:
                words.append(w)
            sel.append((words.index(w), n % epw))
        return q4([xw[w] for w in words], sel, self.kind, rcp)


COMPILED: dict[tuple, Callable] = {}


def vec(n: int, cap: int) -> int:
    v = cap
    while n % v:
        v //= 2
    return v


def compile_kernel(dtype: type, vec_k: int, vec_m: int, rows: int) -> Callable:
    m_sym, k_sym, kb_sym, mb_sym = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    esize = dtype.width // 8
    fake = cute.runtime.make_fake_tensor
    x = fake(dtype, (m_sym, k_sym), stride=(cute.sym_int64(divisibility=vec_k), 1), assumed_align=vec_k * esize)
    q = fake(cutlass.Float8E4M3FN, (m_sym, k_sym), stride=(k_sym, 1), assumed_align=16)
    s = fake(Float32, (m_sym, kb_sym), stride=(1, cute.sym_int64()), assumed_align=16)
    qt = fake(cutlass.Float8E4M3FN, (k_sym, m_sym), stride=(m_sym, 1), assumed_align=16)
    st = fake(Float32, (kb_sym, mb_sym), stride=(mb_sym, 1), assumed_align=16)
    return cute.compile(RowwiseBlockwiseTranspose(dtype, vec_k, vec_m, rows), x, q, s, qt, st, fake_stream(), options="--enable-tvm-ffi")


DISPATCH: dict[tuple, tuple] = {}


def resolve(dtype: torch.dtype, M: int, K: int, sx: int, ptr: int) -> tuple:
    """(compiled kernel or None for an empty input, ceil(K/128), ceil(M/128))."""
    nkb, nmb = (K + BLOCK - 1) // BLOCK, (M + BLOCK - 1) // BLOCK
    if M == 0 or K == 0:
        return None, nkb, nmb
    esize = dtype.itemsize
    vec_k = vec(K, min(16 // esize, 8))
    vec_k = vec(sx, vec_k)
    vec_k = vec(ptr // esize, vec_k)
    # 128 threads x 8 rows wins once the grid fills the GPU (fewer, fuller stores); a small grid
    # is latency-bound, so split each tile over 256 threads x 4 rows.
    rows = 4 if nmb * nkb <= SMALL_GRID_TILES else 8
    vec_m = vec(M, rows)
    ckey = (dtype, vec_k, vec_m, rows)
    fn = COMPILED.get(ckey)
    if fn is None:
        fn = COMPILED[ckey] = compile_kernel(TORCH2CUTE[dtype], vec_k, vec_m, rows)
    if len(DISPATCH) > 4096:
        DISPATCH.clear()
    return fn, nkb, nmb


def fp8cast_rowwise_blockwise_transpose_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_blockwise_transpose with the rowwise scale MN-major."""
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    sx, sk = x.stride()
    if sk != 1:
        # as_strided pins the unit inner stride that torch leaves arbitrary when K == 1.
        x = x.contiguous().as_strided((M, K), (K, 1))
        sx = K
    dtype = x.dtype
    # Host cost matters (hundreds of calls per step): resolve once per (dtype, shape, row stride, pointer alignment).
    key = (dtype, M, K, sx, x.data_ptr() & 15)
    entry = DISPATCH.get(key)
    if entry is None:
        entry = DISPATCH[key] = resolve(dtype, M, K, sx, x.data_ptr())
    fn, nkb, nmb = entry
    dev = x.device
    out_tok = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    scale_tok = mn_major_scale(M, nkb, dev)
    out_blk_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_blk_t = torch.empty((nkb, nmb), dtype=torch.float32, device=dev)
    if fn is not None:
        fn(x, out_tok, scale_tok, out_blk_t, scale_blk_t)
    return out_tok, scale_tok, out_blk_t, scale_blk_t
