"""CuTe-DSL (SM90) kernel of fp8cast_rowwise_kmajor (contract in fp8_rowwise_kmajor), bit-identical to the Triton kernel, with both scales returned MN-major."""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, ld_b32, ld_v4, lds_b32, max_u16x2, prmt, selp, st_b32, st_v2, sts_b32
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale_bf16x2
from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad, to_mn_major

__all__ = ["RowwiseKmajor", "fp8cast_rowwise_kmajor_sm90", "supports_rowwise_kmajor"]

THREADS = 256
TILE = 128


@dsl_user_op
def _qblock(words: list, rr: list, cr: list, *, loc=None, ip=None) -> tuple:
    """
    Quantize a 4 x 4 bf16 block for both outputs, widening each element to fp32 once.

    words[r * 2 + q] holds row r, columns 2q and 2q + 1. Returns four rowwise words (row r, columns 0..3, times rr[r]) and four K-major words (column c, rows 0..3, times cr[c]); byte k of a word is its element k. cvt.rn.satfinite.e4m3x2.f32 is what Triton's .to(tl.float8e4nv) lowers to on SM90, which keeps the bytes identical.
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
        L += ["cvt.rn.satfinite.e4m3x2.f32 l, y1, y0;", "cvt.rn.satfinite.e4m3x2.f32 h, y3, y2;", f"mov.b32 ${r}, {{l, h}};"]
    for c in range(4):  # K-major
        for r in range(4):
            L.append(f"mul.rn.f32 y{r}, x{r * 4 + c}, ${20 + c};")
        L += ["cvt.rn.satfinite.e4m3x2.f32 l, y1, y0;", "cvt.rn.satfinite.e4m3x2.f32 h, y3, y2;", f"mov.b32 ${4 + c}, {{l, h}};"]
    L.append("}")
    st = llvm.StructType.get_literal([T.i32()] * 8)
    args = [Uint32(x).ir_value(loc=loc, ip=ip) for x in list(words) + list(rr) + list(cr)]
    res = asm(st, args, "\n\t".join(L), ",".join(["=r"] * 8 + ["r"] * 16))
    out = [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(8)]
    return out[:4], out[4:]


class RowwiseKmajor:
    """
    One 128 x 128 tile per CTA; 256 threads, each owning an 8 x 8 sub-block, so the K-major bytes of eight consecutive rows of a column come out without a transpose.

    Lane l of warp w owns column vector cv = l % 8 + 8 * (w % 2) (8 columns) and row block rb = l // 8 + 4 * (w // 2) (8 rows). The grid walks column blocks fastest, so co-resident CTAs cover whole rows of the input and outputs (and long runs of each K-major column), which keeps DRAM access local.
    """

    RED_WORDS = 64 * 6
    """Shared-memory words _tile reduces through; subclasses (ScatterQuant) size theirs by it."""

    def __init__(self, n: int, num_groups: int) -> None:
        self.n = n
        self.nb = (n + TILE - 1) // TILE
        self.n_pad = mn_pad(n)  # column stride of the MN-major per-channel scale
        self.num_groups = num_groups

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKm: cute.Tensor, mKmScale: cute.Tensor, stream) -> None:
        mb = mX.shape[0] // TILE
        self.kernel(mX, mOffs, mTok, mTokScale, mKm, mKmScale).launch(grid=[self.nb, mb, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mOffs: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKm: cute.Tensor, mKmScale: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pid_col, pid_row, _ = cute.arch.block_idx()
        gp = (mX.iterator.toint(), mOffs.iterator.toint(), mTok.iterator.toint(), mTokScale.iterator.toint(), mKm.iterator.toint(), mKmScale.iterator.toint())
        smem = cutlass.memory.SmemAllocator()
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)

        v = []
        for i in cutlass.range_constexpr(8):
            addr, ok = self._src(gp[0], pid_row, pid_col, tidx, i)
            v.append(ld_v4(addr, ok))
        self._tile(v, pid_row, pid_col, tidx, mX.shape[0], gp, s_red.iterator.toint())

    def _coords(self, tidx):
        lane = tidx % 32
        warp = tidx // 32
        cv = lane % 8 + 8 * (warp % 2)
        rb = lane // 8 + 4 * (warp // 2)
        return lane, warp, cv, rb

    def _src(self, x_base, pid_row, pid_col, tidx, i):
        """Global address of row i of this thread's sub-block, and whether it is in bounds."""
        _, _, cv, rb = self._coords(tidx)
        col0 = pid_col * TILE + cv * 8
        row = Int64(pid_row) * TILE + rb * 8 + i
        return x_base + (row * self.n + col0) * 2, Int32(col0 < self.n)

    def _tile(self, v, pid_row, pid_col, tidx, M, gp, red_base) -> None:
        x_base, offs_base, tok_base, tok_scale_base, km_base, km_scale_base = gp
        N, G = self.n, self.num_groups
        lane, warp, cv, rb = self._coords(tidx)
        cl = lane % 8  # column vector within the warp
        rl = lane // 8  # row block within the warp

        row_start = Int64(pid_row) * TILE
        col0 = pid_col * TILE + cv * 8  # first of this thread's 8 columns
        col_ok = Int32(col0 < N)
        row0 = row_start + rb * 8  # first of this thread's 8 rows

        # group lookup by warp ballot: gid = min(searchsorted(offs, row_start, right), G - 1)
        cnt = Int32(0)
        for c in range((G + 31) // 32):
            idx = lane + c * 32
            o = ld_b32(offs_base + Int64(idx) * 4, Int32(idx < G), Int32(0x7FFFFFFF))
            cnt = cnt + cute.arch.popc(cute.arch.vote_ballot_sync(Int64(o) <= row_start))
        gid = cutlass.min(cnt, G - 1)
        g_end = Int64(ld_b32(offs_base + Int64(gid) * 4, Int32(1), Int32(0)))
        g_start = Int64(ld_b32(offs_base + Int64(gid - 1) * 4, Int32(gid > 0), Int32(0)))
        k_g = g_end - g_start

        a = [[v[i][k] & Uint32(0x7FFF7FFF) for k in range(4)] for i in range(8)]
        # column amax over this thread's 8 rows: cp[k] packs columns (2k, 2k + 1)
        cp = []
        for k in range(4):
            m = a[0][k]
            for i in range(1, 8):
                m = max_u16x2(m, a[i][k])
            cp.append(m)
        # row amax over this thread's 8 columns: rp[p] packs rows (2p, 2p + 1)
        rp = []
        for p in range(4):
            r0, r1 = (max_u16x2(max_u16x2(a[i][0], a[i][1]), max_u16x2(a[i][2], a[i][3])) for i in (2 * p, 2 * p + 1))
            rp.append(max_u16x2(prmt(r0, r1, 0x5410), prmt(r0, r1, 0x7632)))

        # in-warp butterflies: lane bits 0-2 run along cv, bits 3-4 along rb
        for off in (1, 2, 4):
            for p in range(4):
                rp[p] = max_u16x2(rp[p], cute.arch.shuffle_sync_bfly(rp[p], offset=off))
        for off in (8, 16):
            for k in range(4):
                cp[k] = max_u16x2(cp[k], cute.arch.shuffle_sync_bfly(cp[k], offset=off))

        # cross-warp: 2 warps along cv (rows), 4 along rb (columns), through shared memory
        row_red = red_base
        col_red = red_base + 64 * 2 * 4
        for p in range(4):
            sts_b32(row_red + ((rb * 4 + p) * 2 + warp % 2) * 4, rp[p], Int32(cl == 0))
        for k in range(4):
            sts_b32(col_red + ((cv * 4 + k) * 4 + warp // 2) * 4, cp[k], Int32(rl == 0))
        cute.arch.sync_threads()
        for p in range(4):
            base = row_red + (rb * 4 + p) * 2 * 4
            rp[p] = max_u16x2(lds_b32(base), lds_b32(base + 4))
        for k in range(4):
            base = col_red + (cv * 4 + k) * 4 * 4
            m = lds_b32(base)
            for w in range(1, 4):
                m = max_u16x2(m, lds_b32(base + w * 4))
            cp[k] = m

        # scales: one lane per (warp, row) / (warp, column), broadcast by shuffle
        # this lane: row rb * 8 + cl (half cl % 2 of rp[cl // 2]), columns rl and rl + 4
        rsel = selp(Int32(cl < 4), selp(Int32(cl < 2), rp[0], rp[1]), selp(Int32(cl < 6), rp[2], rp[3]))
        r_sc, r_rc = e8m0_scale_bf16x2(rsel, Int32(cl % 2))
        ca_sc, ca_rc = e8m0_scale_bf16x2(selp(Int32(rl < 2), cp[0], cp[1]), Int32(rl % 2))
        cb_sc, cb_rc = e8m0_scale_bf16x2(selp(Int32(rl < 2), cp[2], cp[3]), Int32(rl % 2))
        row_rcp = [Uint32(cute.arch.shuffle_sync(Int32(r_rc), offset=(lane & 24) | i)) for i in range(8)]
        col_rcp = [Uint32(cute.arch.shuffle_sync(Int32(ca_rc if j < 4 else cb_rc), offset=(j % 4) * 8 + cl)) for j in range(8)]

        # MN-major scales: tok_scale[r, j] at j * M + r, km_scale[c, b] at b * align(N, 4) + c
        st_b32(tok_scale_base + (Int64(pid_col) * M + row0 + cl) * 4, r_sc, Int32(warp % 2 == 0))
        for c, sc in ((col0 + rl, ca_sc), (col0 + rl + 4, cb_sc)):
            addr = km_scale_base + (Int64(pid_row) * self.n_pad + c) * 4
            st_b32(addr, sc, Int32(warp // 2 == 0) & Int32(c < N))

        # FP8: rowwise (M, N) and K-major (group block (N, K_g) at g_start * N)
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
                st_v2(tok_base + (row0 + i) * N + col0, tok[i][0], tok[i][1], col_ok)
        for j in range(8):
            st_v2(km_row + Int64(col0 + j) * k_g, km[j][0], km[j][1], col_ok)


_COMPILED: dict = {}


def _compile(n: int, num_groups: int) -> Callable:
    fake = cute.runtime.make_fake_tensor
    m = cute.sym_int()
    mn = cute.sym_int()
    mb = cute.sym_int()
    nb = (n + TILE - 1) // TILE
    x = fake(cutlass.BFloat16, (m, n), stride=(n, 1), assumed_align=16)
    offs = fake(Int32, (num_groups,), stride=(1,), assumed_align=4)
    tok = fake(cutlass.Float8E4M3FN, (m, n), stride=(n, 1), assumed_align=16)
    tok_scale = fake(cutlass.Float32, (m, nb), stride=(1, cute.sym_int64()), assumed_align=16)
    km = fake(cutlass.Float8E4M3FN, (mn,), stride=(1,), assumed_align=16)
    km_scale = fake(cutlass.Float32, (n, mb), stride=(1, mn_pad(n)), assumed_align=16)
    return cute.compile(RowwiseKmajor(n, num_groups), x, offs, tok, tok_scale, km, km_scale, fake_stream(), options="--enable-tvm-ffi")


def _empty_outputs(x: torch.Tensor, M: int, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = x.device
    return (torch.empty((M, N), dtype=torch.float8_e4m3fn, device=dev), mn_major_scale(M, (N + TILE - 1) // TILE, dev), torch.empty(M * N, dtype=torch.float8_e4m3fn, device=dev), mn_major_scale(N, M // TILE, dev))


def supports_rowwise_kmajor(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> bool:
    """Whether the CuTe kernel covers this input: contiguous, 16-byte aligned bf16 x with N % 8 == 0, and non-empty contiguous int32 grouped_mm_offs."""
    return not (x.dtype != torch.bfloat16 or x.shape[1] % 8 != 0 or not x.is_contiguous() or x.data_ptr() % 16 != 0 or grouped_mm_offs.dtype != torch.int32 or grouped_mm_offs.shape[0] == 0 or not grouped_mm_offs.is_contiguous())


def fp8cast_rowwise_kmajor_sm90(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor on SM90; both scales MN-major. Inputs supports_rowwise_kmajor rejects take the Triton kernel, with its scales converted to the same layout."""
    assert x.ndim == 2 and x.shape[0] % 128 == 0, (f"Expected 2D tensor with M%128==0, got shape {x.shape}")
    M, N = x.shape
    if M == 0:
        return _empty_outputs(x, M, N)
    G = grouped_mm_offs.shape[0]
    if not supports_rowwise_kmajor(x, grouped_mm_offs):
        tok, s_tok, km, s_km = fp8cast_rowwise_kmajor_sm100(x, grouped_mm_offs)
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
