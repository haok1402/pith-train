"""
CuTe-DSL (SM90) kernels of the fp8_weight_quant feature, bit-identical to the Triton ones in
fp8_weight_quant_sm100: the blockwise transposes (not wired into the picker; tests and benchmarks
call them directly), the FP8 parameter all-gather kernels, and one-launch variants of the shard amax
and the scales over a fixed list of tensors.

Numerics follow compute_fp8_scale (SM90 branch) exactly. The tile amax is an integer max over the
magnitude bits (|x| of finite non-negative floats orders like its bit pattern), so a NaN orders
above inf where Triton's float max ignores it: NaN inputs are out of contract.
"""

from typing import Optional, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake_stream

__all__ = ["fp8cast_blockwise_transpose_sm90", "fp8cast_blockwise_transpose_batched_sm90", "fp8_shard_amax_sm90", "fp8_shard_cast_sm90", "fp8_scales_sm90", "fp8_pack_transpose_sm90", "fp8_shard_amax_multi_sm90", "fp8_scales_multi_sm90"]

BLOCK = 128
# Below this many tiles the extra barrier of the shared-memory store path costs more than its
# wider stores save.
SMEM_MIN_TILES = 256
NUM_THREADS = 256
FP8_MAX_RCP = 1.0 / 448.0

def asm(res_type, ptx: str, constraints: str, args: list, side_effects: bool = False):
    return llvm.inline_asm(res_type, [a.ir_value() for a in args], ptx, constraints, has_side_effects=side_effects, is_align_stack=False)


def struct_fields(res, n: int) -> list:
    return [Uint32(llvm.extractvalue(T.i32(), res, [i])) for i in range(n)]


@dsl_user_op
def ld_v4(addr: Int64, pred: Int32, *, loc=None, ip=None) -> list:
    """Predicated 16-byte global load (zero when pred is 0)."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r", [Int64(addr), Int32(pred)], side_effects=True,  # keep loads ahead of the stores; pure asm gets sunk to its use
    )
    return struct_fields(res, 4)


@dsl_user_op
def ld_v2(addr: Int64, pred: Int32, *, loc=None, ip=None) -> list:
    """Predicated 8-byte global load (zero when pred is 0)."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
        "@p ld.global.v2.b32 {$0, $1}, [$2];\n}",
        "=r,=r,l,r", [Int64(addr), Int32(pred)], side_effects=True,  # keep loads ahead of the stores; pure asm gets sunk to its use
    )
    return struct_fields(res, 2)


@dsl_user_op
def ld_scalar(addr: Int64, pred: Int32, nbytes: int, *, loc=None, ip=None) -> Uint32:
    """Predicated zero-extended 1/2/4-byte global load."""
    ty = {1: "u8", 2: "u16", 4: "b32"}[nbytes]
    res = asm(
        T.i32(),
        "{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\nmov.b32 $0, 0;\n"
        f"@p ld.global.{ty} $0, [$1];\n}}",
        "=r,l,r", [Int64(addr), Int32(pred)], side_effects=True,  # keep loads ahead of the stores; pure asm gets sunk to its use
    )
    return Uint32(res)


@dsl_user_op
def st_v2(addr: Int64, a: Uint32, b: Uint32, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, "{\n.reg .pred p;\nsetp.ne.b32 p, $3, 0;\n@p st.global.v2.b32 [$0], {$1, $2};\n}", "l,r,r,r", [Int64(addr), Uint32(a), Uint32(b), Int32(pred)], side_effects=True)


@dsl_user_op
def st_scalar(addr: Int64, v: Uint32, pred: Int32, nbytes: int, *, loc=None, ip=None) -> None:
    ty = {1: "b8", 4: "b32"}[nbytes]
    asm(None, f"{{\n.reg .pred p;\nsetp.ne.b32 p, $2, 0;\n@p st.global.{ty} [$0], $1;\n}}", "l,r,r", [Int64(addr), Uint32(v), Int32(pred)], side_effects=True)


@dsl_user_op
def sts_v2(addr: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None) -> None:
    asm(None, "st.shared.v2.b32 [$0], {$1, $2};", "r,r,r", [Int32(addr), Uint32(a), Uint32(b)], side_effects=True)


@dsl_user_op
def lds_v4(addr: Int32, *, loc=None, ip=None) -> list:
    res = asm(llvm.StructType.get_literal([T.i32()] * 4), "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", [Int32(addr)], side_effects=True)
    return struct_fields(res, 4)


@dsl_user_op
def st_v4(addr: Int64, w: list, pred: Int32, *, loc=None, ip=None) -> None:
    asm(None, "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}", "l,r,r,r,r,r", [Int64(addr), Uint32(w[0]), Uint32(w[1]), Uint32(w[2]), Uint32(w[3]), Int32(pred)], side_effects=True)


@dsl_user_op
def prmt(a: Uint32, b: Uint32, sel: int, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), f"prmt.b32 $0, $1, $2, {sel:#x};", "=r,r,r", [a, b]))


@dsl_user_op
def max_u16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), "max.u16x2 $0, $1, $2;", "=r,r,r", [a, b]))


@dsl_user_op
def max_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), "max.u32 $0, $1, $2;", "=r,r,r", [a, b]))


@dsl_user_op
def redux_max(a: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(asm(T.i32(), "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r", [a], side_effects=True))


@dsl_user_op
def f32x2_to_bf16x2(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """Two fp32 bit patterns -> packed bf16x2 (round to nearest even), lo in bits [15:0]."""
    return Uint32(asm(T.i32(), "{\n.reg .f32 a, b;\nmov.b32 a, $1;\nmov.b32 b, $2;\ncvt.rn.bf16x2.f32 $0, b, a;\n}", "=r,r,r", [lo, hi]))


@dsl_user_op
def cast4(v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, rcp: Uint32, *, loc=None, ip=None) -> Uint32:
    """Four fp32 bit patterns times rcp -> four e4m3 bytes, v0 in bits [7:0]."""
    return Uint32(
        asm(
            T.i32(),
            "{\n.reg .f32 a, b, c, d, r;\n.reg .b16 lo, hi;\n"
            "mov.b32 r, $5;\n"
            "mov.b32 a, $1;\nmov.b32 b, $2;\nmov.b32 c, $3;\nmov.b32 d, $4;\n"
            "mul.rn.f32 a, a, r;\nmul.rn.f32 b, b, r;\nmul.rn.f32 c, c, r;\nmul.rn.f32 d, d, r;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, b, a;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, d, c;\n"
            "mov.b32 $0, {lo, hi};\n}",
            "=r,r,r,r,r,r", [v0, v1, v2, v3, rcp],
        )
    )


@dsl_user_op
def scale_bits(amax_bits: Uint32, *, loc=None, ip=None) -> Uint32:
    """E8M0 scale bits from amax bits, exactly as compute_fp8_scale (SM90 branch)."""
    return Uint32(
        asm(
            T.i32(),
            "{\n.reg .f32 a;\n.reg .b32 bits, m, e;\n"
            "mov.b32 a, $1;\n"
            "max.f32 a, a, 0f38D1B717;\n"  # fp32(1e-4)
            f"mul.rn.f32 a, a, 0f{ptx_f32(FP8_MAX_RCP)};\n"
            "mov.b32 bits, a;\n"
            "and.b32 m, bits, 0x007FFFFF;\n"
            "and.b32 e, bits, 0x7F800000;\n"
            "add.u32 m, m, 0x007FFFFF;\n"
            "shr.u32 m, m, 23;\n"
            "shl.b32 m, m, 23;\n"
            "add.u32 e, e, m;\n"
            "max.u32 $0, e, 0x00800000;\n}",  # max(scale, 2^-126) on non-negative bits
            "=r,r", [amax_bits],
        )
    )


def ptx_f32(x: float) -> str:
    import struct
    return struct.pack(">f", x).hex().upper()


@cute.jit
def thread_coords() -> tuple:
    """256 threads per 128x128 tile, each owning an 8x8 block at (r0, c0)."""
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    r0, c0 = (warp // 2) * 32 + (lane // 8) * 8, (warp % 2) * 64 + (lane % 8) * 8
    return tidx, warp, lane, r0, c0


@cute.jit
def load_rows(base: Int64, row0: Int32, col0: Int32, row_lo: Int32, row_hi: Int32, K: Int32, in_bytes: cutlass.Constexpr[int], round_bf16: cutlass.Constexpr[bool], aligned: cutlass.Constexpr[bool]) -> list:
    """
    Load the thread's 8x8 block, rows row0 + i valid when row_lo <= row0 + i < row_hi.

    base addresses element (0, 0) of a row-major (rows, K) matrix. Returns 8 rows of words: four
    bf16x2 words per row for 2-byte (or bf16-rounded) data, eight fp32 words per row otherwise.
    """
    rows = []
    for i in cutlass.range_constexpr(8):
        row = row0 + i
        row_ok = Int32(row >= row_lo) & Int32(row < row_hi)
        addr = base + (Int64(row) * Int64(K) + Int64(col0)) * in_bytes
        words = []
        if const_expr(aligned):
            pred = row_ok & Int32(col0 < K)
            if const_expr(in_bytes == 2):
                words = ld_v4(addr, pred)
            elif const_expr(in_bytes == 4):
                words = ld_v4(addr, pred) + ld_v4(addr + 16, pred)
            else:
                words = ld_v2(addr, pred)
        else:
            elems = []
            for j in cutlass.range_constexpr(8):
                pred = row_ok & Int32(col0 + j < K)
                elems.append(ld_scalar(addr + j * in_bytes, pred, in_bytes))
            if const_expr(in_bytes == 2):
                words = [elems[2 * q] | (elems[2 * q + 1] << 16) for q in range(4)]
            elif const_expr(in_bytes == 4):
                words = elems
            else:
                words = [elems[4 * q] | (elems[4 * q + 1] << 8) | (elems[4 * q + 2] << 16) | (elems[4 * q + 3] << 24) for q in range(2)]
        if const_expr(in_bytes == 4 and round_bf16):
            words = [f32x2_to_bf16x2(words[2 * q], words[2 * q + 1]) for q in range(4)]
        rows.append(words)
    return rows


@cute.jit
def thread_amax_bits(rows: list, packed_bf16: cutlass.Constexpr[bool]) -> Uint32:
    """Max magnitude bit pattern (as fp32 bits) over the thread's block."""
    m = Uint32(0)
    if const_expr(packed_bf16):
        for i in cutlass.range_constexpr(8):
            for q in cutlass.range_constexpr(4):
                m = max_u16x2(m, rows[i][q] & Uint32(0x7FFF7FFF))
        m = max_u32(m & Uint32(0xFFFF), m >> 16) << 16
    else:
        for i in cutlass.range_constexpr(8):
            for q in cutlass.range_constexpr(8):
                m = max_u32(m, rows[i][q] & Uint32(0x7FFFFFFF))
    return m


@cute.jit
def block_amax_bits(m: Uint32, warp: Int32, lane: Int32, smem: cute.Tensor) -> Uint32:
    """CTA-wide max of m (every thread gets the result)."""
    m = redux_max(m)
    if lane == 0:
        smem[warp] = m
    cute.arch.sync_threads()
    r = smem[0]
    for w in cutlass.range_constexpr(1, NUM_THREADS // 32):
        r = max_u32(r, smem[w])
    return r


@cute.jit
def cast_rows(rows: list, rcp: Uint32, packed_bf16: cutlass.Constexpr[bool]) -> list:
    """Cast 8 rows to e4m3: two words (cols 0-3, cols 4-7) per row."""
    out = []
    for i in cutlass.range_constexpr(8):
        f = []
        if const_expr(packed_bf16):
            for q in cutlass.range_constexpr(4):
                w = rows[i][q]
                f.append(w << 16)
                f.append(w & Uint32(0xFFFF0000))
        else:
            f = rows[i]
        out.append([cast4(f[0], f[1], f[2], f[3], rcp), cast4(f[4], f[5], f[6], f[7], rcp)])
    return out


@cute.jit
def transpose4(a0: Uint32, a1: Uint32, a2: Uint32, a3: Uint32) -> list:
    """4x4 byte transpose: word r holds row r; result word c holds column c."""
    t0, t1, t2, t3 = prmt(a0, a1, 0x5140), prmt(a0, a1, 0x7362), prmt(a2, a3, 0x5140), prmt(a2, a3, 0x7362)
    return [prmt(t0, t2, 0x5410), prmt(t0, t2, 0x7632), prmt(t1, t3, 0x5410), prmt(t1, t3, 0x7632)]


@cute.jit
def transpose8(q: list) -> list:
    """8x8 byte transpose of [row][lo, hi] words into [col][rows 0-3, rows 4-7] words."""
    a = transpose4(q[0][0], q[1][0], q[2][0], q[3][0])  # rows 0-3, cols 0-3
    b = transpose4(q[4][0], q[5][0], q[6][0], q[7][0])  # rows 4-7, cols 0-3
    c = transpose4(q[0][1], q[1][1], q[2][1], q[3][1])  # rows 0-3, cols 4-7
    d = transpose4(q[4][1], q[5][1], q[6][1], q[7][1])  # rows 4-7, cols 4-7
    return [[a[j], b[j]] for j in range(4)] + [[c[j], d[j]] for j in range(4)]


@cute.jit
def store_rows(base: Int64, q: list, row0: Int32, col0: Int32, row_lo: Int32, row_hi: Int32, K: Int32, aligned: cutlass.Constexpr[bool]) -> None:
    """Store [row][lo, hi] fp8 words to a row-major (rows, K) byte matrix at base."""
    for i in cutlass.range_constexpr(8):
        row = row0 + i
        row_ok = Int32(row >= row_lo) & Int32(row < row_hi)
        addr = base + Int64(row) * Int64(K) + Int64(col0)
        if const_expr(aligned):
            st_v2(addr, q[i][0], q[i][1], row_ok & Int32(col0 < K))
        else:
            for j in cutlass.range_constexpr(8):
                byte = (q[i][j // 4] >> (8 * (j % 4))) & Uint32(0xFF)
                st_scalar(addr + j, byte, row_ok & Int32(col0 + j < K), 1)


# A 128x128 byte tile staged in shared memory as 128 rows of eight 16-byte chunks. The chunk index
# is XOR-swizzled by s(row) = ((row >> 3) & 1) << 2 | ((row >> 3) & 7) >> 1, which makes both the
# row-block writes (8 bytes per lane, rows 8 apart) and the transposed-block writes (rows 8 apart
# the other way) hit the minimum two wavefronts; readback reads whole rows, one wavefront each.
TILE_BYTES = BLOCK * BLOCK


@cute.jit
def swz(row: Int32, chunk: Int32) -> Int32:
    x = (row >> 3) & 7
    return row * BLOCK + ((chunk ^ (((x & 1) << 2) | (x >> 1))) << 4)


@cute.jit
def smem_write_block(base: Int32, q: list, rr0: Int32, cc0: Int32) -> None:
    """Write [row][lo, hi] words of the 8x8 block at tile (rr0, cc0) into the staged tile."""
    for i in cutlass.range_constexpr(8):
        sts_v2(base + swz(rr0 + i, cc0 >> 4) + (cc0 & 8), q[i][0], q[i][1])


@cute.jit
def smem_store_tile(base: Int32, gtile: Int64, row_lo: Int32, row_hi: Int32, cols: Int32, ld: Int32, nthreads: cutlass.Constexpr[int] = NUM_THREADS) -> None:
    """
    Store the staged tile with 16-byte stores to the row-major byte matrix whose tile row 0, col 0
    is at gtile (row stride ld); tile row r is valid when row_lo <= r < row_hi, chunk c when 16 *
    c < cols.
    """
    tidx, _, _ = cute.arch.thread_idx()
    for it in cutlass.range_constexpr(TILE_BYTES // 16 // nthreads):
        idx = it * nthreads + tidx
        row = idx >> 3
        c = idx & 7
        w = lds_v4(base + swz(row, c))
        pred = Int32(row >= row_lo) & Int32(row < row_hi) & Int32(c * 16 < cols)
        st_v4(gtile + Int64(row) * Int64(ld) + Int64(c * 16), w, pred)


class BlockwiseTranspose:
    """
    One CTA per 128x128 tile. mode picks the store path: "smem" stages both FP8 tiles in shared
    memory for 16-byte stores (dims % 16 == 0), "v8" stores each thread's 8-byte rows (dims % 8 ==
    0), "scalar" stores bytes (any shape).
    """

    def __init__(self, in_bytes: int, mode: str, batched: bool) -> None:
        self.in_bytes = in_bytes
        self.mode = mode
        self.batched = batched

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, stream: cuda.CUstream):
        if const_expr(self.batched):
            G, M, K = mX.shape[0], mX.shape[1], mX.shape[2]
        else:
            G, M, K = 1, mX.shape[0], mX.shape[1]
        rt, ct = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mX.iterator.toint(), mOut.iterator.toint(), mScale.iterator.toint(), mOutT.iterator.toint(), mScaleT.iterator.toint(), Int32(M), Int32(K), Int32(rt), Int32(ct)).launch(grid=[rt, ct, G], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x: Int64, out: Int64, scale: Int64, out_t: Int64, scale_t: Int64, M: Int32, K: Int32, rt: Int32, ct: Int32):
        bx, by, bz = cute.arch.block_idx()
        tidx, warp, lane, r0, c0 = thread_coords()
        smem = cutlass.memory.SmemAllocator()
        red = smem.allocate_tensor(Uint32, cute.make_layout(NUM_THREADS // 32), byte_alignment=16)
        aligned = const_expr(self.mode != "scalar")

        tile_r, tile_c = bx * BLOCK, by * BLOCK
        row0 = tile_r + r0
        col0 = tile_c + c0
        mk = Int64(M) * Int64(K)
        x_g = x + Int64(bz) * mk * self.in_bytes
        rows = load_rows(x_g, row0, col0, Int32(0), M, K, self.in_bytes, False, aligned)

        packed = const_expr(self.in_bytes == 2)
        amax = block_amax_bits(thread_amax_bits(rows, packed), warp, lane, red)
        sbits = scale_bits(amax)
        q = cast_rows(rows, Uint32(0x7F000000) - sbits, packed)
        # out_t is (K, M): the thread's column j is out_t row col0 + j, its rows are columns.
        qt = transpose8(q)
        out_g = out + Int64(bz) * mk
        out_t_g = out_t + Int64(bz) * mk
        if const_expr(self.mode == "smem"):
            buf = smem.allocate_tensor(Uint32, cute.make_layout(TILE_BYTES // 2), byte_alignment=128)
            base = buf.iterator.toint()
            smem_write_block(base, q, r0, c0)
            smem_write_block(base + TILE_BYTES, qt, c0, r0)
            cute.arch.sync_threads()
            gt = out_g + Int64(tile_r) * Int64(K) + Int64(tile_c)
            smem_store_tile(base, gt, Int32(0), M - tile_r, K - tile_c, K)
            gtt = out_t_g + Int64(tile_c) * Int64(M) + Int64(tile_r)
            smem_store_tile(base + TILE_BYTES, gtt, Int32(0), K - tile_c, M - tile_r, M)
        else:
            store_rows(out_g, q, row0, col0, Int32(0), M, K, aligned)
            store_rows(out_t_g, qt, col0, row0, Int32(0), K, M, aligned)

        if tidx == 0:
            nt = Int64(rt) * Int64(ct)
            s_off, st_off = Int64(bz) * nt + Int64(bx) * Int64(ct) + Int64(by), Int64(bz) * nt + Int64(by) * Int64(rt) + Int64(bx)
            st_scalar(scale + s_off * 4, sbits, Int32(1), 4)
            st_scalar(scale_t + st_off * 4, sbits, Int32(1), 4)


COMPILED: dict = {}


def fake(dtype, ndim: int, div: int) -> cute.Tensor:
    shape = tuple(cute.sym_int() for _ in range(ndim))
    stride = tuple(cute.sym_int64() for _ in range(ndim - 1)) + (1,)
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=div)


def compile_blockwise(dtype: torch.dtype, mode: str, batched: bool):
    key = ("bt", dtype, mode, batched)
    fn = COMPILED.get(key)
    if fn is None:
        nd = 3 if batched else 2
        in_bytes = torch.tensor([], dtype=dtype).element_size()
        fn = cute.compile(BlockwiseTranspose(in_bytes, mode, batched), fake(TORCH2CUTE[dtype], nd, in_bytes), fake(cutlass.Float8E4M3FN, nd, 1), fake(cutlass.Float32, nd, 4), fake(cutlass.Float8E4M3FN, nd, 1), fake(cutlass.Float32, nd, 4), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    return fn


def store_mode(tiles: int, M: int, K: int, *tensors: torch.Tensor) -> str:
    """Store path for the blockwise kernels: the widest the shape and pointers allow."""
    if any(t.data_ptr() % 16 for t in tensors):
        mode = "scalar"
    elif M % 16 == 0 and K % 16 == 0 and tiles >= SMEM_MIN_TILES:
        mode = "smem"
    else:
        mode = "v8" if M % 8 == 0 and K % 8 == 0 else "scalar"
    return mode


def fp8cast_blockwise_transpose_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe-DSL fp8cast_blockwise_transpose: x 2-D (M, K) bf16 / fp32."""
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    if x.stride(1) != 1:
        x = x.contiguous()
    M, K = x.shape
    sr = (M + BLOCK - 1) // BLOCK
    sc = (K + BLOCK - 1) // BLOCK
    dev = x.device
    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    scale = torch.empty((sr, sc), dtype=torch.float32, device=dev)
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = torch.empty((sc, sr), dtype=torch.float32, device=dev)
    if x.stride(0) != K:
        x = x.contiguous()
    fn = compile_blockwise(x.dtype, store_mode(sr * sc, M, K, x, out, out_t), False)
    fn(x, out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


def fp8cast_blockwise_transpose_batched_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe-DSL fp8cast_blockwise_transpose_batched: x 3-D (G, N, K) bf16 / fp32."""
    assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D"
    if not x.is_contiguous():
        x = x.contiguous()
    G, N, K = x.shape
    sr = (N + BLOCK - 1) // BLOCK
    sc = (K + BLOCK - 1) // BLOCK
    dev = x.device
    out = torch.empty((G, N, K), dtype=torch.float8_e4m3fn, device=dev)
    scale = torch.empty((G, sr, sc), dtype=torch.float32, device=dev)
    out_t = torch.empty((G, K, N), dtype=torch.float8_e4m3fn, device=dev)
    scale_t = torch.empty((G, sc, sr), dtype=torch.float32, device=dev)
    if G == 0 or N == 0 or K == 0:
        return out, scale, out_t, scale_t
    fn = compile_blockwise(x.dtype, store_mode(G * sr * sc, N, K, x, out, out_t), True)
    fn(x, out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t


@dsl_user_op
def ld_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(asm(T.i64(), "ld.global.nc.u64 $0, [$1];", "=l,l", [Int64(addr)]))


@cute.jit
def shard_tile(shard: Int64, amax_tile: Int64, out: Int64, local_rows: Int32, K: Int32, tr: Int32, by: Int32, bz: Int32, row_offset: Int32, cast: cutlass.Constexpr[bool], mode: cutlass.Constexpr[str]):
    """
    One 128x128 tile (global tile row tr, column tile by, local group bz) of an fp32 row shard.
    """
    if const_expr(mode == "row"):
        shard_tile_row(shard, amax_tile, out, local_rows, K, tr, by, bz, row_offset, cast)
        return
    aligned = const_expr(mode != "scalar")
    tidx, warp, lane, r0, c0 = thread_coords()
    row0 = tr * BLOCK + r0 - row_offset
    col0 = by * BLOCK + c0
    gk = Int64(bz) * Int64(local_rows) * Int64(K)
    rows = load_rows(shard + gk * 4, row0, col0, Int32(0), local_rows, K, 4, True, aligned)
    if const_expr(cast):
        sbits = scale_bits(ld_scalar(amax_tile, Int32(1), 4))
        q = cast_rows(rows, Uint32(0x7F000000) - sbits, True)
        store_rows(out + gk, q, row0, col0, Int32(0), local_rows, K, aligned)
    else:
        smem = cutlass.memory.SmemAllocator()
        red = smem.allocate_tensor(Uint32, cute.make_layout(NUM_THREADS // 32), byte_alignment=16)
        amax = block_amax_bits(thread_amax_bits(rows, True), warp, lane, red)
        if tidx == 0:
            st_scalar(amax_tile, amax, Int32(1), 4)


@cute.jit
def shard_tile_row(shard: Int64, amax_tile: Int64, out: Int64, local_rows: Int32, K: Int32, tr: Int32, by: Int32, bz: Int32, row_offset: Int32, cast: cutlass.Constexpr[bool]):
    """
    Row-major variant of shard_tile: a warp loads one 128-column tile row (16 bytes per lane), warps
    take rows round-robin, so a shard holding a few rows of a tile still uses all 8 warps. Needs K %
    4 == 0 and a 16-byte aligned shard.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    nw = NUM_THREADS // 32
    col = by * BLOCK + lane * 4
    col_ok = Int32(col < K)
    gk = Int64(bz) * Int64(local_rows) * Int64(K)
    tile_r = tr * BLOCK - row_offset
    words = []
    preds = []
    for rep in cutlass.range_constexpr(BLOCK // nw):
        r = tile_r + rep * nw + warp
        ok = Int32(r >= 0) & Int32(r < local_rows) & col_ok
        w = ld_v4(shard + (gk + Int64(r) * Int64(K) + Int64(col)) * 4, ok)
        words.append([f32x2_to_bf16x2(w[0], w[1]), f32x2_to_bf16x2(w[2], w[3])])
        preds.append(ok)
    if const_expr(cast):
        rcp = Uint32(0x7F000000) - scale_bits(ld_scalar(amax_tile, Int32(1), 4))
        for rep in cutlass.range_constexpr(BLOCK // nw):
            r = tile_r + rep * nw + warp
            a, b = words[rep]
            v = cast4(a << 16, a & Uint32(0xFFFF0000), b << 16, b & Uint32(0xFFFF0000), rcp)
            st_scalar(out + gk + Int64(r) * Int64(K) + Int64(col), v, preds[rep], 4)
    else:
        m = Uint32(0)
        for rep in cutlass.range_constexpr(BLOCK // nw):
            m = max_u16x2(m, words[rep][0] & Uint32(0x7FFF7FFF))
            m = max_u16x2(m, words[rep][1] & Uint32(0x7FFF7FFF))
        m = max_u32(m & Uint32(0xFFFF), m >> 16) << 16
        smem = cutlass.memory.SmemAllocator()
        red = smem.allocate_tensor(Uint32, cute.make_layout(nw), byte_alignment=16)
        amax = block_amax_bits(m, warp, lane, red)
        if tidx == 0:
            st_scalar(amax_tile, amax, Int32(1), 4)


class ShardTile:
    def __init__(self, cast: bool, mode: str) -> None:
        self.cast = cast
        self.mode = mode

    @cute.jit
    def __call__(self, mShard: cute.Tensor, mAmax: cute.Tensor, mOut: cute.Tensor, row_offset: Int32, group_offset: Int32, stream: cuda.CUstream):
        g_local, local_rows, K = mShard.shape[0], mShard.shape[1], mShard.shape[2]
        trs, tre, ct = row_offset // BLOCK, (row_offset + local_rows + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        self.kernel(mShard.iterator.toint(), mAmax.iterator.toint(), mOut.iterator.toint(), Int32(local_rows), Int32(K), Int32(trs), Int32(mAmax.shape[1]), Int32(mAmax.shape[2]), row_offset, group_offset).launch(grid=[tre - trs, ct, g_local], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, shard: Int64, amax: Int64, out: Int64, local_rows: Int32, K: Int32, trs: Int32, R: Int32, C: Int32, row_offset: Int32, group_offset: Int32):
        bx, by, bz = cute.arch.block_idx()
        tr = trs + bx
        a_off = (Int64(group_offset + bz) * Int64(R) + Int64(tr)) * Int64(C) + Int64(by)
        shard_tile(shard, amax + a_off * 4, out, local_rows, K, tr, by, bz, row_offset, self.cast, self.mode)


def shard_tiles(shard: torch.Tensor, amax: torch.Tensor, out: Optional[torch.Tensor], row_offset: int, group_offset: int) -> None:
    if shard.numel() == 0:
        return
    K = shard.shape[2]
    cast = out is not None
    if K % 4 == 0 and shard.data_ptr() % 16 == 0 and (not cast or out.data_ptr() % 4 == 0):
        mode = "row"
    else:
        mode = "scalar"
    key = ("shard", cast, mode, out.dtype if cast else None)
    fn = COMPILED.get(key)
    if fn is None:
        fn = cute.compile(ShardTile(cast, mode), fake(cutlass.Float32, 3, 4), fake(cutlass.Float32, 3, 4), fake(TORCH2CUTE[out.dtype] if cast else cutlass.Float32, 3, 1), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(shard, amax, out if cast else shard, row_offset, group_offset)


def fp8_shard_amax_sm90(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """CuTe-DSL fp8_shard_amax."""
    shard_tiles(shard, amax, None, row_offset, group_offset)


def fp8_shard_cast_sm90(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:
    """CuTe-DSL fp8_shard_cast."""
    shard_tiles(shard, amax, out, row_offset, group_offset)


class Scales:
    @cute.jit
    def __call__(self, mAmax: cute.Tensor, mScale: cute.Tensor, mScaleT: cute.Tensor, stream: cuda.CUstream):
        G, R, C = mAmax.shape[0], mAmax.shape[1], mAmax.shape[2]
        self.kernel(mAmax.iterator.toint(), mScale.iterator.toint(), mScaleT.iterator.toint(), Int32(R), Int32(C)).launch(grid=[(R * C + NUM_THREADS - 1) // NUM_THREADS, G, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, amax: Int64, scale: Int64, scale_t: Int64, R: Int32, C: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, g, _ = cute.arch.block_idx()
        scale_elem(amax, scale, scale_t, Int64(g), bx * NUM_THREADS + tidx, R, C)


@cute.jit
def scale_elem(amax: Int64, scale: Int64, scale_t: Int64, g: Int64, idx: Int32, R: Int32, C: Int32):
    rc = R * C
    ok = Int32(idx < rc)
    base = g * Int64(rc)
    s = scale_bits(ld_scalar(amax + (base + Int64(idx)) * 4, ok, 4))
    r = idx // C
    c = idx - r * C
    st_scalar(scale + (base + Int64(idx)) * 4, s, ok, 4)
    st_scalar(scale_t + (base + Int64(c) * Int64(R) + Int64(r)) * 4, s, ok, 4)


def fp8_scales_sm90(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """CuTe-DSL fp8_scales."""
    fn = COMPILED.get("scales")
    if fn is None:
        f = [fake(cutlass.Float32, 3, 4) for _ in range(3)]
        fn = cute.compile(Scales(), *f, fake_stream(), options="--enable-tvm-ffi")
        COMPILED["scales"] = fn
    fn(amax, scale, scale_t)


PACK_THREADS = 128


class PackTranspose:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.nthreads = PACK_THREADS if mode == "w16" else NUM_THREADS

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, stream: cuda.CUstream):
        G, N, K = mX.shape[0], mX.shape[1], mX.shape[2]
        self.kernel(mX.iterator.toint(), mOut.iterator.toint(), Int32(N), Int32(K), Int32(G)).launch(grid=[(N + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK, G], block=[self.nthreads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x: Int64, out: Int64, N: Int32, K: Int32, G: Int32):
        if const_expr(self.mode == "w16"):
            self.kernel_w16(x, out, N, K, G)
        else:
            self.kernel_8x8(x, out, N, K, G)

    @cute.jit
    def kernel_w16(self, x: Int64, out: Int64, N: Int32, K: Int32, G: Int32):
        """
        128 threads, each owning 8 rows x 16 bytes; the transposed tile is staged in shared memory
        for 16-byte stores.
        """
        bx, by, bz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        lane = tidx % 32
        r0 = warp * 32 + (lane // 8) * 8
        c0 = (lane % 8) * 16
        tile_r = bx * BLOCK
        tile_c = by * BLOCK
        nk = Int64(N) * Int64(K)
        off = Int64(bz) * nk
        col_ok = Int32(tile_c + c0 < K)
        w = []
        for i in cutlass.range_constexpr(8):
            row = tile_r + r0 + i
            addr = x + off + Int64(row) * Int64(K) + Int64(tile_c + c0)
            w.append(ld_v4(addr, Int32(row < N) & col_ok))
        for i in cutlass.range_constexpr(8):
            row = tile_r + r0 + i
            addr = out + off + Int64(row) * Int64(K) + Int64(tile_c + c0)
            st_v4(addr, w[i], Int32(row < N) & col_ok)
        left = transpose8([[w[i][0], w[i][1]] for i in range(8)])
        right = transpose8([[w[i][2], w[i][3]] for i in range(8)])
        smem = cutlass.memory.SmemAllocator()
        buf = smem.allocate_tensor(Uint32, cute.make_layout(TILE_BYTES // 4), byte_alignment=128)
        base = buf.iterator.toint()
        smem_write_block(base, left, c0, r0)
        smem_write_block(base, right, c0 + 8, r0)
        cute.arch.sync_threads()
        gtt = out + Int64(G) * nk + off + Int64(tile_c) * Int64(N) + Int64(tile_r)
        smem_store_tile(base, gtt, Int32(0), K - tile_c, N - tile_r, N, PACK_THREADS)

    @cute.jit
    def kernel_8x8(self, x: Int64, out: Int64, N: Int32, K: Int32, G: Int32):
        bx, by, bz = cute.arch.block_idx()
        _, _, _, r0, c0 = thread_coords()
        row0 = bx * BLOCK + r0
        col0 = by * BLOCK + c0
        nk = Int64(N) * Int64(K)
        off = Int64(bz) * nk
        aligned = const_expr(self.mode != "scalar")
        q = load_rows(x + off, row0, col0, Int32(0), N, K, 1, False, aligned)
        qt = transpose8(q)
        out_t = out + Int64(G) * nk + off
        store_rows(out + off, q, row0, col0, Int32(0), N, K, aligned)
        store_rows(out_t, qt, col0, row0, Int32(0), K, N, aligned)


def fp8_pack_transpose_sm90(x: torch.Tensor, out: torch.Tensor) -> None:
    """CuTe-DSL fp8_pack_transpose."""
    G, N, K = x.shape
    if x.numel() == 0:
        return
    if N % 16 == 0 and K % 16 == 0 and x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0:
        mode = "w16"
    elif N % 8 == 0 and K % 8 == 0 and x.data_ptr() % 8 == 0 and out.data_ptr() % 8 == 0:
        mode = "v8"
    else:
        mode = "scalar"
    key = ("pack", mode, x.dtype, out.dtype)
    fn = COMPILED.get(key)
    if fn is None:
        fn = cute.compile(PackTranspose(mode), fake(TORCH2CUTE[x.dtype], 3, 1), fake(TORCH2CUTE[out.dtype], 1, 1), fake_stream(), options="--enable-tvm-ffi")
        COMPILED[key] = fn
    fn(x, out)


DESC = 8  # int64 words per descriptor row


class ShardAmaxMulti:
    """
    Shard amax over n shards. The table holds two rows per shard: (shard, amax, local_rows, K,
    row_offset, first tile row, element offset of group_offset in the grid, first CTA) and (R, C,
    column tiles, tile rows, 0, 0, 0, 0).
    """

    @cute.jit
    def __call__(self, mTable: cute.Tensor, n: Int32, total: Int32, stream: cuda.CUstream):
        self.kernel(mTable.iterator.toint(), n).launch(grid=[total, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table: Int64, n: Int32):
        bx, _, _ = cute.arch.block_idx()
        # Find the tensor this CTA belongs to: the last row whose first CTA index <= bx.
        t = Int32(0)
        for i in cutlass.range(1, n, unroll=1):
            first = Int32(ld_u64(table + (Int64(i) * 2 * DESC + 7) * 8))
            if first <= bx:
                t = i
        row = table + Int64(t) * 2 * DESC * 8
        shard = ld_u64(row)
        amax = ld_u64(row + 8)
        local_rows = Int32(ld_u64(row + 16))
        K = Int32(ld_u64(row + 24))
        row_offset = Int32(ld_u64(row + 32))
        trs = Int32(ld_u64(row + 40))
        a_base = ld_u64(row + 48)
        first = Int32(ld_u64(row + 56))
        R, C, ct, ntr = Int32(ld_u64(row + 64)), Int32(ld_u64(row + 72)), Int32(ld_u64(row + 80)), Int32(ld_u64(row + 88))
        local = bx - first
        by = local % ct
        rest = local // ct
        bxr = rest % ntr
        bz = rest // ntr
        tr = trs + bxr
        a_off = a_base + (Int64(bz) * Int64(R) + Int64(tr)) * Int64(C) + Int64(by)
        shard_tile(shard, amax + a_off * 4, amax, local_rows, K, tr, by, bz, row_offset, False, "row")


class FP8ShardAmaxGroup:
    """
    One launch of fp8_shard_amax over a fixed list of shards. The table holds raw addresses, so it
    is valid only while the tensors are not reallocated. Shards with K % 4 != 0 or not 16-byte
    aligned get one launch each.
    """

    def __init__(self, shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> None:
        rows = []
        first = 0
        self.fallback = []
        for shard, amax, ro, go in zip(shards, amaxes, row_offsets, group_offsets):
            g_local, local_rows, K = shard.shape
            if shard.numel() == 0:
                continue
            if K % 4 != 0 or shard.data_ptr() % 16 != 0:
                self.fallback.append((shard, amax, ro, go))
                continue
            _, R, C = amax.shape
            trs = ro // BLOCK
            ntr = (ro + local_rows + BLOCK - 1) // BLOCK - trs
            ct = (K + BLOCK - 1) // BLOCK
            rows.append([shard.data_ptr(), amax.data_ptr(), local_rows, K, ro, trs, go * R * C, first])
            rows.append([R, C, ct, ntr, 0, 0, 0, 0])
            first += ntr * ct * g_local
        self.total = first
        self.n = len(rows) // 2
        dev = shards[0].device
        self.table = torch.tensor(rows or [[0] * DESC], dtype=torch.int64, device=dev)
        fn = COMPILED.get("amax_multi")
        if fn is None:
            fn = cute.compile(ShardAmaxMulti(), fake(cutlass.Int64, 2, 8), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
            COMPILED["amax_multi"] = fn
        self.fn = fn

    def __call__(self) -> None:
        if self.total:
            self.fn(self.table, self.n, self.total)
        for shard, amax, ro, go in self.fallback:
            fp8_shard_amax_sm90(shard, amax, ro, go)


class ScalesMulti:
    """
    Scales over n grids. The table holds one row per grid: (amax, scale, scale_t, R, C, CTAs per
    group, 0, first CTA).
    """

    @cute.jit
    def __call__(self, mTable: cute.Tensor, n: Int32, total: Int32, stream: cuda.CUstream):
        self.kernel(mTable.iterator.toint(), n).launch(grid=[total, 1, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, table: Int64, n: Int32):
        bx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        t = Int32(0)
        for i in cutlass.range(1, n, unroll=1):
            first = Int32(ld_u64(table + (Int64(i) * DESC + 7) * 8))
            if first <= bx:
                t = i
        row = table + Int64(t) * DESC * 8
        amax = ld_u64(row)
        scale = ld_u64(row + 8)
        scale_t = ld_u64(row + 16)
        R = Int32(ld_u64(row + 24))
        C, bpg, first = Int32(ld_u64(row + 32)), Int32(ld_u64(row + 40)), Int32(ld_u64(row + 56))
        local = bx - first
        g = local // bpg
        blk = local - g * bpg
        scale_elem(amax, scale, scale_t, Int64(g), blk * NUM_THREADS + tidx, R, C)


class FP8ScalesGroup:
    """
    One launch of fp8_scales over a fixed list of (amax, scale, scale_t) grids; same address
    caveat as FP8ShardAmaxGroup.
    """

    def __init__(self, amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> None:
        rows = []
        first = 0
        for amax, scale, scale_t in zip(amaxes, scales, scales_t):
            G, R, C = amax.shape
            bpg = (R * C + NUM_THREADS - 1) // NUM_THREADS
            rows.append([amax.data_ptr(), scale.data_ptr(), scale_t.data_ptr(), R, C, bpg, 0, first])
            first += G * bpg
        self.total = first
        self.n = len(rows)
        self.table = torch.tensor(rows, dtype=torch.int64, device=amaxes[0].device)
        fn = COMPILED.get("scales_multi")
        if fn is None:
            fn = cute.compile(ScalesMulti(), fake(cutlass.Int64, 2, 8), Int32(0), Int32(0), fake_stream(), options="--enable-tvm-ffi")
            COMPILED["scales_multi"] = fn
        self.fn = fn

    def __call__(self) -> None:
        if self.total:
            self.fn(self.table, self.n, self.total)


def fp8_shard_amax_multi_sm90(shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> FP8ShardAmaxGroup:
    """CuTe-DSL fp8_shard_amax_multi."""
    return FP8ShardAmaxGroup(shards, amaxes, row_offsets, group_offsets)


def fp8_scales_multi_sm90(amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> FP8ScalesGroup:
    """CuTe-DSL fp8_scales_multi."""
    return FP8ScalesGroup(amaxes, scales, scales_t)
