"""
Quantize SM90 activations by row and their transpose by 128x128 tile in one read.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm
from pithtrain.kernels.fp8_e8m0_scale_sm90 import e8m0_scale, exact_rcp
from pithtrain.kernels.fp8_scale_layout import mn_major_scale

__all__ = ["fp8_quant_act_sm90"]

BLOCK = 128

SMALL_GRID_TILES = 200


# The register block is kept as packed 32-bit words of the input dtype (bf16 / f16: two elements
# per word, low half first; f32: one). Every helper widens to fp32 only inside its own asm block,
# so LLVM cannot hoist the widening and keep an fp32 copy of the block live, which would double the
# register footprint, or make ptxas spill under a register cap.


@dsl_user_op
def pack_scaled_e4m3(words: list, element_locations: list, input_format: str, rcp: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Pack four scaled elements as E4M3 bytes in low-to-high order.
    """
    num_words = len(words)
    rcp_operand = f"${num_words + 1}"
    ptx_lines = [".reg .b32 f0, f1, f2, f3;", ".reg .b16 l, h, q0, q1;"]
    for element_idx, (word_idx, half_idx) in enumerate(element_locations):
        word_operand = f"${word_idx + 1}"
        if input_format == "bf16":
            if half_idx == 0:
                ptx_lines.append(f"shl.b32 f{element_idx}, {word_operand}, 16;")
            else:
                ptx_lines.append(f"and.b32 f{element_idx}, {word_operand}, 0xFFFF0000;")
        elif input_format == "f16":
            ptx_lines.append(f"mov.b32 {{l, h}}, {word_operand};")
            ptx_lines.append(f"cvt.f32.f16 f{element_idx}, {'l' if half_idx == 0 else 'h'};")
        else:
            ptx_lines.append(f"mov.b32 f{element_idx}, {word_operand};")
        ptx_lines.append(f"mul.rn.f32 f{element_idx}, f{element_idx}, {rcp_operand};")
    # Put the earlier element in the second source, which lands in the low byte.
    ptx_lines += ["cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0;", "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2;", "mov.b32 $0, {q0, q1};"]
    ptx = "{\n\t" + "\n\t".join(ptx_lines) + "\n\t}"
    constraints = "=r," + ",".join(["r"] * num_words) + ",f"
    operands = [Uint32(word) for word in words] + [Float32(rcp)]
    return Uint32(asm(T.i32(), [operand.ir_value(loc=loc, ip=ip) for operand in operands], ptx, constraints))


def make_global_ptr(base_ptr: cute.Pointer, elem_offset: Int64, elem_bytes: int, dtype: type, alignment: int) -> cute.Pointer:
    elem_addr = base_ptr.toint() + elem_offset * elem_bytes
    return cute.make_ptr(dtype, elem_addr, cute.AddressSpace.gmem, assumed_align=alignment)


class SubBlock:
    """
    Coordinates and bounds for one thread's register block.
    """

    def __init__(self, tidx, row_tile_idx, col_tile_idx, M, K, num_rows_per_thread: int) -> None:
        self.tidx, self.row_tile_idx, self.col_tile_idx = tidx, row_tile_idx, col_tile_idx
        self.M, self.K = M, K
        self.warp, self.lane = tidx // 32, tidx % 32
        self.row_group_idx, self.col_group_idx = self.lane // 8, self.lane % 8
        self.row_start = row_tile_idx * BLOCK + self.warp * (4 * num_rows_per_thread) + self.row_group_idx * num_rows_per_thread
        self.col_start = col_tile_idx * BLOCK + self.col_group_idx * 8


class RowwiseBlockwiseTranspose:
    """
    Fuse rowwise and transposed blockwise FP8 quantization.
    """

    def __init__(self, dtype: type, col_vector_width: int, row_vector_width: int, num_rows_per_thread: int, has_full_tiles: bool, use_coalesced_transpose: bool) -> None:
        self.dtype, self.num_rows_per_thread = dtype, num_rows_per_thread
        self.has_full_tiles, self.use_coalesced_transpose = has_full_tiles, use_coalesced_transpose
        self.num_threads, self.num_warps = 1024 // num_rows_per_thread, 32 // num_rows_per_thread
        self.input_format = {cutlass.BFloat16: "bf16", cutlass.Float16: "f16", Float32: "f32"}[dtype]
        self.elem_bytes, self.elems_per_word = dtype.width // 8, 32 // dtype.width
        self.col_vector_width, self.row_vector_width = col_vector_width, row_vector_width

    @cute.jit
    def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor, stream) -> None:
        M, K = mX.shape[0], mX.shape[1]
        num_row_tiles, num_col_tiles = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
        if cutlass.const_expr(self.use_coalesced_transpose):
            grid = [num_col_tiles, num_row_tiles, 1]
        else:
            grid = [num_row_tiles, num_col_tiles, 1]
        self.kernel(mX, mOut, mScale, mOutT, mScaleT).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, mOutT: cute.Tensor, mScaleT: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.use_coalesced_transpose):
            row_tile_idx, col_tile_idx = by, bx
        else:
            row_tile_idx, col_tile_idx = bx, by
        sub_block = SubBlock(tidx, row_tile_idx, col_tile_idx, Int32(mX.shape[0]), Int32(mX.shape[1]), self.num_rows_per_thread)

        x_words = self.load_block(mX, sub_block)

        row_amax = self.reduce_row_amax(x_words)

        quant_bytes = cute.make_rmem_tensor(cute.make_layout(8), Uint8)  # (element within output vector,)
        self.store_rowwise(x_words, row_amax, quant_bytes, mOut, mScale, sub_block)

        tile_rcp = self.reduce_and_store_tile_scale(row_amax, mScaleT, sub_block)

        if cutlass.const_expr(self.use_coalesced_transpose):
            s_tile = self.stage_transpose(x_words, tile_rcp, sub_block)
            self.store_transposed_coalesced(s_tile, mOutT, sub_block)
        else:
            self.store_transposed_direct(x_words, tile_rcp, quant_bytes, mOutT, sub_block)

    @cute.jit
    def load_block(self, mX: cute.Tensor, sub_block: SubBlock) -> cute.Tensor:
        """
        Load the thread's rows across two eight-column slices, zero-filling the tile edges.
        """
        row_start, col_start, M, K = sub_block.row_start, sub_block.col_start, sub_block.M, sub_block.K
        COL_VECTOR_WIDTH, elem_bytes = self.col_vector_width, self.elem_bytes
        x_row_stride = Int64(mX.stride[0])
        x_block = cute.make_rmem_tensor(cute.make_layout(16 * self.num_rows_per_thread), self.dtype)  # (row, column half, column in half), flattened
        if cutlass.const_expr(not self.has_full_tiles):
            x_block.fill(0.0)
        for row_idx in cutlass.range_constexpr(self.num_rows_per_thread):
            row = row_start + row_idx
            for half_idx in cutlass.range_constexpr(2):
                for vector_idx in cutlass.range_constexpr(8 // COL_VECTOR_WIDTH):
                    col = col_start + half_idx * 64 + vector_idx * COL_VECTOR_WIDTH
                    if (row < M) | self.has_full_tiles:
                        if (col < K) | self.has_full_tiles:
                            x_offset = Int64(row) * x_row_stride + Int64(col)
                            source = cute.make_tensor(make_global_ptr(mX.iterator, x_offset, elem_bytes, self.dtype, COL_VECTOR_WIDTH * elem_bytes), cute.make_layout(COL_VECTOR_WIDTH))
                            dest = cute.make_tensor(x_block.iterator + (row_idx * 16 + half_idx * 8 + vector_idx * COL_VECTOR_WIDTH), cute.make_layout(COL_VECTOR_WIDTH))
                            cute.autovec_copy(source, dest)
        return cute.recast_tensor(x_block, Uint32)

    @cute.jit
    def reduce_row_amax(self, x_words: cute.Tensor) -> cute.Tensor:
        """
        Reduce each of the thread's rows across all 128 columns of the tile.
        """
        max_instruction = {"bf16": "max.bf16x2", "f16": "max.f16x2", "f32": "max.f32"}[self.input_format]
        abs_mask = "0x7FFFFFFF" if cutlass.const_expr(self.input_format == "f32") else "0x7FFF7FFF"
        accumulate_ptx = f"{{\n\t.reg .b32 t;\n\tand.b32 t, $2, {abs_mask};\n\t{max_instruction} $0, $1, t;\n\t}}"
        if cutlass.const_expr(self.input_format == "f32"):
            ptx_body = "mov.b32 $0, $1;"
        else:
            ptx_type = self.input_format
            ptx_body = (
                ".reg .b16 l, h, m;\n\tmov.b32 {l, h}, $1;\n\t"
                f"max.{ptx_type} m, l, h;\n\tcvt.f32.{ptx_type} $0, m;"
            )
        widen_ptx = "{\n\t" + ptx_body + "\n\t}"

        row_amax = cute.make_rmem_tensor(cute.make_layout(8), Float32)  # (row within thread block,)
        for row_idx in cutlass.range_constexpr(self.num_rows_per_thread):
            acc_amax_bits = Uint32(0)
            for element_idx in cutlass.range_constexpr(row_idx * 16, row_idx * 16 + 16, self.elems_per_word):
                word = Uint32(x_words[element_idx // self.elems_per_word])
                acc_amax_bits = Uint32(asm(T.i32(), [Uint32(acc_amax_bits).ir_value(), word.ir_value()], accumulate_ptx, "=r,r,r"))
            amax = Float32(asm(T.f32(), [Uint32(acc_amax_bits).ir_value()], widen_ptx, "=f,r"))
            for step in cutlass.range_constexpr(3):
                amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << step))
            row_amax[row_idx] = amax
        return row_amax

    @cute.jit
    def store_rowwise(self, x_words: cute.Tensor, row_amax: cute.Tensor, quant_bytes: cute.Tensor, mOut: cute.Tensor, mScale: cute.Tensor, sub_block: SubBlock) -> None:
        """
        Quantize each register row and store it with its scale.
        """
        row_start, col_start = sub_block.row_start, sub_block.col_start
        col_group_idx, col_tile_idx = sub_block.col_group_idx, sub_block.col_tile_idx
        M, K = sub_block.M, sub_block.K
        COL_VECTOR_WIDTH = self.col_vector_width
        scale_col_stride = Int64(mScale.stride[1])  # (row, column tile), MN-major
        quant_words = cute.recast_tensor(quant_bytes, Uint32)  # (packed output word,)
        for row_idx in cutlass.range_constexpr(self.num_rows_per_thread):
            row = row_start + row_idx
            scale = e8m0_scale(row_amax[row_idx])
            rcp = exact_rcp(scale)
            if (row < M) | self.has_full_tiles:
                if col_group_idx == row_idx:
                    scale_entry = cute.make_tensor(make_global_ptr(mScale.iterator, Int64(col_tile_idx) * scale_col_stride + Int64(row), 4, Float32, 4), cute.make_layout(1))
                    scale_entry[0] = scale
                for half_idx in cutlass.range_constexpr(2):
                    for word_idx in cutlass.range_constexpr(2):
                        element_idxs = [row_idx * 16 + half_idx * 8 + 4 * word_idx + k for k in range(4)]
                        quant_words[word_idx] = self.quantize_four_elements(x_words, element_idxs, rcp)
                    for vector_idx in cutlass.range_constexpr(8 // COL_VECTOR_WIDTH):
                        col = col_start + half_idx * 64 + vector_idx * COL_VECTOR_WIDTH
                        if (col < K) | self.has_full_tiles:
                            dest = cute.make_tensor(make_global_ptr(mOut.iterator, Int64(row) * Int64(K) + Int64(col), 1, Uint8, COL_VECTOR_WIDTH), cute.make_layout(COL_VECTOR_WIDTH))
                            source = cute.make_tensor(quant_bytes.iterator + vector_idx * COL_VECTOR_WIDTH, cute.make_layout(COL_VECTOR_WIDTH))
                            cute.autovec_copy(source, dest)

    @cute.jit
    def reduce_and_store_tile_scale(self, row_amax: cute.Tensor, mScaleT: cute.Tensor, sub_block: SubBlock) -> Float32:
        """
        Reduce the tile amax, store its scale, and return the reciprocal.
        """
        tidx, warp, lane = sub_block.tidx, sub_block.warp, sub_block.lane
        row_tile_idx, col_tile_idx = sub_block.row_tile_idx, sub_block.col_tile_idx
        num_row_tiles = Int32(mScaleT.shape[1])

        tile_amax = row_amax[0]
        for row_idx in cutlass.range_constexpr(1, self.num_rows_per_thread):
            tile_amax = cute.arch.fmax(tile_amax, row_amax[row_idx])
        tile_amax = cute.arch.fmax(tile_amax, cute.arch.shuffle_sync_bfly(tile_amax, offset=8))
        tile_amax = cute.arch.fmax(tile_amax, cute.arch.shuffle_sync_bfly(tile_amax, offset=16))

        smem = cutlass.memory.SmemAllocator()
        s_amax = smem.allocate_tensor(Float32, cute.make_layout(self.num_warps), byte_alignment=16)  # (warp,)
        if lane == 0:
            s_amax[warp] = tile_amax
        cute.arch.barrier()
        tile_amax = s_amax[0]
        for warp_idx in cutlass.range_constexpr(1, self.num_warps):
            tile_amax = cute.arch.fmax(tile_amax, s_amax[warp_idx])

        tile_scale = e8m0_scale(tile_amax)
        tile_rcp = exact_rcp(tile_scale)
        if tidx == 0:
            scale_t_entry = cute.make_tensor(make_global_ptr(mScaleT.iterator, Int64(col_tile_idx) * Int64(num_row_tiles) + Int64(row_tile_idx), 4, Float32, 4), cute.make_layout(1))
            scale_t_entry[0] = tile_scale
        return tile_rcp

    @cute.jit
    def stage_transpose(self, x_words: cute.Tensor, tile_rcp: Float32, sub_block: SubBlock) -> cute.Tensor:
        warp, row_group_idx, col_group_idx = sub_block.warp, sub_block.row_group_idx, sub_block.col_group_idx
        smem = cutlass.memory.SmemAllocator()
        s_tile = smem.allocate_tensor(Uint32, cute.make_layout(BLOCK * BLOCK // 4), byte_alignment=128)  # (column, swizzled row word), flattened
        word_row_idx = warp * 4 + row_group_idx
        # Permute aligned 16-byte vectors to distribute the column groups over 32 banks.
        swizzled_word_idx = word_row_idx ^ (col_group_idx * 4)
        for half_idx in cutlass.range_constexpr(2):
            for col_in_half in cutlass.range_constexpr(8):
                col = half_idx * 64 + col_group_idx * 8 + col_in_half
                element_idxs = [row_idx * 16 + half_idx * 8 + col_in_half for row_idx in range(4)]
                s_tile[col * (BLOCK // 4) + swizzled_word_idx] = self.quantize_four_elements(x_words, element_idxs, tile_rcp)
        cute.arch.barrier()
        return s_tile

    @cute.jit
    def store_transposed_coalesced(self, s_tile: cute.Tensor, mOutT: cute.Tensor, sub_block: SubBlock) -> None:
        tidx, row_tile_idx, col_tile_idx, M = sub_block.tidx, sub_block.row_tile_idx, sub_block.col_tile_idx, sub_block.M
        # Eight adjacent lanes cover all 128 bytes of one transposed column.
        vector_idx = tidx % 8
        row = row_tile_idx * BLOCK + vector_idx * 16
        vector_layout = cute.make_layout(4)
        for group_idx in cutlass.range_constexpr(4):
            col = tidx // 8 + group_idx * 32
            col_group_idx = (col // 8) % 8
            # Undo the staging permutation at the granularity of a 16-byte vector.
            shared_word_idx = col * (BLOCK // 4) + (vector_idx ^ col_group_idx) * 4
            source_ptr = cute.make_ptr(Uint32, (s_tile.iterator + shared_word_idx).toint(), cute.AddressSpace.smem, assumed_align=16)
            global_col = col_tile_idx * BLOCK + col
            out_t_offset = Int64(global_col) * Int64(M) + Int64(row)
            dest_ptr = make_global_ptr(mOutT.iterator, out_t_offset, 1, Uint32, 16)
            cute.autovec_copy(cute.make_tensor(source_ptr, vector_layout), cute.make_tensor(dest_ptr, vector_layout))

    @cute.jit
    def store_transposed_direct(self, x_words: cute.Tensor, tile_rcp: Float32, quant_bytes: cute.Tensor, mOutT: cute.Tensor, sub_block: SubBlock) -> None:
        """
        Quantize by the tile scale and store each column's rows as vectors.
        """
        row_start, col_start, M, K = sub_block.row_start, sub_block.col_start, sub_block.M, sub_block.K
        ROW_VECTOR_WIDTH = self.row_vector_width
        quant_words = cute.recast_tensor(quant_bytes, Uint32)  # (packed output word,)
        if (row_start < M) | self.has_full_tiles:
            for half_idx in cutlass.range_constexpr(2):
                for col_in_half in cutlass.range_constexpr(8):
                    col = col_start + half_idx * 64 + col_in_half
                    if (col < K) | self.has_full_tiles:
                        for word_idx in cutlass.range_constexpr(self.num_rows_per_thread // 4):
                            quant_words[word_idx] = self.quantize_four_elements(x_words, [(4 * word_idx + k) * 16 + half_idx * 8 + col_in_half for k in range(4)], tile_rcp)
                        for vector_idx in cutlass.range_constexpr(self.num_rows_per_thread // ROW_VECTOR_WIDTH):
                            row = row_start + vector_idx * ROW_VECTOR_WIDTH
                            if (row < M) | self.has_full_tiles:
                                out_t_offset = Int64(col) * Int64(M) + Int64(row)
                                dest = cute.make_tensor(make_global_ptr(mOutT.iterator, out_t_offset, 1, Uint8, ROW_VECTOR_WIDTH), cute.make_layout(ROW_VECTOR_WIDTH))
                                source = cute.make_tensor(quant_bytes.iterator + vector_idx * ROW_VECTOR_WIDTH, cute.make_layout(ROW_VECTOR_WIDTH))
                                cute.autovec_copy(source, dest)

    def quantize_four_elements(self, x_words: cute.Tensor, element_idxs: list, rcp: Float32) -> Uint32:
        """
        Pack four selected register elements as E4M3 bytes after scaling.
        """
        word_idxs, element_locations = [], []
        for element_idx in element_idxs:
            word_idx = element_idx // self.elems_per_word
            if word_idx not in word_idxs:
                word_idxs.append(word_idx)
            element_locations.append((word_idxs.index(word_idx), element_idx % self.elems_per_word))
        return pack_scaled_e4m3([x_words[word_idx] for word_idx in word_idxs], element_locations, self.input_format, rcp)


COMPILE_CACHE: dict[tuple, Callable] = {}


def choose_vector_width(size: int, max_width: int) -> int:
    """
    The largest power-of-2 width up to max_width (a power of 2) that divides size.
    """
    width = max_width
    while size % width:
        width //= 2
    return width


def compile_fp8_quant_act(dtype: type, col_vector_width: int, row_vector_width: int, num_rows_per_thread: int, has_full_tiles: bool, use_coalesced_transpose: bool) -> Callable:
    M, K, num_col_tiles, num_row_tiles = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    elem_bytes = dtype.width // 8
    x = fake(dtype, (M, K), (cute.sym_int64(divisibility=col_vector_width), 1), col_vector_width * elem_bytes)
    out = fake(cutlass.Float8E4M3FN, (M, K), (K, 1))
    scale = fake(Float32, (M, num_col_tiles), (1, cute.sym_int64()))
    out_t = fake(cutlass.Float8E4M3FN, (K, M), (M, 1))
    scale_t = fake(Float32, (num_col_tiles, num_row_tiles), (num_row_tiles, 1))
    return cute.compile(RowwiseBlockwiseTranspose(dtype, col_vector_width, row_vector_width, num_rows_per_thread, has_full_tiles, use_coalesced_transpose), x, out, scale, out_t, scale_t, fake_stream(), options="--enable-tvm-ffi")


DISPATCH_CACHE: dict[tuple, Callable | None] = {}


def resolve_fp8_quant_act(dtype: torch.dtype, M: int, K: int, x_row_stride: int, x_base: int, has_unit_col_stride: bool, num_tiles: int) -> Callable | None:
    if M == 0 or K == 0:
        return None
    elem_bytes = dtype.itemsize
    col_vector_width = choose_vector_width(K, min(16 // elem_bytes, 8))
    col_vector_width = choose_vector_width(x_row_stride, col_vector_width)
    col_vector_width = choose_vector_width(x_base // elem_bytes, col_vector_width)
    # These aligned dense BF16 shapes fill every tile, so loads and stores need no edge predicates.
    has_full_tiles = dtype == torch.bfloat16 and M in (4096, 8192) and K in (5120, 6144, 17408) and x_row_stride == K and col_vector_width == 8
    use_coalesced_transpose = has_unit_col_stride and has_full_tiles and K in (5120, 6144)
    num_rows_per_thread = 4 if use_coalesced_transpose or num_tiles <= SMALL_GRID_TILES else 8
    row_vector_width = choose_vector_width(M, num_rows_per_thread)
    key = ("fp8_quant_act", dtype, col_vector_width, row_vector_width, num_rows_per_thread, has_full_tiles, use_coalesced_transpose)
    fn = COMPILE_CACHE.get(key)
    if fn is None:
        fn = COMPILE_CACHE[key] = compile_fp8_quant_act(TORCH2CUTE[dtype], col_vector_width, row_vector_width, num_rows_per_thread, has_full_tiles, use_coalesced_transpose)
    if len(DISPATCH_CACHE) > 4096:
        DISPATCH_CACHE.clear()
    return fn


def fp8_quant_act_sm90(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SM90 implementation of fp8_quant_act.
    """
    assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D"
    M, K = x.shape
    x_row_stride, x_col_stride = x.stride()
    has_unit_col_stride = x_col_stride == 1
    if not has_unit_col_stride:
        # as_strided pins the unit inner stride that torch leaves arbitrary when K == 1.
        x = x.contiguous().as_strided((M, K), (K, 1))
        x_row_stride = K

    num_row_tiles, num_col_tiles = (M + BLOCK - 1) // BLOCK, (K + BLOCK - 1) // BLOCK
    device = x.device
    out = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    scale = mn_major_scale(M, num_col_tiles, device)
    out_t = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=device)
    scale_t = torch.empty((num_col_tiles, num_row_tiles), dtype=torch.float32, device=device)

    # Cache shape dispatch to avoid recomputing vector alignment for repeated launches.
    dtype, x_base = x.dtype, x.data_ptr()
    key = (dtype, M, K, x_row_stride, x_base & 15, has_unit_col_stride)
    try:
        fn = DISPATCH_CACHE[key]
    except KeyError:
        fn = DISPATCH_CACHE[key] = resolve_fp8_quant_act(dtype, M, K, x_row_stride, x_base, has_unit_col_stride, num_row_tiles * num_col_tiles)

    if fn is not None:
        fn(x, out, scale, out_t, scale_t)
    return out, scale, out_t, scale_t
