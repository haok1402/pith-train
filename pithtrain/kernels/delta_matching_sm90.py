"""
E4M3 operands, FP32 block scales and tile primitives for SM90 Delta-Matching.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float8E4M3FN, Float32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum, SmemAllocator
from cutlass.utils.hopper_helpers import get_smem_layout_atom, make_trivial_tiled_mma
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream

__all__ = [
    "BLOCK_QUERY", "LOG2_E", "THREADS", "allocate_operand", "allocate_staged_operand", "copy_tma_operand",
    "initialize_attention_barriers", "issue_multiply_tile", "launch_delta_matching_kernel", "make_mma",
    "load_quant_block", "make_operand_layout", "make_operand_tma", "multiply_tile", "quantize_delta_matching", "reduce_mma_row", "reduce_quant_block",
]

BLOCK_QUERY = 64
THREADS = 128
LOG2_E = 1.4426950408889634
COMPILE_CACHE = {}


def launch_delta_matching_kernel(name, launcher, tensors, constants=()):
    """
    Compile each tensor layout and launch through the active TVM-FFI stream.
    """
    layouts = tuple((tensor.dtype, tuple(tensor.shape), tensor.stride()) for tensor in tensors)
    key = (name, *layouts, *constants)
    fn = COMPILE_CACHE.get(key)
    if fn is None:
        arguments = [fake(TORCH2CUTE[dtype], shape, strides, align=16 if strides[-1] == 1 else 1) for dtype, shape, strides in layouts]
        fn = cute.compile(launcher, *arguments, *constants, fake_stream(), options="--enable-tvm-ffi")
        COMPILE_CACHE[key] = fn
    fn(*(tensor.detach() for tensor in tensors))


@cute.jit
def make_operand_layout(rows: cutlass.Constexpr, columns: cutlass.Constexpr):
    kind = get_smem_layout_atom(LayoutEnum.ROW_MAJOR, Float8E4M3FN, columns)
    atom = warpgroup.make_smem_layout_atom(kind, Float8E4M3FN)
    return cute.tile_to_shape(atom, (rows, columns), (0, 1))


@cute.jit
def allocate_operand(allocator, rows: cutlass.Constexpr, columns: cutlass.Constexpr):
    layout = make_operand_layout(rows, columns)
    return allocator.allocate_tensor(Float8E4M3FN, layout.outer, 128, swizzle=layout.inner)


@cute.jit
def allocate_staged_operand(allocator, rows: cutlass.Constexpr, columns: cutlass.Constexpr, stages: cutlass.Constexpr):
    kind = get_smem_layout_atom(LayoutEnum.ROW_MAJOR, Float8E4M3FN, columns)
    atom = warpgroup.make_smem_layout_atom(kind, Float8E4M3FN)
    layout = cute.tile_to_shape(atom, (rows, columns, stages), (0, 1, 2))
    return allocator.allocate_tensor(Float8E4M3FN, layout.outer, 128, swizzle=layout.inner)


@cute.jit
def initialize_attention_barriers(allocator, tidx, STAGES: cutlass.Constexpr, CONSUMERS: cutlass.Constexpr):
    """
    Publish resident operands once and reclaim each streamed tile after all consumers release it.
    """
    barriers = allocator.allocate_tensor(Int64, cute.make_layout(1 + 2 * STAGES), 8).iterator
    query_ready, tile_ready, tile_released = barriers, barriers + 1, barriers + 1 + STAGES
    if tidx == 0:
        cute.arch.mbarrier_init(query_ready, 1)
        for stage_idx in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(tile_ready + stage_idx, 1)
            cute.arch.mbarrier_init(tile_released + stage_idx, CONSUMERS)
        cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    return query_ready, tile_ready, tile_released


@cute.jit
def make_operand_tma(mCodes, rows: cutlass.Constexpr, columns: cutlass.Constexpr):
    H, N, D = mCodes.shape
    tensor = cute.make_tensor(mCodes.iterator, cute.make_layout((N, D, H), stride=(D, 1, N * D)))  # (operand row, operand column, head)
    layout = make_operand_layout(rows, columns)
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), tensor, layout, (rows, columns))


@cute.jit
def copy_tma_operand(descriptor, s_operand, row_idx, col_idx, head_idx, barrier):
    atom, tensor = descriptor
    rows, columns = cute.size(s_operand, mode=[0]), cute.size(s_operand, mode=[1])
    tile = cute.local_tile(tensor[None, None, head_idx], (rows, columns), (row_idx, col_idx))  # (tile row, tile column)
    destination, source = cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(s_operand, 0, 2), cute.group_modes(tile, 0, 2))
    cute.copy(atom, source, destination, tma_bar_ptr=barrier)


@cute.jit
def make_mma(columns: cutlass.Constexpr, use_register_operand: cutlass.Constexpr = False):
    source = warpgroup.OperandSource.RMEM if const_expr(use_register_operand) else warpgroup.OperandSource.SMEM
    return make_trivial_tiled_mma(Float8E4M3FN, Float8E4M3FN, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, Float32, (1, 1, 1), (64, columns), a_source=source)


@cute.jit
def issue_multiply_tile(mma, mma_thread, acc_product, left_operand, s_right, accumulate):
    """
    Commit an FP8 WGMMA chain and defer completion to the caller.
    """
    left = left_operand if const_expr(left_operand.memspace == cute.AddressSpace.rmem) else mma_thread.make_fragment_A(mma_thread.partition_A(left_operand))  # (MMA, MMA_M, MMA_K)
    right = mma_thread.make_fragment_B(mma_thread.partition_B(s_right))  # (MMA, MMA_N, MMA_K)
    warpgroup.fence()
    # Keep the per-issue accumulator flag local to this region.
    atom = cute.make_mma_atom(mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, accumulate)
    for col_idx in cutlass.range_constexpr(cute.size(left, mode=[2])):
        cute.gemm(atom, acc_product, left[None, None, col_idx], right[None, None, col_idx], acc_product)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()


@cute.jit
def multiply_tile(mma, mma_thread, acc_product, left_operand, s_right, accumulate):
    issue_multiply_tile(mma, mma_thread, acc_product, left_operand, s_right, accumulate)
    warpgroup.wait_group(0)


@cute.jit
def reduce_mma_row(fragment, row_idx: cutlass.Constexpr, initial, op: cutlass.Constexpr):
    rows = reshape_acc_to_mn(fragment)
    reduction = initial
    for col_idx in cutlass.range_constexpr(cute.size(rows, mode=[1])):
        reduction = op(reduction, rows[row_idx, col_idx])
    return warp_reduce(reduction, op, threads_in_group=4)


@cute.kernel
def sum_key_channels(mKey, mPartial, H: cutlass.Constexpr, N: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    batch_idx, head_idx = bz // H, bz % H
    col_idx, row_idx = bx * 32 + tidx % 32, tidx // 32
    acc_sum = Float32(0.0)
    for row_offset in cutlass.range_constexpr(16):
        acc_sum += mKey[batch_idx, by * 128 + row_idx + row_offset * 8, head_idx, col_idx].to(Float32)
    allocator = SmemAllocator()
    s_partial = allocator.allocate_tensor(Float32, cute.make_layout((8, 32), stride=(32, 1)), 16)
    s_partial[row_idx, tidx % 32] = acc_sum
    cute.arch.barrier()
    if row_idx == 0:
        for row_offset in cutlass.range_constexpr(1, 8):
            acc_sum += s_partial[row_offset, tidx]
        mPartial[bz, by, col_idx] = acc_sum / N


@cute.jit
def launch_key_sum(mKey, mPartial, stream):
    B, N, H, D = mKey.shape
    sum_key_channels(mKey, mPartial, H, N).launch(grid=(D // 32, N // 128, B * H), block=(256, 1, 1), stream=stream)


@cute.jit
def load_quant_block(mInput, mCenter, bx, by, tidx, BLOCK: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr):
    _, _, H, D = mInput.shape
    batch_idx, head_idx = by // H, by % H
    VEC = 8
    REPEATS = BLOCK * D // (THREADS * VEC)
    values = cute.make_rmem_tensor((VEC, REPEATS), Float32)
    col_idx, local_row = tidx % (D // VEC) * VEC, tidx // (D // VEC)
    center = cute.make_rmem_tensor(VEC, Float32)
    if const_expr(center_key):
        center.fill(0.0)
        for partial_idx in range(cute.size(mCenter, mode=[1])):
            for elem_idx in cutlass.range_constexpr(VEC):
                center[elem_idx] += mCenter[by, partial_idx, col_idx + elem_idx]
    for repeat_idx in cutlass.range_constexpr(REPEATS):
        row_idx = bx * BLOCK + local_row + repeat_idx * (THREADS * VEC // D)
        source = cute.local_tile(mInput[batch_idx, row_idx, head_idx, None], (VEC,), (col_idx // VEC,))  # (input vector)
        raw = source.load().to(Float32)
        if const_expr(center_key):
            raw = raw - center.load()
        if const_expr(input_scale != 1.0):
            raw = raw * input_scale
        values[None, repeat_idx].store(raw)
    return values


@cute.jit
def reduce_quant_block(values, s_amax, tidx, is_gradient: cutlass.Constexpr):
    amax = Float32(0.0)
    for elem_idx in cutlass.range_constexpr(cute.size(values)):
        amax = cute.arch.fmax(amax, abs(values[elem_idx]))
    amax = warp_reduce(amax, cute.arch.fmax)
    if tidx % 32 == 0:
        s_amax[tidx // 32] = amax
    cute.arch.barrier()
    for warp_idx in cutlass.range_constexpr(THREADS // 32):
        amax = cute.arch.fmax(amax, s_amax[warp_idx])
    amax = cute.arch.fmax(amax, 1e-30)
    # The dO producer rounds a multiply by 1 / 448 and the input producers divide.
    if const_expr(is_gradient):
        return amax * (1.0 / 448.0)
    return amax / Float32(448.0)


@cute.jit
def store_quant_block(mCodes, mTwin, values, s_codes, bx, by, tidx, scale, BLOCK: cutlass.Constexpr):
    D = cute.size(mCodes, mode=[2])
    VEC = 8
    col_idx, local_row = tidx % (D // VEC) * VEC, tidx // (D // VEC)
    for repeat_idx in cutlass.range_constexpr(cute.size(values, mode=[1])):
        row_idx = local_row + repeat_idx * (THREADS * VEC // D)
        codes = cute.make_rmem_tensor(VEC, Float32)
        for elem_idx in cutlass.range_constexpr(VEC):
            codes[elem_idx] = values[elem_idx, repeat_idx] / scale
        quantized = codes.load().to(Float8E4M3FN)
        destination = cute.local_tile(mCodes[by, bx * BLOCK + row_idx, None], (VEC,), (col_idx // VEC,))  # (code vector)
        destination.store(quantized)
        for elem_idx in cutlass.range_constexpr(VEC):
            s_codes[row_idx, col_idx + elem_idx] = quantized[elem_idx]
    cute.arch.barrier()
    for repeat_idx in cutlass.range_constexpr(BLOCK * D // (THREADS * VEC)):
        tile_idx = tidx + repeat_idx * THREADS
        col_idx, row_idx = tile_idx // (BLOCK // VEC), tile_idx % (BLOCK // VEC) * VEC
        transposed = cute.make_rmem_tensor(VEC, Float8E4M3FN)
        for elem_idx in cutlass.range_constexpr(VEC):
            transposed[elem_idx] = s_codes[row_idx + elem_idx, col_idx]
        destination = cute.local_tile(mTwin[by, col_idx, None], (VEC,), (bx * (BLOCK // VEC) + row_idx // VEC,))  # (transposed vector)
        destination.store(transposed.load())


@cute.kernel
def quantize_block(mInput, mCenter, mCodes, mScale, mTwin, BLOCK: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    D = cute.size(mCodes, mode=[2])
    allocator = SmemAllocator()
    s_amax = allocator.allocate_tensor(Float32, cute.make_layout(THREADS // 32), 16)
    s_codes = allocate_operand(allocator, BLOCK, D)

    values = load_quant_block(mInput, mCenter, bx, by, tidx, BLOCK, center_key, input_scale)
    scale = reduce_quant_block(values, s_amax, tidx, is_gradient)
    if tidx == 0:
        mScale[by, bx] = scale
    store_quant_block(mCodes, mTwin, values, s_codes, bx, by, tidx, scale, BLOCK)


@cute.jit
def launch_quant_block(mInput, mCenter, mCodes, mScale, mTwin, BLOCK: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr, stream):
    B, N, H, _ = mInput.shape
    quantize_block(mInput, mCenter, mCodes, mScale, mTwin, BLOCK, center_key, input_scale, is_gradient).launch(grid=(N // BLOCK, B * H, 1), block=(THREADS, 1, 1), stream=stream)


def quantize_delta_matching(x, block_rows, *, center_key=False, input_scale=1.0, is_gradient=False):
    """
    Quantize dense BSHD BF16 to E4M3 head-major and channel-major codes and per-head FP32 scales.
    """
    B, N, H, D = x.shape
    x = x.contiguous()
    codes = torch.empty((B * H, N, D), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((B * H, N // block_rows), dtype=torch.float32, device=x.device)
    twin = torch.empty((B * H, D, N), dtype=codes.dtype, device=x.device)
    center = torch.empty((B * H, N // 128, D) if center_key else (1, 1, 1), dtype=torch.float32, device=x.device)

    # Centering leaves the exact softmax unchanged and reduces the key quantization range.
    if center_key:
        launch_delta_matching_kernel("delta_matching_key_sum", launch_key_sum, (x, center))
    launch_delta_matching_kernel("delta_matching_quantize", launch_quant_block, (x, center, codes, scales, twin), (block_rows, center_key, input_scale, is_gradient))
    return codes, scales, twin
