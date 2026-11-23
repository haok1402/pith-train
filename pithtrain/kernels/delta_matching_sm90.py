"""
E4M3 operands, FP32 block scales and tile primitives for SM90 Delta-Matching.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float8E4M3FN, Float32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum, SmemAllocator
from cutlass.utils.hopper_helpers import get_smem_layout_atom, make_trivial_tiled_mma
from quack.layout_utils import reshape_acc_to_mn
from quack.reduce import warp_reduce

from pithtrain.kernels.cute_compile_sm90 import TORCH2CUTE, fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, to_ir

__all__ = [
    "BLOCK_QUERY", "LOG2_E", "allocate_operand", "allocate_staged_operand", "copy_tma_operand",
    "initialize_attention_barriers", "issue_multiply_tile", "launch_delta_matching_kernel", "make_mma",
    "make_operand_tma", "multiply_tile", "quantize_centered_kv", "quantize_delta_matching", "reduce_mma_row",
]

BLOCK_QUERY = 64
FORWARD_THREADS = 256
BACKWARD_THREADS = 128
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
def sum_key_channels(mKey, mPartial, H: cutlass.Constexpr, N: cutlass.Constexpr, WIDTH: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    batch_idx, head_idx = bz // H, bz % H
    col_group, row_idx = bx * 32 + tidx % 32, tidx // 32
    acc_sum = cute.make_rmem_tensor(WIDTH, Float32)
    acc_sum.fill(0.0)
    for row_offset in cutlass.range_constexpr(16):
        token = by * 128 + row_idx + row_offset * 8
        part = cute.local_tile(mKey[batch_idx, token, head_idx, None], (WIDTH,), (col_group,))
        acc_sum.store(acc_sum.load() + part.load().to(Float32))
    allocator = SmemAllocator()
    layout = cute.make_layout((8, 32 * WIDTH), stride=(32 * WIDTH, 1))
    s_partial = allocator.allocate_tensor(Float32, layout, 16)
    local_group = tidx % 32
    store = cute.local_tile(s_partial[row_idx, None], (WIDTH,), (local_group,))
    store.store(acc_sum.load())
    cute.arch.barrier()
    if row_idx == 0:
        for row_offset in cutlass.range_constexpr(1, 8):
            shared_part = cute.local_tile(s_partial[row_offset, None], (WIDTH,), (local_group,))
            acc_sum.store(acc_sum.load() + shared_part.load())
        output = cute.local_tile(mPartial[bz, by, None], (WIDTH,), (col_group,))
        output.store(acc_sum.load() / N)


@cute.jit
def launch_key_sum(mKey, mPartial, stream):
    B, N, H, D = mKey.shape
    WIDTH = 4
    sum_key_channels(mKey, mPartial, H, N, WIDTH).launch(
        grid=(D // (32 * WIDTH), N // 128, B * H), block=(256, 1, 1), stream=stream,
    )


@cute.jit
def load_quant_block(mInput, mCenter, bx, by, tidx, BLOCK: cutlass.Constexpr, THREADS: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr):
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
def reduce_quant_block(values, s_amax, tidx, THREADS: cutlass.Constexpr, is_gradient: cutlass.Constexpr):
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
def store_quant_block(mCodes, mTwin, values, s_codes, bx, by, tidx, scale, BLOCK: cutlass.Constexpr, THREADS: cutlass.Constexpr):
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


@dsl_user_op
def reciprocal_rn(scale, *, loc=None, ip=None):
    operand = to_ir(scale, Float32, loc, ip)
    return Float32(asm(T.f32(), [operand], "rcp.rn.f32 $0, $1;", "=f,f"))


@dsl_user_op
def refine_quotient(value, negative_scale, reciprocal, *, loc=None, ip=None):
    # Reuse one correctly rounded reciprocal throughout the block.
    instructions = """{
        .reg .f32 q, residual;
        mul.rn.f32 q, $1, $3;
        fma.rn.f32 residual, $2, q, $1;
        fma.rn.f32 q, residual, $3, q;
        fma.rn.f32 residual, $2, q, $1;
        fma.rn.f32 q, residual, $3, q;
        copysign.f32 $0, $1, q;
    }"""
    operands = [to_ir(v, Float32, loc, ip) for v in (value, negative_scale, reciprocal)]
    return Float32(asm(T.f32(), operands, instructions, "=f,f,f,f"))


@cute.kernel
def quantize_register_twins(mInput, mCenter, mCodes, mScale, mTwin, BLOCK: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    _, _, heads, dim = mInput.shape
    COLUMNS = 4 if const_expr(BLOCK * dim == 16384) else 2
    ROWS = BLOCK * dim // (FORWARD_THREADS * COLUMNS)
    col_start = tidx % (dim // COLUMNS) * COLUMNS
    row_start = bx * BLOCK + tidx // (dim // COLUMNS) * ROWS
    batch_idx, head_idx = by // heads, by % heads
    allocator = SmemAllocator()
    s_amax = allocator.allocate_tensor(Float32, cute.make_layout(FORWARD_THREADS // 32), 16)
    values = cute.make_rmem_tensor((COLUMNS, ROWS), Float32)
    center = cute.make_rmem_tensor(COLUMNS, Float32)
    if const_expr(center_key):
        for col_offset in cutlass.range_constexpr(COLUMNS):
            center[col_offset] = Float32(0.0) + mCenter[by, 0, col_start + col_offset]
    for row_offset in cutlass.range_constexpr(ROWS):
        source = cute.local_tile(mInput[batch_idx, row_start + row_offset, head_idx, None], (COLUMNS,), (col_start // COLUMNS,))
        raw = source.load().to(Float32)
        if const_expr(center_key):
            raw = raw - center.load()
        if const_expr(input_scale != 1.0):
            raw = raw * input_scale
        values[None, row_offset].store(raw)
    scale = reduce_quant_block(values, s_amax, tidx, FORWARD_THREADS, is_gradient)
    if tidx == 0:
        mScale[by, bx] = scale

    PERMUTE_TWIN = const_expr(not is_gradient and not center_key and BLOCK == 32 and dim == 256)
    # Preserve direct division for tiny and nonfinite scales.
    if (scale >= Float32(2.0 ** -88)) and (scale < Float32(float("inf"))):
        store_register_twins(mCodes[by, None, None], mTwin[by, None, None], values, scale, row_start, col_start, ROWS, COLUMNS, True, PERMUTE_TWIN)
    else:
        store_register_twins(mCodes[by, None, None], mTwin[by, None, None], values, scale, row_start, col_start, ROWS, COLUMNS, False, PERMUTE_TWIN)


@cute.jit
def store_register_twins(mCodes, mTwin, values, scale, row_start, col_start,
                         ROWS: cutlass.Constexpr, COLUMNS: cutlass.Constexpr, FAST: cutlass.Constexpr, PERMUTE_TWIN: cutlass.Constexpr):
    codes = cute.make_rmem_tensor(cute.make_ordered_layout((ROWS, COLUMNS), order=(0, 1)), Float8E4M3FN)
    reciprocal = reciprocal_rn(scale) if const_expr(FAST) else Float32(0.0)
    for row_offset in cutlass.range_constexpr(ROWS):
        quotient = cute.make_rmem_tensor(COLUMNS, Float32)
        for col_offset in cutlass.range_constexpr(COLUMNS):
            value = values[col_offset, row_offset]
            if const_expr(FAST):
                quotient[col_offset] = refine_quotient(value, -scale, reciprocal)
            else:
                quotient[col_offset] = value / scale
        quantized = quotient.load().to(Float8E4M3FN)
        destination = cute.local_tile(mCodes[row_start + row_offset, None], (COLUMNS,), (col_start // COLUMNS,))
        destination.store(quantized)
        twin_row = row_offset
        if const_expr(PERMUTE_TWIN):
            # Convert each half of a 32-key group from pairs spaced by eight to consecutive A-register keys.
            twin_row = row_offset // 16 * 16 + row_offset % 8 // 2 * 4 + row_offset // 8 % 2 * 2 + row_offset % 2
        for col_offset in cutlass.range_constexpr(COLUMNS):
            codes[twin_row, col_offset] = quantized[col_offset]
    for col_offset in cutlass.range_constexpr(COLUMNS):
        twin = cute.local_tile(mTwin[col_start + col_offset, None], (ROWS,), (row_start // ROWS,))
        twin.store(codes[None, col_offset].load())


@cute.jit
def make_quant_store(mCodes, rows: cutlass.Constexpr, columns: cutlass.Constexpr):
    heads, length, dim = mCodes.shape
    tensor = cute.make_tensor(mCodes.iterator, cute.make_layout((length, dim, heads), stride=(dim, 1, length * dim)))
    layout = make_operand_layout(rows, columns)
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), tensor, layout, (rows, columns))


@cute.jit
def store_quant_tile(descriptor, source, row, column, head):
    atom, tensor = descriptor
    rows, columns = cute.size(source, mode=[0]), cute.size(source, mode=[1])
    target = cute.local_tile(tensor[None, None, head], (rows, columns), (row, column))
    src, dst = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(source, 0, 2), cute.group_modes(target, 0, 2),
    )
    cute.copy(atom, src, dst)


@cute.jit
def quantize_tma_twins_block(mInput, mCenter, mScale, code_store, twin_store, tidx, bx, by,
                             BLOCK: cutlass.Constexpr, center_key: cutlass.Constexpr,
                             input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr):
    heads = cute.size(mInput, mode=[2])
    DIM, ROWS = 256, 16
    COLUMNS = 4 if const_expr(BLOCK == 64) else 2
    col_start = tidx % (DIM // COLUMNS) * COLUMNS
    row_start = tidx // (DIM // COLUMNS) * ROWS
    allocator = SmemAllocator()
    s_amax = allocator.allocate_tensor(Float32, cute.make_layout(FORWARD_THREADS // 32), 16)
    s_codes = allocate_operand(allocator, BLOCK, DIM)
    s_twin = allocate_operand(allocator, DIM, BLOCK)
    values = cute.make_rmem_tensor((COLUMNS, ROWS), Float32)
    center = cute.make_rmem_tensor(COLUMNS, Float32)
    if const_expr(center_key):
        for column in cutlass.range_constexpr(COLUMNS):
            center[column] = Float32(0.0) + mCenter[by, 0, col_start + column]
    for row in cutlass.range_constexpr(ROWS):
        token = bx * BLOCK + row_start + row
        input_row = mInput[by // heads, token, by % heads, None]
        source = cute.local_tile(input_row, (COLUMNS,), (col_start // COLUMNS,))
        raw = source.load().to(Float32)
        if const_expr(center_key):
            raw = raw - center.load()
        if const_expr(input_scale != 1.0):
            raw = raw * input_scale
        values[None, row].store(raw)
    scale = reduce_quant_block(values, s_amax, tidx, FORWARD_THREADS, is_gradient)
    if tidx == 0:
        mScale[by, bx] = scale
    permute_twin = const_expr(BLOCK == 32 and not center_key)
    if (scale >= Float32(2.0 ** -88)) and (scale < Float32(float("inf"))):
        store_register_twins(s_codes, s_twin, values, scale, row_start, col_start, ROWS, COLUMNS, True, permute_twin)
    else:
        store_register_twins(s_codes, s_twin, values, scale, row_start, col_start, ROWS, COLUMNS, False, permute_twin)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if tidx < 32:
        store_quant_tile(code_store, s_codes, bx, 0, by)
        store_quant_tile(twin_store, s_twin, 0, bx, by)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)


@cute.kernel
def quantize_tma_twins(mInput, mCenter, mScale, code_store, twin_store, BLOCK: cutlass.Constexpr,
                        center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    quantize_tma_twins_block(mInput, mCenter, mScale, code_store, twin_store,
                             tidx, bx, by, BLOCK, center_key, input_scale, is_gradient)


@cute.kernel
def quantize_value_and_fold_key(mValue, mValueScale, code_store, twin_store, mPartial, mCenter, mTicket):
    tidx, _, _ = cute.arch.thread_idx()
    block, head, _ = cute.arch.block_idx()
    BLOCK = 32
    value_blocks = cute.size(mValue, mode=[1]) // BLOCK
    if block < value_blocks:
        quantize_tma_twins_block(mValue, mCenter, mValueScale, code_store, twin_store,
                                 tidx, block, head, BLOCK, False, 1.0, False)
    else:
        # A disjoint CTA per head folds key partials while the value tiles quantize.
        center = Float32(0.0)
        for partial in range(cute.size(mPartial, mode=[1])):
            center += mPartial[head, partial, tidx]
        mCenter[head, 0, tidx] = center
        if tidx == 0 and head == 0:
            mTicket[0] = 0


@cute.jit
def launch_value_and_fold_key(mValue, mValueScale, mValueCodes, mValueTwin, mPartial, mCenter, mTicket, stream):
    batch, length, heads, dim = mValue.shape
    code_store = make_quant_store(mValueCodes, 32, dim)
    twin_store = make_quant_store(mValueTwin, dim, 32)
    quantize_value_and_fold_key(mValue, mValueScale, code_store, twin_store, mPartial, mCenter, mTicket).launch(
        grid=(length // 32 + 1, batch * heads, 1), block=(FORWARD_THREADS, 1, 1), stream=stream,
    )


@cute.kernel
def quantize_block(mInput, mCenter, mCodes, mScale, mTwin, BLOCK: cutlass.Constexpr, THREADS: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    D = cute.size(mCodes, mode=[2])
    allocator = SmemAllocator()
    s_amax = allocator.allocate_tensor(Float32, cute.make_layout(THREADS // 32), 16)
    s_codes = allocate_operand(allocator, BLOCK, D)

    values = load_quant_block(mInput, mCenter, bx, by, tidx, BLOCK, THREADS, center_key, input_scale)
    scale = reduce_quant_block(values, s_amax, tidx, THREADS, is_gradient)
    if tidx == 0:
        mScale[by, bx] = scale
    store_quant_block(mCodes, mTwin, values, s_codes, bx, by, tidx, scale, BLOCK, THREADS)


@cute.jit
def launch_quant_block(mInput, mCenter, mCodes, mScale, mTwin, BLOCK: cutlass.Constexpr, center_key: cutlass.Constexpr, input_scale: cutlass.Constexpr, is_gradient: cutlass.Constexpr, is_forward: cutlass.Constexpr, stream):
    B, N, H, _ = mInput.shape
    register_gradient = const_expr(is_gradient and not center_key and BLOCK * cute.size(mInput, mode=[3]) == 16384)
    THREADS = FORWARD_THREADS if const_expr(is_forward or register_gradient) else BACKWARD_THREADS
    use_tma = const_expr(
        cute.size(mInput, mode=[3]) == 256
        and ((is_forward and BLOCK in (32, 64)) or (register_gradient and BLOCK == 64)),
    )
    if const_expr(use_tma):
        D = cute.size(mInput, mode=[3])
        code_store = make_quant_store(mCodes, BLOCK, D)
        twin_store = make_quant_store(mTwin, D, BLOCK)
        quantize_tma_twins(mInput, mCenter, mScale, code_store, twin_store, BLOCK, center_key, input_scale, is_gradient).launch(
            grid=(N // BLOCK, B * H, 1), block=(FORWARD_THREADS, 1, 1), stream=stream,
        )
    elif const_expr(is_forward or register_gradient):
        quantize_register_twins(mInput, mCenter, mCodes, mScale, mTwin, BLOCK, center_key, input_scale, is_gradient).launch(
            grid=(N // BLOCK, B * H, 1), block=(THREADS, 1, 1), stream=stream,
        )
    else:
        quantize_block(mInput, mCenter, mCodes, mScale, mTwin, BLOCK, THREADS, center_key, input_scale, is_gradient).launch(
            grid=(N // BLOCK, B * H, 1), block=(THREADS, 1, 1), stream=stream,
        )


@cute.kernel
def fold_key_partials(mPartial, mCenter, mTicket, RESET_TICKET: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    channel_block, head_idx, _ = cute.arch.block_idx()
    channel = channel_block * 128 + tidx
    center = Float32(0.0)
    for partial_idx in range(cute.size(mPartial, mode=[1])):
        center += mPartial[head_idx, partial_idx, channel]
    mCenter[head_idx, 0, channel] = center
    if const_expr(RESET_TICKET):
        if channel == 0 and head_idx == 0:
            mTicket[0] = 0


@cute.jit
def launch_key_fold(mPartial, mCenter, mTicket, RESET_TICKET: cutlass.Constexpr, stream):
    heads, _, channels = mPartial.shape
    fold_key_partials(mPartial, mCenter, mTicket, RESET_TICKET).launch(
        grid=(channels // 128, heads, 1), block=(128, 1, 1), stream=stream,
    )


def allocate_quantized_twins(x, block_rows):
    batch, length, heads, channels = x.shape
    codes = torch.empty((batch * heads, length, channels), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((batch * heads, length // block_rows), dtype=torch.float32, device=x.device)
    twin = torch.empty((batch * heads, channels, length), dtype=codes.dtype, device=x.device)
    return codes, scales, twin


def quantize_centered_kv(key, value, ticket):
    key, value = key.contiguous(), value.contiguous()
    batch, length, heads, channels = key.shape
    partial = torch.empty((batch * heads, length // 128, channels), dtype=torch.float32, device=key.device)
    center = torch.empty((batch * heads, 1, channels), dtype=torch.float32, device=key.device)
    key_codes, key_scales, key_twin = allocate_quantized_twins(key, 32)
    value_codes, value_scales, value_twin = allocate_quantized_twins(value, 32)
    launch_delta_matching_kernel("delta_matching_key_sum", launch_key_sum, (key, partial))
    launch_delta_matching_kernel(
        "delta_matching_value_and_key_mean", launch_value_and_fold_key,
        (value, value_scales, value_codes, value_twin, partial, center, ticket),
    )
    launch_delta_matching_kernel(
        "delta_matching_quantize", launch_quant_block, (key, center, key_codes, key_scales, key_twin),
        (32, True, 1.0, False, True),
    )
    return (key_codes, key_scales, key_twin), (value_codes, value_scales, value_twin)


def quantize_delta_matching(x, block_rows, *, center_key=False, input_scale=1.0, is_gradient=False, is_forward=False, scheduler_ticket=None):
    """
    Quantize dense BSHD BF16 to E4M3 head-major and channel-major codes and per-head FP32 scales.
    """
    B, N, H, D = x.shape
    x = x.contiguous()
    codes, scales, twin = allocate_quantized_twins(x, block_rows)
    center = torch.empty((B * H, N // 128, D) if center_key else (1, 1, 1), dtype=torch.float32, device=x.device)

    # Centering leaves the exact softmax unchanged and reduces the key quantization range.
    if center_key:
        launch_delta_matching_kernel("delta_matching_key_sum", launch_key_sum, (x, center))
        if is_forward:
            mean = torch.empty((B * H, 1, D), dtype=torch.float32, device=x.device)
            ticket = scheduler_ticket if scheduler_ticket is not None else mean
            launch_delta_matching_kernel("delta_matching_key_mean", launch_key_fold,
                                         (center, mean, ticket), (scheduler_ticket is not None,))
            center = mean
    launch_delta_matching_kernel(
        "delta_matching_quantize", launch_quant_block, (x, center, codes, scales, twin),
        (block_rows, center_key, input_scale, is_gradient, is_forward),
    )
    return codes, scales, twin
