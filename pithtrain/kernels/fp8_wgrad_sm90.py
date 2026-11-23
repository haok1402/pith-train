import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Float8E4M3FN, Int64
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import select_

from pithtrain.kernels.arch import SM_COUNT
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.fp8_scale_layout import to_mn_major

__all__ = ["fp8_wgrad_sm90"]

BLOCK_K = 128
CACHE = {}


class WeightGradient:
    def __init__(self, M, N, K, rowwise_activation):
        self.block_m = 64 if M <= 64 else 128
        self.block_n = 64 if N <= 64 else (128 if N <= 128 or rowwise_activation else 192)
        self.num_m, self.num_n, self.num_k = cute.ceil_div(M, self.block_m), cute.ceil_div(N, self.block_n), cute.ceil_div(K, BLOCK_K)
        self.rowwise_activation = rowwise_activation
        self.num_activation_blocks = cute.ceil_div(N, BLOCK_K)
        self.cluster_m = 2 if self.num_m % 2 == 0 and self.num_m * self.num_n >= SM_COUNT else 1
        self.grid = min(SM_COUNT // self.cluster_m, self.num_m * self.num_n // self.cluster_m) * self.cluster_m
        self.stages = 3 if self.block_n == 192 else 5
        self.threads_math = 2 * self.block_m
        self.load_warp = self.threads_math // 32
        self.load_regs, self.math_regs = 40, 232 if self.threads_math == 256 else 248

    @cute.jit
    def __call__(self, mGrad: cute.Tensor, mGradScale: cute.Tensor, mActivation: cute.Tensor, mActivationScale: cute.Tensor, mOut: cute.Tensor, stream):
        grad_shape, activation_shape = (self.block_m, BLOCK_K), (self.block_n, BLOCK_K)
        grad_scale_shape, activation_scale_shape = (self.block_m, 1), (self.block_n, 1)
        atom_k = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, Float8E4M3FN)
        grad_layout, activation_layout = cute.tile_to_shape(atom_k, (*grad_shape, self.stages), (0, 1, 2)), cute.tile_to_shape(atom_k, (*activation_shape, self.stages), (0, 1, 2))
        grad_scale_layout, activation_scale_layout = cute.make_layout((*grad_scale_shape, self.stages)), cute.make_layout((*activation_scale_shape, self.stages))
        out_shape = (self.block_m, self.block_n)
        out_layout = cute.make_layout(out_shape, stride=(self.block_n, 1))

        load, multicast = cpasync.CopyBulkTensorTileG2SOp(), cpasync.CopyBulkTensorTileG2SMulticastOp()
        grad_copy = cpasync.make_tiled_tma_atom(load, mGrad, cute.slice_(grad_layout, (None, None, 0)), grad_shape)
        grad_scale_copy = cpasync.make_tiled_tma_atom(load, mGradScale, cute.slice_(grad_scale_layout, (None, None, 0)), grad_scale_shape)
        activation_copy = cpasync.make_tiled_tma_atom(multicast if self.cluster_m == 2 else load, mActivation, cute.slice_(activation_layout, (None, None, 0)), activation_shape, num_multicast=self.cluster_m)
        activation_scale_copy = None
        if cutlass.const_expr(self.rowwise_activation):
            activation_scale_copy = cpasync.make_tiled_tma_atom(load, mActivationScale, cute.slice_(activation_scale_layout, (None, None, 0)), activation_scale_shape)
        out_copy = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mOut, out_layout, out_shape)

        mma = warpgroup.MmaF8Op(Float8E4M3FN, Float8E4M3FN, Float32, (64, self.block_n, 32), warpgroup.OperandSource.SMEM, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K)
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma), (self.block_m // 64, 1, 1))
        layouts = (grad_layout, grad_scale_layout, activation_layout, activation_scale_layout, out_layout)
        copies = (grad_copy, grad_scale_copy, activation_copy, activation_scale_copy, out_copy)
        self.kernel(tiled_mma, layouts, copies, mActivationScale).launch(grid=(self.grid, 1, 1), block=(self.threads_math + 128, 1, 1), cluster=(self.cluster_m, 1, 1), min_blocks_per_mp=1, stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, layouts: tuple, copies: tuple, mActivationScale: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        grad_layout, grad_scale_layout, activation_layout, activation_scale_layout, out_layout = layouts
        smem = cutlass.utils.SmemAllocator()
        s_grad = smem.allocate_tensor(Float8E4M3FN, grad_layout.outer, byte_alignment=1024, swizzle=grad_layout.inner)
        s_activation = smem.allocate_tensor(Float8E4M3FN, activation_layout.outer, byte_alignment=1024, swizzle=activation_layout.inner)
        s_grad_scale, s_out = smem.allocate_tensor(Float32, grad_scale_layout, byte_alignment=16), smem.allocate_tensor(Float32, out_layout, byte_alignment=128)
        s_activation_scale = smem.allocate_tensor(Float32, activation_scale_layout, byte_alignment=16) if cutlass.const_expr(self.rowwise_activation) else None
        s_block_scales = smem.allocate_tensor(Float32, cute.make_layout((self.num_k, 2)), byte_alignment=16) if cutlass.const_expr(not self.rowwise_activation) else None

        barrier_storage = smem.allocate_array(Int64, 2 * self.stages)
        tx_count = (self.block_m + self.block_n) * BLOCK_K + 4 * self.block_m
        if cutlass.const_expr(self.rowwise_activation):
            tx_count += 4 * self.block_n
        producer, consumer = pipeline.CooperativeGroup(pipeline.Agent.Thread), pipeline.CooperativeGroup(pipeline.Agent.Thread, self.cluster_m * self.threads_math // 32)
        cluster_layout = cute.make_layout((1, self.cluster_m, 1, 1))
        load_pipe = pipeline.PipelineTmaAsync.create(barrier_storage=barrier_storage, num_stages=self.stages, producer_group=producer, consumer_group=consumer, tx_count=tx_count, cta_layout_vmnk=cluster_layout)
        inputs = (s_grad, s_grad_scale, s_activation, s_activation_scale)

        if warp >= self.load_warp:
            cute.arch.warpgroup_reg_dealloc(self.load_regs)
            if warp == self.load_warp:
                self.load_loop(bx, copies, inputs, load_pipe)
        else:
            cute.arch.warpgroup_reg_alloc(self.math_regs)
            self.compute_loop(tidx, bx, tiled_mma, load_pipe, inputs, (s_out, s_block_scales), copies[4], mActivationScale)

    @cute.jit
    def tile_coords(self, tile_idx):
        # Visit a band of M tiles before moving to the next N tile to reuse activations in L2.
        tiles_per_band = 16 * self.num_n
        first_m = tile_idx // tiles_per_band * 16
        band_m = cutlass.min(self.num_m - first_m, 16)
        within_band = tile_idx % tiles_per_band
        return first_m + within_band % band_m, within_band // band_m

    @cute.jit
    def load_loop(self, bx, copies, inputs, load_pipe):
        grad_copy, grad_scale_copy, activation_copy, activation_scale_copy, _ = copies
        s_grad, s_grad_scale, s_activation, s_activation_scale = inputs
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stages)
        for tile_idx in cutlass.range(bx, self.num_m * self.num_n, self.grid):
            tile_m, tile_n = self.tile_coords(tile_idx)
            grad = self.partition_copy(grad_copy, s_grad, (self.block_m, BLOCK_K), (tile_m, None))
            grad_scale = self.partition_copy(grad_scale_copy, s_grad_scale, (self.block_m, 1), (tile_m, None))
            activation = self.partition_copy(activation_copy, s_activation, (self.block_n, BLOCK_K), (tile_n, None), self.cluster_m, bx % self.cluster_m)
            activation_scale = None
            if cutlass.const_expr(self.rowwise_activation):
                activation_scale = self.partition_copy(activation_scale_copy, s_activation_scale, (self.block_n, 1), (tile_n, None))
            for k_tile in cutlass.range(self.num_k, unroll=1):
                load_pipe.producer_acquire(state)
                barrier = load_pipe.producer_get_barrier(state)
                self.load_tile(grad_copy[0], grad, barrier, state.index, k_tile)
                self.load_tile(grad_scale_copy[0], grad_scale, barrier, state.index, k_tile)
                if cutlass.const_expr(self.cluster_m == 2):
                    self.load_tile(activation_copy[0], activation, barrier, state.index, k_tile, 3)
                else:
                    self.load_tile(activation_copy[0], activation, barrier, state.index, k_tile)
                if cutlass.const_expr(self.rowwise_activation):
                    self.load_tile(activation_scale_copy[0], activation_scale, barrier, state.index, k_tile)
                state.advance()
        if cutlass.const_expr(self.cluster_m == 2):
            load_pipe.producer_tail(state)

    @cute.jit
    def partition_copy(self, descriptor, smem, tile: cutlass.Constexpr, coord, multicast: cutlass.Constexpr = 1, rank=0):
        atom, gmem = descriptor
        g_tile = cute.local_tile(gmem, tile, coord)
        return cpasync.tma_partition(atom, rank, cute.make_layout(multicast), cute.group_modes(smem, 0, 2), cute.group_modes(g_tile, 0, 2))

    @cute.jit
    def load_tile(self, atom, buffers, barrier, stage, k_tile, mask: cutlass.Constexpr = 0):
        shared, global_tile = buffers
        if cutlass.const_expr(mask):
            cute.copy(atom, global_tile[None, k_tile], shared[None, stage], tma_bar_ptr=barrier, mcast_mask=cutlass.Int16(mask))
        else:
            cute.copy(atom, global_tile[None, k_tile], shared[None, stage], tma_bar_ptr=barrier)

    @cute.jit
    def compute_loop(self, tidx, bx, tiled_mma, load_pipe, inputs, outputs, out_copy, mActivationScale):
        s_grad, s_grad_scale, s_activation, s_activation_scale = inputs
        s_out, s_block_scales = outputs
        mma_slice = tiled_mma.get_slice(tidx)
        grad_desc, activation_desc = tiled_mma.make_fragment_A(mma_slice.partition_A(s_grad)), tiled_mma.make_fragment_B(mma_slice.partition_B(s_activation))
        thread_out = mma_slice.partition_C(s_out)
        acc_tile, acc_wgrad = cute.make_rmem_tensor(thread_out.shape, Float32), cute.make_rmem_tensor(thread_out.shape, Float32)
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        row_lo, col_pair = tidx // 32 * 16 + tidx % 32 // 4, tidx % 4 * 2
        row_hi = row_lo + 8
        for tile_idx in cutlass.range(bx, self.num_m * self.num_n, self.grid):
            tile_m, tile_n = self.tile_coords(tile_idx)
            scale_tile = (tile_n * self.block_n) // BLOCK_K
            if cutlass.const_expr(not self.rowwise_activation):
                self.load_block_scales(tidx, scale_tile, mActivationScale, s_block_scales)
            acc_wgrad.fill(0.0)
            for k_tile in cutlass.range(self.num_k, unroll=1):
                load_pipe.consumer_wait(state)
                stage = state.index
                grad_lo, grad_hi = s_grad_scale[row_lo, 0, stage], s_grad_scale[row_hi, 0, stage]
                if cutlass.const_expr(not self.rowwise_activation):
                    act_lo, act_hi = s_block_scales[k_tile, 0], s_block_scales[k_tile, 1]
                self.mma_tile(tiled_mma, acc_tile, grad_desc, activation_desc, stage)
                if cutlass.const_expr(self.rowwise_activation):
                    self.promote_rows(acc_wgrad, acc_tile, (grad_lo, grad_hi), s_activation_scale, col_pair, stage)
                load_pipe.consumer_release(state)
                if cutlass.const_expr(not self.rowwise_activation):
                    self.promote_blocks(acc_wgrad, acc_tile, (grad_lo, grad_hi), (act_lo, act_hi), tile_n)
                state.advance()
            self.store_tile(tidx, acc_wgrad, thread_out, s_out, out_copy, (tile_m, tile_n))
        if tidx == 0:
            cute.arch.cp_async_bulk_wait_group(0)

    @cute.jit
    def load_block_scales(self, tidx, scale_tile, mActivationScale, s_block_scales):
        for k_tile in cutlass.range(tidx, self.num_k, self.threads_math):
            s_block_scales[k_tile, 0] = mActivationScale[scale_tile, k_tile]
            if cutlass.const_expr(self.block_n > BLOCK_K):
                next_scale = cutlass.min(scale_tile + 1, self.num_activation_blocks - 1)
                s_block_scales[k_tile, 1] = mActivationScale[next_scale, k_tile]
            else:
                s_block_scales[k_tile, 1] = 0.0
        cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_math)

    @cute.jit
    def mma_tile(self, tiled_mma, acc_tile, grad_desc, activation_desc, stage):
        # Reset WGMMA once per scale block and promote only after all four K32 instructions.
        warpgroup.fence()
        for mma_k in cutlass.range_constexpr(BLOCK_K // 32):
            tiled_mma.set(warpgroup.Field.ACCUMULATE, mma_k != 0)
            cute.gemm(tiled_mma, acc_tile, grad_desc[None, None, mma_k, stage], activation_desc[None, None, mma_k, stage], acc_tile)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

    @cute.jit
    def promote_blocks(self, acc_wgrad, acc_tile, grad_scales, activation_scales, tile_n):
        grad_lo, grad_hi = grad_scales
        act_lo, act_hi = activation_scales
        scale_lo, scale_hi = grad_lo * act_lo, grad_hi * act_lo
        if cutlass.const_expr(self.block_n > BLOCK_K):
            next_lo, next_hi = grad_lo * act_hi, grad_hi * act_hi
            boundary = BLOCK_K - (tile_n * self.block_n) % BLOCK_K
        for col_group in cutlass.range_constexpr(self.block_n // 8):
            row_lo_scale, row_hi_scale = scale_lo, scale_hi
            if cutlass.const_expr(self.block_n > BLOCK_K):
                row_lo_scale, row_hi_scale = select_(col_group * 8 < boundary, scale_lo, next_lo), select_(col_group * 8 < boundary, scale_hi, next_hi)
            self.promote_pair(acc_wgrad, acc_tile, col_group, (row_lo_scale, row_hi_scale), (row_lo_scale, row_hi_scale))

    @cute.jit
    def promote_rows(self, acc_wgrad, acc_tile, grad_scales, s_activation_scale, col_pair, stage):
        grad_lo, grad_hi = grad_scales
        for col_group in cutlass.range_constexpr(self.block_n // 8):
            col_lo = col_group * 8 + col_pair
            act_lo, act_hi = s_activation_scale[col_lo, 0, stage], s_activation_scale[col_lo + 1, 0, stage]
            self.promote_pair(acc_wgrad, acc_tile, col_group, (grad_lo * act_lo, grad_hi * act_lo), (grad_lo * act_hi, grad_hi * act_hi))

    @cute.jit
    def promote_pair(self, acc_wgrad, acc_tile, col_group, scales_lo, scales_hi):
        scale_lo_lo, scale_hi_lo = scales_lo
        scale_lo_hi, scale_hi_hi = scales_hi
        scales = (scale_lo_lo, scale_lo_hi, scale_hi_lo, scale_hi_hi)
        for entry in cutlass.range_constexpr(4):
            slot = col_group * 4 + entry
            acc_wgrad[slot] = Float32(math.fma(acc_tile[slot].ir_value(), scales[entry].ir_value(), acc_wgrad[slot].ir_value()))

    @cute.jit
    def store_tile(self, tidx, acc_wgrad, thread_out, s_out, out_copy, coord):
        if tidx == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_math)
        # Preserve the original reduce-add's positive zero without flushing subnormals.
        acc_wgrad.store(acc_wgrad.load() + Float32(0.0))
        cute.autovec_copy(acc_wgrad, thread_out)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=1, number_of_threads=self.threads_math)
        if tidx == 0:
            shared, global_tile = self.partition_copy(out_copy, s_out, (self.block_m, self.block_n), coord)
            cute.copy(out_copy[0], shared, global_tile)
            cute.arch.cp_async_bulk_commit_group()


def fp8_wgrad_sm90(grad, grad_scale, activation, activation_scale, out):
    """
    Use Hopper WGMMA with one FP32 promotion per 128 reduction elements.
    """
    M, K = grad.shape
    N = activation.shape[0]
    num_blocks = (K + BLOCK_K - 1) // BLOCK_K
    rowwise_activation = activation_scale.shape == (N, num_blocks)
    if activation.shape != (N, K) or out.shape != (M, N) or K % 16 or N % 8:
        raise ValueError("fp8_wgrad requires matching (M, K), (N, K) and (M, N) shapes, K % 16 == 0 and N % 8 == 0")
    if grad_scale.shape != (M, num_blocks) or not (rowwise_activation or activation_scale.shape == ((N + BLOCK_K - 1) // BLOCK_K, num_blocks)):
        raise ValueError("fp8_wgrad scales must cover the corresponding row or tile for each 128 tokens")
    if grad.dtype != torch.float8_e4m3fn or activation.dtype != grad.dtype or any(tensor.dtype != torch.float32 for tensor in (grad_scale, activation_scale, out)):
        raise TypeError("fp8_wgrad requires FP8 E4M3 inputs and FP32 scales and output")
    if not out.is_contiguous() or not out.is_cuda or any(tensor.device != out.device for tensor in (grad, grad_scale, activation, activation_scale)):
        raise ValueError("fp8_wgrad requires a contiguous output and tensors on the same CUDA device")
    if not M or not N:
        return
    if not K:
        out.zero_()
        return

    grad, activation = grad.contiguous(), activation.contiguous()
    grad_scale = to_mn_major(grad_scale)
    activation_scale = to_mn_major(activation_scale) if rowwise_activation else activation_scale.contiguous()

    tensors = (grad, grad_scale, activation, activation_scale, out)
    key = ("fp8_wgrad", M, N, K, rowwise_activation, tuple(tensor.stride() for tensor in tensors))
    fn = CACHE.get(key)
    if fn is None:
        dtypes = (Float8E4M3FN, Float32, Float8E4M3FN, Float32, Float32)
        operands = tuple(fake(dtype, tuple(tensor.shape), tensor.stride(), 4 if index == 3 and not rowwise_activation else 16) for index, (dtype, tensor) in enumerate(zip(dtypes, tensors)))
        fn = cute.compile(WeightGradient(M, N, K, rowwise_activation), *operands, fake_stream(), options="--enable-tvm-ffi")
        CACHE[key] = fn
    fn(*tensors)
