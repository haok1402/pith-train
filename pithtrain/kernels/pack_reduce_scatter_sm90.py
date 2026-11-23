"""Pack FSDP FP32 reduce-scatter inputs in rank order on SM90.

Aligned FP32 ranges use a bulk copy through shared memory. Scalar copies handle
unaligned ranges and widen BF16 input gradients into the FP32 destination.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8
from cutlass.cutlass_dsl import dsl_user_op

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import asm, to_ir


__all__ = ["pack_reduce_scatter_sm90"]

WORLD = 8
THREADS = 256
TILE_BYTES = 65536
TILE = TILE_BYTES // 4


@dsl_user_op
def copy_tile_bulk(src: Int64, dst: Int64, staging: Int32, barrier: Int32, size: Int32, *, loc=None, ip=None):
    operands = [
        to_ir(value, typ, loc, ip)
        for value, typ in ((src, Int64), (dst, Int64), (staging, Int32), (barrier, Int32), (size, Int32))
    ]
    asm(
        None, operands,
        "{\n.reg .pred ready;\n"
        "mbarrier.init.shared::cta.b64 [$3], 1;\n"
        "fence.mbarrier_init.release.cluster;\n"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$3], $4;\n"
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$2], [$0], $4, [$3];\n"
        "WAIT_PACK: mbarrier.try_wait.parity.shared::cta.b64 ready, [$3], 0;\n"
        "@!ready bra WAIT_PACK;\n"
        "cp.async.bulk.global.shared::cta.bulk_group [$1], [$2], $4;\n"
        "cp.async.bulk.commit_group;\n"
        "cp.async.bulk.wait_group 0;\n}",
        "l,l,r,r,r", side_effects=True,
    )


class ReduceScatterPack:
    """Copy a tile of a single parameter's shard into its owner's output row."""
    def __init__(self, chunk_sizes: tuple[int, ...], input_bf16: tuple[bool, ...]) -> None:
        self.chunk_sizes = chunk_sizes
        self.input_bf16 = input_bf16
        self.peer_width = sum(chunk_sizes)
        self.tiles = tuple((length + TILE - 1) // TILE for length in chunk_sizes)

    @cute.jit
    def __call__(self, mInputs: tuple, mOutput: cute.Tensor, stream) -> None:
        self.copy_kernel(mInputs, mOutput).launch(
            grid=[sum(self.tiles), WORLD, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def copy_kernel(self, mInputs: tuple, mOutput: cute.Tensor) -> None:
        block, peer, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        shared = cutlass.utils.SmemAllocator()
        staging = shared.allocate_array(Uint8, TILE_BYTES, byte_alignment=128).toint()
        barrier = shared.allocate_array(Int64, 1, byte_alignment=8).toint()
        first_block = 0
        offset = 0
        out_base = mOutput.iterator.toint() + Int64(peer) * (self.peer_width * 4)
        for idx in cutlass.range_constexpr(len(self.chunk_sizes)):
            length = self.chunk_sizes[idx]
            last_block = first_block + self.tiles[idx]
            if (first_block <= block) & (block < last_block):
                element_bytes = 2 if cutlass.const_expr(self.input_bf16[idx]) else 4
                src_base = mInputs[idx].iterator.toint() + Int64(peer) * (length * element_bytes)
                self.copy_tile(src_base, out_base + offset * 4, block - first_block, tidx, length, staging, barrier, self.input_bf16[idx],
                               not self.input_bf16[idx] and offset % 4 == 0 and self.peer_width % 4 == 0 and length % 4 == 0)
            first_block = last_block
            offset += length

    @cute.jit
    def copy_tile(self, src: Int64, dst: Int64, block: Int32, tidx: Int32, length: cutlass.Constexpr[int], staging: Int32, barrier: Int32, bf16: cutlass.Constexpr[bool], vectorized: cutlass.Constexpr[bool]) -> None:
        base = block * TILE
        if cutlass.const_expr(vectorized):
            if tidx == 0:
                size = cutlass.min(TILE, length - base) * 4
                byte_offset = Int64(base) * 4
                copy_tile_bulk(src + byte_offset, dst + byte_offset, staging, barrier, size)
        else:
            for lane_idx in range(cute.ceil_div(cutlass.min(TILE, length - base), THREADS)):
                index = base + tidx + lane_idx * THREADS
                if index < length:
                    if cutlass.const_expr(bf16):
                        src_ptr = cute.make_ptr(BFloat16, src + Int64(index) * 2, cute.AddressSpace.gmem, assumed_align=2)
                        dst_ptr = cute.make_ptr(Float32, dst + Int64(index) * 4, cute.AddressSpace.gmem, assumed_align=4)
                        cute.make_tensor(dst_ptr, cute.make_layout(1))[0] = Float32(cute.make_tensor(src_ptr, cute.make_layout(1))[0])
                    else:
                        src_ptr = cute.make_ptr(cutlass.Uint32, src + Int64(index) * 4, cute.AddressSpace.gmem, assumed_align=4)
                        dst_ptr = cute.make_ptr(cutlass.Uint32, dst + Int64(index) * 4, cute.AddressSpace.gmem, assumed_align=4)
                        cute.make_tensor(dst_ptr, cute.make_layout(1))[0] = cute.make_tensor(src_ptr, cute.make_layout(1))[0]


COMPILED_PACK: dict[tuple, Callable] = {}


def pack_reduce_scatter_sm90(grads: list[torch.Tensor], output: torch.Tensor, world_size: int) -> bool:
    """Pack aligned large FSDP buckets and return whether the launch was supported."""
    if (
        world_size != WORLD or output.dtype != torch.float32 or output.numel() < 2**24
        or not output.is_cuda or not output.is_contiguous() or output.data_ptr() % 16
    ):
        return False
    if any(
        grad.dtype not in (torch.bfloat16, torch.float32)
        or grad.ndim == 0 or grad.size(0) % WORLD
        or grad.device != output.device or not grad.is_contiguous() or grad.data_ptr() % 16
        for grad in grads
    ):
        return False

    lengths = tuple(grad.numel() // WORLD for grad in grads)
    if 0 in lengths or sum(lengths) * WORLD != output.numel():
        return False
    input_bf16 = tuple(grad.dtype == torch.bfloat16 for grad in grads)
    key = lengths, input_bf16, output.device.index
    if key not in COMPILED_PACK:
        inputs = tuple(
            fake(BFloat16 if use_bf16 else Float32, (length * WORLD,), (1,), 16)
            for length, use_bf16 in zip(lengths, input_bf16, strict=True)
        )
        out = fake(Float32, (output.numel(),), (1,), 16)
        COMPILED_PACK[key] = cute.compile(
            ReduceScatterPack(lengths, input_bf16), inputs, out, fake_stream(), options="--enable-tvm-ffi"
        )
    COMPILED_PACK[key](tuple(grad.view(-1) for grad in grads), output.view(-1))
    return True
