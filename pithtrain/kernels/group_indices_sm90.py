"""
Per-row group ids of a grouped GEMM input in one launch (SM90): gi[r] = min(#{g : offs[g] <= r}, G -
1), the same values as arange + searchsorted(right=True) + clamp.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import ld_b32, lds_b32, st_b32, sts_b32

__all__ = ["group_indices", "supports_group_indices"]

THREADS, ROWS_PER_THREAD, MAX_GROUPS = 256, 8, 1024


class GroupIndices:
    def __init__(self, num_groups: int) -> None:
        assert 0 < num_groups <= MAX_GROUPS
        self.G = num_groups

    @cute.jit
    def __call__(self, mOffs: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        per_cta = THREADS * ROWS_PER_THREAD
        grid = (mOut.shape[0] + per_cta - 1) // per_cta
        self.kernel(mOffs, mOut).launch(grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mOffs: cute.Tensor, mOut: cute.Tensor) -> None:
        G = self.G
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        M = mOut.shape[0]
        offs = mOffs.iterator.toint()
        out = mOut.iterator.toint()
        smem = cutlass.memory.SmemAllocator()
        s_offs = smem.allocate(G * 4, byte_alignment=16).toint()
        for j in cutlass.range_constexpr((G + THREADS - 1) // THREADS):
            g = j * THREADS + tid
            v = ld_b32(offs + Int64(cutlass.min(g, G - 1)) * 4, Int32(g < G), Int32(0))
            sts_b32(s_offs + cutlass.min(g, G - 1) * 4, Uint32(v), Int32(g < G))
        cute.arch.sync_threads()
        top = 1 << (G.bit_length() - 1)
        for i in cutlass.range_constexpr(ROWS_PER_THREAD):
            r = (bid * ROWS_PER_THREAD + i) * THREADS + tid
            # count of group ends <= r: binary search for the first end > r
            cnt = Int32(0)
            b = top
            while b > 0:
                nxt = cnt + b
                inb, end = Int32(nxt <= G), Int32(lds_b32(s_offs + cutlass.min(nxt - 1, G - 1) * 4))
                cnt = cnt + b * (inb & Int32(end <= r))
                b //= 2
            st_b32(out + Int64(r) * 4, Uint32(cutlass.min(cnt, G - 1)), Int32(r < M))


GI: dict = {}


def compile_kernel(G: int) -> Callable:
    return cute.compile(GroupIndices(G), fake(Int32, (G,), (1,), 4), fake(Int32, (cute.sym_int(),), (1,), 4), fake_stream(), options="--enable-tvm-ffi")


def supports_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> bool:
    return (grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.dim() == 1 and 0 < grouped_mm_offs.shape[0] <= MAX_GROUPS and grouped_mm_offs.is_contiguous() and 0 <= M < 2**31 and not torch.compiler.is_compiling())


def group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor:
    """(M,) int32 group id of every row, from the inclusive group ends grouped_mm_offs."""
    G = grouped_mm_offs.shape[0]
    out = torch.empty(M, dtype=torch.int32, device=grouped_mm_offs.device)
    if M == 0:
        return out
    fn = GI.get(G)
    if fn is None:
        fn = GI[G] = compile_kernel(G)
    fn(grouped_mm_offs, out)
    return out
