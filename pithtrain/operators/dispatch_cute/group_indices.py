"""
Per-row group ids of a grouped GEMM input (SM90), the Hopper FP8 experts' ``group_indices``, in one
launch: ``gi[r] = min(#{g : offs[g] <= r}, G - 1)``, exactly what ``precompute_group_indices``
computes with ``arange`` + ``searchsorted(right=True)`` + ``clamp`` (three launches). Each CTA
loads the inclusive group ends into shared memory and every row binary-searches them.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.operators.dispatch_cute.common import fake, fake_stream
from pithtrain.operators.fp8_cute.rowwise_kmajor import _ld_b32, _lds_b32, _st_b32, _sts_b32

__all__ = ["group_indices", "supports_group_indices"]

_THREADS = 256
_ROWS_PER_THREAD = 8
_MAX_GROUPS = 1024


class _GroupIndices:
    def __init__(self, num_groups: int) -> None:
        assert 0 < num_groups <= _MAX_GROUPS
        self.G = num_groups

    @cute.jit
    def __call__(self, mOffs: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        per_cta = _THREADS * _ROWS_PER_THREAD
        grid = (mOut.shape[0] + per_cta - 1) // per_cta
        self.kernel(mOffs, mOut).launch(grid=[grid, 1, 1], block=[_THREADS, 1, 1], stream=stream)

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
        for j in cutlass.range_constexpr((G + _THREADS - 1) // _THREADS):
            g = j * _THREADS + tid
            v = _ld_b32(offs + Int64(cutlass.min(g, G - 1)) * 4, Int32(g < G), Int32(0))
            _sts_b32(s_offs + cutlass.min(g, G - 1) * 4, Uint32(v), Int32(g < G))
        cute.arch.sync_threads()
        top = 1 << (G.bit_length() - 1)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            r = (bid * _ROWS_PER_THREAD + i) * _THREADS + tid
            # count of group ends <= r: binary search for the first end > r
            cnt = Int32(0)
            b = top
            while b > 0:
                nxt = cnt + b
                inb = Int32(nxt <= G)
                end = Int32(_lds_b32(s_offs + cutlass.min(nxt - 1, G - 1) * 4))
                cnt = cnt + b * (inb & Int32(end <= r))
                b //= 2
            _st_b32(out + Int64(r) * 4, Uint32(cutlass.min(cnt, G - 1)), Int32(r < M))


_GI: dict = {}


def _compile(G: int) -> Callable:
    return cute.compile(
        _GroupIndices(G),
        fake(Int32, (G,), (1,), 4),
        fake(Int32, (cute.sym_int(),), (1,), 4),
        fake_stream(),
        options="--enable-tvm-ffi",
    )


def supports_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> bool:
    return (
        grouped_mm_offs.dtype == torch.int32
        and grouped_mm_offs.dim() == 1
        and 0 < grouped_mm_offs.shape[0] <= _MAX_GROUPS
        and grouped_mm_offs.is_contiguous()
        and 0 <= M < 2**31
        and not torch.compiler.is_compiling()
    )


def group_indices(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor:
    """``(M,)`` int32 group id of every row, from the inclusive group ends ``grouped_mm_offs``."""
    G = grouped_mm_offs.shape[0]
    out = torch.empty(M, dtype=torch.int32, device=grouped_mm_offs.device)
    if M == 0:
        return out
    fn = _GI.get(G)
    if fn is None:
        fn = _GI[G] = _compile(G)
    fn(grouped_mm_offs, out)
    return out
