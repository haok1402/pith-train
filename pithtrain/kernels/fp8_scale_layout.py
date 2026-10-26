"""
The 1x128 FP8 scale layout DeepGEMM's SM90 GEMMs read without converting.

Every SM90 FP8 GEMM passes each 1x128 scale (the SFA of the dense and m-grouped GEMMs, both
operands of the k-grouped GEMM) through ``get_mn_major_tma_aligned_tensor``, which launches a
``transpose_fp32`` kernel unless the scale of logical shape ``(mn, sf_k)`` already has strides
``(1, align(mn, 4))``: MN-major, each 128-wide K block a contiguous column padded to 16 bytes.
Quantization kernels write that layout directly; the values are unchanged, only where
``scale[r, j]`` lives (``j * mn_pad + r``). 128x128 weight scales are read as they are and stay
row-major. See workspace/kda-tasks/round2/scale_layout/SCALE_LAYOUT.md.
"""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["act_scale_like", "is_mn_major", "mn_major_scale", "mn_pad", "to_mn_major"]


def mn_pad(mn: int) -> int:
    """Row count of one MN-major column: mn rounded up to 16 bytes of fp32."""
    return (mn + 3) & ~3


def mn_major_scale(mn: int, sf_k: int, device: torch.device) -> torch.Tensor:
    """Uninitialized fp32 ``(mn, sf_k)`` scale with strides ``(1, mn_pad(mn))``. The storage ends
    at the last valid element, exactly as DeepGEMM allocates its own transposed copy; a single
    empty_strided is also the cheapest allocation on the host (~1.7 us vs ~2 us for empty)."""
    return torch.empty_strided((mn, sf_k), (1, mn_pad(mn)), dtype=torch.float32, device=device)


def is_mn_major(sf: torch.Tensor) -> bool:
    """Whether DeepGEMM takes the 2D scale *sf* as it is."""
    return sf.dim() == 2 and sf.stride(0) == 1 and sf.stride(1) == mn_pad(sf.shape[0])


def to_mn_major(sf: torch.Tensor) -> torch.Tensor:
    """*sf* in the MN-major layout (a copy unless it already is), for paths that produce a
    row-major scale, so every caller sees one layout."""
    if is_mn_major(sf):
        return sf
    out = mn_major_scale(sf.shape[0], sf.shape[1], sf.device)
    out.copy_(sf)
    return out


def act_scale_like(mn: int, sf_k: int, device: torch.device) -> torch.Tensor:
    """
    Uninitialized fp32 tensor of logical shape ``(mn, sf_k)`` on ``device`` (any device, ``meta``
    included) with the strides this library's quantization ops give a 1x128 activation scale of
    that shape: MN-major (``(1, align(mn, 4))``, above) when the process's device is SM90,
    row-major contiguous otherwise. For fake / meta implementations of ops that return such a
    scale; callers never read the strides. The architecture is the one probed at import (the
    current CUDA device then), not ``device``'s.
    """
    if ARCH_MAJOR == 9:
        return mn_major_scale(mn, sf_k, device)
    return torch.empty((mn, sf_k), dtype=torch.float32, device=device)
