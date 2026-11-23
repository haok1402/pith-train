"""Pack supported FP32 FSDP reduce-scatter buckets."""

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["pack_reduce_scatter"]


def pack_reduce_scatter(grads: list[torch.Tensor], output: torch.Tensor, world_size: int) -> bool:
    """Return whether the bucket was packed into output."""
    if ARCH_MAJOR != 9:
        return False
    from pithtrain.kernels.pack_reduce_scatter_sm90 import pack_reduce_scatter_sm90

    return pack_reduce_scatter_sm90(grads, output, world_size)
