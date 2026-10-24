"""
Picker of the weight-side 128x128 blockwise FP8 quantization: the blockwise transposes of the
weights, and the FP8 parameter all-gather kernels of ``modules/fp8_param_gather.py``.

Both architectures run the Triton kernels (``fp8_weight_quant_sm100``) for the blockwise
transposes: the bit-identical SM90 ports in ``fp8_weight_quant_sm90`` are not wired in (only the
tests run them). The parameter all-gather kernels run the CuTe-DSL ports on SM90 and the Triton
kernels elsewhere; the one-launch-per-shard-group ``fp8_shard_amax_multi`` / ``fp8_scales_multi``
exist on SM90 only and return None elsewhere, where the caller keeps its per-tensor loop. Every
scale is row-major on both architectures.
"""

from typing import Any, Optional, Sequence

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_weight_quant_sm100 import (
    fp8_pack_transpose_sm100,
    fp8_scales_sm100,
    fp8_shard_amax_sm100,
    fp8_shard_cast_sm100,
    fp8cast_blockwise_transpose_batched_sm100,
    fp8cast_blockwise_transpose_sm100,
)

__all__ = [
    "fp8_pack_transpose",
    "fp8_scales",
    "fp8_scales_multi",
    "fp8_shard_amax",
    "fp8_shard_amax_multi",
    "fp8_shard_cast",
    "fp8cast_blockwise_transpose",
    "fp8cast_blockwise_transpose_batched",
]


def fp8cast_blockwise_transpose(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """See fp8cast_blockwise_transpose_sm100 (both architectures)."""
    return fp8cast_blockwise_transpose_sm100(x)


def fp8cast_blockwise_transpose_batched(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """See fp8cast_blockwise_transpose_batched_sm100 (both architectures)."""
    return fp8cast_blockwise_transpose_batched_sm100(x)


def fp8_shard_amax(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:  # fmt: skip
    """
    Write the per-tile amax of the bf16-rounded shard into amax at the tiles it touches.

    amax must be zero where other shards contribute; a MAX across shards then gives the full grid.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_amax_sm90

        return fp8_shard_amax_sm90(shard, amax, row_offset, group_offset)
    fp8_shard_amax_sm100(shard, amax, row_offset, group_offset)


def fp8_shard_cast(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:  # fmt: skip
    """
    Cast the shard to FP8 into out (same shape) with the scales of the reduced amax grid.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_cast_sm90

        return fp8_shard_cast_sm90(shard, amax, out, row_offset, group_offset)
    fp8_shard_cast_sm100(shard, amax, out, row_offset, group_offset)


def fp8_scales(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """
    The (G, R, C) scale grid and its (G, C, R) transpose from a (G, R, C) amax grid.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_scales_sm90

        return fp8_scales_sm90(amax, scale, scale_t)
    fp8_scales_sm100(amax, scale, scale_t)


def fp8_pack_transpose(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    Write x (G, N, K) and its (G, K, N) transpose back to back into the flat byte buffer out.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_pack_transpose_sm90

        return fp8_pack_transpose_sm90(x, out)
    fp8_pack_transpose_sm100(x, out)


def fp8_shard_amax_multi(
    shards: Sequence[torch.Tensor],
    amaxes: Sequence[torch.Tensor],
    row_offsets: Sequence[int],
    group_offsets: Sequence[int],
) -> Optional[Any]:
    """One launch object computing :func:`fp8_shard_amax` of every shard (SM90), or None."""
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_amax_multi_sm90

        return fp8_shard_amax_multi_sm90(shards, amaxes, row_offsets, group_offsets)
    return None


def fp8_scales_multi(
    amaxes: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    scales_t: Sequence[torch.Tensor],
) -> Optional[Any]:
    """One launch object computing :func:`fp8_scales` of every grid (SM90), or None."""
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_scales_multi_sm90

        return fp8_scales_multi_sm90(amaxes, scales, scales_t)
    return None
