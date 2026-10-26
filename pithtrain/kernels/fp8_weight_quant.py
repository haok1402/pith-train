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
from pithtrain.kernels.fp8_weight_quant_sm100 import (fp8_pack_transpose_sm100, fp8_scales_sm100, fp8_shard_amax_sm100, fp8_shard_cast_sm100, fp8cast_blockwise_transpose_batched_sm100, fp8cast_blockwise_transpose_sm100)

__all__ = ["fp8_pack_transpose", "fp8_scales", "fp8_scales_multi", "fp8_shard_amax", "fp8_shard_amax_multi", "fp8_shard_cast", "fp8cast_blockwise_transpose", "fp8cast_blockwise_transpose_batched"]


def fp8cast_blockwise_transpose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    128x128 blockwise FP8 of a weight and of its transpose from one read: ``(w_fp8, w_scale,
    wt_fp8, wt_scale)``.

    - ``x``: 2-D ``(N, K)`` bf16 (fp32 also accepted), any N and K (a partial tile is quantized
      over its valid elements), unit stride along K (rows may be strided, e.g. a column slice; a
      transposed view is not supported: pass ``.contiguous()``). Not modified.
    - ``w_fp8``: e4m3 ``(N, K)`` contiguous; ``w_scale``: fp32 ``(ceil(N / 128), ceil(K / 128))``
      row-major, one scale per 128x128 tile.
    - ``wt_fp8``: e4m3 ``(K, N)`` contiguous, exactly ``w_fp8.T`` (the same tile scales);
      ``wt_scale``: fp32 ``(ceil(K / 128), ceil(N / 128))`` row-major, ``w_scale.T``.

    Per tile: ``amax = max(|x|)``, ``scale = 2^ceil(log2(max(amax, 1e-4) / 448))`` (E8M0,
    exact) and ``fp8 = e4m3(x / scale)`` rounded to nearest even with saturation. Same Triton
    kernel on both architectures; deterministic; fresh outputs.
    """
    return fp8cast_blockwise_transpose_sm100(x)


def fp8cast_blockwise_transpose_batched(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :func:`fp8cast_blockwise_transpose` of each expert of a grouped weight: ``x`` 3-D
    ``(G, N, K)`` bf16 (fp32 also accepted; unit stride along K, as in the 2-D op) gives
    ``w_fp8`` e4m3 ``(G, N, K)``, ``w_scale`` fp32 ``(G, ceil(N / 128), ceil(K / 128))``, ``wt_fp8`` e4m3 ``(G, K, N)`` (each
    expert's ``w_fp8[g].T``) and ``wt_scale`` fp32 ``(G, ceil(K / 128), ceil(N / 128))``, all
    contiguous (scales row-major). Tiles never straddle experts. Bit-identical to the 2-D op per
    expert; same Triton kernel on both architectures; deterministic.
    """
    return fp8cast_blockwise_transpose_batched_sm100(x)


def fp8_shard_amax(shard: torch.Tensor, amax: torch.Tensor, row_offset: int, group_offset: int) -> None:  # fmt: skip
    """
    In place: the 128x128-tile amax of one row shard of a ``(G, N, K)`` weight, written into the
    shared amax grid.

    - ``shard``: fp32 ``(g_local, local_rows, K)`` contiguous, rows ``[row_offset, row_offset +
      local_rows)`` of experts ``[group_offset, group_offset + g_local)`` of the weight (a 2-D
      weight is ``G = 1``). Not modified. Empty shards launch nothing.
    - ``amax``: fp32 ``(G, ceil(N / 128), ceil(K / 128))`` with contiguous rows. For every tile the
      shard touches, the entry is overwritten with ``max(|bf16(x)|)`` over the shard's elements in
      that tile (the values rounded to bf16 first, as FSDP's mixed-precision cast does); every
      other entry is left alone. Zero the grid first, then a MAX all-reduce over the shards gives
      the amax of the whole bf16 weight.

    Exact (a max); bit-identical on both architectures for finite inputs (NaN is out of
    contract). Deterministic.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_amax_sm90

        return fp8_shard_amax_sm90(shard, amax, row_offset, group_offset)
    fp8_shard_amax_sm100(shard, amax, row_offset, group_offset)


def fp8_shard_cast(shard: torch.Tensor, amax: torch.Tensor, out: torch.Tensor, row_offset: int, group_offset: int) -> None:  # fmt: skip
    """
    In place: the FP8 cast of one row shard with the scales of the reduced amax grid.

    - ``shard``, ``row_offset``, ``group_offset``: as in :func:`fp8_shard_amax`; ``amax``: the grid
      after the MAX all-reduce (read only).
    - ``out``: e4m3 tensor of ``shard``'s shape, contiguous, overwritten with ``e4m3(bf16(x) /
      scale)`` (rounded to nearest even with saturation), ``scale = 2^ceil(log2(max(amax, 1e-4)
      / 448))`` the E8M0 scale of the element's tile amax.

    The gathered shards are therefore bit-identical to :func:`fp8cast_blockwise_transpose_batched`'s
    ``w_fp8`` of the full bf16-rounded weight. Same bytes on both architectures; deterministic.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_cast_sm90

        return fp8_shard_cast_sm90(shard, amax, out, row_offset, group_offset)
    fp8_shard_cast_sm100(shard, amax, out, row_offset, group_offset)


def fp8_scales(amax: torch.Tensor, scale: torch.Tensor, scale_t: torch.Tensor) -> None:
    """
    In place: ``scale`` fp32 ``(G, R, C)`` and ``scale_t`` fp32 ``(G, C, R)``, both contiguous,
    overwritten with the E8M0 scale ``2^ceil(log2(max(amax, 1e-4) / 448))`` of each entry of the
    contiguous fp32 ``(G, R, C)`` amax grid (``scale_t[g] = scale[g].T``). Exactly the scales
    :func:`fp8_shard_cast` uses, and bit-identical to :func:`fp8cast_blockwise_transpose_batched`'s
    ``w_scale`` / ``wt_scale`` of the same weight. Same values on both architectures.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_scales_sm90

        return fp8_scales_sm90(amax, scale, scale_t)
    fp8_scales_sm100(amax, scale, scale_t)


def fp8_pack_transpose(x: torch.Tensor, out: torch.Tensor) -> None:
    """
    In place: the bytes of ``x``, an FP8 (one byte per element) ``(G, N, K)`` contiguous tensor,
    and of its per-expert transpose ``(G, K, N)``, written back to back into ``out``: a contiguous
    one-byte-element buffer of at least ``2 * x.numel()`` elements, ``out[:x.numel()]`` the bytes of
    ``x`` and the next ``x.numel()`` those of ``x.transpose(1, 2).contiguous()``. A pure byte copy
    (exact) on both architectures; ``x`` is not modified.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_pack_transpose_sm90

        return fp8_pack_transpose_sm90(x, out)
    fp8_pack_transpose_sm100(x, out)


def fp8_shard_amax_multi(shards: Sequence[torch.Tensor], amaxes: Sequence[torch.Tensor], row_offsets: Sequence[int], group_offsets: Sequence[int]) -> Optional[Any]:
    """
    A callable that runs :func:`fp8_shard_amax` for every ``(shard, amax, row_offset,
    group_offset)`` of the lists in one launch (with a separate launch per shard it cannot take:
    ``K`` not a multiple of 4 or a shard not 16-byte aligned), or None on SM100, where the caller
    loops over :func:`fp8_shard_amax` itself.

    Build it once: it records the tensors' addresses, so it stays valid only while none of them is
    reallocated. Each call writes the same values as the per-shard loop.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_shard_amax_multi_sm90

        return fp8_shard_amax_multi_sm90(shards, amaxes, row_offsets, group_offsets)
    return None


def fp8_scales_multi(amaxes: Sequence[torch.Tensor], scales: Sequence[torch.Tensor], scales_t: Sequence[torch.Tensor]) -> Optional[Any]:
    """
    A callable that runs :func:`fp8_scales` for every ``(amax, scale, scale_t)`` of the lists in
    one launch, or None on SM100, where the caller loops over :func:`fp8_scales` itself.

    Build it once: it records the tensors' addresses, so it stays valid only while none of them is
    reallocated. Each call writes the same values as the per-grid loop.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_weight_quant_sm90 import fp8_scales_multi_sm90

        return fp8_scales_multi_sm90(amaxes, scales, scales_t)
    return None
