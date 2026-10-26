"""
Picker of the MoE stage-3 input path for FP8 experts: dedup expansion + scatter by expert +
rowwise / K-major FP8 quantization, from the received tokens to the quantized grouped GEMM input.

SM90 runs the fused CuTe-DSL kernels (``fp8_scatter_quant_sm90``, stable row placement), elsewhere
the unfused Triton path (``fp8_scatter_quant_sm100``, rows placed in ``atomic_add`` order). The
SM90 kernels return both scales MN-major (``kernels/fp8_scale_layout.py``), the SM100 path
row-major.
"""

from typing import Optional

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_scatter_quant_sm100 import scatter_quant_sm100

__all__ = ["scatter_quant"]


def scatter_quant(
    gathered: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    expand_idx: Optional[torch.Tensor] = None,
    padding_alignment: int = 128,
    group_counts: Optional[list[int]] = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[int],
    torch.Tensor,
]:
    """
    The MoE stage-3 FP8 input in one pass: ``x = gathered[expand_idx]`` scattered by expert as
    :func:`scatter_by_expert` does, then quantized as :func:`quantize_grouped_input` does on SM90.
    Returns ``((x_fp8, x_scale, x_ch_fp8, x_ch_scale), reverse_shuffle_idxs, grouped_mm_offs, ks,
    ks_tensor)``.

    - ``gathered``: bf16 ``(m_src, N)`` received (deduplicated) tokens; ``expert_idxs``: int64
      ``(m,)``, the local expert (``< num_groups``) of each token slot; ``expand_idx``: int64
      ``(m,)``, the row of ``gathered`` of each slot, or None for the identity (then
      ``m_src == m``, else ValueError). None of them is modified.
    - ``padding_alignment`` must be 128 (asserted). ``group_counts``: the true per-expert slot
      counts on the host, if known; it spares the device-to-host copy (and host sync) of the padded
      sizes and must match ``expert_idxs``.
    - The 4-tuple: :func:`quantize_grouped_input`'s SM90 (K-major) layout over the
      ``M = sum(ks)`` grouped rows (each group padded with zero rows to a multiple of 128).
      ``reverse_shuffle_idxs``: int64 ``(m,)``, the grouped row of each slot;
      ``grouped_mm_offs``: int32 ``(G,)`` cumulative padded group ends; ``ks``: host list of the
      padded group sizes, ``ks_tensor`` the same int32 ``(G,)`` on the device. With ``m == 0``
      every output is empty, ``grouped_mm_offs`` and ``ks_tensor`` of shape ``(0,)`` (as
      ``scatter_by_expert`` returns them).

    Row placement within a group: on SM90 the fused kernels place slots stably (slot order), so
    every output is deterministic; inputs they do not cover (m or N zero, more than 512 groups,
    N not a multiple of 8, non-contiguous or misaligned inputs, non-int64 indices) and SM100 place
    them in ``atomic_add`` order, like :func:`scatter_by_expert`. The rowwise FP8 of each slot,
    ``grouped_mm_offs`` and ``ks`` do not depend on placement; the per-channel FP8 does (128-row
    blocks share a scale), and with a given placement it is bit-identical to the unfused
    sequence. Consumers locate a slot only through ``reverse_shuffle_idxs``.

    One of the fused FP8 expert ops: callers use it where :func:`fp8_fused_experts_supported`. On
    SM100 it returns the same K-major layout (row-major scales), which SM100's
    :func:`k_grouped_fp8_gemm` does not read. Scale strides are unspecified.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_scatter_quant_sm90 import scatter_quant_sm90

        return scatter_quant_sm90(
            gathered, expert_idxs, num_groups, expand_idx, padding_alignment, group_counts
        )
    return scatter_quant_sm100(
        gathered, expert_idxs, num_groups, expand_idx, padding_alignment, group_counts
    )
