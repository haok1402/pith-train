"""
The MoE stage-3 input path for FP8 experts, unfused: the dedup expansion, the Triton scatter by
expert (``kernels.token_scatter.scatter_by_expert``) and ``fp8cast_rowwise_kmajor_sm100``. The
``fp8_scatter_quant`` feature's SM100 implementation; ``fp8_scatter_quant_sm90`` fuses it (with
stable row placement). Scales come back row-major.
"""

from typing import Optional

import torch

from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100
from pithtrain.kernels.token_scatter import scatter_by_expert

__all__ = ["scatter_quant_sm100"]


def scatter_quant_sm100(
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
    """``(fp8cast_rowwise_kmajor_sm100(x), rsi, offs, ks, ks_tensor)`` of the scattered
    ``x = scatter_by_expert(gathered[expand_idx], ...)``, in ``scatter_for_grouped_gemm``'s output
    order."""
    assert padding_alignment == 128, "the K-major FP8 layout needs 128-row groups"
    x = gathered if expand_idx is None else gathered[expand_idx]
    out, rsi, offs, ks_t, ks = scatter_by_expert(
        x, expert_idxs, num_groups, group_counts=group_counts
    )
    return fp8cast_rowwise_kmajor_sm100(out, offs), rsi, offs, ks, ks_t
