"""
Picker of the MoE stage-3 input path for FP8 experts: dedup expansion + scatter by expert + rowwise /
K-major FP8 quantization, from the received tokens to the quantized grouped GEMM input.

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
    """``fp8cast_rowwise_kmajor(scatter_for_grouped_gemm(gathered[expand_idx], ...))``; see
    ``fp8_scatter_quant_sm90.scatter_quant_sm90``."""
    if is_sm90():
        from pithtrain.kernels.fp8_scatter_quant_sm90 import scatter_quant_sm90

        return scatter_quant_sm90(
            gathered, expert_idxs, num_groups, expand_idx, padding_alignment, group_counts
        )
    return scatter_quant_sm100(
        gathered, expert_idxs, num_groups, expand_idx, padding_alignment, group_counts
    )
