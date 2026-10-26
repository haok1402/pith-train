"""
Picker of the rowwise + K-major colwise FP8 quantization (the grouped-GEMM input layout).

SM90 runs the CuTe-DSL kernel (``fp8_rowwise_kmajor_sm90``), elsewhere the Triton kernel
(``fp8_rowwise_kmajor_sm100``). FP8 bytes and scale values are bit-identical; the scale strides
are not: the SM90 kernel returns both scales MN-major (``kernels/fp8_scale_layout.py``), the
layout DeepGEMM's SM90 GEMMs read without a transpose, and the SM100 kernel returns them row-major.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100

__all__ = ["fp8cast_rowwise_kmajor"]


def fp8cast_rowwise_kmajor(x: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """See fp8cast_rowwise_kmajor_sm100."""
    if is_sm90():
        from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import fp8cast_rowwise_kmajor_sm90

        return fp8cast_rowwise_kmajor_sm90(x, grouped_mm_offs)
    return fp8cast_rowwise_kmajor_sm100(x, grouped_mm_offs)
