"""
Picker of the rowwise FP8 quantization of x plus the 128x128 blockwise quantization of x.T from one
read of x (the activation operands of the dense FP8 linear).

SM90 runs the CuTe-DSL kernel (``fp8_rowwise_blockwise_transpose_sm90``), elsewhere the Triton
kernel (``fp8_rowwise_blockwise_transpose_sm100``). FP8 bytes and scale values are bit-identical;
the SM90 kernel returns the rowwise scale MN-major (``kernels/fp8_scale_layout.py``), the SM100
kernel row-major. The blockwise scale is row-major on both.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_rowwise_blockwise_transpose_sm100 import (
    fp8cast_rowwise_blockwise_transpose_sm100,
)

__all__ = ["fp8cast_rowwise_blockwise_transpose"]


def fp8cast_rowwise_blockwise_transpose(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """See fp8cast_rowwise_blockwise_transpose_sm100."""
    if is_sm90():
        from pithtrain.kernels.fp8_rowwise_blockwise_transpose_sm90 import (
            fp8cast_rowwise_blockwise_transpose_sm90,
        )

        return fp8cast_rowwise_blockwise_transpose_sm90(x)
    return fp8cast_rowwise_blockwise_transpose_sm100(x)
