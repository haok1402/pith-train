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
    """
    Rowwise 1x128 FP8 of ``x`` and 128x128 blockwise FP8 of ``x.T`` from one read of ``x``:
    ``(x_fp8, x_scale, xt_fp8, xt_scale)``, the activation operands of the dense FP8 linear
    (:func:`fp8_act_weight_gemm`).

    - ``x``: 2-D ``(M, K)`` bf16 (fp16 / fp32 also accepted), any M and K (a partial block is
      quantized over its valid elements), any strides (a non-unit inner stride is copied first).
      Not modified.
    - ``x_fp8``: e4m3 ``(M, K)`` contiguous; ``x_scale``: fp32, logical shape
      ``(M, ceil(K / 128))``, one scale per row and 128-column block, strides unspecified
      (MN-major on SM90, row-major on SM100).
    - ``xt_fp8``: e4m3 ``(K, M)`` contiguous, ``x.T`` with one scale per 128x128 tile;
      ``xt_scale``: fp32 ``(ceil(K / 128), ceil(M / 128))`` row-major on both architectures.

    Per block: ``amax = max(|x|)``, ``scale = 2^ceil(log2(max(amax, 1e-4) / 448))`` (E8M0, exact),
    ``fp8 = e4m3(x / scale)`` rounded to nearest even with saturation. Both architectures return
    the same bytes and scale values; deterministic; fresh outputs.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_rowwise_blockwise_transpose_sm90 import (
            fp8cast_rowwise_blockwise_transpose_sm90,
        )

        return fp8cast_rowwise_blockwise_transpose_sm90(x)
    return fp8cast_rowwise_blockwise_transpose_sm100(x)
