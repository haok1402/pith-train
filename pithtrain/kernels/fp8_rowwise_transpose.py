"""
Picker of the rowwise FP8 quantization of x and of x.T from one read of x (dgrad / wgrad operands).

SM90 runs the CuTe-DSL kernel (``fp8_rowwise_transpose_sm90``), elsewhere the Triton kernel
(``fp8_rowwise_transpose_sm100``). FP8 bytes and scale values are bit-identical; the SM90 kernel
returns both scales MN-major (``kernels/fp8_scale_layout.py``), the SM100 kernel row-major.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_rowwise_transpose_sm100 import fp8cast_rowwise_transpose_sm100

__all__ = ["fp8cast_rowwise_transpose"]


def fp8cast_rowwise_transpose(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rowwise 1x128 FP8 of ``x`` and of ``x.T`` from one read of ``x``: ``(x_fp8, x_scale, xt_fp8,
    xt_scale)``, the grad_output operands of the dense FP8 linear's dgrad and wgrad
    (:func:`fp8_dgrad_wgrad`).

    - ``x``: 2-D ``(M, N)`` bf16 (fp32 also accepted), any M and N (a partial block is quantized
      over its valid elements), any strides. Not modified.
    - ``x_fp8``: e4m3 ``(M, N)`` contiguous; ``x_scale``: fp32, logical shape
      ``(M, ceil(N / 128))``, one scale per row and 128-column block.
    - ``xt_fp8``: e4m3 ``(N, M)`` contiguous, ``x.T`` quantized the same way; ``xt_scale``: fp32,
      logical shape ``(N, ceil(M / 128))``.
    - Scale strides are unspecified (both MN-major on SM90, row-major on SM100).

    Per 128-element block: ``amax = max(|x|)``, ``scale = 2^ceil(log2(max(amax, 1e-4) / 448))``
    (E8M0, exact), ``fp8 = e4m3(x / scale)`` rounded to nearest even with saturation. Both
    architectures return the same bytes and scale values; deterministic; fresh outputs.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_rowwise_transpose_sm90 import fp8cast_rowwise_transpose_sm90

        return fp8cast_rowwise_transpose_sm90(x)
    return fp8cast_rowwise_transpose_sm100(x)
