"""
Picker of gpt-oss's biased clamped SwiGLU fused with the rowwise + K-major FP8 quantization of its
consumers (and, in backward, the gate_up bias gradient).

SM90 runs the fused CuTe-DSL kernels (``fp8_clamped_swiglu_quant_sm90``), elsewhere the unfused
Triton sequence (``fp8_clamped_swiglu_quant_sm100``). FP8 bytes and scale values are
bit-identical; the SM90 kernels return both scales MN-major (``kernels/fp8_scale_layout.py``), the
SM100 path row-major. ``kmajor_channels`` (a K-major layout padded to that many channels for
DeepGEMM's k-grouped wgrad) is honoured by the fused SM90 kernels only.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_clamped_swiglu_quant_sm100 import (
    clamped_swiglu_bwd_quant_sm100,
    clamped_swiglu_quant_fwd_sm100,
    clamped_swiglu_quant_kmajor_sm100,
    clamped_swiglu_quant_rowwise_sm100,
)

__all__ = [
    "clamped_swiglu_bwd_quant",
    "clamped_swiglu_quant_fwd",
    "clamped_swiglu_quant_kmajor",
    "clamped_swiglu_quant_rowwise",
]

_BLOCK = 128


def _check(gate_up: torch.Tensor) -> None:
    assert gate_up.ndim == 2 and gate_up.shape[1] % 2 == 0 and gate_up.shape[0] % _BLOCK == 0


def clamped_swiglu_quant_fwd(
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
    kmajor_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(clamped_swiglu(gate_up + bias[group], alpha, limit), offs)``; see
    ``fp8_clamped_swiglu_quant_sm90.clamped_swiglu_quant_fwd_sm90``."""
    _check(gate_up)
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import (
            clamped_swiglu_quant_fwd_sm90,
        )

        return clamped_swiglu_quant_fwd_sm90(
            gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels
        )
    return clamped_swiglu_quant_fwd_sm100(
        gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels
    )


def clamped_swiglu_quant_rowwise(
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rowwise pair of :func:`clamped_swiglu_quant_fwd`, the same bytes."""
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import (
            clamped_swiglu_quant_rowwise_sm90,
        )

        return clamped_swiglu_quant_rowwise_sm90(gate_up, bias, grouped_mm_offs, alpha, limit)
    return clamped_swiglu_quant_rowwise_sm100(gate_up, bias, grouped_mm_offs, alpha, limit)


def clamped_swiglu_quant_kmajor(
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
    kmajor_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The K-major pair of :func:`clamped_swiglu_quant_fwd`, the same bytes."""
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import (
            clamped_swiglu_quant_kmajor_sm90,
        )

        return clamped_swiglu_quant_kmajor_sm90(
            gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels
        )
    return clamped_swiglu_quant_kmajor_sm100(
        gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels
    )


def clamped_swiglu_bwd_quant(
    dh: torch.Tensor,
    gate_up: torch.Tensor,
    bias: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
    alpha: float,
    limit: float,
    kmajor_channels: int | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """``fp8cast_rowwise_kmajor`` of the clamped SwiGLU input gradient, plus the bias gradient;
    see ``fp8_clamped_swiglu_quant_sm90.clamped_swiglu_bwd_quant_sm90``."""
    assert dh.shape == (gate_up.shape[0], gate_up.shape[1] // 2), (dh.shape, gate_up.shape)
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import (
            clamped_swiglu_bwd_quant_sm90,
        )

        return clamped_swiglu_bwd_quant_sm90(
            dh, gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels
        )
    return clamped_swiglu_bwd_quant_sm100(
        dh, gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels
    )
