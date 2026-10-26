"""
Picker of gpt-oss's biased clamped SwiGLU fused with the rowwise + K-major FP8 quantization of its
consumers (and, in backward, the gate_up bias gradient).

SM90 runs the fused CuTe-DSL kernels, other architectures the unfused Triton sequence. FP8 bytes and
scale values are identical; the SM90 kernels return both scales MN-major, the SM100 path row-major.
padded is honoured by the fused SM90 kernels only. These are fused FP8 expert ops, used where
fp8_fused_experts_supported().
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_clamped_swiglu_quant_sm100 import (clamped_swiglu_bwd_quant_sm100, clamped_swiglu_quant_fwd_sm100, clamped_swiglu_quant_kmajor_sm100, clamped_swiglu_quant_rowwise_sm100)
from pithtrain.kernels.fp8_gemm import k_grouped_safe_size

__all__ = ["clamped_swiglu_bwd_quant", "clamped_swiglu_quant_fwd", "clamped_swiglu_quant_kmajor", "clamped_swiglu_quant_rowwise"]

_BLOCK = 128


def _check(gate_up: torch.Tensor) -> None:
    assert gate_up.ndim == 2 and gate_up.shape[1] % 2 == 0 and gate_up.shape[0] % _BLOCK == 0


def clamped_swiglu_quant_fwd(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """All four outputs (rowwise pair, K-major pair) of fp8cast_rowwise_kmajor(clamped_swiglu(gate_up + bias[group], alpha, limit), grouped_mm_offs)."""
    _check(gate_up)
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import clamped_swiglu_quant_fwd_sm90

        return clamped_swiglu_quant_fwd_sm90(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)
    return clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)


def clamped_swiglu_quant_rowwise(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rowwise FP8 (h_fp8, h_scale) of h = clamped_swiglu(gate_up + bias[group], alpha, limit), the
    m-grouped GEMM operand of the down projection.

    gate_up is bf16 (M, 2N), interleaved (gate at even, up at odd columns), M a multiple of 128 per
    group. bias is bf16 (G, 2N), added in fp32 and rounded to bf16 before the activation; h is
    rounded to bf16 before quantization. h_scale strides are unspecified. Deterministic.
    """
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import clamped_swiglu_quant_rowwise_sm90

        return clamped_swiglu_quant_rowwise_sm90(gate_up, bias, grouped_mm_offs, alpha, limit)
    return clamped_swiglu_quant_rowwise_sm100(gate_up, bias, grouped_mm_offs, alpha, limit)


def clamped_swiglu_quant_kmajor(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, padded: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The per-channel (K-major) pair (h_ch, h_ch_scale) of the same h as clamped_swiglu_quant_rowwise,
    the k-grouped wgrad operand, in quantize_grouped_input's SM90 K-major layout (N channels).

    With padded, the fused SM90 kernel lays it out with N rounded up to a multiple of 256 channels,
    which k_grouped_fp8_gemm takes without a copy; the extra channels are left unwritten. Inputs the
    fused kernel does not cover (N % 8 != 0, more than 1024 groups, a non-bf16 bias) and SM100
    return N channels; the scale's row count is the channel count either way. Scale strides are
    unspecified. Deterministic.
    """
    kmajor_channels = k_grouped_safe_size(gate_up.shape[1] // 2) if padded else None
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import clamped_swiglu_quant_kmajor_sm90

        return clamped_swiglu_quant_kmajor_sm90(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)
    return clamped_swiglu_quant_kmajor_sm100(gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)


def clamped_swiglu_bwd_quant(
    dh: torch.Tensor, gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, padded: bool = False
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Backward of gpt-oss's biased clamped SwiGLU with its consumers' quantization: (dgu_quant, d_bias).

    dh is bf16 (M, N), the gradient of h; gate_up is the saved bf16 (M, 2N) pre-bias GEMM output;
    bias is (G, 2N). dgu_quant is quantize_grouped_input (SM90 K-major layout) of the bf16 gradient
    with respect to gate_up + bias (clamped_swiglu_bwd's arithmetic), its K-major pair padded to a
    multiple of 256 channels as in clamped_swiglu_quant_kmajor when padded. d_bias is (G, 2N) in
    bias's dtype: the per-group sum of that bf16 gradient, as fp32 sums of each 128-row block added
    in block order, rounded once. Deterministic; the architectures agree on d_bias up to fp32
    reassociation within a block.
    """
    kmajor_channels = k_grouped_safe_size(gate_up.shape[1]) if padded else None
    assert dh.shape == (gate_up.shape[0], gate_up.shape[1] // 2), (dh.shape, gate_up.shape)
    if is_sm90():
        from pithtrain.kernels.fp8_clamped_swiglu_quant_sm90 import clamped_swiglu_bwd_quant_sm90

        return clamped_swiglu_bwd_quant_sm90(dh, gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)
    return clamped_swiglu_bwd_quant_sm100(dh, gate_up, bias, grouped_mm_offs, alpha, limit, kmajor_channels)
