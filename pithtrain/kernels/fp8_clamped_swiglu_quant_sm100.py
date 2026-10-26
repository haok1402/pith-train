"""
gpt-oss's biased clamped SwiGLU followed by the rowwise + K-major FP8 quantization of its consumers,
unfused: ``gate_up + bias[group]``, the Triton ``clamped_swiglu`` kernels, then
``fp8cast_rowwise_kmajor_sm100`` (and, in backward, the per-expert bias gradient through
``block_segment_sum``). The ``fp8_clamped_swiglu_quant`` feature's SM100 implementation, and the
sequence ``fp8_clamped_swiglu_quant_sm90`` fuses bit for bit (the bias gradient to fp32
reassociation). Scales come back row-major. ``kmajor_channels`` (a padded K-major layout) applies
to the fused SM90 kernels only and is ignored here, as by their unfused fallback.
"""

import torch

from pithtrain.kernels.clamped_swiglu import clamped_swiglu_bwd, clamped_swiglu_fwd
from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100
from pithtrain.kernels.segment_sum import block_segment_sum

__all__ = ["biased_gate_up", "clamped_swiglu_bwd_biased", "clamped_swiglu_bwd_quant_sm100", "clamped_swiglu_quant_fwd_sm100", "clamped_swiglu_quant_kmajor_sm100", "clamped_swiglu_quant_rowwise_sm100"]

_BLOCK = 128


def _group_ids(grouped_mm_offs: torch.Tensor, M: int) -> torch.Tensor:
    ids = torch.arange(M, device=grouped_mm_offs.device, dtype=torch.int64)
    offs = grouped_mm_offs.to(torch.int64)
    return torch.searchsorted(offs, ids, right=True).clamp_(max=offs.shape[0] - 1)


def biased_gate_up(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor) -> torch.Tensor:
    """``indexed_bias_add``'s forward: gate_up + bias[group of each row]."""
    return gate_up + bias[_group_ids(grouped_mm_offs, gate_up.shape[0])]


def clamped_swiglu_bwd_biased(dh: torch.Tensor, gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """BF16 d(gate_up) from the Triton clamped_swiglu backward kernel, ``dh`` cast to the biased
    gate_up's dtype as autograd does."""
    x = biased_gate_up(gate_up, bias, grouped_mm_offs)
    return clamped_swiglu_bwd(dh.to(x.dtype), x, alpha, limit)


def _empty(M: int, N: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The four outputs of ``fp8cast_rowwise_kmajor_sm100`` for an empty (M, N) input."""
    return (
        torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M, (N + _BLOCK - 1) // _BLOCK), dtype=torch.float32, device=device),
        torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device),
        torch.empty((N, M // _BLOCK), dtype=torch.float32, device=device),
    )


def clamped_swiglu_quant_fwd_sm100(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor_sm100(clamped_swiglu(gate_up + bias[group], alpha, limit), offs)``."""
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0:
        return _empty(M, N, gate_up.device)
    h = clamped_swiglu_fwd(biased_gate_up(gate_up, bias, grouped_mm_offs), alpha, limit)
    return fp8cast_rowwise_kmajor_sm100(h, grouped_mm_offs)


def clamped_swiglu_quant_rowwise_sm100(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The rowwise pair of :func:`clamped_swiglu_quant_fwd_sm100`."""
    return clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit)[:2]


def clamped_swiglu_quant_kmajor_sm100(gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The K-major pair of :func:`clamped_swiglu_quant_fwd_sm100`."""
    return clamped_swiglu_quant_fwd_sm100(gate_up, bias, grouped_mm_offs, alpha, limit)[2:]


def clamped_swiglu_bwd_quant_sm100(
    dh: torch.Tensor, gate_up: torch.Tensor, bias: torch.Tensor, grouped_mm_offs: torch.Tensor, alpha: float, limit: float, kmajor_channels: int | None = None
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """``fp8cast_rowwise_kmajor_sm100`` of the BF16 interleaved (M, 2N) input gradient, and the
    (G, 2N) bias gradient (the per-expert sum of that gradient over 128-row block partials, in
    bias.dtype)."""
    M, N = gate_up.shape[0], gate_up.shape[1] // 2
    if M == 0 or N == 0:
        return _empty(M, 2 * N, gate_up.device), bias.new_zeros(bias.shape)
    dgu = clamped_swiglu_bwd_biased(dh, gate_up, bias, grouped_mm_offs, alpha, limit)
    partials = dgu.float().view(M // _BLOCK, _BLOCK, 2 * N).sum(1)
    return fp8cast_rowwise_kmajor_sm100(dgu, grouped_mm_offs), block_segment_sum(partials, grouped_mm_offs, bias.dtype)
