"""
SM100 fp8_silu_mul_quant: the unfused sequence silu_mul_fwd / silu_mul_bwd, then
fp8cast_rowwise_kmajor_sm100. Scales are row-major.
"""

import torch

from pithtrain.kernels.fp8_rowwise_kmajor_sm100 import fp8cast_rowwise_kmajor_sm100
from pithtrain.kernels.silu_mul import silu_mul_bwd, silu_mul_fwd

__all__ = ["silu_mul_bwd_quant_gu_sm100", "silu_mul_bwd_quant_sm100", "silu_mul_quant_fwd_sm100", "silu_mul_quant_gu_fwd_sm100"]

_BLOCK = 128


def _empty(M: int, N: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The four outputs of fp8cast_rowwise_kmajor_sm100 for an empty (M, N) input."""
    return (
        torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((M, (N + _BLOCK - 1) // _BLOCK), dtype=torch.float32, device=device),
        torch.empty(M * N, dtype=torch.float8_e4m3fn, device=device),
        torch.empty((N, M // _BLOCK), dtype=torch.float32, device=device),
    )


def silu_mul_quant_fwd_sm100(g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor_sm100(silu_mul_fwd(g, u), grouped_mm_offs)."""
    M, N = g.shape
    if M == 0 or N == 0:
        return _empty(M, N, g.device)
    return fp8cast_rowwise_kmajor_sm100(silu_mul_fwd(g, u), grouped_mm_offs)


def silu_mul_bwd_quant_sm100(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """fp8cast_rowwise_kmajor_sm100 of each gradient of silu_mul_bwd(dh, g, u)."""
    M, N = g.shape
    if M == 0 or N == 0:
        return _empty(M, N, g.device), _empty(M, N, g.device)
    dg, du = silu_mul_bwd(dh, g, u)
    return (fp8cast_rowwise_kmajor_sm100(dg, grouped_mm_offs), fp8cast_rowwise_kmajor_sm100(du, grouped_mm_offs))


def silu_mul_quant_gu_fwd_sm100(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """silu_mul_quant_fwd_sm100 of the halves of a fused gate_up output gu (M, 2N)."""
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return _empty(0, N, gu.device)
    g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    return fp8cast_rowwise_kmajor_sm100(silu_mul_fwd(g, u), grouped_mm_offs)


def silu_mul_bwd_quant_gu_sm100(dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp8cast_rowwise_kmajor_sm100(cat([dg, du], 1)) of the SwiGLU gradients of gu."""
    M, N = gu.shape[0], gu.shape[1] // 2
    if M == 0:
        return _empty(0, 2 * N, gu.device)
    g, u = gu[:, :N].contiguous(), gu[:, N:].contiguous()
    dg, du = silu_mul_bwd(dh, g, u)
    return fp8cast_rowwise_kmajor_sm100(torch.cat([dg, du], 1), grouped_mm_offs)
