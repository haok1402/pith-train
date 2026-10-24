"""
Picker of SwiGLU fused with the rowwise + K-major FP8 quantization of its consumers (the FP8
routed experts' ``silu(g) * u`` between the gate/up and down grouped GEMMs).

SM90 runs the fused CuTe-DSL kernels (``fp8_silu_mul_quant_sm90``), elsewhere the unfused Triton
sequence (``fp8_silu_mul_quant_sm100``). FP8 bytes and scale values are bit-identical; the SM90
kernels return both scales MN-major (``kernels/fp8_scale_layout.py``), the SM100 path row-major.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_silu_mul_quant_sm100 import (
    silu_mul_bwd_quant_gu_sm100,
    silu_mul_bwd_quant_sm100,
    silu_mul_quant_fwd_sm100,
    silu_mul_quant_gu_fwd_sm100,
)

__all__ = [
    "silu_mul_bwd_quant",
    "silu_mul_bwd_quant_gu",
    "silu_mul_quant_fwd",
    "silu_mul_quant_gu_fwd",
]

_BLOCK = 128


def _check(xs: list[torch.Tensor], grouped_mm_offs: torch.Tensor) -> None:
    for x in xs:
        assert x.dtype == torch.bfloat16 and x.ndim == 2 and x.is_contiguous()
        assert x.shape == xs[0].shape
    assert xs[0].shape[0] % _BLOCK == 0, f"M must be a multiple of 128, got {xs[0].shape}"
    assert grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous()


def _check_gu(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> None:
    assert gu.dtype == torch.bfloat16 and gu.ndim == 2 and gu.is_contiguous()
    assert gu.shape[1] % 2 == 0
    assert grouped_mm_offs.dtype == torch.int32 and grouped_mm_offs.is_contiguous()


def silu_mul_quant_fwd(
    g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(silu_mul(g, u), grouped_mm_offs)`` for BF16 (M, N) g, u (M a
    multiple of 128), without materializing h on SM90."""
    _check([g, u], grouped_mm_offs)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_quant_fwd_sm90

        return silu_mul_quant_fwd_sm90(g, u, grouped_mm_offs)
    return silu_mul_quant_fwd_sm100(g, u, grouped_mm_offs)


def silu_mul_bwd_quant(
    dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """(dg_quant, du_quant): ``fp8cast_rowwise_kmajor`` of each BF16 gradient of ``silu_mul``."""
    _check([dh, g, u], grouped_mm_offs)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_bwd_quant_sm90

        return silu_mul_bwd_quant_sm90(dh, g, u, grouped_mm_offs)
    return silu_mul_bwd_quant_sm100(dh, g, u, grouped_mm_offs)


def silu_mul_quant_gu_fwd(
    gu: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(silu_mul(gu[:, :N], gu[:, N:]), offs)`` for ``gu`` (M, 2N)."""
    _check_gu(gu, grouped_mm_offs)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_quant_gu_fwd_sm90

        return silu_mul_quant_gu_fwd_sm90(gu, grouped_mm_offs)
    return silu_mul_quant_gu_fwd_sm100(gu, grouped_mm_offs)


def silu_mul_bwd_quant_gu(
    dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``fp8cast_rowwise_kmajor(cat([dg, du], 1), offs)`` of the SwiGLU gradients, ``gu`` (M, 2N)
    the saved gate_up output and ``dh`` (M, N) the gradient of ``silu(g) * u``."""
    _check_gu(gu, grouped_mm_offs)
    assert dh.dtype == torch.bfloat16 and dh.is_contiguous()
    assert dh.shape == (gu.shape[0], gu.shape[1] // 2)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_bwd_quant_gu_sm90

        return silu_mul_bwd_quant_gu_sm90(dh, gu, grouped_mm_offs)
    return silu_mul_bwd_quant_gu_sm100(dh, gu, grouped_mm_offs)
