"""
Picker of SwiGLU fused with the rowwise + K-major FP8 quantization of its consumers (the FP8
routed experts' ``silu(g) * u`` between the gate/up and down grouped GEMMs).

SM90 runs the fused CuTe-DSL kernels (``fp8_silu_mul_quant_sm90``), elsewhere the unfused Triton
sequence (``fp8_silu_mul_quant_sm100``). FP8 bytes and scale values are bit-identical; the SM90
kernels return both scales MN-major (``kernels/fp8_scale_layout.py``), the SM100 path row-major.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.fp8_silu_mul_quant_sm100 import (silu_mul_bwd_quant_gu_sm100, silu_mul_bwd_quant_sm100, silu_mul_quant_fwd_sm100, silu_mul_quant_gu_fwd_sm100)

__all__ = ["silu_mul_bwd_quant", "silu_mul_bwd_quant_gu", "silu_mul_quant_fwd", "silu_mul_quant_gu_fwd"]

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


def silu_mul_quant_fwd(g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The grouped-GEMM FP8 input of ``h = silu(g) * u``: :func:`quantize_grouped_input`'s 4-tuple
    ``(h_fp8, h_scale, h_ch_fp8, h_ch_scale)`` in its SM90 (K-major) layout, without writing h.

    - ``g``, ``u``: bf16 ``(M, N)``, contiguous, same shape, M a multiple of 128; rows grouped by
      expert, each group a multiple of 128 rows. Not modified.
    - ``grouped_mm_offs``: int32 ``(G,)`` contiguous cumulative group ends, multiples of 128, last
      == M.
    - ``h`` is :func:`silu_mul_fwd` of ``g, u`` (fp32 arithmetic, rounded once to bf16); its
      quantization is bit-identical to ``quantize_grouped_input(h, grouped_mm_offs)`` on SM90.
    - M == 0 or N == 0 gives empty outputs of the same shapes.

    One of the fused FP8 expert ops: callers use it where :func:`fp8_fused_experts_supported`.
    Elsewhere (SM100) it runs the unfused sequence and returns the same bytes and scale values in
    the same K-major layout (row-major scales), which SM100's :func:`k_grouped_fp8_gemm` does not
    read. Scale strides are unspecified. Deterministic.
    """
    _check([g, u], grouped_mm_offs)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_quant_fwd_sm90

        return silu_mul_quant_fwd_sm90(g, u, grouped_mm_offs)
    return silu_mul_quant_fwd_sm100(g, u, grouped_mm_offs)


def silu_mul_bwd_quant(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """
    Backward of ``h = silu(g) * u`` quantized for the gate / up projections' backward GEMMs:
    ``(dg_quant, du_quant)``, each the 4-tuple of :func:`silu_mul_quant_fwd` (same layout) of one
    gradient.

    - ``dh``, ``g``, ``u``: bf16 ``(M, N)``, contiguous, same shape, M a multiple of 128; ``dh``
      the gradient of ``h``, ``g`` / ``u`` the saved forward inputs. Not modified.
    - ``grouped_mm_offs``: as in :func:`silu_mul_quant_fwd`.
    - ``dg``, ``du`` are :func:`silu_mul_bwd` of ``(dh, g, u)`` (fp32 arithmetic, each rounded
      once to bf16), then quantized bit-identically to ``quantize_grouped_input`` on SM90; the
      bf16 gradients are never written.

    Fused FP8 expert op (:func:`fp8_fused_experts_supported`); the SM100 fallback as in
    :func:`silu_mul_quant_fwd`. Deterministic.
    """
    _check([dh, g, u], grouped_mm_offs)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_bwd_quant_sm90

        return silu_mul_bwd_quant_sm90(dh, g, u, grouped_mm_offs)
    return silu_mul_bwd_quant_sm100(dh, g, u, grouped_mm_offs)


def silu_mul_quant_gu_fwd(gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :func:`silu_mul_quant_fwd` of the halves of a fused gate_up output: ``g = gu[:, :N]``,
    ``u = gu[:, N:]`` (concatenated, not interleaved).

    - ``gu``: bf16 ``(M, 2N)`` contiguous, M a multiple of 128 per group; not modified.
    - Returns the 4-tuple for the ``(M, N)`` ``h``, the same bytes and scales as
      ``silu_mul_quant_fwd(gu[:, :N].contiguous(), gu[:, N:].contiguous(), grouped_mm_offs)``.

    Fused FP8 expert op (:func:`fp8_fused_experts_supported`); the SM100 fallback as in
    :func:`silu_mul_quant_fwd`. Deterministic.
    """
    _check_gu(gu, grouped_mm_offs)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_quant_gu_fwd_sm90

        return silu_mul_quant_gu_fwd_sm90(gu, grouped_mm_offs)
    return silu_mul_quant_gu_fwd_sm100(gu, grouped_mm_offs)


def silu_mul_bwd_quant_gu(dh: torch.Tensor, gu: torch.Tensor, grouped_mm_offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward of :func:`silu_mul_quant_gu_fwd`: the 4-tuple of :func:`silu_mul_quant_fwd` (same
    layout) of the ``(M, 2N)`` gate_up gradient ``cat([dg, du], 1)``.

    - ``dh``: bf16 ``(M, N)`` contiguous, the gradient of ``h``; ``gu``: bf16 ``(M, 2N)``
      contiguous, the saved gate_up output (``g = gu[:, :N]``, ``u = gu[:, N:]``); M a multiple of
      128 per group. Neither is modified.
    - ``dg``, ``du`` are :func:`silu_mul_bwd` of ``(dh, g, u)`` rounded to bf16; the result is
      bit-identical to ``quantize_grouped_input(cat([dg, du], 1), grouped_mm_offs)`` on SM90.

    Fused FP8 expert op (:func:`fp8_fused_experts_supported`); the SM100 fallback as in
    :func:`silu_mul_quant_fwd`. Deterministic.
    """
    _check_gu(gu, grouped_mm_offs)
    assert dh.dtype == torch.bfloat16 and dh.is_contiguous()
    assert dh.shape == (gu.shape[0], gu.shape[1] // 2)
    if is_sm90():
        from pithtrain.kernels.fp8_silu_mul_quant_sm90 import silu_mul_bwd_quant_gu_sm90

        return silu_mul_bwd_quant_gu_sm90(dh, gu, grouped_mm_offs)
    return silu_mul_bwd_quant_gu_sm100(dh, gu, grouped_mm_offs)
