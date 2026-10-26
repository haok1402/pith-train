"""
Fused SwiGLU ``silu(gate) * up`` Triton kernels and their launchers.

- :func:`silu_mul_fwd` computes ``silu(gate) * up`` in one kernel (two loads, one store);
- :func:`silu_mul_bwd` recomputes ``silu`` / ``sigmoid`` on the fly and writes both input
  gradients in one kernel (three loads, two stores).

Both architectures run them. The autograd op is ``pithtrain.operators.silu_mul``; the fused SM90
FP8 quantization (``fp8_silu_mul_quant_sm90``) reproduces their arithmetic bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_SIZE", "silu_mul_bwd", "silu_mul_fwd"]


@triton.jit
def _silu_mul_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Compute ``out = silu(gate) * up`` element-wise in one pass."""
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(gate_ptr + offs, mask=mask)
    up = tl.load(up_ptr + offs, mask=mask)

    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)
    silu = gate_f * tl.sigmoid(gate_f)
    out = silu * up_f

    tl.store(out_ptr + offs, out.to(up.dtype), mask=mask)


@triton.jit
def _silu_mul_bwd_kernel(grad_out_ptr, gate_ptr, up_ptr, grad_gate_ptr, grad_up_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Compute ``grad_gate`` and ``grad_up`` element-wise in one pass.

    With ``s = silu(gate)`` and ``sigma = sigmoid(gate)``:
        grad_up   = grad_out * s
        grad_gate = grad_out * up * sigma * (1 + gate - s)
    """
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    grad_out = tl.load(grad_out_ptr + offs, mask=mask)
    gate = tl.load(gate_ptr + offs, mask=mask)
    up = tl.load(up_ptr + offs, mask=mask)

    grad_out_f = grad_out.to(tl.float32)
    gate_f = gate.to(tl.float32)
    up_f = up.to(tl.float32)

    sig = tl.sigmoid(gate_f)
    silu = gate_f * sig

    grad_up = grad_out_f * silu
    grad_gate = grad_out_f * up_f * sig * (1.0 + gate_f - silu)

    tl.store(grad_gate_ptr + offs, grad_gate.to(gate.dtype), mask=mask)
    tl.store(grad_up_ptr + offs, grad_up.to(up.dtype), mask=mask)


BLOCK_SIZE = 1024


def silu_mul_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    ``silu(gate) * up``: a fresh contiguous tensor like ``gate``.

    - ``gate``, ``up``: contiguous, same shape and dtype (bf16 in every caller; any float dtype);
      not modified.
    - Computed in fp32 (``gate * sigmoid(gate) * up`` with Triton's ``sigmoid``) and rounded once
      to the dtype; agrees with the PyTorch fp32 formula to fp32 rounding before that final
      rounding, not bit for bit. Deterministic; same kernel on both architectures. The fused FP8
      ops (:func:`silu_mul_quant_fwd`) reproduce this result bit for bit.
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.dtype == up.dtype, f"dtype mismatch: {gate.dtype} vs {up.dtype}"
    assert gate.is_contiguous(), "gate must be contiguous"
    assert up.is_contiguous(), "up must be contiguous"
    out = torch.empty_like(gate)
    n = gate.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _silu_mul_fwd_kernel[grid](gate, up, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def silu_mul_bwd(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of :func:`silu_mul_fwd`: fresh ``(grad_gate, grad_up)`` like ``gate`` / ``up``.

    - ``grad_output``, ``gate``, ``up``: contiguous, same shape and dtype; not modified.
    - With ``s = sigmoid(gate)`` in fp32: ``grad_up = grad_output * gate * s`` and ``grad_gate =
      grad_output * up * s * (1 + gate - gate * s)``, each rounded once to the dtype.
      Deterministic; same kernel on both architectures; :func:`silu_mul_bwd_quant` reproduces
      both bit for bit.
    """
    assert grad_output.shape == gate.shape, (f"grad_output shape {grad_output.shape} != gate shape {gate.shape}")
    assert grad_output.dtype == gate.dtype, (f"grad_output dtype {grad_output.dtype} != gate dtype {gate.dtype}")
    assert grad_output.is_contiguous(), "grad_output must be contiguous"
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    n = gate.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _silu_mul_bwd_kernel[grid](grad_output, gate, up, grad_gate, grad_up, n, BLOCK_SIZE=BLOCK_SIZE)
    return grad_gate, grad_up
