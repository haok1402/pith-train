"""
gpt-oss's clamped SwiGLU, (uc + 1) * gc * sigmoid(alpha * gc) with gc = min(gate, limit) and uc =
clamp(up, -limit, limit), forward and backward Triton kernels over an interleaved gate_up (M, 2N).
Both architectures run them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_SIZE", "clamped_swiglu_bwd", "clamped_swiglu_fwd"]


@triton.jit
def clamped_swiglu_fwd_kernel(gate_up_ptr, out_ptr, n_elements, alpha, limit, BLOCK_SIZE: tl.constexpr):
    # offs indexes the flat (M, N) output; gate is at 2 * offs and up at 2 * offs + 1.
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    gate = tl.load(gate_up_ptr + 2 * offs, mask=mask)
    up = tl.load(gate_up_ptr + 2 * offs + 1, mask=mask)

    g = gate.to(tl.float32)
    u = up.to(tl.float32)

    gc = tl.minimum(g, limit)
    uc = tl.minimum(tl.maximum(u, -limit), limit)

    sig = tl.sigmoid(alpha * gc)
    glu = gc * sig
    out = (uc + 1.0) * glu

    tl.store(out_ptr + offs, out.to(gate.dtype), mask=mask)


@triton.jit
def clamped_swiglu_bwd_kernel(grad_out_ptr, gate_up_ptr, grad_gate_up_ptr, n_elements, alpha, limit, BLOCK_SIZE: tl.constexpr):
    # A saturated clamp passes no gradient; the boundary itself does, as torch.clamp.
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    do = tl.load(grad_out_ptr + offs, mask=mask)
    gate = tl.load(gate_up_ptr + 2 * offs, mask=mask)
    up = tl.load(gate_up_ptr + 2 * offs + 1, mask=mask)

    do_f = do.to(tl.float32)
    g = gate.to(tl.float32)
    u = up.to(tl.float32)

    gc = tl.minimum(g, limit)
    uc = tl.minimum(tl.maximum(u, -limit), limit)

    sig = tl.sigmoid(alpha * gc)
    glu = gc * sig

    grad_uc = do_f * glu
    grad_gc = do_f * (uc + 1.0) * sig * (1.0 + alpha * gc * (1.0 - sig))

    g_active = g <= limit
    u_active = (u >= -limit) & (u <= limit)
    grad_g = tl.where(g_active, grad_gc, 0.0)
    grad_u = tl.where(u_active, grad_uc, 0.0)

    tl.store(grad_gate_up_ptr + 2 * offs, grad_g.to(gate.dtype), mask=mask)
    tl.store(grad_gate_up_ptr + 2 * offs + 1, grad_u.to(up.dtype), mask=mask)


BLOCK_SIZE = 1024


def clamped_swiglu_fwd(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Clamped SwiGLU of an interleaved gate_up: a new contiguous (M, N) tensor in gate_up's dtype, (uc
    + 1) * gc * sigmoid(alpha * gc) with gate = gate_up[:, 0::2], up = gate_up[:, 1::2], gc =
    min(gate, limit), uc = clamp(up, -limit, limit).

    gate_up is contiguous (M, 2N), bf16 (any float dtype). alpha and limit are Python floats
    (gpt-oss: 1.702, 7.0). Computed in fp32 and rounded once to the input dtype. Deterministic.
    """
    assert gate_up.dim() == 2, f"gate_up must be 2-D, got shape {gate_up.shape}"
    M, two_n = gate_up.shape
    assert two_n % 2 == 0, f"gate_up last dim must be even, got {two_n}"
    assert gate_up.is_contiguous(), "gate_up must be contiguous"

    N = two_n // 2
    out = torch.empty((M, N), device=gate_up.device, dtype=gate_up.dtype)

    n = M * N
    if n == 0:
        return out

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    clamped_swiglu_fwd_kernel[grid](gate_up, out, n, float(alpha), float(limit), BLOCK_SIZE=BLOCK_SIZE)
    return out


def clamped_swiglu_bwd(grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """
    Gradient of clamped_swiglu_fwd with respect to the interleaved gate_up: a new (M, 2N) tensor
    like gate_up, gate gradients at even and up gradients at odd columns.

    grad_output is contiguous (M, N) in gate_up's dtype; gate_up is the saved contiguous (M, 2N)
    forward input. The gate gradient is zero where gate > limit, the up gradient zero where |up| >
    limit (the boundary itself passes the gradient, as torch.clamp does). fp32 arithmetic, rounded
    once to the dtype. Deterministic.
    """
    assert grad_output.shape[0] == gate_up.shape[0]
    assert grad_output.shape[1] * 2 == gate_up.shape[1]
    assert grad_output.dtype == gate_up.dtype
    assert grad_output.is_contiguous(), "grad_output must be contiguous"

    grad_gate_up = torch.empty_like(gate_up)
    n = grad_output.numel()
    if n == 0:
        return grad_gate_up

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    clamped_swiglu_bwd_kernel[grid](grad_output, gate_up, grad_gate_up, n, float(alpha), float(limit), BLOCK_SIZE=BLOCK_SIZE)
    return grad_gate_up
