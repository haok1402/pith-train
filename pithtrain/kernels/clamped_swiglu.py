"""
Clamped SwiGLU of OpenAI's gpt-oss, ``(uc + 1) * gc * sigmoid(alpha * gc)`` with
``gc = min(gate, L)`` and ``uc = clamp(up, +/-L)``, as Triton kernels and their launchers.
``gate`` and ``up`` are the even/odd columns of an interleaved ``gate_up [M, 2N]`` input.

One kernel each for forward and backward; the backward recomputes ``gc``, ``uc`` and the sigmoid,
and the clamp branches contribute zero gradient at saturation. Both architectures run them. The
autograd op is ``pithtrain.operators.clamped_swiglu``; the fused SM90 FP8 quantization
(``fp8_clamped_swiglu_quant_sm90``) reproduces their arithmetic bit for bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_SIZE", "clamped_swiglu_bwd", "clamped_swiglu_fwd"]


@triton.jit
def _clamped_swiglu_fwd_kernel(
    gate_up_ptr,
    out_ptr,
    n_elements,
    alpha,
    limit,
    BLOCK_SIZE: tl.constexpr,
):
    # offs indexes the flat (M, N) output; interleaved gate_up pulls
    # gate from 2*offs and up from 2*offs + 1.
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
def _clamped_swiglu_bwd_kernel(
    grad_out_ptr,
    gate_up_ptr,
    grad_gate_up_ptr,
    n_elements,
    alpha,
    limit,
    BLOCK_SIZE: tl.constexpr,
):
    # grad_uc = do * glu
    # grad_gc = do * (uc + 1) * sigma * (1 + alpha * gc * (1 - sigma))
    # Clamp-aware: grad_g = grad_gc if g <= L else 0; grad_u = grad_uc if |u| <= L else 0.
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
    """Clamped SwiGLU ``(M, N)`` of the contiguous interleaved ``gate_up`` ``(M, 2N)``."""
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
    _clamped_swiglu_fwd_kernel[grid](
        gate_up, out, n, float(alpha), float(limit), BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def clamped_swiglu_bwd(
    grad_output: torch.Tensor, gate_up: torch.Tensor, alpha: float, limit: float
) -> torch.Tensor:
    """Gradient ``(M, 2N)`` with respect to the interleaved ``gate_up``."""
    assert grad_output.shape[0] == gate_up.shape[0]
    assert grad_output.shape[1] * 2 == gate_up.shape[1]
    assert grad_output.dtype == gate_up.dtype
    assert grad_output.is_contiguous(), "grad_output must be contiguous"

    grad_gate_up = torch.empty_like(gate_up)
    n = grad_output.numel()
    if n == 0:
        return grad_gate_up

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _clamped_swiglu_bwd_kernel[grid](
        grad_output,
        gate_up,
        grad_gate_up,
        n,
        float(alpha),
        float(limit),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_gate_up
