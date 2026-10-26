"""
Fused clamped SwiGLU used in OpenAI's gpt-oss: ``(uc + 1) * gc * sigmoid(alpha * gc)``
with ``gc = min(gate, L)`` and ``uc = clamp(up, +/-L)``, where ``gate`` and
``up`` are the even/odd columns of an interleaved ``gate_up [M, 2N]`` input.

One Triton kernel each for forward and backward (``pithtrain/kernels/clamped_swiglu.py``). Only
``gate_up`` is saved for backward; ``gc``, ``uc`` and the sigmoid are recomputed, and the clamp
branches contribute zero gradient at saturation.
"""

from __future__ import annotations

import torch

from pithtrain.kernels import clamped_swiglu_bwd, clamped_swiglu_fwd


class _ClampedSwiGLU(torch.autograd.Function):
    """Fused clamped SwiGLU with a single-kernel backward."""

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
        out = clamped_swiglu_fwd(gate_up, alpha, limit)
        ctx.save_for_backward(gate_up)
        ctx.alpha = float(alpha)
        ctx.limit = float(limit)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (gate_up,) = ctx.saved_tensors
        return clamped_swiglu_bwd(grad_output, gate_up, ctx.alpha, ctx.limit), None, None


def clamped_swiglu(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """Fused clamped SwiGLU with interleaved gate/up layout.

    ``gate_up`` is ``(M, 2N)`` with gate at even columns and up at odd columns;
    gate is clamped above at ``+limit``, up symmetrically to ``[-limit, +limit]``.
    Returns ``(M, N)``.
    """
    return _ClampedSwiGLU.apply(gate_up, alpha, limit)
