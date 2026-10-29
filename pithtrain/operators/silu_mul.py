"""
Fused silu(gate) * up with a one-kernel forward and a one-kernel backward. Only gate and up are
saved for backward; silu(gate) is recomputed there.
"""

from __future__ import annotations

import torch

from pithtrain.kernels import silu_mul_bwd, silu_mul_fwd

__all__ = ["silu_mul"]


class SiLUMulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        out = silu_mul_fwd(gate, up)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        return silu_mul_bwd(grad_output, gate, up)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Element-wise silu(gate) * up of two tensors of the same shape and dtype, in that dtype.
    """
    return SiLUMulFunc.apply(gate, up)
