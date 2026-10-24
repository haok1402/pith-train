"""
Fused SwiGLU-style ``silu(gate) * up`` autograd function.

Eager PyTorch implements this as two element-wise kernels in the forward pass
(``silu`` then ``mul``) and several kernels in the backward pass, all of which
are memory-bound. In addition, autograd saves ``sigmoid(gate)``, ``silu(gate)``,
and ``up`` for backward, which wastes activation memory on a recomputable
intermediate.

The kernels (``pithtrain/kernels/silu_mul.py``):

- fuse the forward ``silu(gate) * up`` into one Triton kernel (two loads,
  one store), and
- fuse the backward into one Triton kernel that recomputes ``silu`` /
  ``sigmoid`` on the fly (three loads, two stores).

Only ``gate`` and ``up`` are saved for backward - ``silu(gate)`` is no longer
stored as an activation.
"""

from __future__ import annotations

import torch

from pithtrain.kernels.silu_mul import silu_mul_bwd, silu_mul_fwd


class _SiLUMul(torch.autograd.Function):
    """Fused SwiGLU activation that saves only ``gate`` and ``up``."""

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
    """Fused ``silu(gate) * up`` with a single-kernel backward.

    Parameters
    ----------
    gate, up : torch.Tensor
        Same-shape, same-dtype tensors produced by the gate and up projections
        of a SwiGLU-style MLP.

    Returns
    -------
    torch.Tensor
        Element-wise ``silu(gate) * up`` in the same dtype as the inputs.
    """
    return _SiLUMul.apply(gate, up)
