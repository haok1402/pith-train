"""
Causal depthwise conv1d of the Qwen3.5 Gated DeltaNet, on the channels-last (B, S, C) layout.

``Qwen35MoeGatedDeltaNet`` convolves ``mixed_qkv`` (B, S, C = 2 * key_dim + value_dim) with a
per-channel kernel of width K = 4. The custom ops here (``pithtrain::qwen35_causal_conv1d`` and its
backward) run the Triton kernels of ``pithtrain/kernels/causal_conv1d.py``, which read the input
where the projection GEMM wrote it and write the output in the (B, C, S) storage the sliced
``nn.Conv1d`` output had. The forward is bit-identical to
``conv1d(x.transpose(1, 2), w, padding=K - 1)[..., offset:offset + S]``; see the kernels module for
the exact semantics and the ``offset`` (prepended conv state) convention.
"""

from __future__ import annotations

import torch

from pithtrain.kernels.causal_conv1d import (
    causal_conv1d_bwd,
    causal_conv1d_fwd,
    causal_conv1d_out_like,
)

__all__ = ["causal_conv1d"]


@torch.library.custom_op("pithtrain::qwen35_causal_conv1d", mutates_args=())
def _conv_fwd(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    return causal_conv1d_fwd(x, w, offset)


@_conv_fwd.register_fake
def _(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    return causal_conv1d_out_like(x, offset)


@torch.library.custom_op("pithtrain::qwen35_causal_conv1d_bwd", mutates_args=())
def _conv_bwd(
    g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return causal_conv1d_bwd(g, x, w, offset)


@_conv_bwd.register_fake
def _(g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int):
    return torch.empty_like(x), torch.empty_like(w)


def _setup_context(ctx, inputs, output) -> None:
    x, w, offset = inputs
    ctx.save_for_backward(x, w)
    ctx.offset = offset


def _backward(ctx, g: torch.Tensor):
    x, w = ctx.saved_tensors
    dx, dw = _conv_bwd(g, x, w, ctx.offset)
    return dx, dw, None


_conv_fwd.register_autograd(_backward, setup_context=_setup_context)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """
    Depthwise causal conv of ``x`` (B, offset + S, C) with ``weight`` (C, 1, K) or (C, K), no bias,
    returning (B, S, C) in x's dtype (bf16), stored channels-first (see causal_conv1d_out_like); see the module docstring for the exact semantics.
    """
    assert x.dtype == torch.bfloat16 and x.dim() == 3
    return _conv_fwd(x.contiguous(), weight, offset)
