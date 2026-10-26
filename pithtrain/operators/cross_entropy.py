# Adapted from NVIDIA TransformerEngine (Apache 2.0)
# https://github.com/NVIDIA/TransformerEngine
#   transformer_engine/common/triton/cross_entropy.py  (Triton kernels)
#   transformer_engine/pytorch/triton/cross_entropy.py  (autograd wrapper)
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications: fused online-softmax + cross-entropy into a single kernel,
# removed tensor-parallelism / label-smoothing / distributed support.

import torch

from pithtrain.kernels import cross_entropy_fwd_inplace


class CrossEntropy(torch.autograd.Function):
    """
    Fused cross-entropy that overwrites the logit tensor with gradients
    during the forward pass so no extra activation memory is needed.
    """

    @staticmethod
    def forward(ctx, inp, target, ignore_index):
        loss, inp = cross_entropy_fwd_inplace(inp, target, ignore_index)
        ctx.save_for_backward(inp.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        inp.mul_(grad_output.to(inp.dtype))
        return inp, None, None


def cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    In-place fused cross-entropy loss with mean reduction.

    Overwrites ``inp`` with pre-computed gradients during the forward pass,
    eliminating the need for a separate activation tensor.  All arithmetic
    is performed in FP32; gradients are stored in the original dtype of
    ``inp``.

    Parameters
    ----------
    inp : torch.Tensor
        Logits of shape ``(N, V)`` where N is the number of tokens and V is
        the vocabulary size.  Modified in-place.
    target : torch.Tensor
        Target indices of shape ``(N,)`` with values in ``[0, V-1]``.
    ignore_index : int
        Target value that is ignored in loss and gradient computation.

    Returns
    -------
    torch.Tensor
        Scalar mean cross-entropy loss (float32).
    """
    return CrossEntropy.apply(inp, target, ignore_index)
