# Adapted from NVIDIA TransformerEngine (Apache 2.0)
# https://github.com/NVIDIA/TransformerEngine
#   transformer_engine/common/triton/cross_entropy.py  (Triton kernels)
#   transformer_engine/pytorch/triton/cross_entropy.py  (autograd wrapper)
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications: fused online-softmax + cross-entropy into a single kernel,
# removed tensor-parallelism / label-smoothing / distributed support.

from collections.abc import Callable

import torch

from pithtrain.kernels import cross_entropy_fwd_bwd

__all__ = ["cross_entropy", "prepare_cross_entropy_sum"]


class CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, ignore_index):
        loss, inp = cross_entropy_fwd_bwd(inp, target, ignore_index)
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
    Mean cross-entropy of logits inp (N, V) against target (N,), as a float32 scalar; targets equal
    to ignore_index are left out. Computes in fp32 and, when inp is row-contiguous, overwrites it in
    place with its gradient, in inp's dtype.
    """
    return CrossEntropy.apply(inp, target, ignore_index)


class UnitCrossEntropySum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, count):
        loss, grad = cross_entropy_fwd_bwd(inp, target, -100, sum_grad=True)
        ctx.save_for_backward(grad.detach())
        return loss * count.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad, None, None


def prepare_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor]:
    """Return a unit-gradient backward callback, detached summed loss and token count."""
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    count = (labels != -100).sum()
    loss = UnitCrossEntropySum.apply(logits, labels, count)
    # Expose only unit-gradient backward. Its autograd node releases the logits buffer before
    # the rest of the model's backward runs.
    return lambda: loss.backward(), loss.detach(), count
