"""
Embedding whose weight gradient accumulates into an fp32 main_grad, one row per token, instead of a
dense (vocab, hidden) gradient per micro-batch.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Embedding"]


class EmbeddingFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        main_grad: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input)
        # Kept off the saved tensors: other micro-batches update main_grad in place meanwhile.
        ctx.main_grad = main_grad
        return F.embedding(input, weight)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[None, None, None]:
        (input,) = ctx.saved_tensors
        main_grad = ctx.main_grad
        rows = grad_output.reshape(-1, main_grad.shape[-1]).to(torch.float32)
        # Sorted and deterministic for repeated token ids.
        main_grad.index_put_((input.reshape(-1),), rows, accumulate=True)
        return None, None, None


class Embedding(nn.Embedding):
    """
    nn.Embedding that, with main_grad set, accumulates the weight gradient there in fp32. Only the
    default options are supported (no padding_idx, max_norm, scale_grad_by_freq or sparse).
    """

    main_grad: Optional[torch.Tensor] = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.main_grad is None or not torch.is_grad_enabled():
            return super().forward(input)
        assert self.padding_idx is None and self.max_norm is None
        assert not self.scale_grad_by_freq and not self.sparse
        return EmbeddingFunc.apply(input, self.weight, self.main_grad)
