"""Accumulate dense embedding gradients using a microbatch-sized temporary."""

from functools import partial

import torch
from torch import nn

from pithtrain.contexts.weight_grad_store import WeightGradStore


def _accumulate_embedding_grad(
    main_grad: torch.Tensor, tokens: torch.Tensor, grad_output: torch.Tensor
) -> None:
    sorted_rows = tokens.flatten().sort().values
    # The first position for each row owns its gradient. Duplicate positions
    # have zero gradient, so the temporary can keep a fixed size on GPU.
    compact_rows = torch.searchsorted(sorted_rows, tokens.contiguous())
    updates = torch.ops.aten.embedding_dense_backward(
        grad_output, compact_rows, sorted_rows.numel(), -1, False
    )
    main_grad.index_add_(0, sorted_rows, updates.float())


class _CompactEmbeddingGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, weight, main_grad):
        ctx.save_for_backward(tokens)
        ctx.main_grad = main_grad
        return torch.nn.functional.embedding(tokens, weight)

    @staticmethod
    def backward(ctx, grad_output):
        (tokens,) = ctx.saved_tensors
        if WeightGradStore.enabled:
            WeightGradStore.put(
                partial(_accumulate_embedding_grad, ctx.main_grad, tokens, grad_output)
            )
        else:
            _accumulate_embedding_grad(ctx.main_grad, tokens, grad_output)
        return None, None, None


class CompactEmbedding(nn.Embedding):
    """An unpadded dense embedding with a stock-compatible FP32 weight sum."""

    main_grad: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.main_grad is None or not torch.is_grad_enabled() or not self.weight.requires_grad:
            return super().forward(tokens)
        return _CompactEmbeddingGrad.apply(tokens, self.weight, self.main_grad)
