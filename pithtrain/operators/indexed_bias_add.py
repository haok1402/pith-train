"""
x + bias[group_ids] with a fast segmented-reduction backward for the
common MoE pattern where tokens have already been sorted by destination
expert (so group_ids is non-decreasing and each group is a contiguous
range of rows [grouped_mm_offs[g-1], grouped_mm_offs[g])).

Autograd's default bias[group_ids] backward lowers to index_put_ with
bf16 atomic adds on bias rows — for GPT-OSS expert biases that's the
single dominant kernel by a wide margin (~48% of GPU time, vs <2% for
the actual expert GEMMs).  We replace it with a Triton-fused per-group
reduction (segment_sum_along_rows) that is fully parallel and atomic-free.
"""

import torch

from pithtrain.kernels import segment_sum_along_rows


class _IndexedBiasAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, group_ids, grouped_mm_offs):
        ctx.bias_dtype = bias.dtype
        ctx.bias_shape = bias.shape
        ctx.save_for_backward(grouped_mm_offs)
        return x + bias[group_ids]

    @staticmethod
    def backward(ctx, grad_out):
        (offs,) = ctx.saved_tensors
        if grad_out.shape[0] == 0 or offs.numel() == 0:
            # An EP rank that received no tokens has no rows (and possibly empty offsets): the
            # segment sum would return a (0, H) gradient; the bias gradient is all zeros instead.
            bias_grad = grad_out.new_zeros(ctx.bias_shape, dtype=ctx.bias_dtype)
            return grad_out, bias_grad, None, None
        grad_out_contig = grad_out.contiguous()
        bias_grad = segment_sum_along_rows(grad_out_contig, offs, ctx.bias_dtype)
        # x is identity-add → grad_x = grad_out; group_ids/offs are integer.
        return grad_out, bias_grad, None, None


def indexed_bias_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    group_ids: torch.Tensor,
    grouped_mm_offs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x + bias[group_ids] with fast segmented-sum backward.

    group_ids must be non-decreasing and grouped_mm_offs the matching
    per-group cumulative end offsets (i.e. offs[-1] == M).
    """
    return _IndexedBiasAdd.apply(x, bias, group_ids, grouped_mm_offs)
