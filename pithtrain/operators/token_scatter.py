"""
Token movement around the MoE grouped GEMMs, as autograd functions: the scatter of tokens into the
expert-grouped padded layout (the scatter_by_expert kernel), the padded index gather of the dedup
expansion, and the per-row group ids of the Hopper FP8 grouped GEMM.
"""

from typing import Optional

import torch

from pithtrain.contexts import training
from pithtrain.kernels import m_grouped_group_indices, scatter_by_expert


class ScatterForGroupedGemm(torch.autograd.Function):
    """
    Custom autograd function wrapping the Triton scatter kernels so the
    scatter operation is differentiable.

    Forward: reorders input_tokens by expert assignment with zero-padding.
    Backward: gathers gradients back using reverse_shuffle_idxs.
    """

    @staticmethod
    def forward(
        ctx, input_tokens, expert_idxs, num_groups, padding_alignment=128, group_counts=None
    ):
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = scatter_by_expert(
            input_tokens, expert_idxs, num_groups, padding_alignment, group_counts
        )
        ctx.save_for_backward(reverse_shuffle_idxs)
        return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks

    @staticmethod
    def backward(
        ctx,
        grad_output_tokens,
        grad_reverse_shuffle_idxs,
        grad_grouped_mm_offs,
        grad_ks_tensor,
        grad_ks,
    ):
        (reverse_shuffle_idxs,) = ctx.saved_tensors
        # Forward: output_tokens[reverse_shuffle_idxs[i]] = input_tokens[i]
        # Backward: grad_input_tokens[i] = grad_output_tokens[reverse_shuffle_idxs[i]]
        grad_input_tokens = grad_output_tokens[reverse_shuffle_idxs]
        return grad_input_tokens, None, None, None, None


class _PaddedIndexGather(torch.autograd.Function):
    """Gather rows from a 2-D tensor by a 1-D index, with padded allocation.

    Two optimizations over plain ``input[index]``:

    1. **Avoids saving the input tensor** - the backward is a scatter-add that
       only needs the indices and the input shape, not the input values.
    2. **Pads the output allocation** to a fixed row alignment so the CUDA
       caching allocator sees consistent block sizes across micro-batches,
       reducing memory fragmentation from dynamic MoE token counts.
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, index: torch.Tensor, pad_to_multiple: int
    ) -> torch.Tensor:
        ctx.save_for_backward(index)
        ctx.input_shape = input.shape
        ctx.pad_to_multiple = pad_to_multiple
        actual = index.shape[0]
        padded = (actual + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple
        buf = input.new_empty(padded, *input.shape[1:])
        # Write directly into buf without materializing a temporary.
        torch.index_select(input, 0, index, out=buf[:actual])
        return buf[:actual]  # view - keeps the padded allocation alive

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (index,) = ctx.saved_tensors
        rows = ctx.input_shape[0]
        padded_rows = (rows + ctx.pad_to_multiple - 1) // ctx.pad_to_multiple * ctx.pad_to_multiple
        grad_buf = torch.zeros(
            padded_rows,
            *ctx.input_shape[1:],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        grad_buf[:rows].scatter_add_(0, index.unsqueeze(-1).expand_as(grad_output), grad_output)
        return grad_buf[:rows], None, None


def padded_index_gather(
    input: torch.Tensor, index: torch.Tensor, pad_to_multiple: int = 1024
) -> torch.Tensor:
    """Memory-efficient ``input[index]`` with padded allocation.

    Does not save *input* for backward, and pads the output to
    *pad_to_multiple* rows to reduce CUDA allocator fragmentation.
    """
    return _PaddedIndexGather.apply(input, index, pad_to_multiple)


def scatter_for_grouped_gemm(
    input_tokens: torch.Tensor,  # [m, hidden_size]
    expert_idxs: torch.Tensor,  # [m] int64, values in [0, num_groups-1]
    num_groups: int,
    padding_alignment: int = 128,
    group_counts: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    Fused MoE token scatter: computes output tokens grouped by expert,
    reverse shuffle indices, grouped_mm offsets, and per-group padded sizes
    in a single kernel launch.

    Args:
        input_tokens: [m, hidden_size] input tokens
        expert_idxs: [m] expert assignment for each token (int64, values in [0, num_groups-1])
        num_groups: number of expert groups
        padding_alignment: pad each group's row count to this alignment (default 128)
        group_counts: rows per group on the host, if the caller already has them; skips the
            device-to-host copy of the padded sizes that otherwise blocks the host

    Returns:
        output_tokens: [actual_M, hidden_size] tokens reordered by expert with zero-padding,
            where actual_M = grouped_mm_offs[-1] (no over-allocation)
        reverse_shuffle_idxs: [m] int64, maps each input token to its position in output
        grouped_mm_offs: [num_groups] int32, cumulative padded offsets for grouped_mm
        ks: list[int], per-group padded sizes (derived from ks_tensor)
        ks_tensor: [num_groups] int32, per-group padded sizes
    """
    output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks = (
        ScatterForGroupedGemm.apply(
            input_tokens, expert_idxs, num_groups, padding_alignment, group_counts
        )
    )
    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor


def precompute_group_indices(grouped_mm_offs: torch.Tensor, M: int) -> Optional[torch.Tensor]:
    """
    Precompute per-row group indices for reuse across multiple grouped FP8 projections.

    Converts cumulative offsets to per-row group IDs, e.g.
    ``grouped_mm_offs = [128, 256, 384, 512]`` -> ``[0,0,...,1,1,...,2,2,...,3,3,...]``.

    Only FP8 training's m-grouped GEMMs read it, and only where the kernel library's GEMM takes it
    (m_grouped_group_indices returns None otherwise); None in BF16 training.
    """
    if training.fp8:
        return m_grouped_group_indices(grouped_mm_offs, M)
    return None
