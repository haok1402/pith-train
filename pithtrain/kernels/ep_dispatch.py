"""
Picker of the expert-parallel dispatch index math: the sender-side dedup / counting-sort metadata
(:func:`fused_dedup_prepare_dispatch`) and the receiver-side expert and expand indices
(:func:`expert_and_expand_idx`).

SM90 runs the CuTe-DSL kernels (``ep_dispatch_sm90``, stable placement), which fall back to the
Triton kernels for inputs they do not cover; elsewhere the Triton kernels (``ep_dispatch_sm100``,
``atomic_add`` placement). Order-free outputs are identical.
"""

import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.ep_dispatch_sm100 import (
    expert_and_expand_idx_sm100,
    fused_dedup_prepare_dispatch_sm100,
)

__all__ = ["expert_and_expand_idx", "fused_dedup_prepare_dispatch"]


def fused_dedup_prepare_dispatch(
    topk_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    meta_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """The dispatch metadata of ``topk_ids``; see ``fused_dedup_prepare_dispatch_sm100``."""
    if is_sm90():
        from pithtrain.kernels.ep_dispatch_sm90 import fused_dedup_prepare_dispatch_sm90

        return fused_dedup_prepare_dispatch_sm90(
            topk_ids, num_experts, ep_size, experts_per_rank, meta_out
        )
    return fused_dedup_prepare_dispatch_sm100(
        topk_ids, num_experts, ep_size, experts_per_rank, meta_out
    )


def expert_and_expand_idx(
    recv_meta: torch.Tensor,
    received_expand_idx: torch.Tensor,
    ep_size: int,
    experts_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (expert_idxs, expand_idx) of the received token slots, from the received metadata and the
    received sender-local expand_idx. The slot count is ``received_expand_idx``'s length, which the
    senders decide.
    """
    if is_sm90():
        from pithtrain.kernels.ep_dispatch_sm90 import expert_and_expand_idx_sm90

        return expert_and_expand_idx_sm90(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    return expert_and_expand_idx_sm100(recv_meta, received_expand_idx, ep_size, experts_per_rank)
