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
from pithtrain.kernels.ep_dispatch_sm100 import expert_and_expand_idx_sm100, fused_dedup_prepare_dispatch_sm100

__all__ = ["expert_and_expand_idx", "fused_dedup_prepare_dispatch"]


def fused_dedup_prepare_dispatch(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """
    Sender-side metadata of a deduplicated EP dispatch (each token is sent once to every EP rank
    hosting one of its experts): ``(tokens_per_ep_rank, dedup_tokens_per_gpu,
    dispatch_token_idxs, idxs, expand_idx, send_meta)``, all int64.

    - ``topk_ids``: int64 ``(m, k)`` with unit inner stride, values in ``[0, num_experts)``,
      ``num_experts == ep_size * experts_per_rank`` (expert ``e`` lives on rank
      ``e // experts_per_rank``); a token's k experts must be distinct. ``ep_size <= 32``. Not
      modified.
    - ``meta_out``: optional int64 buffer of at least ``2 * ep_size`` elements; the first two
      outputs are views of its first ``2 * ep_size`` elements (a fresh buffer when None).
    - ``tokens_per_ep_rank`` ``(ep_size,)``: (token, expert) slots routed to each rank;
      ``dedup_tokens_per_gpu`` ``(ep_size,)``: distinct tokens sent to each rank.
    - ``dispatch_token_idxs`` ``(m * ep_size,)``, over-allocated: the first
      ``sum(dedup_tokens_per_gpu)`` entries are the token rows to send, rank-major (rank ``r``'s
      chunk starts at the exclusive prefix sum of ``dedup_tokens_per_gpu``).
    - ``idxs`` ``(m * k,)``: the flat indices into ``topk_ids.view(-1)`` sorted by expert;
      ``expand_idx`` ``(m * k,)``: for each sorted slot, the position of its token within the
      token's dedup chunk for that expert's rank.
    - ``send_meta`` ``(ep_size * (experts_per_rank + 1),)``, viewed ``(ep_size, experts_per_rank +
      1)``: per rank, its per-expert slot counts then its dedup count (the payload of the metadata
      all-to-all). Valid until the next call on the same stream (it may be reused scratch).

    Order within a bucket (a rank's chunk of ``dispatch_token_idxs``, an expert's run of
    ``idxs``): token order on SM90, so every output is deterministic (inputs outside the CuTe
    kernel's coverage, e.g. more than 1024 experts + ranks, take the Triton path); ``atomic_add``
    order on SM100. Counts and the per-bucket sets are identical on both. ``m == 0`` gives zero
    counts and empty index outputs.
    """
    if is_sm90():
        from pithtrain.kernels.ep_dispatch_sm90 import fused_dedup_prepare_dispatch_sm90

        return fused_dedup_prepare_dispatch_sm90(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)
    return fused_dedup_prepare_dispatch_sm100(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)


def expert_and_expand_idx(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Receiver-side indices of the received token slots: ``(expert_idxs, expand_idx)``, fresh int64
    ``(total,)`` with ``total = received_expand_idx.shape[0]``.

    - ``recv_meta``: int64 ``(ep_size * (experts_per_rank + 1),)``, the ``send_meta`` rows every
      sender addressed to this rank, sender-major: per sender, its per-local-expert slot counts
      then its dedup token count. Not modified.
    - ``received_expand_idx``: int64 ``(total,)``, the senders' ``expand_idx`` for this rank,
      received sender-major and expert-minor; ``total`` (decided by the senders) must equal the
      sum of the slot counts in ``recv_meta``. Not modified.
    - ``expert_idxs[i]``: the local expert (``< experts_per_rank``) of slot ``i``, from the
      (sender, expert) segment it falls in.
    - ``expand_idx[i]``: ``received_expand_idx[i]`` plus the dedup counts of the preceding
      senders, i.e. slot ``i``'s row in the received (concatenated) deduplicated tokens.

    Exact integer math, identical on both architectures; deterministic.
    """
    if is_sm90():
        from pithtrain.kernels.ep_dispatch_sm90 import expert_and_expand_idx_sm90

        return expert_and_expand_idx_sm90(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    return expert_and_expand_idx_sm100(recv_meta, received_expand_idx, ep_size, experts_per_rank)
