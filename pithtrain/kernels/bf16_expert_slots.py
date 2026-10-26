"""
The SM90-only BF16 CuTe experts' slot maps, row gathers and segment sums around the grouped
GEMMs: the contract functions of ``bf16_expert_slots_sm90``.

Each function imports ``bf16_expert_slots_sm90`` inside its body, so importing this module (and
``pithtrain.kernels``) never loads cutlass; callers take these kernels only where
``bf16_experts_supported`` says yes. ``operators/bf16_experts.py`` composes them into the BF16
expert autograd node. SM100 has no implementation (``bf16_experts_supported`` is False there).
"""

from typing import NamedTuple

import torch

__all__ = [
    "bf16_expert_gather_rows",
    "bf16_expert_gather_sum",
    "bf16_expert_group_slots",
    "expert_slot_segment_sum",
]


def bf16_expert_group_slots(
    expert_idxs: torch.Tensor,
    expand_idx: torch.Tensor | None,
    num_groups: int,
    num_tokens: int,
) -> NamedTuple:
    """
    Index maps of the grouped expert input (no rows are copied), as an opaque record whose
    fields callers read and pass back to the ``bf16_expert_*`` kernels.

    - ``expert_idxs``: int64 ``(S,)`` contiguous, the local expert of each token slot, values in
      ``[0, num_groups)``, ``0 < num_groups <= 64`` (asserted). ``expand_idx``: int64 ``(S,)``
      contiguous, the received-token row (``< num_tokens``) each slot reads, or None for the
      identity (then the received tokens are the slots). Neither is modified.
    - Grouped rows are expert-major with no padding, slots stable (slot order) within their
      expert. Fields: ``cu_seqlens`` int32 ``(G + 1,)``, the cumulative slot counts per expert;
      ``a_idx`` int32 ``(S,)``, the received-token row read by grouped row ``r``; ``row_of``
      int32 ``(S,)``, the grouped row of slot ``i``; ``slot_of`` int32 ``(S,)``, its inverse;
      ``mask`` int64 ``(num_tokens,)``, per token the bitmask of its local experts, and an
      internal per-(token, expert) grouped-row table, both None without ``expand_idx``.
    - Every map is exact and deterministic; one launch, no host sync. S == 0 gives zero
      ``cu_seqlens`` (and mask).

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_slots_sm90`` (cutlass) on first use.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_group_slots_sm90

    return bf16_expert_group_slots_sm90(expert_idxs, expand_idx, num_groups, num_tokens)


def bf16_expert_gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    ``x[idx]``: a fresh bf16 ``(len(idx), N)``, an exact row copy, for a contiguous bf16
    ``(rows, N)`` ``x`` (N a multiple of 8) and an int32 ``idx`` of row indices. Inputs are not
    modified.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_slots_sm90`` (cutlass) on first use.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_gather_rows_sm90

    return bf16_expert_gather_rows_sm90(x, idx)


def bf16_expert_gather_sum(
    grad: torch.Tensor, maps: NamedTuple, num_groups: int, max_slots: int
) -> torch.Tensor:
    """
    The received tokens' gradient from the grouped input gradient: a fresh bf16
    ``(num_tokens, H)`` whose row ``t`` is the sum of the rows of ``grad`` of token ``t``'s slots,
    added in fp32 in ascending local-expert order starting from zero and rounded once; a token
    with no slot gets zeros.

    - ``grad``: bf16 ``(S, H)`` contiguous in grouped-row order (e.g. :func:`bf16_expert_dgrad`);
      H a multiple of 8. Not modified.
    - ``maps``: :func:`bf16_expert_group_slots` built with an ``expand_idx`` (the mask and table
      are required). ``max_slots``: an upper bound on any token's local experts
      (``min(top_k, num_groups)``).

    Deterministic.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_slots_sm90`` (cutlass) on first use.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import bf16_expert_gather_sum_sm90

    return bf16_expert_gather_sum_sm90(grad, maps, num_groups, max_slots)


def expert_slot_segment_sum(
    x: torch.Tensor, cu_host: list[int], cu: torch.Tensor, idx: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Per-segment column sums: a fresh fp32 ``(G, N)`` whose row ``g`` sums rows ``r`` in
    ``[cu[g], cu[g + 1])`` of ``x``, read as ``x[idx[r]]`` when ``idx`` is given (the bias
    gradient of grouped rows).

    - ``x``: bf16 ``(rows, N)`` contiguous, N a multiple of 8; ``cu``: int32 ``(G + 1,)`` on the
      device and ``cu_host`` the same values as a host list (it sizes the work, so no device sync);
      ``idx``: optional int32 index of ``x``'s rows. None is modified.
    - Order: each segment is cut into 256-row chunks from its start, each chunk summed row by row
      in fp32, the chunk sums added in order; deterministic. An empty segment sums to zero.

    SM90 only: callers check :func:`bf16_experts_supported` first; the call imports
    ``bf16_expert_slots_sm90`` (cutlass) on first use.
    """
    from pithtrain.kernels.bf16_expert_slots_sm90 import expert_slot_segment_sum_sm90

    return expert_slot_segment_sum_sm90(x, cu_host, cu, idx)
