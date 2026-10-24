"""
Expert-parallel dispatch: from the router's top-k to the received, deduplicated tokens and the
routing metadata of stages 3-5 (``RoutingInfo``).

The index math runs in the kernels of ``pithtrain/kernels/ep_dispatch.py``; this module does the
metadata and expand_idx all-to-alls, the host-side split arithmetic and the token gather.
``prepare_dispatch_issue`` / ``PendingDispatch.finish`` split it so that independent GPU work can
be enqueued while the metadata travels to the host.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from pithtrain.kernels.ep_dispatch import expert_and_expand_idx, fused_dedup_prepare_dispatch
from pithtrain.kernels.host_transfer import get_pinned_buffer
from pithtrain.operators.all_to_all import all_to_all_sync
from pithtrain.operators.routing import AllToAllSplits, RoutingInfo
from pithtrain.operators.token_scatter import padded_index_gather

# At most one deferred dispatch is outstanding per process: its metadata lands in the single
# pinned buffer below, which the next issue would overwrite.
_outstanding: list["PendingDispatch"] = []


@dataclass(slots=True, eq=False)
class PendingDispatch:
    """
    A dispatch whose device work has been issued but whose host-side split arithmetic, expand_idx
    all-to-all and token gather have not run yet; finish() completes it.

    Between prepare_dispatch_issue() and finish() the host is free to enqueue independent GPU
    work. finish() then waits on an event that fires once the metadata reached the pinned buffer,
    rather than on the whole stream.
    """

    hidden_states: torch.Tensor
    topk_weight: torch.Tensor
    dispatch_token_idxs: torch.Tensor
    idxs: torch.Tensor
    expand_idx: torch.Tensor
    meta: torch.Tensor
    meta_cpu: torch.Tensor
    done: torch.cuda.Event
    ep_size: int
    experts_per_rank: int
    ep_group: dist.ProcessGroup
    finished: bool = False

    @torch.compiler.disable
    def finish(self) -> tuple[torch.Tensor, RoutingInfo]:
        """
        Complete the dispatch on the current stream: (dispatch_tokens, RoutingInfo).
        """
        assert not self.finished, "finish() called twice on the same pending dispatch"
        self.finished = True
        assert _outstanding and _outstanding[-1] is self
        ep_size, experts_per_rank, ep_group = self.ep_size, self.experts_per_rank, self.ep_group
        self.done.synchronize()
        host = self.meta_cpu.tolist()
        _outstanding.pop()
        stride = experts_per_rank + 1
        input_splits = host[:ep_size]
        dedup_input_splits = host[ep_size : 2 * ep_size]
        recv = host[2 * ep_size :]
        dedup_output_splits = recv[experts_per_rank::stride]
        # Rows this rank receives from each sender (summed over its local experts), and rows per
        # local expert summed over senders: what stage 3 groups its GEMMs by, known here for free
        # so the scatter there need not sync again.
        output_splits = [sum(recv[i * stride : i * stride + experts_per_rank]) for i in range(ep_size)]  # fmt: skip
        expert_counts = [sum(recv[e::stride][:ep_size]) for e in range(experts_per_rank)]

        # -- expand_idx all-to-all, then expert_idxs + global expand_idx in one kernel --
        # One entry per received token slot; stage 3 indexes the gathered rows by expand_idx, so
        # the row count is decided by the senders, never by this rank's own token count.
        expand_idx = self.expand_idx
        received_expand_idx = expand_idx.new_empty(sum(output_splits))
        all_to_all_sync(received_expand_idx, expand_idx, output_splits, input_splits, ep_group)
        recv_meta = self.meta[2 * ep_size :]
        expert_idxs, expand_idx = expert_and_expand_idx(
            recv_meta, received_expand_idx, ep_size, experts_per_rank
        )

        # Trim + gather dedup tokens for dispatch
        total_dedup = sum(dedup_input_splits)
        index = self.dispatch_token_idxs[:total_dedup]
        dispatch_tokens = padded_index_gather(self.hidden_states, index)

        return dispatch_tokens, RoutingInfo(
            self.topk_weight,
            expert_idxs,
            self.idxs,
            expand_idx,
            AllToAllSplits(dedup_input_splits, dedup_output_splits),
            AllToAllSplits(input_splits, output_splits),
            expert_counts,
        )


@torch.compiler.disable
def prepare_dispatch_issue(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    ep_group: dist.ProcessGroup,
) -> PendingDispatch:
    """
    Issue the device side of an expert-parallel dispatch (ep_size > 1) without blocking the host:
    the dedup kernels, the metadata all-to-all and the device-to-host copy of the metadata, all on
    the current stream. Returns a PendingDispatch whose finish() completes the dispatch.
    """
    assert ep_size > 1
    assert not _outstanding, "a deferred dispatch is still outstanding"
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # One int64 buffer holds everything the host needs, so one device-to-host copy fetches it:
    # [tokens_per_ep_rank (ep) | dedup_tokens_per_gpu (ep) | recv_meta (ep * (epr + 1))], the last
    # being, per sender, its token count for each local expert followed by its dedup count.
    stride = experts_per_rank + 1
    meta = torch.empty(2 * ep_size + ep_size * stride, dtype=torch.int64, device=topk_ids.device)
    (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,  # (ep_size, experts_per_rank + 1) interleaved: [tpe_0..., dedup_0, ...]
    ) = fused_dedup_prepare_dispatch(topk_ids, num_experts, ep_size, experts_per_rank, meta)

    # -- Metadata all-to-all (piggyback dedup counts alongside per-expert token counts) --
    recv_meta = meta[2 * ep_size :]
    all_to_all_sync(recv_meta, send_meta, group=ep_group)

    # -- One device-to-host copy, completion marked by an event --
    meta_cpu = get_pinned_buffer("dispatch_meta", meta.shape[0], meta.dtype)
    done = torch.cuda.Event()
    meta_cpu.copy_(meta, non_blocking=True)
    done.record()

    pending = PendingDispatch(
        hidden_states, topk_weight, dispatch_token_idxs, idxs, expand_idx, meta, meta_cpu, done,
        ep_size, experts_per_rank, ep_group,
    )  # fmt: skip
    _outstanding.append(pending)
    return pending


@torch.compiler.disable
def prepare_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    ep_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, RoutingInfo]:
    """
    Expert-parallel dispatch with token deduplication.

    When ep_size > 1, deduplicates tokens routed to the same EP rank via
    multiple experts to cut all-to-all volume, and computes expand_idx to
    reconstruct the full mapping on the receiver side. When ep_size == 1,
    replicates each token once per selected expert.
    """

    k = topk_ids.shape[1]
    expert_idxs = topk_ids.view(-1)

    if ep_size == 1:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        dispatch_tokens = (
            hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_states.shape[-1])
        )
        return dispatch_tokens, RoutingInfo(topk_weight, expert_idxs)

    args = (hidden_states, topk_ids, topk_weight, num_experts, ep_size, experts_per_rank, ep_group)
    return prepare_dispatch_issue(*args).finish()
