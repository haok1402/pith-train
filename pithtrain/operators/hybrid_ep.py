"""
Expert-parallel dispatch and combine through DeepEP's HybridEP (hybrid-ep branch, 10d4dd7), the
alternative to the NCCL all-to-all path of ep_dispatch.py, selected by
DistributedCfg.ep_backend = "hybrid-ep".

HybridEP gathers every rank's routing (one all-gather), scans it so every rank knows where each
token goes, and then moves each token once per destination rank with TMA over NVLink. Combine
pulls the rows back and sums them per token in fp32. The router weights are applied on the expert
side (stage 3), so the combine returns the finished MoE output and stage 5 only adds the residual.

Dropless by construction: the token-drop path of HybridEP is compiled in only when the permute is
fused into the dispatch and the call is non-blocking (jit/compiler.cu enable_token_drop); this
module never fuses, and every output is sized from the exact counts the scan produced.

Two independent switches:

- mode "permute": HybridEP's own permute gives stage 3 one row per (token, local expert),
  expert-major. mode "dedup": the non-permuting dispatch gives one row per (token, rank), and the
  fused BF16 experts read them through expand_idx exactly like the all-to-all path, which saves
  the expanded copy (8 rows per token instead of about 5.3 at Qwen3-30B-A3B) and the permute and
  unpermute kernels; a weighted gather-sum over each row's local experts replaces the unpermute.
- sync_late False: the all-gather, scan and HybridEP's cudaStreamSynchronize all happen in
  stage 2. True: stage 1 issues the all-gather and scan right after the router, with an
  asynchronous copy of the counts to pinned memory (hybrid_ep_dispatch_issue), so they run while
  the host enqueues the other micro-batch's stage-3 backward; stage 2 then waits for that copy
  (by then usually done) and launches the token transfer, ahead of the other micro-batch's
  stage-2 backward on the HybridEP stream.

Either way there is exactly one host wait per forward dispatch; the combine and the whole backward
replay the cached handle with no wait. The per-expert counts stay on the device through the
grouped GEMMs. Every HybridEP call runs on one dedicated stream, in the same order on every rank:
the buffer is shared by dispatch and combine, so two calls must never run concurrently.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from pithtrain.contexts import distributed
from pithtrain.operators.ep_dispatch import EPLoadStats

__all__ = [
    "HybridEPState",
    "StreamWork",
    "hybrid_ep_combine",
    "hybrid_ep_combine_backward",
    "hybrid_ep_dispatch",
    "hybrid_ep_dispatch_backward",
    "hybrid_ep_dispatch_issue",
    "hybrid_ep_drop_check",
    "hybrid_ep_stream",
    "weight_rows",
]

num_sms: int = 0
permute_blocks: int = 0
"""
Grid of HybridEP's permute and unpermute kernels; 0 keeps its default (16 blocks per SM).
"""
custom_allgather: bool = False
"""
Gather the routing with HybridEP's own intra-node all-gather kernel (32 SMs) instead of NCCL.
"""
mode: str = "permute"
sync_late: bool = False
buffer = None
stream: Optional[torch.cuda.Stream] = None


class HybridEPStats:
    """
    Totals for the per-step dropless check: rows received (from the exact counts the host read)
    against rows routed, and two sticky device flags read after the step: HybridEP's overflow flag
    (index 0) and the dedup mode's slot-order check (index 1).
    """

    rows_received: int = 0
    rows_expected: int = 0
    flags: Optional[torch.Tensor] = None


def record_flag(t: torch.Tensor, index: int) -> None:
    if HybridEPStats.flags is None:
        HybridEPStats.flags = torch.zeros(2, dtype=torch.int32, device=t.device)
    HybridEPStats.flags[index : index + 1].bitwise_or_(t.to(torch.int32).view(-1)[:1])


class PinnedRing:
    """
    Pinned host slots for the late-read counts, reused round-robin instead of two pinned
    allocations per dispatch. A slot also backs the handle's dense count, which the backward
    replays read, so it must outlive its handle: the ring is far larger than the number of
    handles in flight at once (about 72-96 layer-micro-batches awaiting backward at
    Qwen3-30B-A3B under DualPipeV), so a slot is reused only long after its handle is gone.
    """

    size = 4096
    counts: Optional[torch.Tensor] = None
    dense: Optional[torch.Tensor] = None
    next = 0

    @classmethod
    def take(cls, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
        if cls.counts is None or cls.counts.shape[1] < num_experts:
            cls.counts = torch.empty(
                cls.size, num_experts, dtype=torch.int64, device="cpu", pin_memory=True
            )
            cls.dense = torch.empty(cls.size, 1, dtype=torch.int32, device="cpu", pin_memory=True)
        i, cls.next = cls.next, (cls.next + 1) % cls.size
        return cls.counts[i, :num_experts], cls.dense[i]


@dataclass(slots=True, eq=False)
class HybridEPState:
    """
    The dispatch of one layer and micro-batch, shared by its stage 2, 3, 4 and 5 records.
    """

    topk_idx: torch.Tensor
    topk_weight: torch.Tensor
    num_experts: int
    experts_per_rank: int
    handle: Optional[object] = None
    """
    hybrid_ep_cpp.HandleImpl from metadata_preprocessing, replayed by the combine and backward.
    """
    num_tokens: int = 0
    num_slots: int = 0
    """
    (token, local expert) rows this rank received.
    """
    num_dense: int = 0
    """
    (token, rank) rows this rank received (the dedup rows).
    """
    recv_tokens: Optional[torch.Tensor] = None
    """
    permute: (num_slots, H) expert-major. dedup: (num_dense, H).
    """
    recv_probs: Optional[torch.Tensor] = None
    """
    The router weight of each slot, fp32 (num_slots,), in the order stage 3 returns its rows.
    """
    tokens_per_expert: Optional[torch.Tensor] = None
    """
    int32 (experts_per_rank,) on the device: rows received per local expert.
    """
    tokens_per_expert_host: Optional[list[int]] = None
    expert_idxs: Optional[torch.Tensor] = None
    """
    dedup: int64 (num_slots,), the local expert of each slot.
    """
    expand_idx: Optional[torch.Tensor] = None
    """
    dedup: int64 (num_slots,), the dense row each slot reads.
    """
    slot_maps: Optional[object] = None
    pending: Optional[tuple] = None
    probs_grad: Optional[torch.Tensor] = None
    topk_weight_grad: Optional[torch.Tensor] = None


class StreamWork:
    """
    The Work-like handle of a HybridEP call: wait() makes the current stream wait for it, so the
    pipeline treats it like an asynchronous all-to-all.
    """

    __slots__ = ("event",)

    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


def hybrid_ep_stream() -> torch.cuda.Stream:
    global stream
    if stream is None:
        stream = torch.cuda.Stream(device=distributed.device)
    return stream


def get_buffer(hidden_dim: int, num_tokens: int, experts_per_rank: int):
    global buffer
    from deep_ep import HybridEPBuffer

    cfg = None if buffer is None else buffer.configurer.buffer_config
    if (
        cfg is None
        or cfg.hidden_dim < hidden_dim
        or cfg.max_num_of_tokens_per_rank < num_tokens
        or cfg.num_of_experts_per_rank < experts_per_rank
    ):
        # A new buffer invalidates every outstanding handle; at a fixed shape this runs once.
        sms = num_sms if num_sms > 0 else None
        buffer = HybridEPBuffer(
            group=distributed.ep_group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=num_tokens,
            num_local_experts=experts_per_rank,
            num_sms_dispatch_api=sms,
            num_sms_combine_api=sms,
            num_blocks_permute=permute_blocks if permute_blocks > 0 else None,
            num_blocks_unpermute=permute_blocks if permute_blocks > 0 else None,
            load_cached_kernels=True,
            use_shared_buffer=True,
            enable_custom_allgather=custom_allgather,
        )
    return buffer


def on_hybrid_stream(wait_event: Optional[torch.cuda.Event], fn, state, *tensors):
    """
    Run fn(state, *tensors) on the HybridEP stream after wait_event (or after the current stream's
    work so far), and return (fn's result, the event recorded after it).
    """
    s = hybrid_ep_stream()
    if wait_event is None:
        wait_event = torch.cuda.current_stream().record_event()
    s.wait_event(wait_event)
    prev = torch.cuda.current_stream()
    # Inputs come from the compute stream and outputs go back to it: tell the caching allocator,
    # so neither side's block is reused while the other stream may still touch it.
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t.record_stream(s)
    torch.cuda.set_stream(s)
    try:
        out = fn(state, *tensors)
        done = s.record_event()
    finally:
        torch.cuda.set_stream(prev)
    for t in (out, state.recv_probs, state.tokens_per_expert, state.topk_weight_grad, state.expert_idxs, state.expand_idx):  # fmt: skip
        if isinstance(t, torch.Tensor):
            t.record_stream(prev)
    return out, done


# ---- forward dispatch ----


@torch.no_grad()
def dispatch_issue(state: HybridEPState, hidden: torch.Tensor):
    """
    Routing all-gather and scan; then either the blocking read of the counts, or (sync_late) an
    asynchronous copy of them to pinned memory, read later by dispatch_finish.
    """
    hidden = hidden.view(-1, hidden.shape[-1])
    T, H = hidden.shape
    G = state.experts_per_rank
    buf = get_buffer(H, T, G)
    topk, routing, probs, _ = buf._prepare_routing_data(
        topk_idx=state.topk_idx,
        topk_weights=state.topk_weight.detach().float(),
        num_of_tokens=T,
        num_of_experts=state.num_experts,
        probs=None,
        routing_map=None,
        num_of_experts_per_rank=G,
    )
    slots = buf._resolve_token_slots(None, T)
    routing = buf._pad_routing_rows(routing, topk, slots)
    config = buf.update_template_config(hidden_dim=H, num_of_tokens_per_rank=slots, num_local_experts=G, topk=topk)  # fmt: skip
    handle = buf.runtime.metadata_preprocessing(
        config=config,
        routing_map=routing,
        num_of_tokens_per_rank=slots,
        num_of_valid_tokens=T,
        enable_permute=True,
        non_blocking=sync_late,
    )
    state.handle, state.num_tokens = handle, T
    state.tokens_per_expert = handle.tokens_per_expert
    record_flag(handle.overflow_flag, 0)
    if sync_late:
        counts, dense = PinnedRing.take(G)
        counts.copy_(handle.padded_tokens_per_expert, non_blocking=True)
        dense.copy_(handle.num_dispatched_tokens_tensor, non_blocking=True)
        state.pending = (hidden, probs, counts, dense, torch.cuda.current_stream().record_event())
    else:
        # Blocking mode: both are pinned and were filled before metadata_preprocessing returned.
        state.pending = (hidden, probs, handle.padded_tokens_per_expert, handle.num_dispatched_tokens_tensor, None)  # fmt: skip
    return None


@torch.no_grad()
def dispatch_transfer(state: HybridEPState, *unused):
    """
    The token transfer, once the counts are on the host: sizes every output exactly.
    """
    hidden, probs, counts, dense, _ = state.pending
    state.pending = None
    handle, G = state.handle, state.experts_per_rank
    state.tokens_per_expert_host = counts.tolist()
    state.num_slots = S = sum(state.tokens_per_expert_host)
    state.num_dense = D = int(dense.item())
    handle.num_permuted_tokens = S
    # The handle's dense count must be readable on the host without a sync for every replay.
    handle.num_dispatched_tokens_tensor = dense
    HybridEPStats.rows_received += S
    HybridEPStats.rows_expected += state.num_tokens * state.topk_idx.shape[1]
    EPLoadStats.rows += S
    if mode == "permute":
        tokens, recv_probs, _ = buffer.runtime.dispatch_with_permute(
            hidden=hidden, probs=probs, handle=handle, with_probs=True
        )
        state.recv_tokens, state.recv_probs = tokens, recv_probs
        return tokens
    tokens, dense_probs, _ = buffer.runtime.dispatch(
        hidden=hidden, probs=probs, handle=handle, with_probs=True
    )
    # Slots in HybridEP's permuted order (expert-major): dense_to_expert_map[d, g] is the slot
    # of dense row d's local expert g, or -1.
    dev = tokens.device
    d2e = handle.dense_to_expert_map[:D].reshape(-1).long()
    valid = d2e >= 0
    target = torch.where(valid, d2e, S)
    rows = torch.arange(D, device=dev).repeat_interleave(G)
    experts = torch.arange(G, device=dev).repeat(D)
    expand_idx = torch.empty(S + 1, dtype=torch.int64, device=dev)
    expert_idxs = torch.empty(S + 1, dtype=torch.int64, device=dev)
    expand_idx.scatter_(0, target, rows)
    expert_idxs.scatter_(0, target, experts)
    expand_idx, expert_idxs = expand_idx[:S], expert_idxs[:S]
    # The dispatched probs are [D, G * ranks per NVLink domain] (the node's experts, not all of
    # them) and only this rank's G columns, at its rank within the node, are written; the rest are
    # stale. On one node that is ep_rank * G; across nodes it is not.
    col = buffer.local_rank * G
    state.recv_probs = dense_probs[expand_idx, col + expert_idxs]
    state.recv_tokens, state.expand_idx, state.expert_idxs = tokens, expand_idx, expert_idxs
    from pithtrain.kernels import bf16_expert_group_slots

    maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, D)
    check_slot_order(maps)
    state.slot_maps = maps
    return tokens


def check_slot_order(maps) -> None:
    """
    The combine's gather-sum reads stage 3's rows in grouped order, which equals slot order only if
    HybridEP's permuted order is stable by dense row within each expert. Checked on the device
    (no sync) into a sticky flag that hybrid_ep_drop_check reports after the step.
    """
    S = maps.row_of.shape[0]
    record_flag((maps.row_of != torch.arange(S, device=maps.row_of.device, dtype=maps.row_of.dtype)).any(), 1)  # fmt: skip


def dispatch_finish(state: HybridEPState) -> torch.cuda.Event:
    event = state.pending[4]
    if event is not None:
        event.synchronize()  # the one host wait of a sync_late dispatch
    _, done = on_hybrid_stream(hybrid_ep_stream().record_event(), dispatch_transfer, state)
    return done


def hybrid_ep_dispatch_issue(state: HybridEPState, hidden: torch.Tensor) -> None:
    """
    sync_late: the routing all-gather and scan, issued from stage 1 right after the router, after
    the work the current stream has so far. A no-op otherwise.
    """
    if sync_late:
        on_hybrid_stream(None, dispatch_issue, state, hidden)


def hybrid_ep_dispatch(state, hidden, wait_event=None):
    """
    Forward dispatch: (None, StreamWork). The received rows are in state.recv_tokens once the work
    is waited on. Issues the all-gather and scan first unless stage 1 already did.
    """
    if state.pending is None:
        on_hybrid_stream(wait_event, dispatch_issue, state, hidden)
    return None, StreamWork(dispatch_finish(state))


# ---- forward combine and the backward ----


@torch.no_grad()
def combine_impl(state: HybridEPState, x: torch.Tensor):
    x = x.contiguous()
    if mode == "permute":
        out, _ = buffer.runtime.combine_with_unpermute(hidden=x, handle=state.handle, with_probs=False)  # fmt: skip
        return out
    from pithtrain.kernels import bf16_expert_gather_sum

    G = state.experts_per_rank
    dense = bf16_expert_gather_sum(x, state.slot_maps, G, min(G, state.topk_idx.shape[1]))
    out, _ = buffer.runtime.combine(hidden=dense, handle=state.handle, with_probs=False)
    return out


def hybrid_ep_combine(state, x, wait_event=None):
    """
    Forward combine of stage 3's weighted rows: ((num_tokens, H), StreamWork).
    """
    out, done = on_hybrid_stream(wait_event, combine_impl, state, x)
    return out, StreamWork(done)


@torch.no_grad()
def combine_backward_impl(state: HybridEPState, grad: torch.Tensor):
    grad = grad.contiguous().view(-1, grad.shape[-1])
    if mode == "permute":
        rows, _, _ = buffer.runtime.dispatch_with_permute(hidden=grad, handle=state.handle, with_probs=False)  # fmt: skip
        return rows
    dense, _, _ = buffer.runtime.dispatch(hidden=grad, handle=state.handle, with_probs=False)
    return dense.index_select(0, state.expand_idx)


def hybrid_ep_combine_backward(state, grad, wait_event=None):
    """
    The combine's backward: the gradient of stage 3's rows, a dispatch replaying the cached handle
    (no all-gather, no scan, no host wait).
    """
    out, done = on_hybrid_stream(wait_event, combine_backward_impl, state, grad)
    return out, StreamWork(done)


@torch.no_grad()
def dispatch_backward_impl(state: HybridEPState, grad_rows: torch.Tensor):
    probs_grad = state.probs_grad
    state.probs_grad = None
    grad_rows = grad_rows.contiguous()
    if mode == "permute":
        grad_hidden, grad_probs = buffer.runtime.combine_with_unpermute(
            hidden=grad_rows,
            probs=probs_grad.float().contiguous() if probs_grad is not None else None,
            handle=state.handle,
            with_probs=probs_grad is not None,
        )
    else:
        # Scatter the slot gradients into this rank's columns of a zero [D, G * ranks per node]
        # tensor (the combine's input layout; its output is [T, E_total]): the combine sums the
        # rows over ranks, so every other column must be zero, not stale.
        G = state.experts_per_rank
        dense_probs = None
        if probs_grad is not None:
            width = G * buffer.num_of_hybrid_ep_ranks_per_nvlink_domain
            dense_probs = torch.zeros(state.num_dense, width, dtype=torch.float32, device=grad_rows.device)  # fmt: skip
            dense_probs[state.expand_idx, buffer.local_rank * G + state.expert_idxs] = probs_grad.float()  # fmt: skip
        grad_hidden, grad_probs = buffer.runtime.combine(
            hidden=grad_rows, probs=dense_probs, handle=state.handle, with_probs=dense_probs is not None
        )  # fmt: skip
    if grad_probs is not None:
        state.topk_weight_grad = grad_probs.gather(1, state.topk_idx.long())
    return grad_hidden


def hybrid_ep_dispatch_backward(state, grad_rows, wait_event=None):
    """
    The dispatch's backward: the gradient of the dispatched hidden states, summed per token over
    its expert ranks, plus (in state.topk_weight_grad) the router weights' gradient.
    """
    # Made on the compute stream and read on the HybridEP stream.
    if state.probs_grad is not None:
        state.probs_grad.record_stream(hybrid_ep_stream())
    out, done = on_hybrid_stream(wait_event, dispatch_backward_impl, state, grad_rows)
    return out, StreamWork(done)


def hybrid_ep_drop_check() -> dict:
    """
    Totals since the last call, summed over the EP group. One small all-reduce; call it after a
    step, never inside the MoE layer. Dropless holds by construction (fusion off, so the drop path
    is not compiled and the overflow flag is never written; outputs are sized from the scan's
    counts), so rows_received == rows_expected only sanity-checks topk_idx (no -1 entries); the live
    check is slot_order_violation.
    """
    dev = distributed.device
    flags = HybridEPStats.flags if HybridEPStats.flags is not None else torch.zeros(2, dtype=torch.int32, device=dev)  # fmt: skip
    t = torch.tensor([HybridEPStats.rows_received, HybridEPStats.rows_expected], dtype=torch.int64, device=dev)  # fmt: skip
    t = torch.cat([t, flags.to(torch.int64)])
    dist.all_reduce(t, group=distributed.ep_group)
    received, expected, overflow, slot_order = t.tolist()
    HybridEPStats.rows_received = HybridEPStats.rows_expected = 0
    if HybridEPStats.flags is not None:
        HybridEPStats.flags.zero_()
    return dict(rows_received=received, rows_expected=expected, overflow=overflow, slot_order_violation=slot_order, dropped=expected - received)  # fmt: skip


@torch.compile(dynamic=True, fullgraph=True)
def weight_rows_fwd(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.float() * w.unsqueeze(-1)).to(x.dtype)


@torch.compile(dynamic=True, fullgraph=True)
def weight_rows_bwd(grad: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    grad_f = grad.float()
    return (grad_f * w.unsqueeze(-1)).to(x.dtype), (grad_f * x.float()).sum(-1)


class WeightRowsFunc(torch.autograd.Function):
    """
    (x.float() * w[:, None]).to(x.dtype), saving x in its own dtype. Forward and backward are each
    one fused Inductor kernel (dynamic in the row count), not eager fp32 temporaries: stage 3 is
    not a compiled region, unlike stage 5 where the all-to-all path weights its rows.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        return weight_rows_fwd(x, w)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_x, grad_w = weight_rows_bwd(grad.contiguous(), x, w)
        return grad_x, grad_w


def weight_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return WeightRowsFunc.apply(x, w)
