"""
Triton kernels of the ep_dispatch feature (SM100 path, and the SM90 fallback). The sender side is
three launches: a per-CTA bincount (tl.histogram, no atomics), a single-CTA reduce + prefix sum that
also builds send_meta, and a dedup scatter + counting sort that places rows with atomic_add, so the
order within a bucket is launch-dependent.
"""

import torch
import triton
import triton.language as tl

__all__ = ["expert_and_expand_idx_sm100", "fused_dedup_prepare_dispatch_sm100"]


@triton.jit
def dedup_bincount_kernel(
    topk_ids_ptr,
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64, output
    per_cta_gpu_hist_ptr,  # [num_ctas * EP_PADDED] int64, output
    m,
    stride_topk_m,  # topk_ids.stride(0)
    K: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NE_PADDED: tl.constexpr,  # next_power_of_2(NUM_EXPERTS + 1)
    EP_PADDED: tl.constexpr,  # next_power_of_2(EP_SIZE + 1)
    BLOCK: tl.constexpr,
):
    """Per-CTA histograms of slots per expert and of distinct tokens per EP rank."""
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)

    local_expert_hist = tl.zeros([NE_PADDED], dtype=tl.int64)
    local_gpu_hist = tl.zeros([EP_PADDED], dtype=tl.int64)

    for start in range(pid * BLOCK, m, num_ctas * BLOCK):
        tok_offs = start + tl.arange(0, BLOCK)
        mask = tok_offs < m

        # per-token visited-rank bitmask: int32 limits ep_size to 32
        visited = tl.zeros([BLOCK], dtype=tl.int32)

        for j in tl.static_range(K):
            expert_id = tl.load(topk_ids_ptr + tok_offs * stride_topk_m + j, mask=mask, other=0)

            # masked lanes count into the sentinel bin NUM_EXPERTS
            safe_expert_id = tl.where(mask, expert_id, NUM_EXPERTS).to(tl.int32)
            local_expert_hist += tl.histogram(safe_expert_id, NE_PADDED).to(tl.int64)

            gpu_id = expert_id // EXPERTS_PER_RANK
            gpu_bit = (1 << gpu_id).to(tl.int32)
            already_visited = (visited & gpu_bit) != 0
            first_visit = mask & ~already_visited
            visited = visited | gpu_bit

            safe_gpu_id = tl.where(first_visit, gpu_id, EP_SIZE).to(tl.int32)
            local_gpu_hist += tl.histogram(safe_gpu_id, EP_PADDED).to(tl.int64)

    expert_offs = tl.arange(0, NE_PADDED)
    tl.store(per_cta_expert_hist_ptr + pid * NE_PADDED + expert_offs, local_expert_hist)
    gpu_offs = tl.arange(0, EP_PADDED)
    tl.store(per_cta_gpu_hist_ptr + pid * EP_PADDED + gpu_offs, local_gpu_hist)


@triton.jit
def reduce_and_prefix_sum_kernel(
    per_cta_expert_hist_ptr,  # [num_ctas * NE_PADDED] int64
    per_cta_gpu_hist_ptr,  # [num_ctas * EP_PADDED] int64
    dedup_tokens_per_gpu_ptr,  # [EP_SIZE] int64
    tokens_per_ep_rank_ptr,  # [EP_SIZE] int64
    expert_starts_ptr,  # [NUM_EXPERTS] int64
    gpu_starts_ptr,  # [EP_SIZE] int64
    dedup_counters_ptr,  # [EP_SIZE] int64 (zeroed)
    sort_counters_ptr,  # [NUM_EXPERTS] int64 (zeroed)
    send_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64
    NUM_CTAS: tl.constexpr,
    NE_PADDED: tl.constexpr,
    EP_PADDED: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """
    Single CTA: reduce the per-CTA histograms into the counts, their exclusive starts and send_meta,
    and zero kernel 3's atomic counters.
    """
    expert_prefix = tl.zeros([], dtype=tl.int64)
    gpu_prefix = tl.zeros([], dtype=tl.int64)
    for g in tl.static_range(EP_SIZE):
        rank_total = tl.zeros([], dtype=tl.int64)
        for e in tl.static_range(EXPERTS_PER_RANK):
            expert_val = tl.zeros([], dtype=tl.int64)
            for c in range(NUM_CTAS):
                expert_val += tl.load(per_cta_expert_hist_ptr + c * NE_PADDED + g * EXPERTS_PER_RANK + e)
            tl.store(expert_starts_ptr + g * EXPERTS_PER_RANK + e, expert_prefix)
            expert_prefix += expert_val
            rank_total += expert_val
            tl.store(send_meta_ptr + g * (EXPERTS_PER_RANK + 1) + e, expert_val)

        tl.store(tokens_per_ep_rank_ptr + g, rank_total)

        gpu_val = tl.zeros([], dtype=tl.int64)
        for c in range(NUM_CTAS):
            gpu_val += tl.load(per_cta_gpu_hist_ptr + c * EP_PADDED + g)
        tl.store(dedup_tokens_per_gpu_ptr + g, gpu_val)
        tl.store(gpu_starts_ptr + g, gpu_prefix)
        gpu_prefix += gpu_val
        tl.store(send_meta_ptr + g * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK, gpu_val)

        tl.store(dedup_counters_ptr + g, tl.zeros([], dtype=tl.int64))

    for e in tl.static_range(NUM_EXPERTS):
        tl.store(sort_counters_ptr + e, tl.zeros([], dtype=tl.int64))


@triton.jit
def dedup_scatter_expand_kernel(
    topk_ids_ptr,
    expert_starts_ptr,  # [num_experts] int64 (from kernel 1)
    gpu_starts_ptr,  # [ep_size] int64 (from kernel 1)
    dedup_counters_ptr,  # [ep_size] int64, pre-zeroed (atomics)
    sort_counters_ptr,  # [num_experts] int64, pre-zeroed (atomics)
    dispatch_token_idxs_ptr,  # [total_dedup] int64, output
    dedup_local_pos_ptr,  # [m * ep_size] int64, scratch
    idxs_ptr,  # [m * k] int64, output
    expand_idx_ptr,  # [m * k] int64, output
    m,
    stride_topk_m,
    K: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Pass 1 places each token once per visited rank in dispatch_token_idxs and records its position
    in dedup_local_pos; pass 2 counting-sorts the slots by expert into idxs and looks up
    expand_idx from dedup_local_pos.
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_M
    tok_offs = tok_start + tl.arange(0, BLOCK_M)
    mask = tok_offs < m

    visited = tl.zeros([BLOCK_M], dtype=tl.int32)

    for j in tl.static_range(K):
        expert_id = tl.load(topk_ids_ptr + tok_offs * stride_topk_m + j, mask=mask, other=0)
        gpu_id = expert_id // EXPERTS_PER_RANK

        gpu_bit = (1 << gpu_id).to(tl.int32)
        already_visited = (visited & gpu_bit) != 0
        first_visit = mask & ~already_visited
        visited = visited | gpu_bit

        dedup_pos = tl.atomic_add(dedup_counters_ptr + gpu_id, 1, mask=first_visit)
        gpu_start = tl.load(gpu_starts_ptr + gpu_id, mask=first_visit, other=0)
        tl.store(dispatch_token_idxs_ptr + gpu_start + dedup_pos, tok_offs, mask=first_visit)
        tl.store(dedup_local_pos_ptr + tok_offs * EP_SIZE + gpu_id, dedup_pos, mask=first_visit)

    for j in tl.static_range(K):
        expert_id = tl.load(topk_ids_ptr + tok_offs * stride_topk_m + j, mask=mask, other=0)
        gpu_id = expert_id // EXPERTS_PER_RANK

        expert_start = tl.load(expert_starts_ptr + expert_id, mask=mask, other=0)
        sort_pos = tl.atomic_add(sort_counters_ptr + expert_id, 1, mask=mask)
        sorted_idx = expert_start + sort_pos

        flat_idx = tok_offs * K + j
        tl.store(idxs_ptr + sorted_idx, flat_idx, mask=mask)

        local_pos = tl.load(dedup_local_pos_ptr + tok_offs * EP_SIZE + gpu_id, mask=mask, other=0)
        tl.store(expand_idx_ptr + sorted_idx, local_pos, mask=mask)


# Scratch of fused_dedup_prepare_dispatch_sm100, one int64 allocation per shape and stream. Only
# temporaries consumed in stream order before the next call on the same stream live here (and
# send_meta, documented as valid until then); outputs that outlive the call are fresh.
scratch: dict[tuple, tuple[torch.Tensor, ...]] = {}
SCRATCH_ENTRIES = 8


def dispatch_scratch(device: torch.device, m: int, num_ctas: int, ne_padded: int, ep_padded: int, num_experts: int, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, ...]:
    key = (device, torch.cuda.current_stream(device).cuda_stream, m, num_experts, ep_size)
    views = scratch.get(key)
    if views is None:
        sizes = (
            num_ctas * ne_padded,  # per_cta_expert_hist
            num_ctas * ep_padded,  # per_cta_gpu_hist
            num_experts,  # expert_starts
            ep_size,  # gpu_starts
            ep_size,  # dedup_counters
            num_experts,  # sort_counters
            ep_size * (experts_per_rank + 1),  # send_meta
            m * ep_size,  # dedup_local_pos
        )
        flat = torch.empty(sum(sizes), dtype=torch.int64, device=device)
        views = tuple(flat.split(sizes))
        if len(scratch) >= SCRATCH_ENTRIES:
            scratch.clear()
        scratch[key] = views
    return views


def fused_dedup_prepare_dispatch_sm100(topk_ids: torch.Tensor, num_experts: int, ep_size: int, experts_per_rank: int, meta_out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Triton path of fused_dedup_prepare_dispatch (atomic order within each bucket)."""
    m, k = topk_ids.shape
    device = topk_ids.device
    assert ep_size <= 32, f"ep_size={ep_size} exceeds int32 bitmask limit of 32"

    if m == 0:
        send_meta = torch.zeros(ep_size * (experts_per_rank + 1), dtype=torch.int64, device=device)
        if meta_out is None:
            meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
        meta_out[: 2 * ep_size].zero_()
        return (meta_out[:ep_size], meta_out[ep_size : 2 * ep_size], torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device), torch.empty(0, dtype=torch.int64, device=device), send_meta)

    # at most 32 CTAs: kernel 2 reduces the per-CTA histograms in one CTA with NUM_CTAS constexpr
    BLOCK = 1024
    num_ctas = min(triton.cdiv(m, BLOCK), 32)
    # tl.histogram needs a power-of-2 bin count; +1 for the sentinel bin of masked-out lanes
    ne_padded = triton.next_power_of_2(num_experts + 1)
    ep_padded = triton.next_power_of_2(ep_size + 1)

    (per_cta_expert_hist, per_cta_gpu_hist, expert_starts, gpu_starts, dedup_counters, sort_counters, send_meta, dedup_local_pos) = dispatch_scratch(device, m, num_ctas, ne_padded, ep_padded, num_experts, ep_size, experts_per_rank)

    dedup_bincount_kernel[(num_ctas,)](topk_ids, per_cta_expert_hist, per_cta_gpu_hist, m, topk_ids.stride(0), K=k, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, NUM_EXPERTS=num_experts, NE_PADDED=ne_padded, EP_PADDED=ep_padded, BLOCK=BLOCK)

    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=device)
    tokens_per_ep_rank, dedup_tokens_per_gpu = meta_out[:ep_size], meta_out[ep_size : 2 * ep_size]

    reduce_and_prefix_sum_kernel[(1,)](per_cta_expert_hist, per_cta_gpu_hist, dedup_tokens_per_gpu, tokens_per_ep_rank, expert_starts, gpu_starts, dedup_counters, sort_counters, send_meta, NUM_CTAS=num_ctas, NE_PADDED=ne_padded, EP_PADDED=ep_padded, NUM_EXPERTS=num_experts, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank)

    dispatch_token_idxs = torch.empty(m * ep_size, dtype=torch.int64, device=device)
    idxs = torch.empty(m * k, dtype=torch.int64, device=device)
    expand_idx = torch.empty(m * k, dtype=torch.int64, device=device)

    BLOCK_M = 128
    grid = (triton.cdiv(m, BLOCK_M),)
    dedup_scatter_expand_kernel[grid](topk_ids, expert_starts, gpu_starts, dedup_counters, sort_counters, dispatch_token_idxs, dedup_local_pos, idxs, expand_idx, m, topk_ids.stride(0), K=k, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, BLOCK_M=BLOCK_M)

    return (tokens_per_ep_rank, dedup_tokens_per_gpu, dispatch_token_idxs, idxs, expand_idx, send_meta)


@triton.jit
def expert_and_expand_idx_kernel(
    recv_meta_ptr,  # [EP_SIZE * (EXPERTS_PER_RANK + 1)] int64, the received metadata
    received_expand_idx_ptr,  # [total] int64, input
    expert_idxs_ptr,  # [total] int64, output
    expand_idx_ptr,  # [total] int64, output
    total,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Per received slot: its local expert, found by an unrolled pass over the (sender, expert)
    segments, and its expand_idx plus the dedup counts of the preceding senders.

    total is a kernel argument, not a device-side sum, so the store mask and the caller's allocation
    are the same number (a mismatch would write out of bounds).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    expert_idx = tl.zeros([BLOCK], dtype=tl.int64)
    offset = tl.zeros([BLOCK], dtype=tl.int64)
    start = tl.zeros([], dtype=tl.int64)
    dedup_start = tl.zeros([], dtype=tl.int64)
    for g in tl.static_range(EP_SIZE):
        for e in tl.static_range(EXPERTS_PER_RANK):
            count = tl.load(recv_meta_ptr + g * (EXPERTS_PER_RANK + 1) + e)
            in_seg = (offs >= start) & (offs < start + count)
            expert_idx = tl.where(in_seg, e, expert_idx)
            offset = tl.where(in_seg, dedup_start, offset)
            start += count
        dedup_start += tl.load(recv_meta_ptr + g * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK)

    expand_val = tl.load(received_expand_idx_ptr + offs, mask=mask)
    tl.store(expert_idxs_ptr + offs, expert_idx, mask=mask)
    tl.store(expand_idx_ptr + offs, expand_val + offset, mask=mask)


def expert_and_expand_idx_sm100(recv_meta: torch.Tensor, received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The Triton path of expert_and_expand_idx."""
    # the kernel indexes both inputs with unit stride
    recv_meta, received_expand_idx = recv_meta.contiguous(), received_expand_idx.contiguous()
    total = received_expand_idx.shape[0]
    expert_idxs = torch.empty_like(received_expand_idx)
    expand_idx = torch.empty_like(received_expand_idx)
    if total == 0:
        return expert_idxs, expand_idx
    BLOCK = 1024
    expert_and_expand_idx_kernel[(triton.cdiv(total, BLOCK),)](recv_meta, received_expand_idx, expert_idxs, expand_idx, total, EP_SIZE=ep_size, EXPERTS_PER_RANK=experts_per_rank, BLOCK=BLOCK)
    return expert_idxs, expand_idx
