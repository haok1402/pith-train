"""Validates the dedup EP dispatch pipeline (single-process simulation)."""

import pytest
import torch


def simulate_sender(hidden_states, topk_ids, ep_size, experts_per_rank):
    m, k = topk_ids.shape
    num_experts = ep_size * experts_per_rank
    expert_idxs = topk_ids.view(-1)

    cnts = topk_ids.new_zeros((m, num_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_idxs.argsort()

    tokens_per_ep_rank = tokens_per_expert.view(ep_size, -1).sum(dim=1)
    input_splits = tokens_per_ep_rank.tolist()

    gpu_ids = topk_ids // experts_per_rank
    cnts_dedup = topk_ids.new_zeros((m, ep_size))
    cnts_dedup.scatter_(1, gpu_ids, 1)

    nz = cnts_dedup.T.nonzero()
    dispatch_token_idxs = nz[:, 1]
    dedup_sorted_tokens = hidden_states[dispatch_token_idxs]

    dedup_tokens_per_gpu = cnts_dedup.sum(dim=0)
    dedup_input_splits = dedup_tokens_per_gpu.tolist()

    token_ids = idxs // k
    gpu_ids_sorted = gpu_ids.view(-1)[idxs]
    nz_keys = nz[:, 0] * m + nz[:, 1]
    query_keys = gpu_ids_sorted * m + token_ids
    global_pos = torch.searchsorted(nz_keys, query_keys)
    gpu_starts = dedup_tokens_per_gpu.cumsum(0) - dedup_tokens_per_gpu
    expand_idx = global_pos - gpu_starts[gpu_ids_sorted]

    ref_tokens = hidden_states[idxs // k]

    def _split(tensor, splits):
        return dict(zip(range(ep_size), tensor.split(splits)))

    return {
        "input_splits": input_splits,
        "dedup_input_splits": dedup_input_splits,
        "tokens_per_expert": tokens_per_expert,
        "dedup_chunks": _split(dedup_sorted_tokens, dedup_input_splits),
        "expand_idx_chunks": _split(expand_idx, input_splits),
        "ref_chunks": _split(ref_tokens, input_splits),
    }


def simulate_receiver(sender_data_list, receiver_gpu, ep_size, experts_per_rank):
    g = receiver_gpu
    dedup_output_splits = []
    output_splits = []

    for sender in sender_data_list:
        tpe_local = sender["tokens_per_expert"][g * experts_per_rank : (g + 1) * experts_per_rank]
        output_splits.append(tpe_local.sum().item())
        dedup_output_splits.append(sender["dedup_input_splits"][g])

    dedup_gathered = torch.cat([s["dedup_chunks"][g] for s in sender_data_list])
    received_expand_idx = torch.cat([s["expand_idx_chunks"][g] for s in sender_data_list])

    dedup_counts = torch.tensor(dedup_output_splits, dtype=received_expand_idx.dtype)
    dedup_starts = dedup_counts.cumsum(0) - dedup_counts
    offset_adj = dedup_starts.repeat_interleave(torch.tensor(output_splits, dtype=torch.long))
    adjusted = received_expand_idx + offset_adj

    expanded = dedup_gathered[adjusted]
    reference = torch.cat([s["ref_chunks"][g] for s in sender_data_list])
    return expanded, reference, adjusted, sum(dedup_output_splits)


CONFIGS = [
    (32, 8, 32, 8),
    (64, 4, 16, 4),
    (128, 8, 64, 8),
    (16, 2, 8, 4),
    (100, 8, 32, 4),
    (256, 8, 128, 16),
    (10, 4, 8, 2),
    (32, 1, 8, 4),  # k=1: no dedup
    (32, 4, 8, 2),  # many experts per rank
    (2, 2, 4, 2),  # tiny
    (1, 2, 8, 4),  # single token
    (2048, 8, 128, 2),  # Qwen3-30B-A3B: many experts, few EP ranks
]


def _reference_dispatch_token_idxs(topk_ids_cpu, ep_size, experts_per_rank):
    """Derive per-GPU dispatch token index sets from the reference nonzero path."""
    m = topk_ids_cpu.shape[0]
    gpu_ids = topk_ids_cpu // experts_per_rank
    cnts_dedup = topk_ids_cpu.new_zeros((m, ep_size))
    cnts_dedup.scatter_(1, gpu_ids, 1)
    nz = cnts_dedup.T.nonzero()
    dispatch_idxs = nz[:, 1]
    dedup_per_gpu = cnts_dedup.sum(dim=0).long()
    gpu_starts = dedup_per_gpu.cumsum(0) - dedup_per_gpu
    return dispatch_idxs, dedup_per_gpu, gpu_starts


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_fused_dedup_dispatch(ms, k, num_experts, ep_size, seed):
    """Compare fused Triton kernel outputs against the PyTorch reference."""
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch

    torch.manual_seed(seed)
    experts_per_rank = num_experts // ep_size
    H = 64
    device = "cuda"

    hidden_states = torch.randn(ms, H, device=device)
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:k] for _ in range(ms)])

    # Reference (CPU)
    ref = simulate_sender(hidden_states.cpu(), topk_ids.cpu(), ep_size, experts_per_rank)

    # Fused kernel
    (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,
    ) = fused_dedup_prepare_dispatch(topk_ids, num_experts, ep_size, experts_per_rank)

    # -- Deterministic checks (exact match) --
    ref_tokens_per_ep_rank = ref["tokens_per_expert"].view(ep_size, -1).sum(dim=1)
    assert torch.equal(tokens_per_ep_rank.cpu(), ref_tokens_per_ep_rank)
    ref_dedup_tokens_per_gpu = torch.tensor(ref["dedup_input_splits"], dtype=torch.int64)
    assert torch.equal(dedup_tokens_per_gpu.cpu(), ref_dedup_tokens_per_gpu)

    # -- send_meta interleaved layout (embeds tokens_per_expert + dedup counts) --
    ref_send_meta = torch.cat(
        [
            ref["tokens_per_expert"].view(ep_size, experts_per_rank),
            ref_dedup_tokens_per_gpu.unsqueeze(1),
        ],
        dim=1,
    ).view(-1)
    assert torch.equal(send_meta.cpu(), ref_send_meta), "send_meta layout mismatch"

    # -- Set equality per GPU chunk for dispatch_token_idxs --
    gpu_starts = dedup_tokens_per_gpu.cumsum(0) - dedup_tokens_per_gpu
    ref_dispatch, _, ref_gpu_starts = _reference_dispatch_token_idxs(
        topk_ids.cpu(), ep_size, experts_per_rank
    )
    for g in range(ep_size):
        count = ref["dedup_input_splits"][g]
        if count == 0:
            continue
        our_start = gpu_starts[g].item()
        our_set = set(dispatch_token_idxs[our_start : our_start + count].cpu().tolist())
        ref_start = ref_gpu_starts[g].item()
        ref_set = set(ref_dispatch[ref_start : ref_start + count].tolist())
        assert our_set == ref_set, f"GPU {g}: dispatch_token_idxs mismatch"

    # -- Semantic consistency: expand_idx correctness --
    if ms > 0 and k > 0:
        token_ids = idxs // k
        expert_ids_sorted = topk_ids.view(-1)[idxs]
        gpu_ids_sorted = expert_ids_sorted // experts_per_rank
        gpu_starts_dev = gpu_starts.to(device)
        gathered = dispatch_token_idxs[gpu_starts_dev[gpu_ids_sorted] + expand_idx]
        assert torch.equal(gathered, token_ids), "expand_idx semantic invariant violated"


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_fused_dedup_end_to_end(ms, k, num_experts, ep_size, seed):
    """Full sender->receiver pipeline using fused kernel, verifying expanded == reference."""
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch

    torch.manual_seed(seed)
    experts_per_rank = num_experts // ep_size
    H = 64
    device = "cuda"

    sender_data_list = []
    for _ in range(ep_size):
        h = torch.randn(ms, H, device=device)
        ids = torch.stack([torch.randperm(num_experts, device=device)[:k] for _ in range(ms)])
        (
            tokens_per_ep_rank,
            dedup_tokens_per_gpu,
            dispatch_token_idxs,
            idxs,
            expand_idx,
            send_meta,
        ) = fused_dedup_prepare_dispatch(ids, num_experts, ep_size, experts_per_rank)

        # Extract tokens_per_expert from send_meta interleaved layout
        meta_2d = send_meta.view(ep_size, experts_per_rank + 1)
        tokens_per_expert = meta_2d[:, :experts_per_rank].reshape(-1)

        total_dedup = dedup_tokens_per_gpu.sum().item()
        dispatch_token_idxs = dispatch_token_idxs[:total_dedup]
        dedup_sorted_tokens = h[dispatch_token_idxs]

        dedup_input_splits = dedup_tokens_per_gpu.cpu().tolist()
        input_splits = tokens_per_ep_rank.cpu().tolist()
        ref_tokens = h[idxs // k]

        def _split(tensor, splits, ep=ep_size):
            return dict(zip(range(ep), tensor.cpu().split(splits)))

        sender_data_list.append(
            {
                "input_splits": input_splits,
                "dedup_input_splits": dedup_input_splits,
                "tokens_per_expert": tokens_per_expert.cpu(),
                "dedup_chunks": _split(dedup_sorted_tokens, dedup_input_splits),
                "expand_idx_chunks": _split(expand_idx, input_splits),
                "ref_chunks": _split(ref_tokens, input_splits),
            }
        )

    for g in range(ep_size):
        expanded, reference, adjusted, dedup_total = simulate_receiver(
            sender_data_list, g, ep_size, experts_per_rank
        )
        assert torch.equal(expanded, reference), f"GPU {g}: expanded != reference"
        if adjusted.numel() > 0:
            assert adjusted.min() >= 0
            assert adjusted.max() < dedup_total


def test_fused_dedup_dispatch_m_zero():
    """Edge case: m=0 should return empty tensors."""
    from pithtrain.operators.ep_dispatch import fused_dedup_prepare_dispatch

    topk_ids = torch.empty((0, 4), dtype=torch.int64, device="cuda")
    (
        tokens_per_ep_rank,
        dedup_tokens_per_gpu,
        dispatch_token_idxs,
        idxs,
        expand_idx,
        send_meta,
    ) = fused_dedup_prepare_dispatch(topk_ids, num_experts=16, ep_size=4, experts_per_rank=4)

    assert tokens_per_ep_rank.shape == (4,)
    assert dedup_tokens_per_gpu.shape == (4,)
    assert idxs.numel() == 0
    assert expand_idx.numel() == 0
    assert send_meta.shape == (4 * (4 + 1),)
    assert send_meta.sum() == 0


# -- Unit tests for post-all-to-all fused kernels --


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_adjust_expand_idx(ms, k, num_experts, ep_size, seed):
    """Compare fused adjust_expand_idx against PyTorch reference."""
    from pithtrain.operators.ep_dispatch import adjust_expand_idx

    torch.manual_seed(seed)
    device = "cuda"

    # Generate realistic inputs: simulate received data from ep_size senders
    dedup_tokens_from_each_gpu = torch.randint(1, max(ms, 2), (ep_size,), device=device)
    output_splits_tensor = torch.randint(1, max(ms * k // ep_size, 2), (ep_size,), device=device)
    total = output_splits_tensor.sum().item()
    total_dedup = dedup_tokens_from_each_gpu.sum().item()

    received_expand_idx = torch.randint(0, max(total_dedup, 1), (total,), device=device)

    # Reference (PyTorch)
    dedup_starts = dedup_tokens_from_each_gpu.cumsum(0) - dedup_tokens_from_each_gpu
    offset_adj = dedup_starts.repeat_interleave(output_splits_tensor)
    ref = received_expand_idx + offset_adj

    # Fused kernel
    result = adjust_expand_idx(
        received_expand_idx, dedup_tokens_from_each_gpu, output_splits_tensor
    )

    assert torch.equal(result, ref), "adjust_expand_idx mismatch"


@pytest.mark.parametrize("ms,k,num_experts,ep_size", CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_build_expert_idxs(ms, k, num_experts, ep_size, seed):
    """Compare fused build_expert_idxs against PyTorch reference."""
    from pithtrain.operators.ep_dispatch import build_expert_idxs

    torch.manual_seed(seed)
    experts_per_rank = num_experts // ep_size
    device = "cuda"

    # Generate realistic tokens_per_expert_group
    tokens_per_expert_group = torch.randint(0, max(ms, 2), (num_experts,), device=device)

    # Reference (PyTorch)
    ref_expert_idxs = (
        torch.arange(num_experts, device=device) % experts_per_rank
    ).repeat_interleave(tokens_per_expert_group)

    total = int(tokens_per_expert_group.sum())
    expert_idxs = build_expert_idxs(tokens_per_expert_group, experts_per_rank, total)

    assert expert_idxs.numel() == total, "one entry per received token slot"
    assert torch.equal(expert_idxs, ref_expert_idxs), "expert_idxs mismatch"


def test_build_expert_idxs_does_not_write_past_its_buffer():
    """The store mask must come from the allocation, not from a device-side sum.

    The kernel fills ``sum(tokens_per_expert_group)`` entries but is launched over whole
    BLOCK-sized CTAs, so the last CTA covers offsets past the end. Sizing the output from
    the same number the mask uses is what keeps that tail masked off. The previous version
    took the mask bound from ``tokens_per_expert_group`` and the size from the caller, so
    whenever the caller guessed low it wrote the difference into the next allocation --
    which in practice was ``output_splits_tensor``, i.e. it corrupted the all-to-all split
    sizes rather than any tensor the MoE math would notice.
    """
    from pithtrain.operators.ep_dispatch import build_expert_idxs

    device = "cuda"
    ep_size, experts_per_rank = 8, 16
    num_experts = ep_size * experts_per_rank

    # Not a multiple of the kernel's BLOCK, so the last CTA really does have a tail.
    total = 5501
    counts = torch.full((num_experts,), total // num_experts, dtype=torch.int64, device=device)
    counts[: total % num_experts] += 1
    assert int(counts.sum()) == total

    # Canary slab behind the output, so an overrun is observable rather than left to
    # whatever the caching allocator happened to hand out next.
    slab = torch.full((total + 4096,), -1, dtype=torch.int64, device=device)
    slab[:total] = build_expert_idxs(counts, experts_per_rank, total)
    torch.cuda.synchronize()
    assert int((slab[total:] != -1).sum()) == 0, "kernel wrote past the end of expert_idxs"


def _ragged_dispatch_worker(rank, world_size, tokens_per_rank, out):
    """One EP rank of test_prepare_dispatch_ragged_token_counts."""
    import os

    import torch.distributed as dist

    from pithtrain.operators.all_to_all import direct_all_to_all
    from pithtrain.operators.ep_dispatch import prepare_dispatch
    from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    ep_size, experts_per_rank, k, hidden = world_size, 4, 4, 64
    num_experts = ep_size * experts_per_rank
    device = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank)

    m = tokens_per_rank[rank]
    hidden_states = torch.randn(1, m, hidden, device=device, dtype=torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:k] for _ in range(m)])
    topk_weight = torch.rand(m, k, device=device)

    dispatch_tokens, routing = prepare_dispatch(
        hidden_states,
        topk_ids,
        topk_weight,
        num_experts,
        ep_size,
        experts_per_rank,
        dist.group.WORLD,
    )
    gathered = direct_all_to_all(
        dispatch_tokens.detach(),
        routing.dispatch_splits.output_splits,
        routing.dispatch_splits.input_splits,
        dist.group.WORLD,
    )
    work = getattr(gathered, "comm_work", None)
    if work is not None:
        work.wait()

    # The two lines of forward_stage3 that consume the routing info.
    expanded = padded_index_gather(gathered, routing.expand_idx)
    scatter_for_grouped_gemm(expanded, routing.expert_idxs, experts_per_rank)

    out.append(
        (
            rank,
            m,
            expanded.shape[0],
            routing.expert_idxs.numel(),
            sum(routing.combine_splits.output_splits),
        )
    )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_prepare_dispatch_ragged_token_counts():
    """A rank holding a short packed micro-batch still receives every slot routed at it.

    The received row count is decided by the *senders*, so it has nothing to do with this
    rank's own token count: ``expert_idxs`` must have one entry per received row, the same
    length as ``expand_idx``, which is what stage 3 gathers by.

    The skew (8 tokens against 4096) is what packed / ragged micro-batching produces and
    fixed-shape pretraining never does. Every other case in this file gives all senders the
    same ``ms``, and with equal token counts the old local bound ``m * k * ep_size`` is
    always large enough, so nothing goes wrong.
    """
    import torch.multiprocessing as mp

    world_size = 2
    manager = mp.Manager()
    out = manager.list()
    mp.spawn(_ragged_dispatch_worker, args=(world_size, [8, 4096], out), nprocs=world_size)

    assert len(out) == world_size
    for rank, m, rows, idxs, expected in sorted(out):
        assert rows == expected, f"rank {rank}: gathered {rows} rows, expected {expected}"
        assert idxs == expected, (
            f"rank {rank} (m={m}): expert_idxs has {idxs} entries but {expected} rows were "
            f"received; the old code capped it at m * k * ep_size = {m * 4 * world_size}"
        )
