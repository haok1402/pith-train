"""
The split expert-parallel dispatch (prepare_dispatch_issue + PendingDispatch.finish) produces
exactly what the one-shot r5-base prepare_dispatch did.

    torchrun --nproc-per-node=8 tests/test_dispatch_split.py

Runs at EP=8 (all ranks) and EP=4 (two groups of four). Both paths are fed the same outputs of the
dedup kernels (whose atomics order tokens nondeterministically from one launch to the next), so
everything downstream of them, which is what the split changes, is compared with torch.equal:
the metadata all-to-all, the host split arithmetic, the expand_idx all-to-all, the expert index
kernel and the token gather. Routings include senders that send nothing to some ranks and a rank
with no tokens at all.
"""

import os

import torch
import torch.distributed as dist

import pithtrain.operators.ep_dispatch as ep_dispatch
from pithtrain.kernels import expert_and_expand_idx
from pithtrain.operators.ep_dispatch import prepare_dispatch, prepare_dispatch_issue
from pithtrain.operators.routing import AllToAllSplits, RoutingInfo
from pithtrain.operators.token_scatter import padded_index_gather


def reference_prepare_dispatch(hidden_states, topk_ids, topk_weight, num_experts, ep_size, experts_per_rank, ep_group):  # fmt: skip
    """r5-base prepare_dispatch for ep_size > 1, verbatim but for the kernel call."""
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    stride = experts_per_rank + 1
    meta = torch.empty(2 * ep_size + ep_size * stride, dtype=torch.int64, device=topk_ids.device)
    (tokens_per_ep_rank, dedup_tokens_per_gpu, dispatch_token_idxs, idxs, expand_idx, send_meta) = (
        ep_dispatch.fused_dedup_prepare_dispatch(topk_ids, num_experts, ep_size, experts_per_rank, meta)
    )  # fmt: skip
    recv_meta = meta[2 * ep_size :]
    torch.compiler.disable(dist.all_to_all_single)(recv_meta, send_meta, group=ep_group)
    meta_cpu = torch.empty(meta.shape[0], dtype=meta.dtype, pin_memory=True)
    meta_cpu.copy_(meta, non_blocking=True)
    torch.cuda.current_stream(hidden_states.device).synchronize()
    host = meta_cpu.tolist()
    input_splits = host[:ep_size]
    dedup_input_splits = host[ep_size : 2 * ep_size]
    recv = host[2 * ep_size :]
    dedup_output_splits = recv[experts_per_rank::stride]
    output_splits = [sum(recv[s * stride : s * stride + experts_per_rank]) for s in range(ep_size)]
    expert_counts = [sum(recv[e::stride][:ep_size]) for e in range(experts_per_rank)]
    received_expand_idx = expand_idx.new_empty(sum(output_splits))
    torch.compiler.disable(dist.all_to_all_single)(received_expand_idx, expand_idx, output_splits, input_splits, ep_group)  # fmt: skip
    expert_idxs, expand_idx = expert_and_expand_idx(recv_meta, received_expand_idx, ep_size, experts_per_rank)  # fmt: skip
    total_dedup = sum(dedup_input_splits)
    dispatch_tokens = padded_index_gather(hidden_states, dispatch_token_idxs[:total_dedup])
    return dispatch_tokens, RoutingInfo(
        topk_weight, expert_idxs, idxs, expand_idx,
        AllToAllSplits(dedup_input_splits, dedup_output_splits),
        AllToAllSplits(input_splits, output_splits),
        expert_counts,
    )  # fmt: skip


class FrozenKernels:
    """Replay one launch of the dedup kernels to every caller, as fresh copies."""

    def __init__(self, topk_ids, num_experts, ep_size, experts_per_rank):
        self.real = ep_dispatch.fused_dedup_prepare_dispatch
        meta = torch.empty(2 * ep_size, dtype=torch.int64, device=topk_ids.device)
        outs = self.real(topk_ids, num_experts, ep_size, experts_per_rank, meta)
        self.outs = [t.clone() for t in outs]

    def __call__(self, topk_ids, num_experts, ep_size, experts_per_rank, meta_out=None):
        ep = ep_size
        if meta_out is None:
            meta_out = torch.empty(2 * ep, dtype=torch.int64, device=topk_ids.device)
        meta_out[:ep].copy_(self.outs[0])
        meta_out[ep : 2 * ep].copy_(self.outs[1])
        return (meta_out[:ep], meta_out[ep : 2 * ep], *[t.clone() for t in self.outs[2:]])


def make_routing(rank, m, k, num_experts, ep_size, experts_per_rank, seed):
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    if rank == 1:
        m = 0  # a rank without tokens
    scores = torch.rand(m, num_experts, generator=g, device="cuda")
    if rank % 3 == 0 and m:
        # this sender only routes to the first half of the EP ranks: zero splits elsewhere
        scores[:, (ep_size // 2) * experts_per_rank :] = -1
    topk_ids = scores.topk(k, dim=-1).indices
    topk_weight = torch.rand(m, k, generator=g, device="cuda").requires_grad_()
    hidden = torch.randn(m, 256, generator=g, device="cuda", dtype=torch.bfloat16)
    return hidden.requires_grad_(), topk_ids, topk_weight


def assert_same(a, b, what):
    tokens_a, info_a = a
    tokens_b, info_b = b
    assert torch.equal(tokens_a, tokens_b), what
    for name, x, y in zip(RoutingInfo._fields, info_a, info_b):
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and torch.equal(x, y), f"{what}: {name}"
        else:
            assert x == y, f"{what}: {name} {x} != {y}"


def run_group(ep_group, ep_size, rank):
    experts_per_rank, k, m = 4, 6, 300
    num_experts = ep_size * experts_per_rank
    for seed in range(3):
        hidden, topk_ids, topk_weight = make_routing(rank, m, k, num_experts, ep_size, experts_per_rank, seed)  # fmt: skip
        args = (hidden, topk_ids, topk_weight, num_experts, ep_size, experts_per_rank, ep_group)
        frozen = FrozenKernels(topk_ids, num_experts, ep_size, experts_per_rank)
        ep_dispatch.fused_dedup_prepare_dispatch = frozen
        try:
            ref = reference_prepare_dispatch(*args)
            one_shot = prepare_dispatch(*args)
            assert_same(ref, one_shot, "one-shot")
            pending = prepare_dispatch_issue(*args)
            # Independent GPU work enqueued between issue and finish changes nothing.
            x = torch.randn(2048, 2048, device="cuda")
            for _ in range(4):
                x = x @ x / 2048
            split = pending.finish()
            assert_same(ref, split, "split")
        finally:
            ep_dispatch.fused_dedup_prepare_dispatch = frozen.real
        # The finished outputs keep the autograd link to the inputs.
        tokens, info = one_shot
        if tokens.numel():
            assert tokens.requires_grad and info.topk_weight is topk_weight


def check_misuse(ep_group, ep_size, rank):
    experts_per_rank, k, m = 4, 6, 64
    num_experts = ep_size * experts_per_rank
    hidden, topk_ids, topk_weight = make_routing(rank, m, k, num_experts, ep_size, experts_per_rank, 7)  # fmt: skip
    args = (hidden, topk_ids, topk_weight, num_experts, ep_size, experts_per_rank, ep_group)
    pending = prepare_dispatch_issue(*args)
    # A second dispatch would overwrite the pinned metadata buffer the first still needs.
    try:
        prepare_dispatch_issue(*args)
        raise RuntimeError("issue with a dispatch outstanding did not raise")
    except AssertionError:
        pass
    pending.finish()
    try:
        pending.finish()
        raise RuntimeError("second finish() did not raise")
    except AssertionError:
        pass


def check_no_aliasing(ep_group, ep_size, rank):
    """A dispatch's outputs are untouched by the next dispatch (scratch reuse is internal)."""
    experts_per_rank, k, m = 4, 6, 300
    num_experts = ep_size * experts_per_rank
    first = prepare_dispatch(*make_routing(rank, m, k, num_experts, ep_size, experts_per_rank, 11)[:3], num_experts, ep_size, experts_per_rank, ep_group)  # fmt: skip
    saved = [t.clone() if isinstance(t, torch.Tensor) else t for t in (first[0], *first[1])]
    prepare_dispatch(*make_routing(rank, m, k, num_experts, ep_size, experts_per_rank, 12)[:3], num_experts, ep_size, experts_per_rank, ep_group)  # fmt: skip
    torch.cuda.synchronize()
    after = (first[0], *first[1])
    assert not outputs_aliased(saved, after), "a dispatch output was overwritten by the next one"
    # The check itself catches an overwritten output.
    corrupted = list(after)
    corrupted[4] = after[4].clone().fill_(-1) if after[4].numel() else torch.ones(1)
    assert outputs_aliased(saved, corrupted)


def outputs_aliased(saved, now):
    for a, b in zip(saved, now):
        if isinstance(a, torch.Tensor) and (a.shape != b.shape or not torch.equal(a, b)):
            return True
    return False


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert world == 8, "run with --nproc-per-node=8"
    groups = {8: dist.new_group(list(range(8)))}
    halves = [dist.new_group([0, 1, 2, 3]), dist.new_group([4, 5, 6, 7])]
    groups[4] = halves[rank // 4]
    for ep_size, group in groups.items():
        run_group(group, ep_size, dist.get_rank(group))
        check_misuse(group, ep_size, dist.get_rank(group))
        check_no_aliasing(group, ep_size, dist.get_rank(group))
        dist.barrier()
        if rank == 0:
            print(f"dispatch split: EP={ep_size} OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
