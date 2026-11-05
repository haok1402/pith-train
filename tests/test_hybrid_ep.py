"""
HybridEP dispatch / combine (pithtrain.operators.hybrid_ep) against a dense single-rank reference,
forward and backward, on the Qwen3-30B-A3B MoE shape with skewed routing, for every combination of
mode (permute, dedup) and sync_late. A per-row "expert" x * w[e] stands in for the MLP, so the
input gradient, the router-weight gradient and every expert's weight gradient are checked. Run
with 8 GPUs:

    torchrun --nproc-per-node=8 tests/test_hybrid_ep.py
"""

import os

import torch
import torch.distributed as dist

from pithtrain.contexts import distributed
from pithtrain.operators import hybrid_ep
from pithtrain.operators.hybrid_ep import (
    HybridEPState,
    hybrid_ep_combine,
    hybrid_ep_combine_backward,
    hybrid_ep_dispatch,
    hybrid_ep_dispatch_backward,
    hybrid_ep_dispatch_issue,
    hybrid_ep_drop_check,
    weight_rows,
)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])  # the device on this node; rank is global
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    distributed.device = torch.device("cuda", local_rank)
    distributed.ep_group, distributed.ep_size, distributed.ep_rank = dist.group.WORLD, world, rank
    hybrid_ep.custom_allgather = os.environ.get("HYBRID_EP_CUSTOM_ALLGATHER", "0") == "1"
    T, H, E, K = 4096, 2048, 128, 8
    G = E // world

    torch.manual_seed(1234)
    # Every rank builds every rank's inputs, so the reference sees all of them.
    xs = torch.randn(world, T, H, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(world, T, E, device="cuda")
    logits[:, :, :8] += 2.0  # skew the load toward rank 0's experts
    logits[3, :, 40:44] += 3.0  # and rank 3's tokens toward rank 2's
    scores = logits.softmax(-1)
    ws, idxs = torch.topk(scores, K, dim=-1)
    ws = ws / ws.sum(-1, keepdim=True)
    wexp = torch.randn(E, H, device="cuda", dtype=torch.bfloat16) * 0.1  # expert e: x * wexp[e]
    gy = torch.randn(world, T, H, device="cuda", dtype=torch.bfloat16)

    # Reference: y_t = sum_k w_tk * (x_t * wexp[e_tk])
    xr = xs.float().requires_grad_()
    wr = ws.clone().requires_grad_()
    er = wexp.float().requires_grad_()
    yr = (wr.unsqueeze(-1) * (xr.unsqueeze(2) * er[idxs])).sum(2)
    yr.backward(gy.float())

    for mode in ("permute", "dedup"):
        for sync_late in (False, True):
            hybrid_ep.mode, hybrid_ep.sync_late = mode, sync_late
            for it in range(2):
                x, w = xs[rank].clone(), ws[rank].clone()
                state = HybridEPState(idxs[rank].contiguous(), w, E, G)
                hybrid_ep_dispatch_issue(state, x)  # stage 1's early issue (no-op unless sync_late)
                _, work = hybrid_ep_dispatch(state, x)
                work.wait()
                S = state.num_slots
                assert int(state.tokens_per_expert.sum()) == S
                rows = state.recv_tokens.detach().requires_grad_()
                probs = state.recv_probs.detach().requires_grad_()
                # fp32 weights: the weight gradient sums thousands of rows per expert
                local_w = wexp[rank * G : (rank + 1) * G].float().requires_grad_()
                if mode == "dedup":
                    expert_of_slot, slot_rows = state.expert_idxs, rows[state.expand_idx]
                else:
                    counts = state.tokens_per_expert.long()
                    expert_of_slot = torch.repeat_interleave(torch.arange(G, device="cuda"), counts, output_size=S)  # fmt: skip
                    slot_rows = rows
                outs = (slot_rows.float() * local_w[expert_of_slot]).to(torch.bfloat16)
                weighted = weight_rows(outs, probs)
                y, work = hybrid_ep_combine(state, weighted.detach())
                work.wait()
                err = rel(y, yr[rank])

                g_rows, work = hybrid_ep_combine_backward(state, gy[rank])
                work.wait()
                weighted.backward(g_rows)
                state.probs_grad = probs.grad
                g_x, work = hybrid_ep_dispatch_backward(state, rows.grad)
                work.wait()
                torch.cuda.synchronize()
                ex = rel(g_x, xr.grad[rank])
                ew = rel(state.topk_weight_grad, wr.grad[rank])
                ee = rel(local_w.grad, er.grad[rank * G : (rank + 1) * G])
                stats = hybrid_ep_drop_check()
                errs = torch.tensor([err, ex, ew, ee], device="cuda")
                dist.all_reduce(errs, op=dist.ReduceOp.MAX)
                ok = (errs < 1e-2).all().item() and stats["dropped"] == 0 and stats["overflow"] == 0
                ok = ok and stats["slot_order_violation"] == 0
                if rank == 0:
                    e = errs.tolist()
                    print(f"{mode:7s} sync_late={sync_late!s:5s} iter {it}: rank0 slots {S} dense {state.num_dense} | max rel err y {e[0]:.1e} dx {e[1]:.1e} dw_router {e[2]:.1e} dw_expert {e[3]:.1e} | {stats}", flush=True)  # fmt: skip
                assert ok, (mode, sync_late, errs.tolist(), stats)
    interleaved(rank, world, xs, ws, idxs, wexp, gy, T, H, E, K, G)
    deep_interleave(rank, xs, ws, idxs, wexp, gy, E, G)
    ring_reuse(rank, xs, ws, idxs, wexp, gy, E, G)
    stage3_bf16_experts(rank, world, T, H, E, K, G)
    slot_order_negative(G)
    dist.barrier()
    if rank == 0:
        print("OK")
    dist.destroy_process_group()


def reference(xs, ws, idxs, wexp, gy):
    xr, wr, er = (
        xs.float().requires_grad_(),
        ws.clone().requires_grad_(),
        wexp.float().requires_grad_(),
    )
    yr = (wr.unsqueeze(-1) * (xr.unsqueeze(2) * er[idxs])).sum(2)
    yr.backward(gy.float())
    return yr, xr.grad, wr.grad, er.grad


class Run:
    """
    One layer-micro-batch through HybridEP with the stand-in expert, phase by phase.
    """

    def __init__(self, rank, x, w, idx, wexp, gy, E, G):
        self.rank, self.x, self.gy, self.G = rank, x, gy, G
        self.state = HybridEPState(idx.contiguous(), w, E, G)
        self.local_w = wexp[rank * G : (rank + 1) * G].float().requires_grad_()

    def issue(self):
        hybrid_ep_dispatch_issue(self.state, self.x)

    def transfer(self):
        _, work = hybrid_ep_dispatch(self.state, self.x)
        work.wait()
        st = self.state
        self.rows = st.recv_tokens.detach().requires_grad_()
        self.probs = st.recv_probs.detach().requires_grad_()
        if st.expand_idx is not None:
            expert, slot_rows = st.expert_idxs, self.rows[st.expand_idx]
        else:
            counts = st.tokens_per_expert.long()
            expert = torch.repeat_interleave(torch.arange(self.G, device="cuda"), counts, output_size=st.num_slots)  # fmt: skip
            slot_rows = self.rows
        outs = (slot_rows.float() * self.local_w[expert]).to(torch.bfloat16)
        self.weighted = weight_rows(outs, self.probs)

    def combine(self):
        self.y, work = hybrid_ep_combine(self.state, self.weighted.detach())
        work.wait()

    def combine_bwd(self):
        g_rows, work = hybrid_ep_combine_backward(self.state, self.gy)
        work.wait()
        self.weighted.backward(g_rows)
        self.state.probs_grad = self.probs.grad

    def dispatch_bwd(self):
        self.gx, work = hybrid_ep_dispatch_backward(self.state, self.rows.grad)
        work.wait()

    def errors(self, ref, rank, G):
        yr, gxr, gwr, ger = ref
        torch.cuda.synchronize()
        return [rel(self.y, yr[rank]), rel(self.gx, gxr[rank]), rel(self.state.topk_weight_grad, gwr[rank]), rel(self.local_w.grad, ger[rank * G : (rank + 1) * G])]  # fmt: skip


def interleaved(rank, world, xs, ws, idxs, wexp, gy, T, H, E, K, G):
    """
    Two layer-micro-batches in DualPipeV order: both handles alive at once, a backward replaying
    after a later forward used the shared buffer, ring slots reused.
    """
    idxs_b = (idxs + 16) % E  # a different skew: the hot experts move to rank 1
    xs_b, gy_b = xs.flip(1).contiguous(), gy.flip(1).contiguous()
    ref_a, ref_b = reference(xs, ws, idxs, wexp, gy), reference(xs_b, ws, idxs_b, wexp, gy_b)
    for mode in ("permute", "dedup"):
        for sync_late in (False, True):
            hybrid_ep.mode, hybrid_ep.sync_late = mode, sync_late
            a = Run(rank, xs[rank].clone(), ws[rank].clone(), idxs[rank], wexp, gy[rank], E, G)
            b = Run(
                rank, xs_b[rank].clone(), ws[rank].clone(), idxs_b[rank], wexp, gy_b[rank], E, G
            )
            order = [a.issue, a.transfer, a.combine, b.issue, a.combine_bwd, b.transfer]
            order += [a.dispatch_bwd, b.combine, b.combine_bwd, b.dispatch_bwd]
            for phase in order:
                phase()
            errs = torch.tensor(a.errors(ref_a, rank, G) + b.errors(ref_b, rank, G), device="cuda")
            dist.all_reduce(errs, op=dist.ReduceOp.MAX)
            stats = hybrid_ep_drop_check()
            if rank == 0:
                print(f"interleaved {mode:7s} sync_late={sync_late!s:5s}: max rel err A/B {[round(e, 4) for e in errs.tolist()]} | {stats}", flush=True)  # fmt: skip
            assert (errs < 1e-2).all().item() and stats["dropped"] == 0 and stats["slot_order_violation"] == 0, (mode, sync_late, errs.tolist(), stats)  # fmt: skip


def deep_interleave(rank, xs, ws, idxs, wexp, gy, E, G, depth=12):
    """
    DualPipeV replays a handle long after its forward: depth forwards (each with its own routing)
    before the first backward, then the backwards in reverse order.
    """
    from pithtrain.operators.hybrid_ep import PinnedRing

    shifts = [(3 * i) % E for i in range(depth)]
    refs = [reference(xs.roll(i, 1), ws, (idxs + sh) % E, wexp, gy.roll(i, 1)) for i, sh in enumerate(shifts)]  # fmt: skip
    for mode in ("permute", "dedup"):
        hybrid_ep.mode, hybrid_ep.sync_late = mode, True
        runs = [Run(rank, xs[rank].roll(i, 0).clone(), ws[rank].clone(), (idxs[rank] + sh) % E, wexp, gy[rank].roll(i, 0).contiguous(), E, G) for i, sh in enumerate(shifts)]  # fmt: skip
        for r in runs:
            for phase in (r.issue, r.transfer, r.combine):
                phase()
        for r in reversed(runs):
            r.combine_bwd()
            r.dispatch_bwd()
        errs = torch.tensor([max(r.errors(ref, rank, G)) for r, ref in zip(runs, refs)], device="cuda")  # fmt: skip
        dist.all_reduce(errs, op=dist.ReduceOp.MAX)
        stats = hybrid_ep_drop_check()
        if rank == 0:
            print(f"deep interleave {mode:7s} depth {depth}: max rel err per state {[round(e, 4) for e in errs.tolist()]} (ring next {PinnedRing.next}) | {stats}", flush=True)  # fmt: skip
        assert (errs < 1e-2).all().item() and stats["slot_order_violation"] == 0, (mode, errs.tolist(), stats)  # fmt: skip


def ring_reuse(rank, xs, ws, idxs, wexp, gy, E, G):
    """
    The pinned ring backs each handle's dense count until its backward replay. With one slot,
    B's early issue overwrites A's count before A's combine backward reads it (the corruption is
    detected on the handle, without running the corrupted replay); with two slots it must not.
    """
    from pithtrain.operators.hybrid_ep import PinnedRing

    saved = PinnedRing.size, PinnedRing.counts, PinnedRing.dense, PinnedRing.next
    hybrid_ep.mode, hybrid_ep.sync_late = "dedup", True
    idxs_b = (idxs + 16) % E
    try:
        for size in (1, 2):
            PinnedRing.size, PinnedRing.counts, PinnedRing.dense, PinnedRing.next = (
                size,
                None,
                None,
                0,
            )
            a = Run(rank, xs[rank].clone(), ws[rank].clone(), idxs[rank], wexp, gy[rank], E, G)
            b = Run(rank, xs[rank].flip(0).contiguous(), ws[rank].clone(), idxs_b[rank], wexp, gy[rank], E, G)  # fmt: skip
            for phase in (a.issue, a.transfer, a.combine, b.issue):
                phase()
            torch.cuda.synchronize()  # B's count copy has landed
            seen = int(a.state.handle.num_dispatched_tokens_tensor.item())
            corrupt = torch.tensor([float(seen != a.state.num_dense)], device="cuda")
            dist.all_reduce(corrupt, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"ring size {size}: A's handle dense count {'overwritten by B' if corrupt.item() else 'intact'}", flush=True)  # fmt: skip
            assert bool(corrupt.item()) == (size == 1), size
            if size == 2:
                ref_a = reference(xs, ws, idxs, wexp, gy)
                for phase in (a.combine_bwd, b.transfer, a.dispatch_bwd, b.combine, b.combine_bwd, b.dispatch_bwd):  # fmt: skip
                    phase()
                err = torch.tensor([max(a.errors(ref_a, rank, G))], device="cuda")
                dist.all_reduce(err, op=dist.ReduceOp.MAX)
                assert err.item() < 1e-2, err.item()
            else:
                # Leave the buffer consistent on every rank: finish B, drop A's corrupted handle.
                for phase in (b.transfer, b.combine, b.combine_bwd, b.dispatch_bwd):
                    phase()
            hybrid_ep_drop_check()
    finally:
        PinnedRing.size, PinnedRing.counts, PinnedRing.dense, PinnedRing.next = saved


class Proj(torch.nn.Module):
    def __init__(self, G, out_f, in_f, gen):
        super().__init__()
        self.num_groups, self.main_grad = G, None
        w = torch.randn(G, out_f, in_f, device="cuda", generator=gen) * in_f**-0.5
        self.weight = torch.nn.Parameter(w.to(torch.bfloat16))


def stage3_bf16_experts(rank, world, T, H, E, K, G):
    """
    One layer's stage 3 on the real fused BF16 SwiGLU experts, dedup against permute on identical
    routing: outputs, input and router-weight gradients, and the experts' weight gradients.
    """
    from pithtrain.kernels import bf16_experts_supported

    if not bf16_experts_supported(G):
        return
    from pithtrain.operators.bf16_experts import bf16_swiglu_experts

    gen = torch.Generator(device="cuda").manual_seed(99)
    inter = 768
    gate, up, down = Proj(G, inter, H, gen), Proj(G, inter, H, gen), Proj(G, H, inter, gen)
    g2 = torch.Generator(device="cuda").manual_seed(1000 + rank)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g2)
    logits = torch.randn(T, E, device="cuda", generator=g2)
    logits[:, :8] += 2.0
    w, idx = torch.topk(logits.softmax(-1), K, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).contiguous()
    gy = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g2)
    res = {}
    for mode in ("permute", "dedup"):
        hybrid_ep.mode, hybrid_ep.sync_late = mode, True
        for p in (gate, up, down):
            p.weight.grad = None
        st = HybridEPState(idx.contiguous(), w, E, G)
        hybrid_ep_dispatch_issue(st, x)
        _, work = hybrid_ep_dispatch(st, x)
        work.wait()
        rows = st.recv_tokens.detach().requires_grad_()
        probs = st.recv_probs.detach().requires_grad_()
        if mode == "dedup":
            outs = bf16_swiglu_experts(gate, up, down, rows, st.expert_idxs, st.expand_idx, K)
        else:
            counts = st.tokens_per_expert.long()
            expert = torch.repeat_interleave(torch.arange(G, device="cuda"), counts, output_size=st.num_slots)  # fmt: skip
            outs = bf16_swiglu_experts(gate, up, down, rows, expert, None, K)
        weighted = weight_rows(outs, probs)
        y, work = hybrid_ep_combine(st, weighted.detach())
        work.wait()
        g_rows, work = hybrid_ep_combine_backward(st, gy)
        work.wait()
        weighted.backward(g_rows)
        st.probs_grad = probs.grad
        gx, work = hybrid_ep_dispatch_backward(st, rows.grad)
        work.wait()
        torch.cuda.synchronize()
        res[mode] = [y.clone(), gx.clone(), st.topk_weight_grad.clone()] + [p.weight.grad.clone() for p in (gate, up, down)]  # fmt: skip
    # Bitwise: both modes run the same experts on the same rows in the same order, so any drift
    # between them is a bug; an error they share is the fp32 reference test's to catch.
    same = torch.tensor([float(not torch.equal(d, p)) for d, p in zip(res["dedup"], res["permute"])], device="cuda")  # fmt: skip
    dist.all_reduce(same, op=dist.ReduceOp.MAX)
    stats = hybrid_ep_drop_check()
    if rank == 0:
        print(f"stage-3 BF16 experts, dedup vs permute: bitwise equal y/dx/dw_router/dgate/dup/ddown {[v == 0 for v in same.tolist()]} | {stats}", flush=True)  # fmt: skip
    assert same.sum().item() == 0 and stats["slot_order_violation"] == 0, (same.tolist(), stats)


def slot_order_negative(G):
    """
    The slot-order check must fire on slots that are not stable by dense row within each expert.
    """
    from pithtrain.kernels import bf16_expert_group_slots
    from pithtrain.operators.hybrid_ep import check_slot_order

    D, per = 64, 4
    expert_idxs = torch.arange(G, device="cuda").repeat_interleave(per)
    expand_idx = torch.randint(0, D, (G * per,), device="cuda")
    ok_maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, D)
    check_slot_order(ok_maps)
    assert hybrid_ep_drop_check()["slot_order_violation"] == 0
    perm = torch.randperm(G * per, device="cuda")
    bad_maps = bf16_expert_group_slots(expert_idxs[perm].contiguous(), expand_idx[perm].contiguous(), G, D)  # fmt: skip
    check_slot_order(bad_maps)
    stats = hybrid_ep_drop_check()
    assert stats["slot_order_violation"] > 0, stats
    if int(os.environ["RANK"]) == 0:
        print(f"slot-order check fires on shuffled slots: {stats['slot_order_violation']} ranks flagged", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
