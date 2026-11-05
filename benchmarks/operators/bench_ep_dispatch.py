"""
Op-level benchmark of the EP dispatch and combine at the Qwen3-30B-A3B shape (hidden 2048, 128
experts, top-8, 4096 tokens per rank, BF16): PithTrain's NCCL all-to-all path against blocking
HybridEP at several SM counts. Forward and backward of each, latency per call, achieved bandwidth,
tokens/s, the SMs each kernel holds (from a profiler trace), and the slowdown of a grouped GEMM
running concurrently on another stream.

    torchrun --nproc-per-node=8 benchmarks/operators/bench_ep_dispatch.py [--sms 8,16,24,32]

Bandwidth is the useful payload over the latency, the same payload for both backends: the
(token, destination rank) rows a rank receives in dispatch (and sends back in combine), times
hidden * 2 bytes. The all-to-all path moves more than that in combine (one row per (token,
expert), not per (token, rank)), which shows up as lower useful bandwidth.
"""

import argparse
import json
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pithtrain.contexts import distributed
from pithtrain.kernels import bf16_expert_group_slots
from pithtrain.operators import hybrid_ep
from pithtrain.operators.all_to_all import all_to_all_async
from pithtrain.operators.ep_dispatch import prepare_dispatch
from pithtrain.operators.hybrid_ep import (
    HybridEPState,
    hybrid_ep_combine,
    hybrid_ep_combine_backward,
    hybrid_ep_dispatch,
    hybrid_ep_dispatch_backward,
)

T, H, E, K = 4096, 2048, 128, 8
INTER = 768  # Qwen3-30B-A3B moe_intermediate_size


def timed(fn, iters=20, warmup=5):
    """
    Wall time per call with the GPU drained at both ends: counts host syncs inside fn.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    t = torch.tensor([dt], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def kernel_sms(fn, keys):
    """
    Max grid size (blocks; these kernels run one block per SM) of kernels whose name has a key.
    """
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        events = json.load(open(f.name))["traceEvents"]
    out = {}
    for ev in events:
        if ev.get("cat") != "kernel":
            continue
        name = ev["name"]
        for key in keys:
            if key in name:
                g = ev["args"].get("grid", [0, 0, 0])
                out[key] = max(out.get(key, 0), g[0] * g[1] * g[2])
    return out


class GemmLoad:
    """
    The stage-3 gate_up grouped GEMM on this rank's received rows, as the compute to overlap.
    """

    def __init__(self, rows: int, G: int):
        self.x = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
        self.w = torch.randn(G, 2 * INTER, H, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        counts = torch.full((G,), rows // G, dtype=torch.int32, device="cuda")
        counts[-1] += rows - counts.sum()
        self.offs = counts.cumsum(0).to(torch.int32)
        self.stream = torch.cuda.Stream()

    def run(self, n: int):
        for _ in range(n):
            F.grouped_mm(self.x, self.w, offs=self.offs)

    def window(self, n: int, during=None, during_iters=0):
        """
        Time of n GEMMs on self.stream, with during() called during_iters times meanwhile.
        """
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start.record()
            self.run(n)
            end.record()
        for _ in range(during_iters):
            during()
        torch.cuda.synchronize()
        t = torch.tensor([start.elapsed_time(end) / n], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sms", default="8,16,24,32")
    p.add_argument("--modes", default="permute,dedup", help="HybridEP modes to time")
    p.add_argument(
        "--routing",
        default="uniform",
        help="uniform; hot:F (rank 0's experts receive F times the mean); or a prefix P of dumped P.rank<R>.pt topk_idx",
    )
    args = p.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])  # the device on this node; rank is global
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    distributed.device = torch.device("cuda", local_rank)
    distributed.ep_group, distributed.ep_size, distributed.ep_rank = group, world, rank
    G = E // world

    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g)
    if args.routing == "uniform" or args.routing.startswith("hot:"):
        logits = torch.randn(T, E, device="cuda", generator=g)
        if args.routing.startswith("hot:"):
            # Bias rank 0's experts until they take the target share of this rank's slots.
            target, lo, hi = float(args.routing[4:]) / world, 0.0, 10.0
            for _ in range(40):
                b = (lo + hi) / 2
                biased = logits.clone()
                biased[:, :G] += b
                share = (torch.topk(biased, K, dim=-1).indices < G).float().mean().item()
                lo, hi = (b, hi) if share < target else (lo, b)
            logits[:, :G] += (lo + hi) / 2
        scores = logits.softmax(-1)
        w, idx = torch.topk(scores, K, dim=-1)
    else:
        idx = torch.load(f"{args.routing}.rank{rank}.pt").cuda()  # a dumped (T, K) topk_idx
        w = torch.rand(T, K, device="cuda", generator=g)
    w = w / w.sum(-1, keepdim=True)
    idx = idx.contiguous()

    rows = {}
    results = []

    # ---- PithTrain's all-to-all path ----
    def a2a_prepare():
        return prepare_dispatch(x, idx, w, E, world, G, group)

    tokens, routing = a2a_prepare()
    ds, cs = routing.dispatch_splits, routing.combine_splits
    recv_dedup = sum(ds.output_splits)
    slots = sum(cs.output_splits)
    rows["a2a"] = recv_dedup

    def a2a_send(t, out_splits, in_splits):
        out, work = all_to_all_async(t, out_splits, in_splits, group)
        work.wait()
        return out

    gathered = a2a_send(tokens, ds.output_splits, ds.input_splits)
    moe_outs = torch.randn(slots, H, device="cuda", dtype=torch.bfloat16, generator=g)
    grad_y = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g)

    def a2a_dispatch_fwd():
        tok, r = a2a_prepare()
        out = a2a_send(tok, r.dispatch_splits.output_splits, r.dispatch_splits.input_splits)
        bf16_expert_group_slots(r.expert_idxs, r.expand_idx, G, out.shape[0])  # the "permute"
        return out

    def a2a_combine_fwd():
        back = a2a_send(moe_outs, cs.input_splits, cs.output_splits)
        # stage 5's unpermute + weighted sum per token
        probs = w.view(-1)[routing.moe_local_idxs]
        token_idx = routing.moe_local_idxs // K
        weighted = (back.float() * probs.unsqueeze(-1)).to(back.dtype)
        agg = back.new_zeros(T, H)
        agg.scatter_add_(0, token_idx[:, None].expand_as(weighted), weighted)
        return agg

    def a2a_combine_bwd():
        # grad of the combined output to the expert rows: gather per slot, weight, send
        probs = w.view(-1)[routing.moe_local_idxs]
        token_idx = routing.moe_local_idxs // K
        gs = (grad_y[token_idx].float() * probs.unsqueeze(-1)).to(grad_y.dtype)
        return a2a_send(gs, cs.output_splits, cs.input_splits)

    grad_gathered = torch.randn_like(gathered)

    def a2a_dispatch_bwd():
        return a2a_send(grad_gathered, ds.input_splits, ds.output_splits)

    a2a_sms = kernel_sms(lambda: a2a_send(tokens, ds.output_splits, ds.input_splits), ["nccl"])
    a2a_ops = dict(dispatch_fwd=a2a_dispatch_fwd, dispatch_bwd=a2a_dispatch_bwd, combine_fwd=a2a_combine_fwd, combine_bwd=a2a_combine_bwd)  # fmt: skip
    for name, fn in a2a_ops.items():
        results.append(dict(backend="all-to-all", sms=a2a_sms.get("nccl", 0), op=name, latency=timed(fn)))  # fmt: skip
    # pure NCCL transfer, without metadata / permute
    results.append(dict(backend="all-to-all", sms=a2a_sms.get("nccl", 0), op="dispatch_fwd(a2a only)", latency=timed(lambda: a2a_send(tokens, ds.output_splits, ds.input_splits))))  # fmt: skip
    results.append(dict(backend="all-to-all", sms=a2a_sms.get("nccl", 0), op="combine_fwd(a2a only)", latency=timed(lambda: a2a_send(moe_outs, cs.input_splits, cs.output_splits))))  # fmt: skip

    # concurrent GEMM: alone, then with each comm op running on another stream
    comm_stream = torch.cuda.Stream()
    gemm = GemmLoad(slots, G)
    n_gemm = 200
    gemm_alone = gemm.window(n_gemm)

    def overlapped(fn, latency):
        n = max(1, int(gemm_alone * n_gemm / 1e3 / latency * 1.2) + 1)
        n_t = torch.tensor([n], device="cuda")
        dist.all_reduce(n_t, op=dist.ReduceOp.MAX)

        def during():
            with torch.cuda.stream(comm_stream):
                fn()

        return gemm.window(n_gemm, during, int(n_t.item()))

    for r in results:
        if r["backend"] == "all-to-all" and r["op"] in a2a_ops:
            r["gemm_slowdown"] = overlapped(a2a_ops[r["op"]], r["latency"]) / gemm_alone - 1

    # ---- HybridEP, blocking ----
    configs = [(int(a), m) for a in args.sms.split(",") for m in args.modes.split(",")]
    for sms, mode in configs:
        if sms != hybrid_ep.num_sms:
            hybrid_ep.buffer = None
            torch.cuda.empty_cache()
        hybrid_ep.num_sms, hybrid_ep.mode = sms, mode
        state = HybridEPState(idx, w, E, G)
        _, work = hybrid_ep_dispatch(state, x)
        work.wait()
        P, D = state.num_slots, state.num_dense
        weighted = torch.randn(P, H, device="cuda", dtype=torch.bfloat16, generator=g)
        grad_rows = torch.randn(D if mode == "dedup" else P, H, device="cuda", dtype=torch.bfloat16, generator=g)  # fmt: skip
        probs_grad = torch.randn(P, device="cuda", generator=g)

        def he(fn, *a):
            def run():
                res, work = fn(*a)
                work.wait()
                return res

            return run

        def dispatch_bwd():
            state.probs_grad = probs_grad
            r, work = hybrid_ep_dispatch_backward(state, grad_rows)
            work.wait()
            return r

        def dispatch_fwd():
            s = HybridEPState(idx, w, E, G)
            _, work = hybrid_ep_dispatch(s, x)
            work.wait()
            return s.recv_tokens

        ops = dict(dispatch_fwd=dispatch_fwd, dispatch_bwd=dispatch_bwd, combine_fwd=he(hybrid_ep_combine, state, weighted), combine_bwd=he(hybrid_ep_combine_backward, state, grad_y))  # fmt: skip
        ksms = kernel_sms(dispatch_fwd, ["dispatch_kernel", "combine_kernel", "permute", "scan"])
        ksms.update(kernel_sms(ops["combine_fwd"], ["combine_kernel", "unpermute"]))
        hybrid_ep.HybridEPStats.rows_received = hybrid_ep.HybridEPStats.rows_expected = 0
        for name, fn in ops.items():
            lat = timed(fn)
            r = dict(backend=f"hep-{mode}", sms=sms, op=name, latency=lat, kernels=ksms)
            r["gemm_slowdown"] = overlapped(fn, lat) / gemm_alone - 1
            results.append(r)
        # rows received per rank in dispatch: one per (token, local expert), permuted. The rows
        # on the wire are per (token, rank), the same count as the all-to-all's dedup rows.
        rows[f"hybrid{sms}_{mode}"] = (P, D)

    # payload: dedup rows per rank (identical routing), averaged over ranks
    t = torch.tensor([rows["a2a"], slots], dtype=torch.float64, device="cuda")
    dist.all_reduce(t)
    dedup_rows, slot_rows = (t / world).tolist()
    # (token, local expert) rows each rank receives as the expert side: the load each rank carries
    recv = torch.tensor([float(sum(routing.expert_counts))], device="cuda")
    per_rank = [torch.empty_like(recv) for _ in range(world)]
    dist.all_gather(per_rank, recv)
    per_rank = torch.cat(per_rank)
    if rank == 0:
        print(f"shape T={T} H={H} E={E} K={K} ep={world} routing={args.routing}")
        print(f"rows per rank: dedup (token, rank) {dedup_rows:.0f}, (token, expert) {slot_rows:.0f} ({dedup_rows / T:.2f} / {slot_rows / T:.2f} per token)")  # fmt: skip
        print(f"expert-side rows received per rank: {[int(v) for v in per_rank.tolist()]} (max/mean {(per_rank.max() / per_rank.mean()).item():.2f})")  # fmt: skip
        print(f"grouped GEMM alone: {gemm_alone * 1e3:.0f} us")
        payload = dedup_rows * H * 2
        hdr = f"{'backend':<11} {'SMs':>4} {'op':<24} {'latency us':>10} {'GB/s':>7} {'Mtok/s':>7} {'GEMM slowdown':>13}"  # fmt: skip
        print(hdr)
        for r in results:
            gbps = payload / r["latency"] / 1e9
            mtok = T / r["latency"] / 1e6
            sd = r.get("gemm_slowdown")
            sd = f"{sd * 100:+.1f}%" if sd is not None else "-"
            print(f"{r['backend']:<11} {r['sms']:>4} {r['op']:<24} {r['latency'] * 1e6:>10.0f} {gbps:>7.0f} {mtok:>7.2f} {sd:>13}")  # fmt: skip
        for r in results:
            if "kernels" in r and r["op"] == "dispatch_fwd":
                print("hybrid kernels (blocks)", r["sms"], r["kernels"])
        print("all-to-all nccl blocks", a2a_sms)
        out = os.environ.get("BENCH_OUT")
        if out:
            json.dump(dict(results=results, rows=dict(dedup=dedup_rows, slots=slot_rows), gemm_alone_ms=gemm_alone), open(out, "w"), indent=1, default=str)  # fmt: skip
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
