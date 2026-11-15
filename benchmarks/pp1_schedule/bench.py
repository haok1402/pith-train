"""
Profile the pp_size 1 schedules of DualPipeV: the overlapped default (forward of one micro-batch
paired with the backward of another) against the experimental sequential one
(TrainingCfg.pp1_sequential_schedule: F0 F1 B1 B0 per micro-batch), on the pretraining benchmark
setup (random init, force-balanced routing, Muon) over uniform and RL-like packed workloads.

One process serves one (model, expert_parallel_size) and loops over schedules and workloads,
appending one JSON line per (schedule, workload) to --out; finished keys are skipped on a rerun, so
a preempted suite resumes. With --nsys, one step per (schedule, workload) runs inside a
cudaProfilerStart/Stop range; run under nsys --capture-range=cudaProfilerApi
--capture-range-end=repeat to get one report per range, numbered in the order of the "capture"
field.

Suites:
- main: the uniform pretraining workload and the RL-like packed workloads.
- oom: equal packs of growing length at a fixed micro-batch count, until the first OOM, which is
  recorded and ends the process (the other ranks then fail on the broken collective).
- check: losses and gradient norms of the two schedules on the same batches, without optimizer
  updates.
"""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from pithtrain.contexts import distributed, training
from pithtrain.kernels import cross_entropy_fwd_bwd, tune_kernels
from pithtrain.modules.distributed import setup_distributed
from pithtrain.modules.load_balance import MoELoadBalanceLossTracker
from pithtrain.modules.logging import setup_logging
from pithtrain.modules.optimizer import clip_grad_norm, local_grads, scale_grads_
from pithtrain.modules.training import (
    make_constant_scheduler,
    make_muon_optimizer,
    setup_training,
)
from pithtrain.pipeline import Microbatch
from pithtrain.tasks.pretrain_lm import PretrainLMCfg

MODELS = {
    "deepseek-v2-lite": "deepseek-v2",
    "qwen3-30b-a3b": "qwen3",
}

ALIGN = 128


def round_align(x: float) -> int:
    return max(ALIGN * 2, int(round(x / ALIGN)) * ALIGN)


def split_fixed(total: int, weights: np.ndarray) -> list[int]:
    """
    Split total tokens by weights into ALIGN multiples that sum to total exactly.
    """
    parts = [round_align(total * w) for w in weights]
    parts[int(np.argmax(parts))] += total - sum(parts)
    assert sum(parts) == total and min(parts) > 0, parts
    return parts


def pack_lengths(kind: str, M: int, per_rank: int, world: int, seed: int) -> list[list[int]]:
    """
    Pack lengths [rank][micro-batch] for a fixed token budget of per_rank * world per step.

    even: every pack the same length.
    intra: lengths differ across the micro-batches of a rank, the same list on every rank.
    full: the per-rank budget differs across ranks and the lengths differ within each rank.
    """
    rng = np.random.default_rng(seed)
    total = per_rank * world
    if kind == "even":
        return [[per_rank // M] * M for _ in range(world)]
    if kind == "intra":
        lens = split_fixed(per_rank, rng.dirichlet([1.5] * M)) if M > 1 else [per_rank]
        return [list(lens) for _ in range(world)]
    if kind == "full":
        weights = rng.lognormal(0.0, 0.4, world)
        budgets = split_fixed(total, weights / weights.sum())
        out = []
        for b in budgets:
            out.append(split_fixed(b, rng.dirichlet([1.5] * M)) if M > 1 else [b])
        return out
    raise ValueError(kind)


class TokenSource:
    """
    Contiguous token slices of the tokenized corpus, addressed by a running offset.
    """

    def __init__(self, root: Path):
        files = sorted(root.rglob("*.bin"))
        assert files, f"no .bin under {root}"
        self.tokens = np.load(files[0], mmap_mode="r")

    def take(self, offset: int, n: int) -> np.ndarray:
        offset = offset % (len(self.tokens) - n - 2)
        return np.asarray(self.tokens[offset : offset + n + 1], dtype=np.int64)


def make_packed(src: TokenSource, length: int, offset: int, rng: np.random.Generator):
    """
    One packed micro-batch (1, length) with synthetic document boundaries (lognormal lengths, mean
    about 1.5k, as in RL rollouts) and -100 on the last token of each document.
    """
    raw = src.take(offset, length)
    tokens = torch.from_numpy(raw[:length].copy()).unsqueeze(0)
    labels = torch.from_numpy(raw[1 : length + 1].copy()).unsqueeze(0)
    bounds = [0]
    while bounds[-1] < length:
        doc = int(np.clip(rng.lognormal(7.0, 0.8), 64, 8192))
        bounds.append(min(length, bounds[-1] + doc))
    for end in bounds[1:]:
        labels[0, end - 1] = -100
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
    return tokens, labels, cu_seqlens


class ChunkedCrossEntropy(torch.autograd.Function):
    """
    The token-summed loss of pretrain_lm.objective through the same fused kernel, run on row slices.
    The SM90 kernel indexes the flattened (N * V) logits with int32, so a micro-batch of more than
    (2^31 - 1) / V tokens (20,971 for DeepSeek-V2-Lite, 14,134 for Qwen3) fails there. Each slice is
    row-contiguous, so the kernel still overwrites the logits with their gradient in place and no
    memory is added; a micro-batch within the bound runs as one slice, as pretrain_lm does.
    """

    @staticmethod
    def forward(ctx, inp, target, ignore_index):
        rows = (2**31 - 1) // inp.shape[1]
        total = inp.new_zeros((), dtype=torch.float32)
        for begin in range(0, inp.shape[0], rows):
            end = min(begin + rows, inp.shape[0])
            mean, grad = cross_entropy_fwd_bwd(inp[begin:end], target[begin:end], ignore_index)
            count = (target[begin:end] != ignore_index).sum().clamp_min(1)
            # The kernel's gradient is of the slice mean; scale it to the gradient of the sum.
            grad.mul_(count.to(grad.dtype))
            total = total + mean * count
        ctx.save_for_backward(inp.detach())
        return total

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        inp.mul_(grad_output.to(inp.dtype))
        return inp, None, None


def objective(model_outputs, objective_inputs):
    """
    pretrain_lm.objective with the cross-entropy split by ChunkedCrossEntropy.
    """
    (logits,) = model_outputs
    (labels,) = objective_inputs
    logits = logits.view(-1, logits.size(-1))
    loss = ChunkedCrossEntropy.apply(logits, labels.view(-1), -100)
    return loss, loss.detach()


def build_microbatches(src, workload: dict, step: int) -> list[Microbatch]:
    device = distributed.device
    rank, world = distributed.rank, distributed.world_size
    out = []
    if workload["kind"] == "uniform":
        seq, M = workload["seq"], workload["M"]
        base = (step * world + rank) * M * (seq + 1)
        for i in range(M):
            raw = src.take(base + i * (seq + 1), seq)
            tokens = torch.from_numpy(raw[:seq].copy()).unsqueeze(0).to(device)
            labels = torch.from_numpy(raw[1:].copy()).unsqueeze(0).to(device)
            out.append(Microbatch(model_inputs=(tokens,), cu_seqlens=None, objective_inputs=(labels,)))  # fmt: skip
        return out
    lens = workload["lengths"][rank]
    rng = np.random.default_rng(1000003 * step + 7919 * rank + 17)
    offset = (step * world + rank) * sum(lens) * 3
    for L in lens:
        tokens, labels, cu = make_packed(src, L, offset, rng)
        offset += L + 1
        out.append(
            Microbatch(
                model_inputs=(tokens.to(device),),
                cu_seqlens=cu.to(device),
                objective_inputs=(labels.to(device),),
            )
        )
    return out


def all_max(x: float) -> float:
    t = torch.tensor(float(x), device=distributed.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def train_step(microbatches, update: bool = True, label: str | None = None) -> dict:
    """
    One step as pretrain_lm.train_step runs it (model.step, token-weighted grad scale, clip,
    optimizer), with the scale taken from the global non-ignored token count, since packs differ
    across ranks.
    """
    device = distributed.device
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, optimizers, schedulers = training.model, training.optimizers, training.schedulers
    model.train()
    if label is not None:
        torch.cuda.nvtx.range_push(label)
    outs = model.step(microbatches, objective)
    if label is not None:
        torch.cuda.nvtx.range_pop()
    counted = sum((mb.objective_inputs[0] != -100).sum() for mb in microbatches)
    stats = torch.stack([counted.float(), torch.stack(outs).sum().float()]).to(device)
    dist.all_reduce(stats)
    global_tokens = stats[0].clamp_min(1)
    loss = stats[1] / global_tokens
    scale_grads_(local_grads(model), 1.0 / global_tokens)
    gnorm = clip_grad_norm(model, max_norm=1.0, norm_type=2, hsdp_replica=1)
    for optimizer, scheduler in zip(optimizers, schedulers):
        if update:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if update:
            scheduler.step()
    MoELoadBalanceLossTracker.reset()
    loss_v, gnorm_v = loss.item(), float(gnorm)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    real_tokens = sum(mb.model_inputs[0].numel() for mb in microbatches)
    tok = torch.tensor(float(real_tokens), device=device)
    dist.all_reduce(tok)
    return dict(
        step_time=all_max(elapsed),
        loss=loss_v,
        grad_norm=gnorm_v,
        tokens=int(tok.item()),
        peak_alloc_gib=all_max(torch.cuda.max_memory_allocated() / 2**30),
        peak_reserved_gib=all_max(torch.cuda.max_memory_reserved() / 2**30),
    )


def main_workloads(per_rank: int, world: int, uniform_gbs: list[int]) -> list[dict]:
    wls = []
    for gbs in uniform_gbs:
        M = gbs // world
        wls.append(dict(name=f"uniform-seq2048-mb{M}", kind="uniform", seq=2048, M=M))
    for M in (1, 2, 4, 8):
        kinds = ("even", "full") if M == 1 else ("even", "intra", "full")
        for kind in kinds:
            lens = pack_lengths(kind, M, per_rank, world, seed=100 + M)
            wls.append(dict(name=f"pack{per_rank // 1024}k-M{M}-{kind}", kind="packed", M=M, lengths=lens))  # fmt: skip
    return wls


def is_oom(e: BaseException) -> bool:
    """
    An allocation failure, also when raised from a compiled graph: AOTInductor reports a failed
    allocation as RuntimeError from aoti_torch_empty_strided, not as torch.OutOfMemoryError.
    """
    if isinstance(e, torch.OutOfMemoryError):
        return True
    text = str(e)
    return "out of memory" in text.lower() or "aoti_torch_empty_strided" in text


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_records(path: Path) -> list[dict]:
    recs = []
    for file in [path, *path.parent.glob(path.stem + ".oom.rank*.jsonl")]:
        if not file.exists():
            continue
        for line in file.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def pending_runs(recs: list[dict], workloads: list[dict], schedules: list[str]) -> list[tuple]:
    """
    (workload, schedule) pairs still to run, workload-major, so both schedules of a workload run
    back to back in one step on one node. A workload is done when every schedule has a measurement
    on one common host, or has been found to OOM (memory does not depend on the node). Otherwise
    every schedule of it that did not OOM runs again here, so its comparison stays on one node.
    """
    runs = []
    for wl in workloads:
        ok_hosts = {sched: set() for sched in schedules}
        oom = {sched: False for sched in schedules}
        for r in recs:
            if r.get("workload") != wl["name"] or r.get("schedule") not in ok_hosts:
                continue
            if r.get("status") == "ok" and r.get("host"):
                ok_hosts[r["schedule"]].add(r["host"])
            elif r.get("status") == "oom":
                oom[r["schedule"]] = True
        if wl.get("probe_len") is not None:
            # An OOM probe: a schedule that has OOMed at a shorter pack is not run longer.
            for sched in schedules:
                for r in recs:
                    if r.get("schedule") == sched and r.get("status") == "oom" and r.get("probe_len") is not None and r["probe_len"] <= wl["probe_len"]:  # fmt: skip
                        oom[sched] = True
        hosts = set().union(*ok_hosts.values()) or {None}
        if any(all(oom[x] or h in ok_hosts[x] for x in schedules) for h in hosts):
            continue
        runs.extend((wl, sched) for sched in schedules if not oom[sched])
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--ep", type=int, required=True)
    p.add_argument("--suite", default="main", choices=["main", "oom", "check"])
    p.add_argument("--schedules", default="overlapped,sequential")
    p.add_argument("--per-rank-tokens", type=int, default=16384)
    p.add_argument("--uniform-gbs", default="256,1024")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timed", type=int, default=3)
    p.add_argument("--nsys", action="store_true")
    p.add_argument("--nsys-max-mb", type=int, default=32, help="skip the nsys step above this M")
    p.add_argument("--oom-M", type=int, default=2)
    p.add_argument("--oom-lengths", default="16384,24576,32768,49152,65536,98304,131072")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    world = int(os.environ["WORLD_SIZE"])
    schedules = args.schedules.split(",")
    if args.suite == "main":
        workloads = main_workloads(args.per_rank_tokens, world, [int(x) for x in args.uniform_gbs.split(",")])  # fmt: skip
    elif args.suite == "check":
        workloads = [
            dict(name="uniform-seq2048-mb4", kind="uniform", seq=2048, M=4),
            dict(
                name="pack16k-M4-full",
                kind="packed",
                M=4,
                lengths=pack_lengths("full", 4, 16384, world, 104),
            ),  # fmt: skip
        ]
    else:
        workloads = []
        for L in [int(x) for x in args.oom_lengths.split(",")]:
            lens = [[L] * args.oom_M for _ in range(world)]
            workloads.append(dict(name=f"oom-M{args.oom_M}-L{L}", kind="packed", M=args.oom_M, lengths=lens, probe_len=L))  # fmt: skip
    max_len = 2048
    for w in workloads:
        if w["kind"] == "packed":
            max_len = max(max_len, max(max(r) for r in w["lengths"]))

    cfg = PretrainLMCfg()
    cfg.dataset = Path(f"workspace/datasets/dclm-baseline/toktxt/{MODELS[args.model]}")
    cfg.distributed.pipeline_parallel_size = 1
    cfg.distributed.expert_parallel_size = args.ep
    # Short, so a rank left in a collective by a peer's OOM fails fast instead of waiting 15 min.
    cfg.distributed.timeout = timedelta(minutes=5)
    t = cfg.training
    t.model = Path(f"benchmarks/pretraining/{args.model}")
    t.optimizer = make_muon_optimizer
    t.scheduler = make_constant_scheduler
    t.lr = 1.0e-6
    t.max_steps = 1
    t.micro_batch_size = 1
    t.global_batch_size = world
    t.sequence_length = max_len
    t.moe_load_balance_type = "global-batch"
    t.moe_load_balance_coef = 1e-3
    t.benchmark = True
    t.pp1_sequential_schedule = schedules[0] == "sequential"

    out = Path(args.out)
    setup_logging(cfg)
    setup_distributed(cfg)
    rank = distributed.rank
    tag = dict(model=args.model, ep=args.ep, dp_attn=distributed.dp_size, world=world, suite=args.suite)  # fmt: skip

    def record(rec):
        if rank == 0:
            append_jsonl(out, {**tag, **rec})
            print("RESULT " + json.dumps({**tag, **rec}), flush=True)

    try:
        setup_training(cfg)
    except torch.OutOfMemoryError as e:
        record(dict(schedule="*", workload="setup", status="oom", error=str(e)[:300]))
        sys.stdout.flush()
        os._exit(3)
    src = TokenSource(cfg.dataset)
    tune_kernels()
    static_alloc = all_max(torch.cuda.memory_allocated() / 2**30)
    host = socket.gethostname()
    if args.suite == "check":
        runs = [(wl, sched) for sched in schedules for wl in workloads]
    else:
        obj = [pending_runs(load_records(out), workloads, schedules) if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        runs = obj[0]
    capture = 0
    tuned = False

    for wl, schedule in runs:
        training.model.sequential_at_pp1 = schedule == "sequential"
        if True:
            key = (schedule, wl["name"])
            torch.cuda.empty_cache()
            base = dict(schedule=schedule, workload=wl["name"], M=wl["M"], static_alloc_gib=static_alloc, host=host, probe_len=wl.get("probe_len"))  # fmt: skip
            if wl["kind"] == "packed":
                lens = wl["lengths"]
                base["lengths"] = lens
                base["max_pack"] = max(max(r) for r in lens)
            try:
                if args.suite == "check":
                    for s in range(3):
                        mbs = build_microbatches(src, wl, s)
                        r = train_step(mbs, update=False)
                        record({**base, "status": "ok", "check_step": s, **r})
                    continue
                steps = []
                n = args.warmup + args.timed
                for s in range(n):
                    mbs = build_microbatches(src, wl, s)
                    r = train_step(mbs)
                    if not tuned:
                        tune_kernels()
                        tuned = True
                    if s >= args.warmup:
                        steps.append(r)
                rec = dict(base, status="ok")
                rec["step_times"] = [x["step_time"] for x in steps]
                rec["step_time"] = float(np.median(rec["step_times"]))
                rec["tokens"] = steps[-1]["tokens"]
                rec["tokens_per_s_per_gpu"] = rec["tokens"] / rec["step_time"] / world
                rec["peak_alloc_gib"] = max(x["peak_alloc_gib"] for x in steps)
                rec["peak_reserved_gib"] = max(x["peak_reserved_gib"] for x in steps)
                rec["loss"] = steps[-1]["loss"]
                if args.nsys and wl["M"] <= args.nsys_max_mb:
                    mbs = build_microbatches(src, wl, n)
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.cudart().cudaProfilerStart()
                    mesh = distributed
                    torch.cuda.nvtx.range_push(
                        f"rank={mesh.rank}; pp={mesh.pp_rank}/{mesh.pp_size} dp={mesh.dp_rank}/{mesh.dp_size} "
                        f"cp={mesh.cp_rank}/{mesh.cp_size} ep={mesh.ep_rank}/{mesh.ep_size}; "
                        f"mbs=1 seq={max(mb.model_inputs[0].shape[1] for mb in mbs)}"
                    )
                    r = train_step(mbs, label=f"workload {wl['name']} schedule {schedule}")
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                    capture += 1
                    rec["capture"] = capture
                    rec["nsys_prefix"] = os.environ.get("PP1_NSYS_PREFIX")
                    rec["nsys_step_time"] = r["step_time"]
                record(rec)
            except Exception as e:
                if not is_oom(e):
                    traceback.print_exc()
                    sys.stdout.flush()
                    os._exit(4)
                # The other ranks may sit in a collective this rank never reaches, so the process
                # group cannot continue: record and leave.
                msg = str(e).splitlines()[0][:300]
                print(f"[rank {rank}] OOM in {key}: {msg}", flush=True)
                rec = dict(base, status="oom", oom_rank=rank, error=msg)
                rec.update({**tag})
                append_jsonl(out.with_suffix(f".oom.rank{rank}.jsonl"), rec)
                sys.stdout.flush()
                os._exit(3)
    if rank == 0:
        print("SUITE DONE", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
