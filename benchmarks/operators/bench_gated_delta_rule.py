"""
Benchmark forward and backward latency of the chunked gated delta rule operator at a Qwen3.5
Gated DeltaNet geometry, with and without context parallelism, and check its output against the
fp32 token-by-token recurrence.

Usage (one process per CP rank; nproc 1 runs without CP):
    torchrun --nproc-per-node=<cp> benchmarks/operators/bench_gated_delta_rule.py \
        --key-heads 16 --value-heads 64 --seq 8192 [--batch 1] [--no-check]

--seq is the full sequence length; each rank holds seq / cp tokens in the contiguous layout.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pithtrain.operators.gated_delta_rule import gated_delta_rule

D = 128


def make_inputs(B: int, S: int, H: int, HV: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=gen)

    q = F.normalize(randn(B, S, H, D), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(B, S, H, D), dim=-1).to(torch.bfloat16)
    v = randn(B, S, HV, D).to(torch.bfloat16)
    a_log = torch.log(torch.rand(HV, device="cuda", generator=gen) * 16)
    g = -a_log.exp() * F.softplus(randn(B, S, HV) + 1.0)
    beta = randn(B, S, HV).to(torch.bfloat16).sigmoid()
    do = randn(B, S, HV, D).to(torch.bfloat16)
    return q, k, v, g, beta, do


def recurrence(q, k, v, g, beta):
    """
    The fp32 recurrence over the full sequence, key heads repeated to the value heads.
    """
    groups = v.shape[2] // q.shape[2]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q, k = (x.repeat_interleave(groups, dim=2) for x in (q, k))
    state = q.new_zeros(q.shape[0], v.shape[2], D, D)
    out = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state))
        state = state + k[:, t][..., :, None] * u[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * D**-0.5, state))
    return torch.stack(out, 1)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    cp_group = None
    if world > 1:
        dist.init_process_group("nccl")
        cp_group = dist.new_group(list(range(world)))
    B, H, HV, S = args.batch, args.key_heads, args.value_heads, args.seq
    assert S % world == 0
    T = S // world

    full = make_inputs(B, S, H, HV, seed=0)
    q, k, v, g, beta, do = (x[:, rank * T : (rank + 1) * T].contiguous() for x in full)
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
    tag = f"B{B} S{S} H{H} HV{HV} cp{world}"

    def run_once() -> torch.Tensor:
        for leaf in leaves:
            leaf.grad = None
        out = gated_delta_rule(*leaves, cp_group=cp_group)
        out.backward(do)
        return out

    try:
        out = run_once()
    except Exception as e:  # noqa: BLE001 - report the failure as the measurement
        if rank == 0:
            print(f"{tag} | FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        if world > 1:
            dist.destroy_process_group()
        return

    if not args.no_check:
        ref = recurrence(*(x[:, : (rank + 1) * T] for x in full[:5]))[:, rank * T :]
        diff = torch.tensor([calc_diff(out, ref)], device="cuda")
        if world > 1:
            dist.all_reduce(diff, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"{tag} | out calc_diff vs fp32 recurrence (max over ranks): {diff.item():.2e}",
                flush=True,
            )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    fwd_ms = bwd_ms = 0.0
    for _ in range(args.iters):
        for leaf in leaves:
            leaf.grad = None
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        out = gated_delta_rule(*leaves, cp_group=cp_group)
        mid.record()
        out.backward(do)
        end.record()
        torch.cuda.synchronize()
        fwd_ms += start.elapsed_time(mid)
        bwd_ms += mid.elapsed_time(end)
    times = torch.tensor([fwd_ms / args.iters, bwd_ms / args.iters], device="cuda")
    if world > 1:
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"{tag} | fwd {times[0].item():7.3f} ms, bwd {times[1].item():7.3f} ms (max over ranks)",
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
