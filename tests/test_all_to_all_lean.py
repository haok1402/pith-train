"""
The lean all-to-all (process-group binding with prebuilt options) moves exactly the bytes the
public torch.distributed.all_to_all_single does.

    torchrun --nproc-per-node=4 tests/test_all_to_all_lean.py

Uneven splits including zero-row ones, bf16 / fp32 / int64 rows, the asynchronous path the stage
2/4 functions use (on a side stream, as there) and the synchronous one the dispatch metadata uses;
mismatched splits must raise instead of reading past the tensor.
"""

import os

import torch
import torch.distributed as dist

import pithtrain.operators.all_to_all as a2a


def splits_for(rank, world, seed):
    g = torch.Generator().manual_seed(seed * 100 + rank)
    sizes = torch.randint(0, 7, (world,), generator=g).tolist()
    sizes[(rank + seed) % world] = 0  # every rank sends nothing to someone
    return sizes


def exchange_splits(input_splits, world):
    t = torch.tensor(input_splits, dtype=torch.int64, device="cuda")
    out = torch.empty_like(t)
    dist.all_to_all_single(out, t)
    return out.tolist()


def run_async(x, out_splits, in_splits, group, stream):
    with torch.cuda.stream(stream):
        out, work = a2a.all_to_all_async(x, out_splits, in_splits, group)
    work.wait()
    torch.cuda.current_stream().wait_stream(stream)
    return out


def run_sync(x, out_splits, in_splits, group):
    out = x.new_empty(sum(out_splits), *x.shape[1:])
    a2a.all_to_all_sync(out, x, out_splits, in_splits, group)
    return out


def run_public(x, out_splits, in_splits, group):
    out = x.new_empty(sum(out_splits), *x.shape[1:])
    dist.all_to_all_single(out, x, out_splits, in_splits, group)
    return out


def expect_raise(fn):
    try:
        fn()
    except AssertionError:
        return
    raise RuntimeError("mismatched splits did not raise")


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    group = dist.new_group(list(range(world)))
    stream = torch.cuda.Stream()
    for seed in range(4):
        in_splits = splits_for(rank, world, seed)
        out_splits = exchange_splits(in_splits, world)
        for dtype in (torch.bfloat16, torch.float32, torch.int64):
            g = torch.Generator(device="cuda").manual_seed(seed * 10 + rank)
            if dtype.is_floating_point:
                x = torch.randn(sum(in_splits), 64, generator=g, device="cuda").to(dtype)
            else:
                x = torch.randint(-(2**40), 2**40, (sum(in_splits), 3), generator=g, device="cuda")
            for path in (run_async, run_sync):
                args = (x, out_splits, in_splits, group) + ((stream,) if path is run_async else ())
                lean, public = path(*args), run_public(x, out_splits, in_splits, group)
                assert lean.dtype == public.dtype and torch.equal(lean, public), (path, dtype)
    # Splits that do not cover the tensor raise on the lean path before any NCCL call.
    x = torch.zeros(5, 8, device="cuda")
    bad = [5 + 1] + [0] * (world - 1)
    expect_raise(lambda: a2a.all_to_all_async(x, [0] * world, bad, group))
    expect_raise(lambda: a2a.all_to_all_sync(x.new_empty(3, 8), x, [0] * world, bad, group))
    good = [5] + [0] * (world - 1)
    expect_raise(lambda: a2a.all_to_all_sync(x.new_empty(3, 8), x, [4] + [0] * (world - 1), good, group))  # fmt: skip
    dist.barrier()
    if rank == 0:
        print(f"lean all-to-all == public on {world} ranks: OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
