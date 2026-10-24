"""
FP8 parameter all-gather under FSDP2 (pithtrain/modules/fp8_param_gather.py).

Checks the invariant that the gathered FP8 operands are bit-identical to quantizing the bf16 weight
FSDP gathers with the flag off, before and after a weight update, plus forward/backward agreement
with the default FP8 path and the memory held while unsharded.

Usage:
    torchrun --nproc-per-node=1 tests/test_fp8_param_gather.py
    torchrun --nproc-per-node=4 tests/test_fp8_param_gather.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels.fp8_weight_quant import (
    fp8cast_blockwise_transpose,
    fp8cast_blockwise_transpose_batched,
)
from pithtrain.modules.fp8_param_gather import install_fp8_param_gather, precompute_fp8_param_scales
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.linear import FP8Linear
from pithtrain.operators.token_scatter import precompute_group_indices


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item() if denom > 0 else 0.0


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Ragged rows (576) and K (10944) and a row count below 128 * world size, so tiles
        # straddle shards and shards are uneven.
        self.a = FP8Linear(2048, 576, bias=False)
        self.b = FP8Linear(576, 10944, bias=False)
        self.c = FP8Linear(10944, 64 * 3, bias=False)
        self.experts = FP8GroupedLinear(5, 192, 384)
        self.norm = nn.RMSNorm(192)

    @torch.compile(fullgraph=True)
    def dense(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.c(self.b(self.a(x))))

    def forward(self, x: torch.Tensor, offs: torch.Tensor, ks: list, ks_t: torch.Tensor) -> torch.Tensor:  # fmt: skip
        h = self.dense(x)
        gi = precompute_group_indices(offs, h.shape[0])
        return self.experts(h, offs, ks, ks_t, group_indices=gi)


def build(mesh, gather: bool, seed: int) -> Block:
    torch.manual_seed(seed)
    model = Block().cuda()
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, std=0.02)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model.experts, mesh=mesh, reshard_after_forward=False, mp_policy=mp)
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)
    model.set_gradient_divide_factor(1.0)
    model.experts.set_gradient_divide_factor(1.0)
    if gather:
        install_fp8_param_gather(model, (FP8Linear, FP8GroupedLinear))
    return model


def check_invariant(ref: Block, imp: Block) -> None:
    """After unshard, imp's FP8 operands equal quantizing ref's bf16 unsharded weights."""
    for m in (ref, ref.experts, imp, imp.experts):
        m.unshard()
    for name in ("a", "b", "c", "experts"):
        r, i = getattr(ref, name), getattr(imp, name)
        w = r.weight.detach()
        fn = fp8cast_blockwise_transpose_batched if w.dim() == 3 else fp8cast_blockwise_transpose
        expect = fn(w)
        got = i._get_quantized_weight()
        for e, g, what in zip(expect, got, ("fp8", "scale", "fp8_t", "scale_t")):
            assert e.shape == g.shape, (name, what, e.shape, g.shape)
            if e.dtype == torch.float8_e4m3fn:
                e, g = e.view(torch.uint8), g.view(torch.uint8)
            assert torch.equal(e, g), f"{name}.{what} differs"


def run_step(model: Block, x: torch.Tensor, offs, ks, ks_t) -> torch.Tensor:
    FP8WeightCacheControl.step()
    out = model(x, offs, ks, ks_t)
    out.float().square().sum().backward()
    return out.detach()


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (world,))
    training.fp8 = True

    ref = build(mesh, gather=False, seed=0)
    imp = build(mesh, gather=True, seed=0)

    torch.manual_seed(1)
    M = 1024
    x = torch.randn(M, 2048, device="cuda", dtype=torch.bfloat16)
    counts = [128, 256, 0, 384, 256]  # one empty expert
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks, ks_t = counts, torch.tensor(counts, device="cuda", dtype=torch.int32)

    for step in range(2):
        check_invariant(ref, imp)
        out_r = run_step(ref, x, offs, ks, ks_t)
        out_i = run_step(imp, x, offs, ks, ks_t)
        assert torch.equal(out_r, out_i), "forward differs"
        for (n, pr), (_, pi) in zip(ref.named_parameters(), imp.named_parameters()):
            gr, gi = pr.grad.full_tensor(), pi.grad.full_tensor()
            d = calc_diff(gr, gi)
            assert d < 1e-6, (n, d)
        # A deterministic in-place update, then a re-gather through the out= path.
        with torch.no_grad():
            for pr, pi in zip(ref.parameters(), imp.parameters()):
                upd = pr.grad.sign() * 1e-3
                pr.sub_(upd)
                pi.sub_(upd)
                pr.grad = pi.grad = None
        precompute_fp8_param_scales()
        if rank == 0:
            print(f"step {step}: invariant, forward and gradients match")

    # Memory: the unsharded FP8 weights take 2 B/param (fp8 + fp8_t), as the bf16 weight would,
    # and nothing else; the default path holds the bf16 weight plus its 2 B/param FP8 cache.
    for m in (ref, ref.experts, imp, imp.experts):
        m.reshard()
    torch.cuda.synchronize()
    n = sum(m.weight.numel() for m in (imp.a, imp.b, imp.c, imp.experts))
    base = torch.cuda.memory_allocated()
    imp.unshard()
    imp.experts.unshard()
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - base
    small = sum(p.numel() * 2 for p in imp.norm.parameters())
    # The unsharded fp8 + fp8_t take 2 B/param; the gather also releases this rank's 1 B/param
    # FP8 shard (regrown by the next precompute), so the net growth is 2 - 1/world B/param.
    expect = 2 * n - n / world
    if rank == 0:
        print(f"unshard grew {grown / n:.3f} B/param (expect ~{expect / n:.3f})")
    assert abs(grown - small - expect) < 2 * 1024**2, (grown, n, world)
    imp.reshard()
    imp.experts.reshard()

    dist.barrier()
    if rank == 0:
        print("PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
