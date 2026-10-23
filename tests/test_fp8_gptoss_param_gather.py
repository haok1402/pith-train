"""
FP8 parameter all-gather of gpt-oss's raw expert weights (pithtrain/operators/fp8_gptoss_param.py).

The gathered FP8 operands of gate_up_proj / down_proj must be bit-identical to quantizing the bf16
weights FSDP gathers with the flag off, before and after an update; the expert forward must be
bit-identical and the gradients equal. The state_dict keys must not change.

Usage:
    torchrun --nproc-per-node=1 tests/test_fp8_gptoss_param_gather.py
    torchrun --nproc-per-node=4 tests/test_fp8_gptoss_param_gather.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from pithtrain.contexts import training
from pithtrain.models.gpt_oss import GptOssExperts
from pithtrain.operators.deepgemm_quantize import fp8cast_blockwise_transpose_batched
from pithtrain.operators.fp8_gptoss_param import fp8_param_operands, install_raw_fp8_param_gather
from pithtrain.operators.fp8_param import install_fp8_param_gather, precompute_fp8_param_scales
from pithtrain.operators.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.linear import FP8Linear

G, H, INTER = 6, 2880, 2880


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item() if denom > 0 else 0.0


def build(mesh, gather: bool, seed: int) -> GptOssExperts:
    torch.manual_seed(seed)
    model = GptOssExperts(GptOssConfig(hidden_size=H, intermediate_size=INTER), G).cuda()
    for p in model.parameters():
        nn.init.normal_(p, std=0.05 if p.dim() == 3 else 0.5)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)
    model.set_gradient_divide_factor(1.0)
    if gather:
        install_fp8_param_gather(model, (FP8Linear, FP8GroupedLinear))
        install_raw_fp8_param_gather(model)
    return model


def check_invariant(ref: GptOssExperts, imp: GptOssExperts) -> None:
    ref.unshard()
    imp.unshard()
    for name in GptOssExperts.fp8_raw_params:
        expect = fp8cast_blockwise_transpose_batched(getattr(ref, name).detach())
        got = fp8_param_operands(imp, name)
        assert got is not None
        for e, g, what in zip(expect, got, ("fp8", "scale", "fp8_t", "scale_t")):
            assert e.shape == g.shape, (name, what, e.shape, g.shape)
            if e.dtype == torch.float8_e4m3fn:
                e, g = e.view(torch.uint8), g.view(torch.uint8)
            assert torch.equal(e, g), f"{name}.{what} differs"


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (world,))
    training.fp8 = True

    ref = build(mesh, gather=False, seed=0)
    imp = build(mesh, gather=True, seed=0)
    assert list(ref.state_dict()) == list(imp.state_dict())

    torch.manual_seed(1)
    counts = [256, 0, 384, 128, 512, 256]
    M = sum(counts)
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks, ks_t = counts, torch.tensor(counts, device="cuda", dtype=torch.int32)

    for step in range(2):
        check_invariant(ref, imp)
        outs = []
        for m in (ref, imp):
            FP8WeightCacheControl.step()
            out = m(x, offs, ks=ks, ks_tensor=ks_t)
            out.float().square().sum().backward()
            outs.append(out.detach())
        assert torch.equal(outs[0], outs[1]), "forward differs"
        for (n, pr), (_, pi) in zip(ref.named_parameters(), imp.named_parameters()):
            d = calc_diff(pr.grad.full_tensor(), pi.grad.full_tensor())
            assert d < 1e-6, (n, d)
        with torch.no_grad():
            for pr, pi in zip(ref.parameters(), imp.parameters()):
                upd = pr.grad.sign() * 1e-3
                pr.sub_(upd)
                pi.sub_(upd)
                pr.grad = pi.grad = None
        precompute_fp8_param_scales()
        if rank == 0:
            print(f"step {step}: invariant, forward and gradients match", flush=True)

    # Memory: the unsharded FP8 weights take 2 B/param (fp8 + fp8_t), the bf16 size, and nothing
    # else is held for them; the default path holds bf16 weights plus a 2 B/param FP8 cache.
    for m in (ref, imp):
        m.reshard()
    torch.cuda.synchronize()
    n = imp.gate_up_proj.numel() + imp.down_proj.numel()
    base = torch.cuda.memory_allocated()
    imp.unshard()
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - base
    if rank == 0:
        print(f"unshard grew {grown / n:.3f} B/param (expect ~2)", flush=True)
    imp.reshard()
    dist.barrier()
    if rank == 0:
        print("PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
