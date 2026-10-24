"""
fp32 main_grad accumulation of gpt-oss's raw expert weights (install_raw_main_grad in pithtrain/modules/wgrad_accumulation.py)
under FSDP2 with two micro-batches per step: the reduced gradients must match the default path
(bf16 .grad per micro-batch, summed by FSDP2 in fp32) up to that path's bf16 rounding, and the
forward is untouched. Run: torchrun --nproc-per-node=2 tests/test_fp8_gptoss_main_grad.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.models.gpt_oss import GptOssExperts
from pithtrain.modules.wgrad_accumulation import install_raw_main_grad
from pithtrain.operators.main_grad import raw_main_grad

G, H, INTER = 4, 2880, 2880


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def build(mesh, main_grad: bool) -> GptOssExperts:
    torch.manual_seed(0)
    model = GptOssExperts(GptOssConfig(hidden_size=H, intermediate_size=INTER), G).cuda()
    for p in model.parameters():
        nn.init.normal_(p, std=0.05 if p.dim() == 3 else 0.5)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)
    model.set_gradient_divide_factor(1.0)
    if main_grad:
        install_raw_main_grad(model)
        assert raw_main_grad(model, "gate_up_proj") is not None
    return model


def main() -> None:
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    training.fp8 = True
    models = [build(mesh, False), build(mesh, True)]
    counts = [512, 256, 0, 768]
    offs = torch.tensor(counts, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    ks_t = torch.tensor(counts, device="cuda", dtype=torch.int32)
    torch.manual_seed(1 + dist.get_rank())
    xs = [torch.randn(sum(counts), H, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    FP8WeightCacheControl.step()
    outs = []
    for m in models:
        for i, x in enumerate(xs):
            m.set_requires_gradient_sync(i == 1)
            m.set_is_last_backward(i == 1)
            y = m(x, offs, ks=counts, ks_tensor=ks_t)
            y.float().square().sum().backward()
            outs.append(y.detach())
    assert torch.equal(outs[0], outs[2]) and torch.equal(outs[1], outs[3]), "forward differs"
    for (n, pr), (_, pi) in zip(models[0].named_parameters(), models[1].named_parameters()):
        d = calc_diff(pr.grad.full_tensor(), pi.grad.full_tensor())
        if dist.get_rank() == 0:
            print(f"{n}: diff {d:.2e}", flush=True)
        assert d < 1e-5, (n, d)
    mg = raw_main_grad(models[1], "gate_up_proj")
    assert mg.untyped_storage().size() == 0, "main_grad not released after reduce"
    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
