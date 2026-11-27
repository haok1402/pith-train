"""Check direct dense FSDP2's FP32 gradient shards and optimizer updates on Hopper."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("requires SM90")
    from pithtrain.kernels import fp8_wgrad_supported

    if not fp8_wgrad_supported():
        pytest.skip("requires SM90")
    if "RANK" in os.environ:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())
        dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    else:
        store = tmp_path_factory.mktemp("fsdp2") / "store"
        dist.init_process_group("nccl", init_method=store.as_uri(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cuda", (dist.get_world_size(),))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("uses", "defer", "mode"),
    [
        pytest.param((1,), False, "eager", id="single-use"),
        pytest.param((1, 1, 1), False, "eager", id="three-microbatches"),
        pytest.param((1, 1, 0), True, "eager", id="deferred-with-final-skip"),
        pytest.param((2,), False, "eager", id="twice-in-one-backward"),
        pytest.param((0,), False, "eager", id="skipped"),
        pytest.param((1, 1, 1), True, "compiled", id="compiled-deferred"),
        pytest.param((2,), False, "compiled", id="compiled-twice"),
        pytest.param((2,), False, "tied", id="tied-weight"),
        pytest.param((1,), False, "rank-skip", id="skipped-on-one-rank"),
    ],
)
def test_direct_dense_main_grad(mesh, uses, defer, mode):
    from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
    from pithtrain.kernels import fp8_quant_act, fp8_quant_grad, fp8_wgrad
    from pithtrain.modules.wgrad_accumulation import install_wgrad_accumulation
    from pithtrain.operators.linear import BF16Linear, FP8Linear

    width = rows = 128
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    if mode == "rank-skip" and dist.get_world_size() < 2:
        pytest.skip("requires at least two ranks")
    inputs = torch.Generator(device=device).manual_seed(42 + rank)

    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.fp8 = FP8Linear(width, width, bias=False, device=device)
            self.head = BF16Linear(width, width, bias=False, device=device)
            if mode == "tied":
                self.tied = FP8Linear(width, width, bias=False, device=device)
                self.tied.weight = self.fp8.weight

        def project(self, x):
            return self.fp8(x)

        def forward(self, x, head_in):
            if mode == "tied" and len(x) == 2:
                return (self.project(x[0]), self.tied(x[1])), self.head(head_in)
            return tuple(self.project(value) for value in x), self.head(head_in)

    torch.manual_seed(17)
    model = Dense()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    model.set_gradient_divide_factor(1.0)
    install_wgrad_accumulation(model, overwrite=True)
    if mode == "compiled":
        model.project = torch.compile(model.project, fullgraph=True, backend="aot_eager")

    group = fully_shard.state(model)._fsdp_param_groups[0]
    pre_backward = group.pre_backward

    def poison_fresh_buffers(*args):
        # Make accidental reduction of an unwritten buffer deterministic.
        fresh = [
            layer
            for layer in (model.fp8, model.head)
            if not layer.main_grad.untyped_storage().size()
        ]
        pre_backward(*args)
        for layer in fresh:
            layer.main_grad.fill_(float("nan"))

    group.pre_backward = poison_fresh_buffers
    weights = (model.fp8.weight, model.head.weight)
    reference = [nn.Parameter(weight.to_local().detach().clone()) for weight in weights]
    optimizer = torch.optim.AdamW(weights, lr=0.01, weight_decay=0.1, fused=True)
    reference_optimizer = torch.optim.AdamW(reference, lr=0.01, weight_decay=0.1, fused=True)

    def samples():
        return torch.randn(rows, width, generator=inputs, device=device, dtype=torch.bfloat16)

    # Reverse the use pattern on the second optimizer step to verify reset across steps.
    for step_uses in (uses, (int(not any(uses)),)):
        local_sums = [torch.zeros(width, width, device=device) for _ in weights]
        expected = [None for _ in weights]
        has_fp8_grad = False
        for index, calls in enumerate(step_uses):
            if mode == "rank-skip" and rank % 2:
                calls = 0
            sync = not defer or index == len(step_uses) - 1
            model.set_requires_gradient_sync(sync)
            acts = [samples().requires_grad_() for _ in range(calls)]
            grad_out = tuple(samples() for _ in range(calls))
            # BF16 addmm's product and FP32 accumulation are exact for these small values.
            head_in = (samples().round() / 32).requires_grad_()
            head_grad = samples().round() / 32
            output, head = model(acts, head_in)
            torch.autograd.backward((*output, head), (*grad_out, head_grad))

            for x, dy in zip(acts, grad_out):
                _, _, x_t, x_s = fp8_quant_act(x.detach())
                _, _, dy_t, dy_s = fp8_quant_grad(dy)
                contribution = torch.empty_like(local_sums[0])
                fp8_wgrad(dy_t, dy_s, x_t, x_s, contribution)
                if has_fp8_grad:
                    local_sums[0].add_(contribution)
                else:
                    local_sums[0].copy_(contribution)
                    has_fp8_grad = True
            local_sums[1].addmm_(head_grad.float().T, head_in.detach().float())
            if not sync:
                assert all(weight.grad is None for weight in weights)
                continue
            # Round in the same order as FSDP2: reduce each synchronization window,
            # then add that result to the optimizer's shard.
            for i, total in enumerate(local_sums):
                dist.all_reduce(total)
                shard = total.chunk(dist.get_world_size(), dim=0)[rank]
                if expected[i] is None:
                    expected[i] = shard.clone()
                else:
                    expected[i].add_(shard)
                total.zero_()
            has_fp8_grad = False

        for weight, ref, shard in zip(weights, reference, expected):
            ref.grad = shard
            local_grad = weight.grad.to_local()
            assert local_grad.dtype == torch.float32
            # Compare FP32 bits, including the sign of zero.
            assert torch.equal(local_grad.view(torch.int32), ref.grad.view(torch.int32))

        optimizer.step()
        reference_optimizer.step()
        for weight, ref in zip(weights, reference):
            torch.testing.assert_close(weight.to_local(), ref, atol=0, rtol=0)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        FP8WeightCacheControl.step()
