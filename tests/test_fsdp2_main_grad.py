"""Check FP32 main gradients in direct dense FSDP2."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels import fp8_quant_act, fp8_quant_grad, fp8_wgrad, fp8_wgrad_supported
from pithtrain.modules.wgrad_accumulation import install_wgrad_accumulation
from pithtrain.operators.linear import BF16Linear, FP8Linear

pytestmark = pytest.mark.skipif(not fp8_wgrad_supported(), reason="requires Hopper")


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    store = tmp_path_factory.mktemp("fsdp2") / "store"
    dist.init_process_group("nccl", init_method=store.as_uri(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cuda", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "uses,mode",
    [
        pytest.param((0,), "eager", id="skipped"),
        pytest.param((2,), "eager", id="reused"),
        pytest.param((2,), "tied", id="tied-weight"),
        pytest.param((1, 1, 0), "compiled", id="compiled-deferred-with-final-skip"),
    ],
)
def test_dense_main_grad(mesh, uses, mode):
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.fp8 = FP8Linear(128, 128, bias=False, device="cuda")
            self.head = BF16Linear(128, 128, bias=False, device="cuda")
            self.tied = None
            if mode == "tied":
                self.tied = FP8Linear(128, 128, bias=False, device="cuda")
                self.tied.weight = self.fp8.weight

        def project(self, x):
            return self.fp8(x)

        def forward(self, inputs, head_input):
            outputs = tuple(
                self.tied(x) if i == 1 and self.tied is not None else self.project(x)
                for i, x in enumerate(inputs)
            )
            return (*outputs, self.head(head_input), self.head(head_input))

    torch.manual_seed(17)
    model = Dense()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    install_wgrad_accumulation(model, overwrite=True)
    if mode == "compiled":
        model.project = torch.compile(model.project, fullgraph=True, backend="aot_eager")
    if model.tied is not None:
        assert model.tied.main_grad is model.fp8.main_grad

    group = fully_shard.state(model)._fsdp_param_groups[0]
    pre_backward = group.pre_backward

    def poison_fresh_fp8_buffer(*args):
        fresh = not model.fp8.main_grad.untyped_storage().size()
        pre_backward(*args)
        if fresh:
            model.fp8.main_grad.fill_(torch.nan)

    group.pre_backward = poison_fresh_fp8_buffer

    def samples():
        return torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)

    # Reverse the use pattern on the next step to check the write state resets.
    for step_uses in (uses, (int(not any(uses)),)):
        contributions = []
        expected_head = torch.zeros((128, 128), device="cuda")
        for index, calls in enumerate(step_uses):
            sync = index == len(step_uses) - 1
            model.set_requires_gradient_sync(sync)
            inputs = [samples().requires_grad_() for _ in range(calls)]
            head_input = (samples().round() / 32).requires_grad_()
            head_grad = samples().round() / 32
            grads = [samples() for _ in inputs]
            torch.autograd.backward(model(inputs, head_input), (*grads, head_grad, head_grad))
            expected_head.addmm_(head_grad.float().T, head_input.detach().float(), alpha=2)
            for x, dy in reversed(list(zip(inputs, grads, strict=True))):
                _, _, x_t, x_scale = fp8_quant_act(x.detach())
                _, _, dy_t, dy_scale = fp8_quant_grad(dy)
                grad = torch.empty_like(expected_head)
                fp8_wgrad(dy_t, dy_scale, x_t, x_scale, grad)
                contributions.append(grad)
            if not sync:
                assert model.fp8.weight.grad is None

        expected_fp8 = contributions[0] if contributions else torch.zeros_like(expected_head)
        for grad in contributions[1:]:
            expected_fp8.add_(grad)
        for layer, expected in ((model.fp8, expected_fp8), (model.head, expected_head)):
            actual = layer.weight.grad.to_local()
            assert actual.dtype == torch.float32
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
            layer.weight.grad = None
        FP8WeightCacheControl.step()
