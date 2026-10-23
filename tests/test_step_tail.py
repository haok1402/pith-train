"""The foreach gradient scaling and the sync-free clip are bit-identical to per-parameter ops."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from pithtrain.modules.optimizer import clip_grad_norm, local_grads, scale_grads_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def mesh():
    if not dist.is_initialized():
        store = tempfile.NamedTemporaryFile(delete=False)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        dist.init_process_group(
            "nccl", init_method=f"file://{store.name}", rank=0, world_size=1,
            device_id=torch.device("cuda", 0),
        )  # fmt: skip
    yield init_device_mesh("cuda", (1,))


def reference_scale(model, scale):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


@torch.no_grad()
def reference_clip(model, max_norm, norm_type=2.0, hsdp_replica=1):
    """r5-base clip_grad_norm, verbatim."""
    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g = g.to_local()
        grads.append(g)
    local_norm = torch.nn.utils.get_total_norm(grads, norm_type=norm_type)
    local_norm_sq = local_norm**norm_type
    torch.distributed.all_reduce(local_norm_sq, op=torch.distributed.ReduceOp.SUM)
    local_norm_sq = local_norm_sq / hsdp_replica
    total_norm = (local_norm_sq ** (1.0 / norm_type)).clamp(min=1e-6)
    clip_coef = (max_norm / total_norm).clamp(max=1.0)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm


def make_model(mesh, grad_magnitude, special=None, seed=0):
    """Mixed DTensor / plain parameters, bf16 and fp32 gradients, one parameter without a grad."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    model = torch.nn.Module()
    shapes = [((64, 32), torch.bfloat16, True), ((128,), torch.float32, True),
              ((48, 16), torch.float32, False), ((33,), torch.bfloat16, False),
              ((8, 8), torch.bfloat16, True)]  # fmt: skip
    for i, (shape, dtype, sharded) in enumerate(shapes):
        data = torch.randn(shape, generator=g, device="cuda").to(dtype)
        grad = (torch.randn(shape, generator=g, device="cuda") * grad_magnitude).to(dtype)
        if special is not None and i == 1:
            grad.view(-1)[:3] = torch.tensor([special, -0.0, 1e-30], device="cuda")
        if sharded:
            data, grad = distribute_tensor(data, mesh, [Shard(0)]), distribute_tensor(grad, mesh, [Shard(0)])  # fmt: skip
        p = torch.nn.Parameter(data)
        if i != 4:
            p.grad = grad
        model.register_parameter(f"p{i}", p)
    return model


def grads_of(model):
    return [g.clone() for g in local_grads(model)]


def assert_bitwise(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert torch.equal(x.view(-1).view(torch.int16 if x.element_size() == 2 else torch.int32),
                           y.view(-1).view(torch.int16 if y.element_size() == 2 else torch.int32))  # fmt: skip


@pytest.mark.parametrize("magnitude", [1e-4, 10.0])
@pytest.mark.parametrize("special", [None, float("nan"), float("inf")])
def test_scale_and_clip_bitwise(mesh, magnitude, special):
    ref, new = make_model(mesh, magnitude, special), make_model(mesh, magnitude, special)
    scale = torch.tensor(1.0 / 3000.0, device="cuda")
    reference_scale(ref, scale)
    scale_grads_(local_grads(new), scale)
    assert_bitwise(grads_of(ref), grads_of(new))
    n_ref = reference_clip(ref, max_norm=1e-3 if magnitude > 1 else 1e3)
    n_new = clip_grad_norm(new, max_norm=1e-3 if magnitude > 1 else 1e3)
    assert_bitwise(grads_of(ref), grads_of(new))
    assert_bitwise([n_ref.view(1)], [n_new.view(1)])


def test_clip_coefficient_is_applied(mesh):
    # Large gradients against a small max_norm: the coefficient is < 1 and must be applied.
    model = make_model(mesh, 10.0)
    before = grads_of(model)
    clip_grad_norm(model, max_norm=1e-3)
    assert not any(torch.equal(a, b) for a, b in zip(before, grads_of(model)))


def test_skipping_a_parameter_is_detected(mesh):
    ref, new = make_model(mesh, 10.0), make_model(mesh, 10.0)
    scale = torch.tensor(0.5, device="cuda")
    reference_scale(ref, scale)
    scale_grads_(local_grads(new)[1:], scale)
    with pytest.raises(AssertionError):
        assert_bitwise(grads_of(ref), grads_of(new))


def test_clip_has_no_host_sync(mesh):
    model, ref = make_model(mesh, 10.0), make_model(mesh, 10.0, seed=1)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        clip_grad_norm(model, max_norm=1e-3)
        # r5-base branches on the coefficient on the host, which the debug mode reports.
        with pytest.raises(RuntimeError):
            reference_clip(ref, max_norm=1e-3)
    finally:
        torch.cuda.set_sync_debug_mode("default")
