"""The lean FSDP forward-method wrapper matches FSDP's own wrapper call for call."""

import copy
import inspect
import os
import tempfile
from collections import Counter, namedtuple

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp import register_fsdp_forward_method as upstream_register
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    FSDPParamGroup,
    RegisterPostBackwardFunction,
)
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

import pithtrain.modules.fsdp_forward as fsdp_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module", autouse=True)
def world():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        dist.init_process_group(
            "nccl", init_method=f"file://{tempfile.mktemp()}", rank=0, world_size=1,
            device_id=torch.device("cuda", 0),
        )  # fmt: skip
    torch.cuda.set_device(0)


Pair = namedtuple("Pair", "y z")


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 64, bias=False)
        self.b = torch.nn.Linear(64, 64, bias=False)

    def stage(self, x, w, rot=(None, None), flag=False, splits=()):
        y = self.a(x) * w  # w is fp32 router-probability-like: FSDP casts it to bf16
        if rot[0] is not None:
            y = y + rot[0]
        return Pair(y, self.b(x))


class Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])
        self.inside = None  # (inputs, outs list) to run the layers inside the root's forward

    def forward(self, x):
        if self.inside is not None:
            inputs, outs = self.inside
            outs.extend(forward_layers(self, inputs))
        return x


def build(register, seed=0):
    torch.manual_seed(seed)
    root = Root().cuda()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in root.layers:
        fully_shard(layer, reshard_after_forward=False, mp_policy=mp)
        register(layer, "stage")
    fully_shard(root, reshard_after_forward=False, mp_policy=mp)
    root.set_requires_gradient_sync(False)
    root.set_reshard_after_backward(False)
    root(torch.zeros(1, device="cuda"))  # lazy init with the model as the FSDP root
    return root


def accumulated_grads(root):
    """Each layer parameter's unsharded gradient as FSDP accumulates it across micro-batches."""
    grads = []
    for layer in root.layers:
        for group in layer._get_fsdp_state()._fsdp_param_groups:
            for fsdp_param in group.fsdp_params:
                g = fsdp_param.unsharded_accumulated_grad
                grads.append((g if g is not None else fsdp_param.unsharded_param.grad).clone())
    return grads


class Spy:
    """Counts FSDP's hook-side calls and the lean wrapper's full-path calls."""

    def __init__(self, monkeypatch):
        self.counts = Counter()
        for cls, name in ((FSDPParamGroup, "pre_forward"), (FSDPParamGroup, "post_backward"),
                          (FSDPParamGroup, "pre_backward"), (FSDPState, "_pre_backward")):  # fmt: skip
            orig = getattr(cls, name)

            def wrapped(*a, _orig=orig, _name=name, **k):
                self.counts[_name] += 1
                return _orig(*a, **k)

            monkeypatch.setattr(cls, name, wrapped)
        forward = RegisterPostBackwardFunction.forward

        def register(*a):
            self.counts["RegisterPostBackwardFunction"] += 1
            return forward(*a)

        monkeypatch.setattr(RegisterPostBackwardFunction, "forward", staticmethod(register))


def forward_layers(root, inputs):
    outs = []
    for x, w, rot in inputs:
        h = x
        for layer in root.layers:
            y, z = layer.stage(h, w, rot, flag=True, splits=[3, 1, 4])  # scalars, as in routing
            h = (y + z).float().to(torch.bfloat16)
        outs.append(h)
    return outs


def run(root, inputs, inside_root=False):
    """
    Micro-batch forwards through every layer, then the backwards, as the pipeline does: called
    directly (the overlapped loop) or from inside the root module's forward (sequential passes).
    """
    if inside_root:
        outs = []
        root.inside = (inputs, outs)
        root(torch.zeros(1, device="cuda"))
        root.inside = None
    else:
        outs = forward_layers(root, inputs)
    for h in reversed(outs):
        h.float().pow(2).sum().backward()
    return [o.detach() for o in outs], accumulated_grads(root)


def make_inputs():
    g = torch.Generator(device="cuda").manual_seed(1)
    inputs = []
    for _ in range(4):
        x = torch.randn(16, 64, generator=g, device="cuda", dtype=torch.bfloat16).requires_grad_()
        w = torch.rand(16, 1, generator=g, device="cuda", dtype=torch.float32).requires_grad_()
        rot = (torch.randn(16, 64, generator=g, device="cuda", dtype=torch.bfloat16), None)
        inputs.append((x, w, rot))
    return inputs


@pytest.mark.parametrize("inside_root", [False, True])
def test_lean_wrapper_matches_upstream(monkeypatch, inside_root):
    spy = Spy(monkeypatch)
    ref_out, ref_grads = run(build(upstream_register), make_inputs(), inside_root)
    ref_counts = copy.copy(spy.counts)
    spy.counts.clear()
    out, grads = run(build(fsdp_forward.register_fsdp_forward_method), make_inputs(), inside_root)
    for a, b in zip(ref_out + ref_grads, out + grads):
        assert a.dtype == b.dtype and torch.equal(a, b)
    for name in ("post_backward", "pre_backward", "_pre_backward", "RegisterPostBackwardFunction"):
        assert spy.counts[name] == ref_counts[name], (name, spy.counts, ref_counts)
    # The repeated calls took the lean path: FSDP's own pre_forward ran only for each layer's
    # first call (4 micro-batches x 3 layers upstream, 1 x 3 lean).
    assert ref_counts["pre_forward"] == 12 and spy.counts["pre_forward"] == 3


def run_interleaved(root, inputs):
    """
    The overlapped loop's order: each micro-batch's backward before the next one's forward, with
    FSDP's root post-backward callback deferred to the end as DualPipeV._step does, so every layer
    is still in PRE_BACKWARD when the next micro-batch reaches it.
    """
    state_ctx = root._get_fsdp_state()._state_ctx
    state_ctx.post_backward_final_callback_queued = True
    outs = []
    for x, w, rot in inputs:
        (h,) = forward_layers(root, [(x, w, rot)])
        h.float().pow(2).sum().backward()
        outs.append(h.detach())
    root._get_fsdp_state()._root_post_backward_final_callback()
    return outs, accumulated_grads(root)


def test_backward_state_calls_match_upstream(monkeypatch):
    spy = Spy(monkeypatch)
    pre = Counter()
    orig = FSDPState._pre_forward

    def counted(*a, **k):
        pre["_pre_forward"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(FSDPState, "_pre_forward", counted)
    ref_root = build(upstream_register)
    spy.counts.clear()
    pre.clear()
    ref_out, ref_grads = run_interleaved(ref_root, make_inputs())
    ref_counts, ref_pre = copy.copy(spy.counts), pre["_pre_forward"]
    root = build(fsdp_forward.register_fsdp_forward_method)
    spy.counts.clear()
    pre.clear()
    out, grads = run_interleaved(root, make_inputs())
    for a, b in zip(ref_out + ref_grads, out + grads):
        assert a.dtype == b.dtype and torch.equal(a, b)
    for name in ("post_backward", "pre_backward", "_pre_backward", "RegisterPostBackwardFunction"):
        assert spy.counts[name] == ref_counts[name], (name, spy.counts, ref_counts)
    # Upstream: every call goes through _pre_forward. Lean: only the first micro-batch's (the
    # later ones find their layer in PRE_BACKWARD with its parameters unsharded).
    assert ref_pre == 12 and pre["_pre_forward"] == 3


def test_full_path_when_not_applicable(monkeypatch):
    spy = Spy(monkeypatch)
    root = build(fsdp_forward.register_fsdp_forward_method)
    x, w, rot = make_inputs()[0]
    with torch.no_grad():  # no grad: always FSDP's own path
        for _ in range(3):
            root.layers[0].stage(x, w, rot)
    assert spy.counts["pre_forward"] == 3


def test_lean_hooks(monkeypatch):
    for cls, name in fsdp_forward._HOOKS:  # restore FSDP's own hooks after the test
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    originals = {(cls, name): cls.__dict__[name].__wrapped__ for cls, name in fsdp_forward._HOOKS}
    calls = Counter()
    disable = torch._dynamo.disable

    def counted(*a, **k):
        calls["disable"] += 1
        return disable(*a, **k)

    monkeypatch.setattr(torch._dynamo, "disable", counted)
    ref_out, ref_grads = run_interleaved(
        build(fsdp_forward.register_fsdp_forward_method), make_inputs()
    )
    assert calls["disable"] > 0  # FSDP's decorator: one disable wrapper built per hook call
    fsdp_forward.install_lean_hooks()
    hooks = {key: key[0].__dict__[key[1]] for key in originals}
    for key, hook in hooks.items():
        assert hook._torchdynamo_disable and inspect.unwrap(hook) is originals[key]
    fsdp_forward.install_lean_hooks()  # idempotent
    assert all(key[0].__dict__[key[1]] is hook for key, hook in hooks.items())
    root = build(fsdp_forward.register_fsdp_forward_method)
    calls.clear()
    out, grads = run_interleaved(root, make_inputs())
    assert calls["disable"] == 0
    for a, b in zip(ref_out + ref_grads, out + grads):
        assert a.dtype == b.dtype and torch.equal(a, b)
