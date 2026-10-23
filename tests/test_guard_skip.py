"""Guard skipping applies only to step signatures that already ran with full Dynamo guards."""

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo import eval_frame

from pithtrain.contexts import distributed, training
from pithtrain.pipeline.guard_skip import GuardSkip

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def contexts(monkeypatch):
    for name, value in dict(ep_size=4, cp_size=1, pp_size=2).items():
        monkeypatch.setattr(distributed, name, value, raising=False)
    monkeypatch.setattr(training, "fp8", True, raising=False)
    torch._dynamo.reset()


def microbatches(seq=64, cu_seqlens=None):
    tokens = torch.zeros(1, seq, dtype=torch.long, device="cuda")
    return [SimpleNamespace(model_inputs=(tokens,), cu_seqlens=cu_seqlens) for _ in range(4)]


def stance_skips():
    return eval_frame._stance.skip_guard_eval_unsafe


def test_only_repeated_signatures_skip():
    model, skip = torch.nn.Linear(2, 2), GuardSkip()
    run = lambda mbs: skip.run(model, mbs, stance_skips)  # noqa: E731
    assert run(microbatches()) is False  # first step: guarded
    assert run(microbatches()) is True  # same signature: skipped
    cu = torch.tensor([0, 30, 64], dtype=torch.int32, device="cuda")
    assert run(microbatches(cu_seqlens=cu)) is False  # packed: new signature, guarded
    assert run(microbatches(seq=128)) is False  # new shapes
    with torch.no_grad():
        assert run(microbatches()) is False  # inference / forward_only
        assert run(microbatches()) is True
    model.eval()
    assert run(microbatches()) is False  # training mode is part of the key
    assert stance_skips() is False  # restored after the step


class Variant(torch.nn.Module):
    def __init__(self, gated: bool):
        super().__init__()
        self.gated = gated
        self.proj = torch.nn.Linear(16, 16, device="cuda")

    @torch.compile(fullgraph=True)
    def compute(self, x, extra):
        y = self.proj(x)
        if self.gated:
            y = torch.sigmoid(y) * y
        if extra is not None:
            y = y + extra
        return y


def test_layer_variants_keep_their_entries_under_skip():
    """Two variants of one compiled method (attribute and None-ness differ) stay distinguished."""
    a, b = Variant(True), Variant(False)
    x = torch.randn(4, 16, device="cuda")
    extra = torch.randn(4, 16, device="cuda")
    calls = [(a, None), (b, None), (a, extra), (b, extra)]
    guarded = [m.compute(x, e) for m, e in calls]
    with torch.compiler.set_stance("default", skip_guard_eval_unsafe=True):
        skipped = [m.compute(x, e) for m, e in calls]
        eager = [Variant.compute.__wrapped__(m, x, e) for m, e in calls]
    for g, s, ref in zip(guarded, skipped, eager):
        assert torch.equal(g, s)
        torch.testing.assert_close(s, ref)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx, self.mlp = 0, torch.nn.Identity()
        self.self_attn = torch.nn.Linear(2, 2)


class LinearAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.idx, self.mlp = 1, torch.nn.Identity()
        self.linear_attn = torch.nn.Linear(2, 2)


def test_structurally_different_layers_never_skip():
    """Layer variants with different submodules make Dynamo's reduced guards raise: no skipping."""
    mixed = torch.nn.ModuleList([Attention(), LinearAttention()])
    skip = GuardSkip()
    for _ in range(3):
        assert skip.run(mixed, microbatches(), stance_skips) is False
    assert skip.signature(mixed, microbatches()) is None
    same = torch.nn.ModuleList([Attention(), Attention()])
    skip = GuardSkip()
    skip.run(same, microbatches(), stance_skips)
    assert skip.run(same, microbatches(), stance_skips) is True


class Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router_replay = None
        self.load_balance_loss_fn = None


class LoadBalance:
    def __init__(self, lb_coef):
        self.lb_coef = lb_coef


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Attention()
        self.layer.mlp = torch.nn.Sequential(Gate())


def armed():
    """A model and a GuardSkip whose signature already ran guarded once, and skips on repeat."""
    model, skip = Model(), GuardSkip()
    assert skip.run(model, microbatches(), stance_skips) is False
    assert skip.run(model, microbatches(), stance_skips) is True
    return model, skip


class NoopFunctionMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def test_hidden_condition_changes_run_guarded():
    """Every condition Dynamo guards outside the tensors takes the guarded path when it changes."""
    gate = lambda m: m.layer.mlp[0]  # noqa: E731

    def check(mutate, ctx=None):
        model, skip = armed()
        mutate(model)
        if ctx is not None:
            with ctx():
                assert skip.run(model, microbatches(), stance_skips) is False
        else:
            assert skip.run(model, microbatches(), stance_skips) is False

    check(lambda m: None, ctx=torch.inference_mode)  # inference mode
    check(lambda m: m.layer.self_attn.eval())  # a child module's training flag
    check(lambda m: None, ctx=NoopFunctionMode)  # torch-function mode
    hooks = lambda: torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)  # noqa: E731
    check(lambda m: None, ctx=hooks)  # saved-tensor hooks
    check(lambda m: setattr(gate(m), "router_replay", lambda n, k: None))  # replay installed
    check(lambda m: setattr(gate(m), "load_balance_loss_fn", LoadBalance(1e-3)))  # lb installed
    check(lambda m: m.layer.add_module("extra", torch.nn.Linear(2, 2)))  # layer structure

    model, skip = armed()  # replacing one replay callable by another
    gate(model).router_replay = lambda n, k: None
    skip.run(model, microbatches(), stance_skips)
    gate(model).router_replay = lambda n, k: None
    assert skip.run(model, microbatches(), stance_skips) is False

    model, skip = armed()  # changing a load-balance setting in place
    gate(model).load_balance_loss_fn = lb = LoadBalance(1e-3)
    skip.run(model, microbatches(), stance_skips)
    lb.lb_coef = 1e-2
    assert skip.run(model, microbatches(), stance_skips) is False
