"""
Saved-tensor accounting of the Qwen3.5 token-mixing blocks under selective checkpointing.

A compiled block's forward is run under saved_tensors_hooks to list every tensor autograd keeps
for backward (shape, dtype) and their bytes, and the memory held between forward and backward is
measured. A launch-path change to an op inside the block must leave both unchanged; in particular
the recomputed ops (the GDN conv, the FP8 input projections) must stay recomputed rather than have
their outputs saved.
"""

import pytest
import torch

from pithtrain.operators import gated_delta_rule as gdr_module
from tests.operators.test_gated_delta_rule_pinned_sm90 import fla_pinned  # noqa: F401  (fixture)


def account(f, args: tuple, grad_seed: int = 1) -> dict:
    """Saved tensors (sorted (shape, dtype) list), their bytes and the held memory of one call."""
    f(*[a.clone().requires_grad_() if a.is_floating_point() else a for a in args]).float().sum().backward()  # fmt: skip
    torch.cuda.synchronize()
    saved = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append((tuple(t.shape), str(t.dtype), t.numel() * t.element_size()))
        return t

    x = args[0].clone().requires_grad_()
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = f(x, *args[1:])
    held = torch.cuda.memory_allocated() - m0
    torch.manual_seed(grad_seed)
    y.backward(torch.randn_like(y))
    out = {"saved": sorted((s, d) for s, d, _ in saved), "bytes": sum(b for *_, b in saved)}
    return {**out, "held": held, "y": y.detach(), "dx": x.grad}


def gdn_block(monkeypatch: pytest.MonkeyPatch, fp8: bool, S: int = 4096):
    from transformers import AutoConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models import qwen35_moe
    from pithtrain.operators.linear import BF16Linear, FP8Linear

    monkeypatch.setattr(training, "Linear", FP8Linear if fp8 else BF16Linear, raising=False)
    monkeypatch.setattr(training, "fp8", fp8, raising=False)
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    cfg = AutoConfig.from_pretrained("examples/pretrain_lm/qwen3.5-35b-a3b")
    torch.manual_seed(0)
    gdn = qwen35_moe.Qwen35MoeGatedDeltaNet(cfg, 0).cuda().to(torch.bfloat16)
    for p in gdn.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(1, S, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    return gdn, x


def compiled(module):
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True)


@pytest.mark.usefixtures("fla_pinned")
@pytest.mark.parametrize("new_ops", [("delta_rule",)])
def test_gdn_launch_paths_keep_saved_set(monkeypatch: pytest.MonkeyPatch, new_ops: tuple) -> None:
    """
    FP8 GDN block: the pinned delta rule saves what the custom op saves minus the repeated q / k,
    and the conv (with the input projections) stays recomputed.
    """
    from pithtrain.models import qwen35_moe

    gdn, x = gdn_block(monkeypatch, fp8=True)
    real_gdr = gdr_module.gated_delta_rule

    def select(use_new: bool) -> None:
        on = lambda op: use_new and op in new_ops  # noqa: E731
        monkeypatch.setattr(qwen35_moe, "gated_delta_rule", lambda *a, pinned=False, **kw: real_gdr(*a, pinned=on("delta_rule"), **kw))  # fmt: skip

    select(True)
    new = account(compiled(gdn), (x,))
    select(False)
    ref = account(compiled(gdn), (x,))
    print(new_ops, "held", new["held"] / 2**20, "MiB vs", ref["held"] / 2**20, "MiB; saved", new["saved"])  # fmt: skip
    assert torch.equal(new["y"], ref["y"]) and torch.equal(new["dx"], ref["dx"])
    # The conv / in_proj_qkv output (1, S, 8192) bf16 is recomputed, never saved.
    assert ((1, x.shape[1], 8192), "torch.bfloat16") not in new["saved"]
    # The pinned delta rule reads q / k unrepeated: of the [1, S, 32, 128] bf16 tensors the
    # custom op saved (repeat_interleave'd q and k, and v) only v is left, and the op holds the
    # 16-head q / k instead (saved as their per-group expands); nothing else changes.
    per_value_head = ((1, x.shape[1], 32, 128), "torch.bfloat16")
    assert ref["saved"].count(per_value_head) == 3 and new["saved"].count(per_value_head) == 1
    repeated, unrepeated = 2 * x.shape[1] * 32 * 128 * 2, 2 * x.shape[1] * 16 * 128 * 2
    assert ref["held"] - new["held"] == repeated - unrepeated


def test_gdn_saved_set_detects_policy_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: with the conv dropped from the recompute policy its output gets saved."""
    from pithtrain.models import qwen35_moe

    gdn, x = gdn_block(monkeypatch, fp8=True)
    ref = account(compiled(gdn), (x,))
    policy = qwen35_moe._recompute_policy
    conv = torch.ops.pithtrain.qwen35_causal_conv1d.default

    def leaky(ctx, func, *args, **kwargs):
        if func is conv:
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_SAVE
        return policy(ctx, func, *args, **kwargs)

    monkeypatch.setattr(qwen35_moe, "_recompute_policy", leaky)
    got = account(compiled(gdn), (x,))
    assert got["bytes"] > ref["bytes"]
