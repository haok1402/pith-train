"""
ZeRO-2 and ZeRO-3 for the FSDP2 groups of the two V-chunks, driven per decoder layer.

DistributedCfg.attn_sharding_stage covers every parameter outside the routed experts, and
expert_sharding_stage the routed experts. Each picks what a rank holds unsharded during a step:

- 1: the full bf16 weights and the full fp32 gradient for the whole step, reduce-scattered once at
  its end by DualPipeV. Nothing here is installed for these groups.
- 2: the full weights, but a layer's gradient is reduce-scattered after each micro-batch's backward
  through it, and its main_grad freed.
- 3: as 2, and a layer's weights are also resharded after its forward and after its backward, and
  all-gathered again for the next use.

FSDP2's own hooks act per call, and a decoder layer is called three times per micro-batch (stages 1,
3 and 5) while the routed experts defer their weight gradients to DualPipeV's W phase: reducing per
call would reduce a layer three times per micro-batch, and the experts before their weight
gradients exist. For the groups at stage 2 or 3, DualPipeV.run_step instead holds the FSDP2 state
in PRE_BACKWARD for the whole step, where FSDP2 only all-gathers on demand, and the hooks installed
here drive the groups per layer:

- A layer's backward begins at its stage 3 output, since stage 5 reads no parameters: main_grad is
  allocated, the weights are all-gathered (stage 3) and the previous layer's are prefetched.
- It ends where the gradient reaches the stage 1 input for the attention group, and the stage 3
  input for the experts, which reshard there (their weight gradients read only activations) and
  reduce after their weight gradients, queued behind them in WeightGradStore when the W phase
  defers them.
- At stage 3 a layer reshards after its stage 3 forward, and its stage 1 prefetches its experts and
  the next layer's attention group. qwen3's fused forward_stage5_stage1, which FSDP2 does not wrap,
  all-gathers the next layer itself.

Prefetching follows the order of a V-chunk, embed_tokens, then the layers, then the final norm and
lm_head, which take the attention stage; FSDP2's implicit order, every call of the step in forward
order, interleaves micro-batches and chunks under DualPipeV. A chunk's first group is all-gathered
when first used. The embedding's input needs no gradient, so its backward ends with the backward
pass that runs it.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.modules.fsdp_forward import replace_grad_tensors, scan_call

__all__ = ["install_sharding_stages"]


@dataclass(eq=False)
class ChunkItem:
    """
    One entry of a V-chunk's order: a decoder layer, or embed_tokens, norm or lm_head. attn and
    experts hold the item's groups at stage 2 or 3; groups at stage 1 are left to DualPipeV.
    """

    attn: List[FSDPParamGroup]
    experts: List[FSDPParamGroup] = field(default_factory=list)
    prev: Optional["ChunkItem"] = None
    next: Optional["ChunkItem"] = None


@dataclass
class ShardingStages:
    attn: int
    experts: int

    def attn_resharded(self, item: Optional[ChunkItem]) -> List[FSDPParamGroup]:
        return item.attn if item is not None and self.attn == 3 else []

    def experts_resharded(self, item: Optional[ChunkItem]) -> List[FSDPParamGroup]:
        return item.experts if item is not None and self.experts == 3 else []


class RunInBackward(torch.autograd.Function):
    """
    Identity on some tensors whose backward calls fn once their gradients are all in.
    """

    @staticmethod
    def forward(ctx: Any, fn: Callable[[], None], *tensors: torch.Tensor):
        ctx.fn = fn
        ctx.set_materialize_grads(False)
        return tensors

    @staticmethod
    def backward(ctx: Any, *grads: Optional[torch.Tensor]):
        ctx.fn()
        return (None, *grads)


def run_in_backward(fn: Callable[[], None], args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """
    args and kwargs with their grad-requiring tensors passed through one RunInBackward(fn), or as
    they are when they hold none or grad is disabled.
    """
    tensors = []
    scan_call(args, kwargs, None, tensors)
    if not tensors or not torch.is_grad_enabled():
        return args, kwargs
    replacements = iter(RunInBackward.apply(fn, *tensors))
    args = replace_grad_tensors(args, replacements)
    return args, {name: replace_grad_tensors(value, replacements) for name, value in kwargs.items()}


def prefetch(groups: List[FSDPParamGroup], pass_type: str) -> None:
    """
    Start the all-gather of each sharded group on FSDP2's all-gather streams; the next use waits.
    """
    for group in groups:
        FSDPParamGroup._prefetch_unshard(group, pass_type)


def gather(groups: List[FSDPParamGroup]) -> None:
    for group in groups:
        group.unshard()
        group.wait_for_unshard()


def reshard(groups: List[FSDPParamGroup]) -> None:
    for group in groups:
        group.reshard()


def begin_backward(groups: List[FSDPParamGroup]) -> None:
    # pre_backward, which wgrad_accumulation extends to allocate main_grad, all-gathers any
    # resharded weights.
    for group in groups:
        group.pre_backward(False)


def end_backward(groups: List[FSDPParamGroup], reshard_after: bool) -> None:
    # post_backward with this one reduction enabled (run_step disables reduction for the step),
    # which wgrad_accumulation extends to hand over and free main_grad.
    for group in groups:
        group.reduce_grads = group.all_reduce_grads = True
        group.reshard_after_backward = reshard_after
        group.post_backward()
        group.reduce_grads = group.all_reduce_grads = group.reshard_after_backward = False


def drive(module: nn.Module, stage: int) -> List[FSDPParamGroup]:
    """
    The groups of the FSDP2 module to drive here, none at stage 1. Records the stage on the module
    for DualPipeV.run_step.
    """
    if stage == 1:
        return []
    module.sharding_stage = stage
    return list(fully_shard.state(module)._fsdp_param_groups)


def install_layer(layer: nn.Module, item: ChunkItem, stages: ShardingStages) -> None:
    """
    Wrap the decoder layer's forward_stage1, forward_stage3 and fused forward_stage5_stage1, if it
    has one, around the bound methods FSDP2 registered.
    """
    attn, experts = item.attn, item.experts

    def end_attn() -> None:
        end_backward(attn, stages.attn == 3)

    def end_experts() -> None:
        # The input gradient is done, and the weight gradients read only activations.
        reshard(stages.experts_resharded(item))
        if WeightGradStore.enabled:
            WeightGradStore.put(lambda: end_backward(experts, False))
        else:
            end_backward(experts, False)

    def begin() -> None:
        begin_backward(attn + experts)
        prefetch(stages.attn_resharded(item.prev) + stages.experts_resharded(item.prev), "backward")  # fmt: skip

    forward_stage1 = layer.forward_stage1

    def stage1(*args, **kwargs):
        prefetch(stages.experts_resharded(item) + stages.attn_resharded(item.next), "forward")
        args, kwargs = run_in_backward(end_attn, args, kwargs) if attn else (args, kwargs)
        return forward_stage1(*args, **kwargs)

    forward_stage3 = layer.forward_stage3

    def stage3(*args, **kwargs):
        args, kwargs = run_in_backward(end_experts, args, kwargs) if experts else (args, kwargs)
        out = forward_stage3(*args, **kwargs)
        # The layer's weights are done for this forward: stage 5 reads none.
        reshard(stages.attn_resharded(item) + stages.experts_resharded(item))
        (out,), _ = run_in_backward(begin, (out,), {})
        return out

    layer.forward_stage1, layer.forward_stage3 = stage1, stage3

    forward_stage5_stage1 = getattr(layer, "forward_stage5_stage1", None)
    if forward_stage5_stage1 is not None:

        def stage5_stage1(*args, **kwargs):
            # FSDP2 does not wrap the fused method, so its weights are all-gathered here.
            gather(attn)
            prefetch(stages.experts_resharded(item) + stages.attn_resharded(item.next), "forward")
            args, kwargs = run_in_backward(end_attn, args, kwargs) if attn else (args, kwargs)
            return forward_stage5_stage1(*args, **kwargs)

        layer.forward_stage5_stage1 = stage5_stage1


def install_module(module: nn.Module, item: ChunkItem, stages: ShardingStages) -> None:
    """
    Hook embed_tokens, norm or lm_head, each called once per micro-batch, after FSDP2's hooks.
    """
    groups = item.attn

    def end() -> None:
        end_backward(groups, stages.attn == 3)

    def begin() -> None:
        begin_backward(groups)
        prefetch(stages.attn_resharded(item.prev) + stages.experts_resharded(item.prev), "backward")  # fmt: skip
        if isinstance(module, nn.Embedding):
            # No gradient reaches the token ids; the weight's lands by the end of this pass.
            Variable._execution_engine.queue_callback(end)

    def pre_hook(_: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        prefetch(stages.attn_resharded(item.next), "forward")
        return run_in_backward(end, args, kwargs)

    def post_hook(_: nn.Module, args: tuple, output: Any) -> Any:
        reshard(stages.attn_resharded(item))
        (output,), _ = run_in_backward(begin, (output,), {})
        return output

    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook)


def install_sharding_stages(model: nn.Module, attn_stage: int, expert_stage: int) -> None:
    """
    Drive the FSDP2 groups of model, the nn.Sequential of the two V-chunks, at these stages. Call
    after fully_shard and register_fsdp_forward_method; a no-op at stages (1, 1).
    """
    assert attn_stage in (1, 2, 3) and expert_stage in (1, 2, 3), (attn_stage, expert_stage)
    if attn_stage == expert_stage == 1:
        return
    stages = ShardingStages(attn_stage, expert_stage)
    for chunk in model:
        order: List[tuple[nn.Module, ChunkItem]] = []
        if chunk.embed_tokens is not None:
            order.append((chunk.embed_tokens, ChunkItem(drive(chunk.embed_tokens, attn_stage))))
        for layer in chunk.layers.values():
            experts = getattr(layer.mlp, "experts", None)
            expert_groups = drive(experts, expert_stage) if isinstance(experts, FSDPModule) else []
            order.append((layer, ChunkItem(drive(layer, attn_stage), expert_groups)))
        for module in (chunk.norm, chunk.lm_head):
            if module is not None:
                order.append((module, ChunkItem(drive(module, attn_stage))))
        for (_, before), (_, after) in zip(order, order[1:]):
            before.next, after.prev = after, before
        for module, item in order:
            if not (item.attn or item.experts):
                continue
            if hasattr(module, "forward_stage1"):
                install_layer(module, item, stages)
            else:
                install_module(module, item, stages)
