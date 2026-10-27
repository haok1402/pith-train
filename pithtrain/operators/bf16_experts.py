"""
BF16 SwiGLU routed experts on SM90 from the received tokens, as one autograd node.

Forward, for slots grouped by expert (bf16_expert_group_slots; no expanded or scattered copy of
the tokens is written):

    preact, h = gated(gathered[a_idx] @ [W_gate | W_up]^T)   one GEMM, SwiGLU in its epilogue
    out[slot_of[r]] = (h @ W_down^T)[r]                        per expert, rows stored to their
                                                              received slots by the epilogue

Backward:

    grad_preact, h = dgated(grad_out[slot_of] @ W_down, preact)   dSwiGLU epilogue
    grad_x = grad_preact @ [W_gate; W_up]                           one GEMM over K = 2I
    grad_gathered[t] = sum of grad_x over the slots of token t      fp32, in expert order

and, deferred to DualPipeV's W phase through WeightGradStore, the weight gradients accumulated
straight into the fp32 main_grads:

    main_grad_down    += grad_out[slot_of]^T h
    main_grad_gate|up += grad_preact^T gathered[a_idx]    one GEMM over N = 2I, split on store

Only gathered, preact and the index maps are saved. h is the SwiGLU of the bf16-rounded preact in
both directions, so the backward recomputes the forward's h bit for bit; the grouped expert input
is never written. Every reduction runs in a fixed order, so the node is deterministic. The math is
that of the unfused path (gather, scatter, gate/up/down GroupedLinear and silu_mul) with fp32
accumulation everywhere: the dgrad sums the gate and up contributions and the token's rows in fp32
before rounding once.

gpt-oss (bf16_clamped_swiglu_experts): per-expert biases added in fp32 inside the two forward
epilogues, the clamped SwiGLU of the model's limit, and the bias gradients as per-expert row sums
(expert_slot_segment_sum) returned with the input gradient.

SM90 only: the models import this module lazily, after pithtrain.kernels.bf16_experts_supported
said yes.
"""

from functools import partial
from typing import NamedTuple

import torch
import torch.nn as nn

from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import (
    bf16_expert_dgated_backward,
    bf16_expert_dgrad,
    bf16_expert_down_forward_scattered,
    bf16_expert_gated_forward,
    bf16_expert_gather_rows,
    bf16_expert_gather_sum,
    bf16_expert_group_slots,
    bf16_expert_wgrad_down,
    bf16_expert_wgrad_gate_up,
    expert_slot_segment_sum,
)
from pithtrain.operators.main_grad import raw_main_grad

__all__ = ["BF16SwiGLUExpertsFunc", "bf16_swiglu_experts", "bf16_clamped_swiglu_experts"]


class _WeightGradTargets:
    """
    Where the weight gradients go: (owner, parameter name, row slice) for gate, up and down. A
    weight with an fp32 main_grad (GroupedLinear.main_grad, or raw_main_grad for a raw parameter)
    accumulates there; one without gets an fp32 temporary added to its bf16 .grad.
    """

    def __init__(self, targets):
        self.targets = targets

    def run(self, accumulate) -> None:
        main_grads, temporaries = {}, []
        for owner, name, _ in self.targets:
            if (owner, name) in main_grads:
                continue
            main_grad = owner.main_grad if name == "weight" else raw_main_grad(owner, name)
            if main_grad is None:
                main_grad = torch.zeros_like(getattr(owner, name), dtype=torch.float32)
                temporaries.append((owner, name, main_grad))
            main_grads[owner, name] = main_grad
        accumulate(
            *(
                main_grads[owner, name] if rows is None else main_grads[owner, name][rows]
                for owner, name, rows in self.targets
            )
        )
        for owner, name, main_grad in temporaries:
            param = getattr(owner, name)
            grad = main_grad.to(param.dtype)
            param.grad = grad if param.grad is None else param.grad.add_(grad)


def _accumulate_wgrads(
    targets: _WeightGradTargets,
    x,
    grad_out,
    h,
    grad_preact,
    is_grad_preact_swapped,
    maps: NamedTuple,
) -> None:
    def accumulate(main_grad_gate, main_grad_up, main_grad_down):
        bf16_expert_wgrad_down(grad_out, h, main_grad_down, maps.cu_seqlens, maps.slot_of)
        bf16_expert_wgrad_gate_up(
            x,
            grad_preact,
            is_grad_preact_swapped,
            main_grad_gate,
            main_grad_up,
            maps.cu_seqlens,
            maps.a_idx,
        )

    targets.run(accumulate)


class BF16SwiGLUExpertsFunc(torch.autograd.Function):
    """
    See the module docstring. Returns the expert outputs in received-slot order.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gathered: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        bias_gate_up: torch.Tensor | None,
        bias_down: torch.Tensor | None,
        maps: NamedTuple,
        wgrad_targets: _WeightGradTargets,
        max_slots: int,
        act_params: tuple,
        cu_seqlens_host: list[int] | None,
    ) -> torch.Tensor:
        cu_seqlens = maps.cu_seqlens
        preact, h = bf16_expert_gated_forward(
            gathered, w_gate, w_up, cu_seqlens, maps.a_idx, bias_gate_up, act_params
        )
        out = bf16_expert_down_forward_scattered(h, w_down, cu_seqlens, maps.slot_of, bias_down)
        del h
        # The received tokens (and, in backward, grad_out) come from an all-to-all on the comm
        # stream and are read again by the deferred weight gradient on this stream: tie their reuse
        # to this stream, so the allocator cannot hand the block to the next all-to-all before the
        # wgrad has run.
        gathered.record_stream(torch.cuda.current_stream())
        ctx.save_for_backward(gathered, preact)
        ctx.weights = (w_gate, w_up, w_down)
        ctx.maps, ctx.wgrad_targets, ctx.max_slots = maps, wgrad_targets, max_slots
        ctx.act_params, ctx.cu_seqlens_host = act_params, cu_seqlens_host
        ctx.bias_dtypes = tuple(
            bias.dtype if bias is not None else None for bias in (bias_gate_up, bias_down)
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gathered, preact = ctx.saved_tensors
        w_gate, w_up, w_down = ctx.weights
        maps = ctx.maps
        cu_seqlens = maps.cu_seqlens
        grad_out = grad_out.contiguous()
        grad_out.record_stream(torch.cuda.current_stream())
        grad_preact, h, is_grad_preact_swapped = bf16_expert_dgated_backward(
            grad_out, w_down, preact, cu_seqlens, maps.slot_of, w_gate, w_up, ctx.act_params
        )
        del preact
        grad_bias_gate_up = grad_bias_down = None
        if ctx.bias_dtypes[0] is not None:
            # gpt-oss: gate is the lower address, so grad_preact's pairs interleave like the bias
            cu_seqlens_host = ctx.cu_seqlens_host
            if cu_seqlens_host is None:
                cu_seqlens_host = cu_seqlens.tolist()
            grad_bias_gate_up = expert_slot_segment_sum(
                grad_preact, cu_seqlens_host, cu_seqlens
            ).to(ctx.bias_dtypes[0])
            grad_bias_down = expert_slot_segment_sum(
                grad_out, cu_seqlens_host, cu_seqlens, maps.slot_of
            ).to(ctx.bias_dtypes[1])
        grad_x = bf16_expert_dgrad(grad_preact, w_gate, w_up, cu_seqlens, is_grad_preact_swapped)
        if maps.mask is not None:
            grad_gathered = bf16_expert_gather_sum(grad_x, maps, w_gate.shape[0], ctx.max_slots)
        else:
            grad_gathered = bf16_expert_gather_rows(grad_x, maps.row_of)
        del grad_x
        wgrad = partial(
            _accumulate_wgrads,
            ctx.wgrad_targets,
            gathered.detach(),
            grad_out.detach(),
            h,
            grad_preact,
            is_grad_preact_swapped,
            maps,
        )
        if WeightGradStore.enabled:
            WeightGradStore.put(wgrad)
        else:
            wgrad()
        return (
            grad_gathered,
            None,
            None,
            None,
            grad_bias_gate_up,
            grad_bias_down,
            None,
            None,
            None,
            None,
            None,
        )


def bf16_swiglu_experts(
    gate_proj: nn.Module,
    up_proj: nn.Module | None,
    down_proj: nn.Module,
    gathered: torch.Tensor,
    expert_idxs: torch.Tensor,
    expand_idx: torch.Tensor | None,
    top_k: int,
) -> torch.Tensor:
    """
    down_proj(silu(gate(x)) * up(x)) per expert for the slots of the received tokens, x =
    gathered[expand_idx] grouped by expert_idxs, returned in slot order. gate and up are gate_proj
    and up_proj (GroupedLinear), or with up_proj None the two halves of a fused gate_proj whose
    rows are [gate; up].
    """
    G = gate_proj.num_groups
    maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, gathered.shape[0])
    if up_proj is not None:
        w_gate, w_up = gate_proj.weight, up_proj.weight
        targets = [(gate_proj, "weight", None), (up_proj, "weight", None)]
    else:
        intermediate_size = gate_proj.weight.shape[1] // 2
        w_gate = gate_proj.weight[:, :intermediate_size]
        w_up = gate_proj.weight[:, intermediate_size:]
        gate_rows = (slice(None), slice(None, intermediate_size))
        up_rows = (slice(None), slice(intermediate_size, None))
        targets = [(gate_proj, "weight", gate_rows), (gate_proj, "weight", up_rows)]
    wgrad_targets = _WeightGradTargets(targets + [(down_proj, "weight", None)])
    return BF16SwiGLUExpertsFunc.apply(
        gathered,
        w_gate,
        w_up,
        down_proj.weight,
        None,
        None,
        maps,
        wgrad_targets,
        min(top_k, G),
        (),
        None,
    )


def bf16_clamped_swiglu_experts(
    experts: nn.Module,
    gathered: torch.Tensor,
    expert_idxs: torch.Tensor,
    expand_idx: torch.Tensor | None,
    expert_counts: list[int] | None,
    top_k: int,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """
    gpt-oss's experts (raw gate_up_proj (G, 2I, H) with interleaved gate / up rows, down_proj and
    their biases): down(clamped_swiglu(gate_up(x) + b_gate_up)) + b_down per expert for the slots of
    the received tokens, returned in slot order.
    """
    G = experts.num_experts
    maps = bf16_expert_group_slots(expert_idxs, expand_idx, G, gathered.shape[0])
    gate_up_weight = experts.gate_up_proj
    gate_rows, up_rows = (slice(None), slice(0, None, 2)), (slice(None), slice(1, None, 2))
    wgrad_targets = _WeightGradTargets(
        [
            (experts, "gate_up_proj", gate_rows),
            (experts, "gate_up_proj", up_rows),
            (experts, "down_proj", None),
        ]
    )
    cu_seqlens_host = None
    if expert_counts is not None:
        cu_seqlens_host = [0]
        for count in expert_counts:
            cu_seqlens_host.append(cu_seqlens_host[-1] + count)
    return BF16SwiGLUExpertsFunc.apply(
        gathered,
        gate_up_weight[gate_rows],
        gate_up_weight[up_rows],
        experts.down_proj,
        experts.gate_up_proj_bias,
        experts.down_proj_bias,
        maps,
        wgrad_targets,
        min(top_k, G),
        (alpha, limit),
        cu_seqlens_host,
    )
