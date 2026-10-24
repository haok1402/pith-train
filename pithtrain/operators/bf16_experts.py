"""
BF16 SwiGLU routed experts on SM90 from the received tokens, as one autograd node.

Forward, for slots grouped by expert (``kernels/bf16_expert_slots_sm90.py``: group_slots; no expanded or scattered copy of the
tokens is written):

    preact, h = gated(gathered[a_idx] @ [W_gate | W_up]^T)   one GEMM, SwiGLU in its epilogue
    out[slot_of[r]] = (h @ W_down^T)[r]                        per expert, rows stored to their
                                                              received slots by the epilogue

Backward:

    dpre, h = dgated(dout[slot_of] @ W_down, preact)    dout read in slot order, dSwiGLU epilogue
    dx      = dpre @ [W_gate; W_up]                       one GEMM over K = 2I
    dgathered[t] = sum of dx over the slots of token t    fp32, in expert order (gather_sum)

and, deferred to DualPipeV's W phase through WeightGradStore, the weight gradients accumulated
straight into the fp32 main_grads:

    main_grad_down    += dout[slot_of]^T h
    main_grad_gate|up += dpre^T gathered[a_idx]           one GEMM over N = 2I, split on store

Only ``gathered``, ``preact`` and the index maps are saved. h is the SwiGLU of the bf16-rounded
preact in both directions, so the backward recomputes the forward's h bit for bit; the grouped
expert input is never written. Every reduction runs in a fixed order, so the node is
deterministic. The math is that of the unfused path (gather, scatter, gate/up/down GroupedLinear and
silu_mul) with fp32 accumulation everywhere: the dgrad sums the gate and up contributions and the
token's rows in fp32 before rounding once.

gpt-oss (bf16_clamped_swiglu_experts): per-expert biases added in fp32 inside the two forward
epilogues, the clamped SwiGLU of the model's limit, and the bias gradients as per-expert row sums
(expert_slot_segment_sum) returned with the input gradient.

SM90 only: this module imports the SM90 kernels (and QuACK) at the top, so the models import it
lazily, after ``kernels.bf16_expert_support.supported`` said yes.
"""

from functools import partial

import torch
import torch.nn as nn

from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.kernels import bf16_expert_gemm_sm90 as gemm
from pithtrain.kernels.bf16_expert_slots_sm90 import (
    SlotMaps,
    expert_slot_segment_sum,
    gather_rows,
    gather_sum,
    group_slots,
)
from pithtrain.operators.main_grad import raw_main_grad

__all__ = ["BF16SwiGLUExpertsFunc", "bf16_swiglu_experts", "bf16_clamped_swiglu_experts"]


class _Grads:
    """
    Where the weight gradients go: (owner, parameter name, row slice) for gate, up and down. A
    weight with an fp32 main_grad (GroupedLinear.main_grad, or raw_main_grad for a raw parameter)
    accumulates there; one without gets an fp32 temporary added to its bf16 .grad.
    """

    def __init__(self, targets):
        self.targets = targets

    def run(self, fn) -> None:
        bufs, temps = {}, []
        for owner, name, _ in self.targets:
            if (owner, name) in bufs:
                continue
            mg = owner.main_grad if name == "weight" else raw_main_grad(owner, name)
            if mg is None:
                mg = torch.zeros_like(getattr(owner, name), dtype=torch.float32)
                temps.append((owner, name, mg))
            bufs[owner, name] = mg
        fn(*(bufs[o, n] if sl is None else bufs[o, n][sl] for o, n, sl in self.targets))
        for owner, name, mg in temps:
            p = getattr(owner, name)
            g = mg.to(p.dtype)
            p.grad = g if p.grad is None else p.grad.add_(g)


def _wgrads(grads: _Grads, x, dy, h, dpre, dpre_swapped, maps: SlotMaps) -> None:
    def accumulate(mg_gate, mg_up, mg_down):
        gemm.wgrad_down(dy, h, mg_down, maps.cu_seqlens, maps.slot_of)
        gemm.wgrad_gate_up(x, dpre, dpre_swapped, mg_gate, mg_up, maps.cu_seqlens, maps.a_idx)

    grads.run(accumulate)


class BF16SwiGLUExpertsFunc(torch.autograd.Function):
    """See the module docstring. Returns the expert outputs in received-slot order."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        gathered: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        b_gate_up: torch.Tensor | None,
        b_down: torch.Tensor | None,
        maps: SlotMaps,
        grads: _Grads,
        max_slots: int,
        act: tuple,
        cu_host: list[int] | None,
    ) -> torch.Tensor:
        cu = maps.cu_seqlens
        preact, h = gemm.gated_forward(gathered, w_gate, w_up, cu, maps.a_idx, b_gate_up, act)
        # rows written straight to their slots by the down GEMM's epilogue
        out = gemm.down_forward_scattered(h, w_down, cu, maps.slot_of, b_down)
        del h
        # The received tokens (and, in backward, dout) come from an all-to-all on the comm stream
        # and are read again by the deferred weight gradient on this stream: tie their reuse to
        # this stream, so the allocator cannot hand the block to the next all-to-all before the
        # wgrad has run.
        gathered.record_stream(torch.cuda.current_stream())
        ctx.save_for_backward(gathered, preact)
        ctx.weights = (w_gate, w_up, w_down)
        ctx.maps, ctx.grads, ctx.max_slots = maps, grads, max_slots
        ctx.act, ctx.cu_host = act, cu_host
        ctx.bias_dtypes = tuple(b.dtype if b is not None else None for b in (b_gate_up, b_down))
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gathered, preact = ctx.saved_tensors
        w_gate, w_up, w_down = ctx.weights
        maps = ctx.maps
        cu = maps.cu_seqlens
        dout = dout.contiguous()
        dout.record_stream(torch.cuda.current_stream())
        dpre, h, swapped = gemm.dgated_backward(dout, w_down, preact, cu, maps.slot_of, w_gate, w_up, ctx.act)  # fmt: skip
        del preact
        db_gate_up = db_down = None
        if ctx.bias_dtypes[0] is not None:
            # gpt-oss: gate is the lower address, so dpre's pairs interleave like the bias
            cu_host = ctx.cu_host if ctx.cu_host is not None else cu.tolist()
            db_gate_up = expert_slot_segment_sum(dpre, cu_host, cu).to(ctx.bias_dtypes[0])
            db_down = expert_slot_segment_sum(dout, cu_host, cu, maps.slot_of).to(
                ctx.bias_dtypes[1]
            )
        dx = gemm.dgrad(dpre, w_gate, w_up, cu, swapped)
        if maps.mask is not None:
            dgathered = gather_sum(dx, maps, w_gate.shape[0], ctx.max_slots)
        else:
            dgathered = gather_rows(dx, maps.row_of)
        del dx
        x = gathered.detach()
        wgrad = partial(_wgrads, ctx.grads, x, dout.detach(), h, dpre, swapped, maps)
        if WeightGradStore.enabled:
            WeightGradStore.put(wgrad)
        else:
            wgrad()
        return dgathered, None, None, None, db_gate_up, db_down, None, None, None, None, None


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
    ``down_proj(silu(gate(x)) * up(x))`` per expert for the slots of the received tokens,
    ``x = gathered[expand_idx]`` grouped by ``expert_idxs``, returned in slot order. gate and up
    are ``gate_proj`` and ``up_proj`` (GroupedLinear), or with ``up_proj=None`` the two halves of a
    fused ``gate_proj`` whose rows are [gate; up].
    """
    G = gate_proj.num_groups
    maps = group_slots(expert_idxs, expand_idx, G, gathered.shape[0])
    if up_proj is not None:
        w_gate, w_up = gate_proj.weight, up_proj.weight
        targets = [(gate_proj, "weight", None), (up_proj, "weight", None)]
    else:
        inter = gate_proj.weight.shape[1] // 2
        w_gate, w_up = gate_proj.weight[:, :inter], gate_proj.weight[:, inter:]
        halves = (slice(None), slice(None, inter)), (slice(None), slice(inter, None))
        targets = [(gate_proj, "weight", halves[0]), (gate_proj, "weight", halves[1])]
    grads = _Grads(targets + [(down_proj, "weight", None)])
    return BF16SwiGLUExpertsFunc.apply(
        gathered, w_gate, w_up, down_proj.weight, None, None, maps, grads, min(top_k, G), (), None
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
    gpt-oss's experts (raw ``gate_up_proj`` (G, 2I, H) with interleaved gate / up rows,
    ``down_proj`` and their biases): ``down(clamped_swiglu(gate_up(x) + b_gu)) + b_down`` per
    expert for the slots of the received tokens, returned in slot order.
    """
    G = experts.num_experts
    maps = group_slots(expert_idxs, expand_idx, G, gathered.shape[0])
    w = experts.gate_up_proj
    even, odd = (slice(None), slice(0, None, 2)), (slice(None), slice(1, None, 2))
    grads = _Grads(
        [(experts, "gate_up_proj", even), (experts, "gate_up_proj", odd),
         (experts, "down_proj", None)]
    )  # fmt: skip
    cu_host = None
    if expert_counts is not None:
        cu_host = [0]
        for c in expert_counts:
            cu_host.append(cu_host[-1] + c)
    return BF16SwiGLUExpertsFunc.apply(
        gathered, w[even], w[odd], experts.down_proj, experts.gate_up_proj_bias,
        experts.down_proj_bias, maps, grads, min(top_k, G), (alpha, limit), cu_host,
    )  # fmt: skip
