"""
Gated DeltaNet chunked delta rule as a torch.library custom op.

FLA's chunk_gated_delta_rule is @torch.compiler.disable'd (a hard Dynamo graph break).
Wrapping its low-level fwd/bwd in a custom_op + register_fake makes it an opaque,
shape-known graph node so the linear-attention region stays fullgraph-compilable.

A CP group turns on FLA's context-parallel path, which needs the contiguous layout because it
splits the chain by rank order. Each rank all-gathers a state-sized transition summary once, so
the exchange does not grow with the sequence. That path takes batch 1, and the recurrence is
independent per head, so the batch folds into the head axis.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import fla.ops.common.chunk_o
import torch
from fla.ops.cp import build_cp_context
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd
from fla.ops.utils import prepare_chunk_indices
from torch.distributed import ProcessGroup, get_world_size
from torch.distributed.distributed_c10d import _resolve_process_group

from pithtrain.kernels import (
    chunk_gdr_bwd_pinned,
    chunk_gdr_fwd_pinned,
    gated_delta_rule_pinned_supported,
)

__all__ = ["dense_gated_delta_rule", "gated_delta_rule"]

# fmt: off
# mypy: ignore-errors

# Disable FLA's issue#640 guard: the BK=64 backward miscompile doesn't apply at head_k_dim=128.
fla.ops.common.chunk_o.TRITON_ABOVE_3_4_0 = False

CHUNK_SIZE = 64  # FLA chunk size (BT) for the gated delta rule


@lru_cache(maxsize=None)
def _cp_state(group_name: Optional[str], seq_len: int) -> Tuple:
    """
    FLA context-parallel context and chunk indices for this rank, or (None, None) without one.

    Cached because build_cp_context reaches host memory, and the result depends only on the
    group and the local sequence length.
    """
    if group_name is None:
        return None, None
    group = _resolve_process_group(group_name)
    device = torch.cuda.current_device()
    # One global sequence spanning the group, which is the shape FLA merges state over.
    cu_seqlens = torch.tensor([0, seq_len * get_world_size(group)], dtype=torch.int32, device=device)  # fmt: skip
    context = build_cp_context(cu_seqlens, group)
    chunk_indices = prepare_chunk_indices(context.cu_seqlens, CHUNK_SIZE, cu_seqlens_cpu=context.cu_seqlens_cpu)  # fmt: skip
    return context, chunk_indices


@torch.library.custom_op("pithtrain::gated_delta_rule_fwd", mutates_args=())
def _gdr_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    cp_context, chunk_indices = _cp_state(cp_group_name, q.shape[1])
    g_out, o, A, _, initial_state, _ = chunk_gated_delta_rule_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=q.shape[-1] ** -0.5, initial_state=None, output_final_state=False, cu_seqlens=None if cp_context is None else cp_context.cu_seqlens, cp_context=cp_context, chunk_indices=chunk_indices, state_v_first=False, use_gate_in_kernel=False, A_log=None, dt_bias=None)
    # Under CP initial_state is the merged predecessor state; the backward recomputes h from it
    # rather than paying a second all-gather. It stays absent on the dense path.
    return o.to(q.dtype), g_out, A, [] if initial_state is None else [initial_state]

@_gdr_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group_name: Optional[str]):
    b, s, hv, vd = v.shape
    o = torch.empty((b, s, hv, vd), dtype=q.dtype, device=q.device)
    A = torch.empty((b, s, hv, CHUNK_SIZE), dtype=q.dtype, device=q.device)
    state = [] if cp_group_name is None else [torch.empty((1, hv, q.shape[-1], vd), dtype=torch.float32, device=q.device)]  # fmt: skip
    return o, torch.empty_like(g, dtype=torch.float32), A, state

@torch.library.custom_op("pithtrain::gated_delta_rule_bwd", mutates_args=())
def _gdr_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: List[torch.Tensor], cp_group_name: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, g_out, beta, A, do = (t.contiguous() for t in (q, k, v, g_out, beta, A, do))
    cp_context, chunk_indices = _cp_state(cp_group_name, q.shape[1])
    dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(q=q, k=k, v=v, g=g_out, beta=beta, A=A, scale=q.shape[-1] ** -0.5, initial_state=initial_state[0] if initial_state else None, do=do, dht=None, cu_seqlens=None if cp_context is None else cp_context.cu_seqlens, cp_context=cp_context, chunk_indices=chunk_indices, state_v_first=False, use_gate_in_kernel=False, g_input=None, A_log=None, dt_bias=None)
    return dq.to(q.dtype).contiguous(), dk.to(k.dtype).contiguous(), dv.to(v.dtype).contiguous(), dg.to(g_out.dtype).contiguous(), db.to(beta.dtype).contiguous()

@_gdr_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_out: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor, initial_state: List[torch.Tensor], cp_group_name: Optional[str]):
    cf = torch.contiguous_format # match the .contiguous() grads; empty_like would inherit input strides
    return torch.empty_like(q, memory_format=cf), torch.empty_like(k, memory_format=cf), torch.empty_like(v, memory_format=cf), torch.empty_like(g_out, memory_format=cf), torch.empty_like(beta, memory_format=cf)

def _gdr_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q, k, v, _, beta, cp_group_name = inputs
    _, g_out, A, state = output
    ctx.save_for_backward(q, k, v, g_out, beta, A, *state)
    ctx.cp_group_name = cp_group_name

def _gdr_backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor, *_unused) -> Tuple:
    q, k, v, g_out, beta, A, *state = ctx.saved_tensors
    return *_gdr_bwd(q, k, v, g_out, beta, A, do, state, ctx.cp_group_name), None

_gdr_fwd.register_autograd(_gdr_backward, setup_context=_gdr_setup_context)

def _select_key_heads(x: torch.Tensor) -> torch.Tensor:
    """
    [b, s, heads, dim], from either that or its [b, s, heads, group, dim] expand over a group.
    """
    if x.dim() == 5:
        x = x[:, :, :, 0]
    return x if x.is_contiguous() else x.contiguous()

@torch.library.custom_op("pithtrain::gated_delta_rule_dense_fwd", mutates_args=())
def _dense_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    chunk_gated_delta_rule_fwd, dense: returns the output, the chunk-cumulated gate and the solved
    WY matrix A, the two forward intermediates the backward reads. v, g and beta have HV heads;
    q and k have HV too, or H with HV a multiple of H, given as their [b, s, H, HV // H, dim]
    expand (grouped value heads share a key head).
    """
    return chunk_gdr_fwd_pinned(_select_key_heads(q), _select_key_heads(k), v, g, beta)

@_dense_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
    B, T, HV = g.shape
    A = torch.empty(B, T, HV, CHUNK_SIZE, device=k.device, dtype=k.dtype)
    return torch.empty_like(v), torch.empty_like(g, dtype=torch.float), A

@torch.library.custom_op("pithtrain::gated_delta_rule_dense_bwd", mutates_args=())
def _dense_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    chunk_gated_delta_rule_bwd, dense: (dq, dk, dv, dg, dbeta), dg w.r.t. the raw gate.

    With grouped value heads the q / k gradients are returned per value head, shaped like the
    [b, s, H, HV // H, dim] expands q and k came in as, and autograd's backward of the expand sums
    them over each group: the same reduction, at the same point, as the backward of a
    repeat_interleave of q / k ahead of an H == HV call. (FLA's own wrappers instead reduce each of
    dk's two terms over the group before adding them, a different rounding.)
    """
    dq, dk, dv, dg, dbeta = chunk_gdr_bwd_pinned(_select_key_heads(q), _select_key_heads(k), v, g_cum, beta, A, do)
    return dq.view(q.shape), dk.view(k.shape), dv, dg, dbeta

@_dense_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g_cum: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, do: torch.Tensor):
    return tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in (q, k, v, g_cum, beta))

def _dense_setup_context(ctx, inputs, output) -> None:
    q, k, v, _, beta = inputs
    _, g_cum, A = output
    ctx.save_for_backward(q, k, v, g_cum, beta, A)

def _dense_backward(ctx, do: torch.Tensor, *_unused) -> tuple:
    q, k, v, g_cum, beta, A = ctx.saved_tensors
    return _dense_bwd(q, k, v, g_cum, beta, A, do.contiguous())

_dense_fwd.register_autograd(_dense_backward, setup_context=_dense_setup_context)

def dense_gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Chunked gated delta rule: q / k [b, s, heads, dim], v [b, s, value_heads, value_dim], g and
    beta [b, s, value_heads], value_heads a multiple of heads, through the pinned FLA launches
    (chunk_gdr_fwd_pinned / chunk_gdr_bwd_pinned).
    """
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    B, S, H, K = k.shape
    group_size = v.shape[2] // H
    if group_size > 1:
        # A view, not a copy: the kernels read each key head once per group by index.
        q, k = (t.unsqueeze(3).expand(B, S, H, group_size, K) for t in (q, k))
    o, _, _ = _dense_fwd(q, k, v, g, beta)
    return o

def _fold_batch(t: torch.Tensor) -> torch.Tensor:
    """
    [b, s, heads, ...] -> [1, s, b * heads, ...]; a view at batch 1, else a copy.
    """
    return t.transpose(0, 1).reshape(1, t.shape[1], -1, *t.shape[3:])

def gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, cp_group: Optional[ProcessGroup] = None) -> torch.Tensor:
    """
    Chunked gated delta rule over [b, s, heads, dim], returning [b, s, value_heads, value_dim].
    q and k may have fewer heads than v, g and beta (a divisor): each key head then serves a
    group of consecutive value heads.

    With a cp_group the recurrence is sharded over it, s is the rank-local length, and the
    caller must already hold the contiguous layout. Without a cp_group the dense path goes
    through FLA's kernels launched with their autotuned configs pinned (dense_gated_delta_rule)
    wherever gated_delta_rule_pinned_supported says the installed versions, GPU and call geometry
    match the recorded configs; it reads grouped q / k without repeating them, and is
    bit-identical and far cheaper on the host. The other paths repeat_interleave grouped q / k
    first.
    """
    if cp_group is None and gated_delta_rule_pinned_supported(q, k, v, g, beta):
        return dense_gated_delta_rule(q, k, v, g, beta)
    repeats = v.shape[2] // q.shape[2]
    if repeats > 1:
        q = q.repeat_interleave(repeats, dim=2)
        k = k.repeat_interleave(repeats, dim=2)
    if cp_group is None:
        o, *_ = _gdr_fwd(q, k, v, g, beta, None)
        return o
    b, s = q.shape[:2]
    folded = (_fold_batch(t) for t in (q, k, v, g, beta))
    o, *_ = _gdr_fwd(*folded, cp_group.group_name)
    return o.reshape(s, b, -1, o.shape[-1]).transpose(0, 1)
