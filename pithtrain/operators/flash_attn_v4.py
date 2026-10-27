"""
Flash Attention 4 (CuTeDSL).

Wraps FA4's internal _flash_attn_fwd/_flash_attn_bwd with torch.library.custom_op
so that torch.compile can trace through them. Supports both symmetric (GQA/MHA)
and asymmetric (MLA) head dimensions under BSHD layout, plus sliding-window
attention (window_size) and attention sinks (learnable_sink). Each call goes through
memoized_launch, which replays FA4's launches for a repeated signature.
"""

from typing import Optional, Tuple

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from pithtrain.kernels import memoized_launch

# fmt: off

# FA4's backward computes the sink gradient itself when given learnable_sink: the dQ postprocess
# kernel reduces dL/ds_h = -sum_rows exp(s_h - LSE) * D (LSE includes the sink, D = rowsum(dO * O))
# from the per-row LSE and D its preprocess already wrote, one CTA per head, no extra pass over
# attention. A custom_op cannot return None, so a sink-less call returns an empty placeholder.
def _grad_sink_or_empty(grad_sink: list, q: torch.Tensor) -> torch.Tensor:
    return grad_sink[0] if grad_sink else q.new_empty(0)

def _empty_grad_sink_like(learnable_sink: Optional[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(learnable_sink) if learnable_sink is not None else q.new_empty(0)

@torch.library.custom_op("pithtrain::flash_attn4_mha_fwd", mutates_args=())
def _mha_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    entry = lambda: tuple(_flash_attn_fwd(q, k, v, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink, return_lse=True)[:2])
    inputs = dict(q=q, k=k, v=v, learnable_sink=learnable_sink)
    o, lse = memoized_launch("fwd", inputs, (softmax_scale, causal, window_size_left, window_size_right), entry)
    return o, lse

@_mha_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):
    (b, s, h, _), dv = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::flash_attn4_mha_bwd", mutates_args=())
def _mha_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    entry = lambda: tuple(_flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink))
    inputs = dict(q=q, k=k, v=v, o=o, lse=lse, do=do, learnable_sink=learnable_sink)
    dq, dk, dv, *grad_sink = memoized_launch("bwd", inputs, (softmax_scale, causal, window_size_left, window_size_right), entry)
    return dq, dk, dv, _grad_sink_or_empty(grad_sink, q)

@_mha_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), _empty_grad_sink_like(learnable_sink, q)

def _mha_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q, k, v, softmax_scale, causal, window_size_left, window_size_right, learnable_sink = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, learnable_sink)
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right

def _mha_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:
    q, k, v, o, lse, learnable_sink = ctx.saved_tensors
    dq, dk, dv, grad_sink = _mha_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale, ctx.causal, ctx.window_size_left, ctx.window_size_right, learnable_sink)
    return dq, dk, dv, None, None, None, None, grad_sink if learnable_sink is not None else None

_mha_fwd.register_autograd(_mha_backward, setup_context=_mha_setup_context)

def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = False, window_size: Tuple[Optional[int], Optional[int]] = (None, None), learnable_sink: Optional[torch.Tensor] = None) -> torch.Tensor:
    o, _ = _mha_fwd(q, k, v, softmax_scale, causal, window_size[0], window_size[1], learnable_sink)
    return o

@torch.library.custom_op("pithtrain::flash_attn4_varlen_fwd", mutates_args=())
def _varlen_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    o, lse, *_ = _flash_attn_fwd(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink, return_lse=True)
    return o, lse

@_varlen_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):
    (t, h, _), dv = q.shape, v.shape[-1]
    o = torch.empty((t, h, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((h, t), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::flash_attn4_varlen_bwd", mutates_args=())
def _varlen_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, *grad_sink = _flash_attn_bwd(q, k, v, o, do, lse, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, causal=causal, window_size_left=window_size_left, window_size_right=window_size_right, learnable_sink=learnable_sink)
    return dq, dk, dv, _grad_sink_or_empty(grad_sink, q)

@_varlen_bwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float, causal: bool, window_size_left: Optional[int], window_size_right: Optional[int], learnable_sink: Optional[torch.Tensor]):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), _empty_grad_sink_like(learnable_sink, q)

def _varlen_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, output: Tuple) -> None:
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, window_size_left, window_size_right, learnable_sink = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, learnable_sink)
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right

def _varlen_backward(ctx: torch.autograd.function.FunctionCtx, grad_o: torch.Tensor, grad_lse: torch.Tensor) -> Tuple:
    q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, learnable_sink = ctx.saved_tensors
    dq, dk, dv, grad_sink = _varlen_bwd(q, k, v, o, lse, grad_o, cu_seqlens_q, cu_seqlens_k, ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.softmax_scale, ctx.causal, ctx.window_size_left, ctx.window_size_right, learnable_sink)
    return dq, dk, dv, None, None, None, None, None, None, None, None, grad_sink if learnable_sink is not None else None

_varlen_fwd.register_autograd(_varlen_backward, setup_context=_varlen_setup_context)

def flash_attn_varlen_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, softmax_scale: float, causal: bool = False, window_size: Tuple[Optional[int], Optional[int]] = (None, None), learnable_sink: Optional[torch.Tensor] = None) -> torch.Tensor:
    o, _ = _varlen_fwd(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, softmax_scale, causal, window_size[0], window_size[1], learnable_sink)
    return o
