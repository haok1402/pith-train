"""
Fused FP8 q/k/v projection for Qwen3 attention.

q_proj, k_proj and v_proj read the same input. Instead of one quantization plus three GEMMs in the
forward and three dgrad + three wgrad GEMMs in the backward (each its own custom-op call), the
three FP8 weights are stacked along N and the projection runs as one GEMM per direction:

- forward: ``qkv = x @ [Wq; Wk; Wv].T`` -- one activation quantization, one GEMM (N = 5120 for
  Qwen3-30B-A3B). Every output element is the same K-reduction as in the separate GEMMs, and the
  stacked weight / 128x128 scale tensors are the separate ones concatenated (N blocks never
  straddle two projections), so the forward is bit-identical.
- backward: the q/k/v output gradients arrive as one (M, Nq + Nk + Nv) tensor. Its 1x128 and
  128x1 quantizations equal the per-projection ones block for block, so the wgrad
  ``[dWq; dWk; dWv] = dqkv.T @ x`` is bit-identical; the dgrad ``dx = dqkv @ [Wq; Wk; Wv]`` is one
  GEMM accumulating over all of N in fp32 instead of three bf16 GEMM outputs summed in bf16.

The stacked weights are concatenated per call (a few MiB of FP8) and never saved: the backward
saves the three separate transposed weights, which are the cached FP8 weights themselves.
"""

from typing import Tuple

import torch

from pithtrain.operators.deepgemm_quantize import (
    fp8cast_rowwise_blockwise_transpose,
    fp8cast_rowwise_transpose,
)
from pithtrain.operators.linear import _fp8_gemm_nt, fp8_dgrad_wgrad


@torch.library.custom_op("pithtrain::qwen3_qkv_fwd", mutates_args=())
def _qkv_fwd(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wq_fp8: torch.Tensor,
    sq: torch.Tensor,
    wq_t_fp8: torch.Tensor,
    sq_t: torch.Tensor,
    wk_fp8: torch.Tensor,
    sk: torch.Tensor,
    wk_t_fp8: torch.Tensor,
    sk_t: torch.Tensor,
    wv_fp8: torch.Tensor,
    sv: torch.Tensor,
    wv_t_fp8: torch.Tensor,
    sv_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (M, K) x -> (q (M, Nq), k (M, Nk), v (M, Nv), x_t_fp8 (K, M), scale_x_t). wq/wk/wv are taken
    only to give them gradients and the transposed FP8 weights only to save them for the
    backward; the GEMM reads the FP8 weights. q/k/v are copied out of the stacked GEMM output
    into contiguous tensors (one kernel): consumers see the same layouts as with three GEMMs, so
    the compiled q/k norms keep their persistent-reduction kernels and their exact bits.
    """
    m = x.shape[0]
    x_fp8, scale_x, x_t_fp8, scale_x_t = fp8cast_rowwise_blockwise_transpose(x)
    w = torch.cat((wq_fp8, wk_fp8, wv_fp8), dim=0)
    s = torch.cat((sq, sk, sv), dim=0)
    qkv = _fp8_gemm_nt(x_fp8, scale_x, w, s, m, w.shape[0], x.device, x.dtype)
    q, k, v = torch.split_with_sizes_copy(qkv, (wq.shape[0], wk.shape[0], wv.shape[0]), dim=1)
    return q, k, v, x_t_fp8, scale_x_t


@_qkv_fwd.register_fake
def _(x, wq, wk, wv, *quantized_weights):
    m, k = x.shape
    q = x.new_empty((m, wq.shape[0]))
    kk = x.new_empty((m, wk.shape[0]))
    v = x.new_empty((m, wv.shape[0]))
    x_t_fp8 = torch.empty((k, m), dtype=torch.float8_e4m3fn, device=x.device)
    scale_x_t = torch.empty(((k + 127) // 128, (m + 127) // 128), dtype=torch.float32, device=x.device)  # fmt: skip
    return q, kk, v, x_t_fp8, scale_x_t


@torch.library.custom_op("pithtrain::qwen3_qkv_bwd", mutates_args=())
def _qkv_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    wq_t_fp8: torch.Tensor,
    sq_t: torch.Tensor,
    wk_t_fp8: torch.Tensor,
    sk_t: torch.Tensor,
    wv_t_fp8: torch.Tensor,
    sv_t: torch.Tensor,
    x_t_fp8: torch.Tensor,
    scale_x_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k/v output grads -> (dx (M, K), dW (Nq + Nk + Nv, K) of the stacked weight)."""
    grad = torch.cat((dq, dk, dv), dim=1)
    m, n = grad.shape
    k = x_t_fp8.shape[0]
    g_fp8, scale_g, g_t_fp8, scale_g_t = fp8cast_rowwise_transpose(grad)
    del grad
    w_t = torch.cat((wq_t_fp8, wk_t_fp8, wv_t_fp8), dim=1)
    s_t = torch.cat((sq_t, sk_t, sv_t), dim=1)
    dx = _fp8_gemm_nt(g_fp8, scale_g, w_t, s_t, m, k, dq.device, dq.dtype)
    dw = _fp8_gemm_nt(g_t_fp8, scale_g_t, x_t_fp8, scale_x_t, n, k, dq.device, dq.dtype)
    return dx, dw


@_qkv_bwd.register_fake
def _(dq, dk, dv, wq_t_fp8, sq_t, wk_t_fp8, sk_t, wv_t_fp8, sv_t, x_t_fp8, scale_x_t):
    m = dq.shape[0]
    n = dq.shape[1] + dk.shape[1] + dv.shape[1]
    k = x_t_fp8.shape[0]
    return dq.new_empty((m, k)), dq.new_empty((n, k))


def _qkv_setup_context(ctx, inputs, output) -> None:
    _, wq, wk, wv, _, _, wq_t, sq_t, _, _, wk_t, sk_t, _, _, wv_t, sv_t = inputs
    *_, x_t_fp8, scale_x_t = output
    ctx.save_for_backward(wq_t, sq_t, wk_t, sk_t, wv_t, sv_t, x_t_fp8, scale_x_t)
    ctx.n = (wq.shape[0], wk.shape[0], wv.shape[0])


def _qkv_backward(ctx, dq, dk, dv, grad_x_t, grad_scale_x_t):
    m = ctx.saved_tensors[-2].shape[1]
    dq, dk, dv = (g if g is not None else torch.zeros(m, n, dtype=torch.bfloat16, device=ctx.saved_tensors[0].device) for g, n in zip((dq, dk, dv), ctx.n))  # fmt: skip
    dx, dw = _qkv_bwd(dq.contiguous(), dk.contiguous(), dv.contiguous(), *ctx.saved_tensors)
    dwq, dwk, dwv = dw.split(ctx.n, dim=0)
    return (dx, dwq, dwk, dwv) + (None,) * 12


_qkv_fwd.register_autograd(_qkv_backward, setup_context=_qkv_setup_context)


def fp8_qkv_projection(
    x: torch.Tensor,
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    v_proj: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The three FP8Linear projections of x (no bias), each (..., N_proj).
    """
    weights = []
    for proj in (q_proj, k_proj, v_proj):
        weights.extend(proj._get_quantized_weight())
    x2 = x.flatten(0, -2)
    q, k, v, _, _ = _qkv_fwd(x2, q_proj.weight, k_proj.weight, v_proj.weight, *weights)
    lead = x.shape[:-1]
    return q.view(*lead, q.shape[-1]), k.view(*lead, k.shape[-1]), v.view(*lead, v.shape[-1])


@torch.library.custom_op("pithtrain::qwen3_o_proj_fwd", mutates_args=())
def _o_proj_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fp8: torch.Tensor,
    s: torch.Tensor,
    w_t_fp8: torch.Tensor,
    s_t: torch.Tensor,
) -> torch.Tensor:
    """
    ``_fp8_linear_fwd`` (same quantization, same GEMM, same bits) that returns only the output.
    The backward re-quantizes x instead of keeping its transposed FP8 copy alive: x is the
    attention output, which the attention backward keeps anyway, so the saved FP8 copy
    (M * K bytes per micro-batch) was pure extra memory.
    """
    m, n = x.shape[0], w_fp8.shape[0]
    x_fp8, scale_x, _, _ = fp8cast_rowwise_blockwise_transpose(x)
    return _fp8_gemm_nt(x_fp8, scale_x, w_fp8, s, m, n, x.device, x.dtype)


@_o_proj_fwd.register_fake
def _(x, w, w_fp8, s, w_t_fp8, s_t):
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


@torch.library.custom_op("pithtrain::qwen3_o_proj_bwd", mutates_args=())
def _o_proj_bwd(
    grad: torch.Tensor,
    x: torch.Tensor,
    w_t_fp8: torch.Tensor,
    s_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """dgrad and wgrad of ``_o_proj_fwd``, re-quantizing x (bit-identical to the forward's)."""
    _, _, x_t_fp8, scale_x_t = fp8cast_rowwise_blockwise_transpose(x)
    return fp8_dgrad_wgrad(grad, w_t_fp8, s_t, x_t_fp8, scale_x_t, x.shape[1])


@_o_proj_bwd.register_fake
def _(grad, x, w_t_fp8, s_t):
    m, n = grad.shape
    k = x.shape[1]
    return grad.new_empty((m, k)), grad.new_empty((n, k))


def _o_proj_setup_context(ctx, inputs, output) -> None:
    x, _, _, _, w_t_fp8, s_t = inputs
    ctx.save_for_backward(x, w_t_fp8, s_t)


def _o_proj_backward(ctx, grad):
    dx, dw = _o_proj_bwd(grad, *ctx.saved_tensors)
    return dx, dw, None, None, None, None


_o_proj_fwd.register_autograd(_o_proj_backward, setup_context=_o_proj_setup_context)


def fp8_o_proj(x: torch.Tensor, o_proj: torch.nn.Module) -> torch.Tensor:
    """FP8 o_proj (no bias) that saves its bf16 input instead of an FP8 transposed copy."""
    w_fp8, s, w_t_fp8, s_t = o_proj._get_quantized_weight()
    out = _o_proj_fwd(x.flatten(0, -2), o_proj.weight, w_fp8, s, w_t_fp8, s_t)
    return out.view(*x.shape[:-1], out.shape[-1])
