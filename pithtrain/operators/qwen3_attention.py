"""
FP8 q/k/v and o projections for Qwen3 attention.

The q/k/v projection stacks the three FP8 weights along N and runs one GEMM per direction:
qkv = x @ [Wq; Wk; Wv].T in the forward, and in the backward one dgrad and one wgrad on the
stacked (M, Nq + Nk + Nv) gradient. The forward and the wgrad are bit-identical to three separate
FP8Linear layers; the dgrad sums over all of N in fp32 instead of adding three bf16 outputs.

The o projection matches FP8Linear but saves its bf16 input instead of an FP8 transposed copy.
"""

from typing import Tuple

import torch

from pithtrain.kernels import (
    fp8_dgrad_wgrad,
    fp8_gemm_nt,
    fp8cast_rowwise_blockwise_transpose,
    fp8cast_rowwise_transpose,
)

__all__ = ["fp8_o_proj", "fp8_qkv_projection"]


@torch.library.custom_op("pithtrain::qwen3_qkv_fwd", mutates_args=())
def qwen3_qkv_fwd(
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
    (M, K) x -> (q (M, Nq), k (M, Nk), v (M, Nv), x_t_fp8 (K, M), scale_x_t). wq, wk and wv are
    taken only to give them gradients and the transposed FP8 weights only to save them for the
    backward; the GEMM reads the FP8 weights. q, k and v are contiguous, as three GEMMs would give
    them, so the compiled q/k norms that read them keep the same kernels and bits.
    """
    M = x.shape[0]
    x_fp8, scale_x, x_t_fp8, scale_x_t = fp8cast_rowwise_blockwise_transpose(x)
    w_qkv_fp8 = torch.cat((wq_fp8, wk_fp8, wv_fp8), dim=0)
    scale_w_qkv = torch.cat((sq, sk, sv), dim=0)
    qkv = fp8_gemm_nt(
        x_fp8, scale_x, w_qkv_fp8, scale_w_qkv, M, w_qkv_fp8.shape[0], x.device, x.dtype
    )
    q, k, v = torch.split_with_sizes_copy(qkv, (wq.shape[0], wk.shape[0], wv.shape[0]), dim=1)
    return q, k, v, x_t_fp8, scale_x_t


@qwen3_qkv_fwd.register_fake
def _(x, wq, wk, wv, *quantized_weights):
    M, K = x.shape
    q = x.new_empty((M, wq.shape[0]))
    k = x.new_empty((M, wk.shape[0]))
    v = x.new_empty((M, wv.shape[0]))
    x_t_fp8 = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x.device)
    scale_x_t = torch.empty(
        ((K + 127) // 128, (M + 127) // 128), dtype=torch.float32, device=x.device
    )
    return q, k, v, x_t_fp8, scale_x_t


@torch.library.custom_op("pithtrain::qwen3_qkv_bwd", mutates_args=())
def qwen3_qkv_bwd(
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
    """
    q/k/v output grads -> (grad_x (M, K), grad_w (Nq + Nk + Nv, K) of the stacked weight).
    """
    grad_qkv = torch.cat((dq, dk, dv), dim=1)
    M, N = grad_qkv.shape
    K = x_t_fp8.shape[0]
    grad_fp8, scale_grad, grad_t_fp8, scale_grad_t = fp8cast_rowwise_transpose(grad_qkv)
    del grad_qkv
    w_qkv_t_fp8 = torch.cat((wq_t_fp8, wk_t_fp8, wv_t_fp8), dim=1)
    scale_w_qkv_t = torch.cat((sq_t, sk_t, sv_t), dim=1)
    grad_x = fp8_gemm_nt(grad_fp8, scale_grad, w_qkv_t_fp8, scale_w_qkv_t, M, K, dq.device, dq.dtype)  # fmt: skip
    grad_w = fp8_gemm_nt(grad_t_fp8, scale_grad_t, x_t_fp8, scale_x_t, N, K, dq.device, dq.dtype)
    return grad_x, grad_w


@qwen3_qkv_bwd.register_fake
def _(dq, dk, dv, wq_t_fp8, sq_t, wk_t_fp8, sk_t, wv_t_fp8, sv_t, x_t_fp8, scale_x_t):
    M = dq.shape[0]
    N = dq.shape[1] + dk.shape[1] + dv.shape[1]
    K = x_t_fp8.shape[0]
    return dq.new_empty((M, K)), dq.new_empty((N, K))


def qwen3_qkv_setup_context(ctx, inputs, output) -> None:
    _, wq, wk, wv, _, _, wq_t, sq_t, _, _, wk_t, sk_t, _, _, wv_t, sv_t = inputs
    *_, x_t_fp8, scale_x_t = output
    ctx.save_for_backward(wq_t, sq_t, wk_t, sk_t, wv_t, sv_t, x_t_fp8, scale_x_t)
    ctx.proj_sizes = (wq.shape[0], wk.shape[0], wv.shape[0])


def qwen3_qkv_backward(ctx, grad_q, grad_k, grad_v, grad_x_t, grad_scale_x_t):
    M = ctx.saved_tensors[-2].shape[1]
    device = ctx.saved_tensors[0].device
    grad_q, grad_k, grad_v = (
        grad if grad is not None else torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        for grad, N in zip((grad_q, grad_k, grad_v), ctx.proj_sizes)
    )
    grad_x, grad_w = qwen3_qkv_bwd(
        grad_q.contiguous(), grad_k.contiguous(), grad_v.contiguous(), *ctx.saved_tensors
    )
    grad_wq, grad_wk, grad_wv = grad_w.split(ctx.proj_sizes, dim=0)
    return (grad_x, grad_wq, grad_wk, grad_wv) + (None,) * 12


qwen3_qkv_fwd.register_autograd(qwen3_qkv_backward, setup_context=qwen3_qkv_setup_context)


def fp8_qkv_projection(
    x: torch.Tensor,
    q_proj: torch.nn.Module,
    k_proj: torch.nn.Module,
    v_proj: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The three FP8Linear projections of x (no bias), each (..., N_proj).
    """
    quantized_weights = []
    for proj in (q_proj, k_proj, v_proj):
        quantized_weights.extend(proj.get_quantized_weight())
    x_2d = x.flatten(0, -2)
    q, k, v, _, _ = qwen3_qkv_fwd(x_2d, q_proj.weight, k_proj.weight, v_proj.weight, *quantized_weights)  # fmt: skip
    lead_shape = x.shape[:-1]
    return (
        q.view(*lead_shape, q.shape[-1]),
        k.view(*lead_shape, k.shape[-1]),
        v.view(*lead_shape, v.shape[-1]),
    )


@torch.library.custom_op("pithtrain::qwen3_o_proj_fwd", mutates_args=())
def qwen3_o_proj_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fp8: torch.Tensor,
    s: torch.Tensor,
    w_t_fp8: torch.Tensor,
    s_t: torch.Tensor,
) -> torch.Tensor:
    """
    The output of fp8_linear_fwd, bit for bit, without the transposed FP8 copy of x. x is the
    attention output, which the attention backward keeps anyway, so the backward quantizes it
    again instead of saving M * K more bytes.
    """
    M, N = x.shape[0], w_fp8.shape[0]
    x_fp8, scale_x, _, _ = fp8cast_rowwise_blockwise_transpose(x)
    return fp8_gemm_nt(x_fp8, scale_x, w_fp8, s, M, N, x.device, x.dtype)


@qwen3_o_proj_fwd.register_fake
def _(x, w, w_fp8, s, w_t_fp8, s_t):
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


@torch.library.custom_op("pithtrain::qwen3_o_proj_bwd", mutates_args=())
def qwen3_o_proj_bwd(
    grad: torch.Tensor,
    x: torch.Tensor,
    w_t_fp8: torch.Tensor,
    s_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dgrad and wgrad of qwen3_o_proj_fwd, re-quantizing x (bit-identical to the forward's).
    """
    _, _, x_t_fp8, scale_x_t = fp8cast_rowwise_blockwise_transpose(x)
    return fp8_dgrad_wgrad(grad, w_t_fp8, s_t, x_t_fp8, scale_x_t, x.shape[1])


@qwen3_o_proj_bwd.register_fake
def _(grad, x, w_t_fp8, s_t):
    M, N = grad.shape
    K = x.shape[1]
    return grad.new_empty((M, K)), grad.new_empty((N, K))


def qwen3_o_proj_setup_context(ctx, inputs, output) -> None:
    x, _, _, _, w_t_fp8, s_t = inputs
    ctx.save_for_backward(x, w_t_fp8, s_t)


def qwen3_o_proj_backward(ctx, grad):
    grad_x, grad_w = qwen3_o_proj_bwd(grad, *ctx.saved_tensors)
    return grad_x, grad_w, None, None, None, None


qwen3_o_proj_fwd.register_autograd(qwen3_o_proj_backward, setup_context=qwen3_o_proj_setup_context)


def fp8_o_proj(x: torch.Tensor, o_proj: torch.nn.Module) -> torch.Tensor:
    """
    FP8 o_proj (no bias) that saves its bf16 input instead of an FP8 transposed copy.
    """
    w_fp8, scale, w_t_fp8, scale_t = o_proj.get_quantized_weight()
    out = qwen3_o_proj_fwd(x.flatten(0, -2), o_proj.weight, w_fp8, scale, w_t_fp8, scale_t)
    return out.view(*x.shape[:-1], out.shape[-1])
