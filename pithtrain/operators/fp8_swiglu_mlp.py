"""
Dense FP8 SwiGLU MLP (DeepSeek-V2 shared experts and dense layers) as one custom op on Hopper.

``down_proj(silu(gate_proj(x)) * up_proj(x))`` over FP8Linear layers costs, besides the five GEMMs,
one quantization of x per projection, a SwiGLU pass that writes h in BF16 for down_proj to read
back and quantize, and in backward a SwiGLU-backward pass that writes dg and du in BF16 for gate_proj
and up_proj to read back and quantize. Here:

- x is quantized once for gate_proj and up_proj (as ``FP8Linear.quantize_input`` does);
- backward quantizes dg and du straight from dh (``pithtrain.kernels.silu_mul_bwd_quant``
  with one group of all M rows, whose per-channel K-major output is exactly the transposed half of
  ``fp8cast_rowwise_transpose``), so dg and du never reach HBM in BF16;
- the whole MLP is one autograd node: one op dispatch in place of five, and the saved state is the
  FP8 operands and g, u (what the unfused path saves too).

Every kernel and every GEMM operand is bit-identical to the FP8Linear -> silu_mul -> FP8Linear path,
so the forward output and the weight gradients equal it bit for bit; the input gradient is the same
dx_gate + dx_up sum. A custom op (not an autograd.Function) so it runs inside the compiled stage 1.
Rows must be a multiple of 128 (the fused backward's tile); other shapes take the unfused path.
"""

from typing import Tuple

import torch

from pithtrain.kernels import (
    fp8_gemm_nt,
    fp8cast_rowwise_blockwise_transpose,
    fp8cast_rowwise_transpose,
)

__all__ = ["fp8_swiglu_mlp", "fusable"]


def fusable(x: torch.Tensor) -> bool:
    """Shapes the fused op handles (Hopper FP8, rows a multiple of 128)."""
    return x.numel() > 0 and x.flatten(0, -2).shape[0] % 128 == 0


@torch.library.custom_op("pithtrain::fp8_swiglu_mlp_fwd", mutates_args=())
def _fwd(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    wg_fp8: torch.Tensor,
    wg_scale: torch.Tensor,
    wg_t: torch.Tensor,
    wg_t_scale: torch.Tensor,
    wu_fp8: torch.Tensor,
    wu_scale: torch.Tensor,
    wu_t: torch.Tensor,
    wu_t_scale: torch.Tensor,
    wd_fp8: torch.Tensor,
    wd_scale: torch.Tensor,
    wd_t: torch.Tensor,
    wd_t_scale: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Returns (y, g, u, x_t_fp8, x_t_scale, h_t_fp8, h_t_scale): the output and the tensors the
    backward needs (g and u for the SwiGLU gradient, the blockwise-transposed FP8 x and h for the
    weight gradients). The BF16 weights are taken only to give them gradients and the transposed
    FP8 weights only to save them for backward; the arguments follow FP8Linear's quantized-weight
    tuple (fp8, scale, t_fp8, t_scale) per projection.
    """
    from pithtrain.operators.silu_mul import silu_mul

    M, inter, hidden = x.shape[0], wg_fp8.shape[0], wd_fp8.shape[0]
    x_fp8, x_scale, x_t, x_t_scale = fp8cast_rowwise_blockwise_transpose(x)
    g = fp8_gemm_nt(x_fp8, x_scale, wg_fp8, wg_scale, M, inter, x.device, x.dtype)
    u = fp8_gemm_nt(x_fp8, x_scale, wu_fp8, wu_scale, M, inter, x.device, x.dtype)
    del x_fp8, x_scale
    h_fp8, h_scale, h_t, h_t_scale = fp8cast_rowwise_blockwise_transpose(silu_mul(g, u))
    y = fp8_gemm_nt(h_fp8, h_scale, wd_fp8, wd_scale, M, hidden, x.device, x.dtype)
    return y, g, u, x_t, x_t_scale, h_t, h_t_scale


@_fwd.register_fake
def _(x, w_gate, w_up, w_down, wg_fp8, *rest):
    M, K = x.shape
    inter, hidden = wg_fp8.shape[0], w_down.shape[0]
    dev, fp8 = x.device, torch.float8_e4m3fn
    cdiv = lambda a: (a + 127) // 128  # noqa: E731
    return (
        x.new_empty((M, hidden)),
        x.new_empty((M, inter)),
        x.new_empty((M, inter)),
        torch.empty((K, M), dtype=fp8, device=dev),
        torch.empty((cdiv(K), cdiv(M)), dtype=torch.float32, device=dev),
        torch.empty((inter, M), dtype=fp8, device=dev),
        torch.empty((cdiv(inter), cdiv(M)), dtype=torch.float32, device=dev),
    )


@torch.library.custom_op("pithtrain::fp8_swiglu_mlp_bwd", mutates_args=())
def _bwd(
    dy: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    x_t: torch.Tensor,
    x_t_scale: torch.Tensor,
    h_t: torch.Tensor,
    h_t_scale: torch.Tensor,
    wg_t: torch.Tensor,
    wg_t_scale: torch.Tensor,
    wu_t: torch.Tensor,
    wu_t_scale: torch.Tensor,
    wd_t: torch.Tensor,
    wd_t_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dw_gate, dw_up, dw_down), each as FP8Linear's backward computes it."""
    from pithtrain.kernels import silu_mul_bwd_quant

    (M, hidden), inter = dy.shape, g.shape[1]
    dev, dt = dy.device, dy.dtype
    dy_fp8, dy_scale, dy_t, dy_t_scale = fp8cast_rowwise_transpose(dy)
    dh = fp8_gemm_nt(dy_fp8, dy_scale, wd_t, wd_t_scale, M, inter, dev, dt)
    dw_down = fp8_gemm_nt(dy_t, dy_t_scale, h_t, h_t_scale, hidden, inter, dev, dt)
    del dy_fp8, dy_scale, dy_t, dy_t_scale
    offs = torch.full((1,), M, dtype=torch.int32, device=dev)
    dgq, duq = silu_mul_bwd_quant(dh, g, u, offs)
    del dh
    dx = fp8_gemm_nt(dgq[0], dgq[1], wg_t, wg_t_scale, M, hidden, dev, dt)
    dx += fp8_gemm_nt(duq[0], duq[1], wu_t, wu_t_scale, M, hidden, dev, dt)
    # One group of all M rows: the K-major per-channel half is the (inter, M) transpose.
    dw_gate = fp8_gemm_nt(dgq[2].view(inter, M), dgq[3], x_t, x_t_scale, inter, hidden, dev, dt)
    dw_up = fp8_gemm_nt(duq[2].view(inter, M), duq[3], x_t, x_t_scale, inter, hidden, dev, dt)
    return dx, dw_gate, dw_up, dw_down


@_bwd.register_fake
def _(dy, g, u, x_t, x_t_scale, h_t, h_t_scale, *weights):
    (M, hidden), inter = dy.shape, g.shape[1]
    return (
        dy.new_empty((M, hidden)),
        dy.new_empty((inter, hidden)),
        dy.new_empty((inter, hidden)),
        dy.new_empty((hidden, inter)),
    )


def _setup_context(ctx, inputs, output):
    q = inputs[4:]
    _, g, u, x_t, x_t_scale, h_t, h_t_scale = output
    ctx.save_for_backward(g, u, x_t, x_t_scale, h_t, h_t_scale, *q[2:4], *q[6:8], *q[10:12])


def _backward(ctx, dy, *_):
    dx, dw_gate, dw_up, dw_down = _bwd(dy, *ctx.saved_tensors)
    return (dx, dw_gate, dw_up, dw_down) + (None,) * 12


_fwd.register_autograd(_backward, setup_context=_setup_context)


def fp8_swiglu_mlp(gate_proj, up_proj, down_proj, x: torch.Tensor) -> torch.Tensor:
    """down_proj(silu(gate_proj(x)) * up_proj(x)) over FP8Linear layers (no bias), fused."""
    x2d = x.flatten(0, -2)
    weights = (gate_proj.weight, up_proj.weight, down_proj.weight)
    quantized = (*gate_proj._get_quantized_weight(), *up_proj._get_quantized_weight(), *down_proj._get_quantized_weight())  # fmt: skip
    y = _fwd(x2d, *weights, *quantized)[0]
    return y.view(*x.shape[:-1], y.shape[-1])
