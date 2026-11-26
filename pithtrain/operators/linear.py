"""
FP8Linear runs its GEMMs in FP8 E4M3 with one E8M0 scale per 128-element block.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.lowering import fallback_handler, register_lowering

from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels import (
    act_scale_like,
    fp8_channels_as_rows,
    fp8_gemm_bwd,
    fp8_gemm_fwd,
    fp8_gemm_nt,
    fp8_quant_act,
    fp8_quant_delta_gate,
    fp8_quant_delta_grad,
    fp8_quant_swiglu,
    fp8_quant_weight,
    fp8_silu_mul_bwd,
    silu_mul_bwd,
)

__all__ = ["BF16Linear", "BF16LinearFunc", "FP8Linear", "fp8_swiglu_main_grad"]


@torch.library.custom_op("pithtrain::fp8_linear_fwd", mutates_args=())
def fp8_linear_fwd(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    output = input_2d @ weight.T for an (M, K) input_2d. Returns output (M, N) in the input dtype
    and, saved for the wgrad, input_t_fp8 (K, M) with its scale_input_t. weight is taken only to
    give it a gradient; the GEMM reads weight_fp8.
    """
    return fp8_gemm_fwd(input_2d, weight_fp8, scale_weight)


@fp8_linear_fwd.register_fake
def _(input_2d, weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t):
    (M, K), N = input_2d.shape, weight.shape[0]
    output = torch.empty((M, N), dtype=input_2d.dtype, device=input_2d.device)
    input_t_fp8 = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=input_2d.device)
    size = ((K + 127) // 128, (M + 127) // 128)
    scale_input_t = torch.empty(size, dtype=torch.float32, device=input_2d.device)
    return output, input_t_fp8, scale_input_t


@torch.library.custom_op("pithtrain::fp8_linear_bwd", mutates_args=())
def fp8_linear_bwd(
    grad_output_2d: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fp8_gemm_bwd(grad_output_2d, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K)


@fp8_linear_bwd.register_fake
def _(grad_output_2d, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K):
    M, N = grad_output_2d.shape
    grad_input = torch.empty((M, K), dtype=grad_output_2d.dtype, device=grad_output_2d.device)
    weight_grad = torch.empty((N, K), dtype=grad_output_2d.dtype, device=grad_output_2d.device)
    return grad_input, weight_grad


def fp8_backward_into_main_grad(
    grad_output,
    weight_t_fp8,
    scale_weight_t,
    input_t_fp8,
    scale_input_t,
    K,
    main_grad,
    written,
    quantized_grad=None,
):
    # Keep the original wgrad route for every contribution. Only later writes need a temporary.
    accumulate = written is not None and written.item()
    dest = torch.empty_like(main_grad) if accumulate else main_grad
    grad_input, _ = fp8_gemm_bwd(
        grad_output,
        weight_t_fp8,
        scale_weight_t,
        input_t_fp8,
        scale_input_t,
        K,
        dest,
        quantized_grad,
    )
    if accumulate:
        main_grad.add_(dest)
    elif written is not None:
        written.fill_(True)
    return grad_input


@torch.library.custom_op("pithtrain::fp8_linear_bwd_into", mutates_args=("main_grad", "written"))
def fp8_linear_bwd_into(
    grad_output: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    K: int,
    main_grad: torch.Tensor,
    written: torch.Tensor | None = None,
) -> torch.Tensor:
    return fp8_backward_into_main_grad(
        grad_output, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, K, main_grad, written
    )


@fp8_linear_bwd_into.register_fake
def _(
    grad_output,
    weight_t_fp8,
    scale_weight_t,
    input_t_fp8,
    scale_input_t,
    K,
    main_grad,
    written=None,
):
    return grad_output.new_empty((grad_output.shape[0], K))


def fp8_linear_setup_context(ctx, inputs, output):
    input_2d, _, _, _, weight_t_fp8, scale_weight_t = inputs
    _, input_t_fp8, scale_input_t = output
    ctx.save_for_backward(weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t)
    ctx.K = input_2d.shape[1]


def fp8_linear_backward(ctx, grad_output, grad_input_t_fp8, grad_scale_input_t):
    weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t = ctx.saved_tensors
    grad_input, weight_grad = fp8_linear_bwd(
        grad_output, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, ctx.K
    )
    return grad_input, weight_grad, None, None, None, None


fp8_linear_fwd.register_autograd(fp8_linear_backward, setup_context=fp8_linear_setup_context)


@torch.library.custom_op("pithtrain::fp8_quantize_activation", mutates_args=())
def fp8_quantize_activation(
    input_2d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The input quantization of fp8_linear_fwd on its own, so projections that share an input
    quantize it once. Returns (input_fp8, scale_input, input_t_fp8, scale_input_t).
    """
    return fp8_quant_act(input_2d)


@fp8_quantize_activation.register_fake
def _(input_2d):
    M, K = input_2d.shape
    device = input_2d.device
    input_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    scale_input = act_scale_like(M, (K + 127) // 128, device)
    input_t_fp8 = torch.empty((K, M), dtype=torch.float8_e4m3fn, device=device)
    size = ((K + 127) // 128, (M + 127) // 128)
    scale_input_t = torch.empty(size, dtype=torch.float32, device=device)
    return input_fp8, scale_input, input_t_fp8, scale_input_t


@torch.library.custom_op("pithtrain::fp8_quantize_delta_gate", mutates_args=())
def fp8_quantize_delta_gate(
    value: torch.Tensor, gate: torch.Tensor, norm_weight: torch.Tensor, inv_rms: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return fp8_quant_delta_gate(value, norm_weight, inv_rms, gate)


@fp8_quantize_delta_gate.register_fake
def _(value, gate, norm_weight, inv_rms):
    return fp8_quantize_activation(value)


@torch.library.custom_op("pithtrain::fp8_linear_fwd_prequantized", mutates_args=())
def fp8_linear_fwd_prequantized(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_weight: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_fp8: torch.Tensor,
    scale_input: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
) -> torch.Tensor:
    """
    The output of fp8_linear_fwd for an input already quantized by fp8_quantize_activation.
    input_2d and weight are taken only to give them gradients; the GEMM reads the FP8 operands.
    """
    M, N = input_fp8.shape[0], weight_fp8.shape[0]
    return fp8_gemm_nt(
        input_fp8, scale_input, weight_fp8, scale_weight, M, N, input_2d.device, input_2d.dtype
    )


@fp8_linear_fwd_prequantized.register_fake
def _(input_2d, weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t, *quantized_input):
    M, N = input_2d.shape[0], weight.shape[0]
    return torch.empty((M, N), dtype=input_2d.dtype, device=input_2d.device)


def fp8_linear_prequantized_setup_context(ctx, inputs, output):
    input_2d, _, _, _, weight_t_fp8, scale_weight_t, _, _, input_t_fp8, scale_input_t = inputs
    ctx.save_for_backward(weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t)
    ctx.K = input_2d.shape[1]


def fp8_linear_prequantized_backward(ctx, grad_output):
    weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t = ctx.saved_tensors
    grad_input, weight_grad = fp8_linear_bwd(
        grad_output, weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t, ctx.K
    )
    return grad_input, weight_grad, None, None, None, None, None, None, None, None


fp8_linear_fwd_prequantized.register_autograd(
    fp8_linear_prequantized_backward, setup_context=fp8_linear_prequantized_setup_context
)


class FP8LinearMainGradFunc(torch.autograd.Function):
    """Write the weight gradient into the FSDP-owned FP32 accumulation buffer."""

    @staticmethod
    def forward(
        ctx,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_fp8: torch.Tensor,
        scale_weight: torch.Tensor,
        weight_t_fp8: torch.Tensor,
        scale_weight_t: torch.Tensor,
        main_grad: torch.Tensor,
        quantized_input: Optional[Tuple[torch.Tensor, ...]],
        written: torch.Tensor | None,
    ) -> torch.Tensor:
        if quantized_input is None:
            output, input_t_fp8, scale_input_t = fp8_linear_fwd(
                input_2d, weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t
            )
        else:
            output = fp8_linear_fwd_prequantized(
                input_2d,
                weight,
                weight_fp8,
                scale_weight,
                weight_t_fp8,
                scale_weight_t,
                *quantized_input,
            )
            _, _, input_t_fp8, scale_input_t = quantized_input
        ctx.save_for_backward(weight_t_fp8, scale_weight_t, input_t_fp8, scale_input_t)
        ctx.main_grad = main_grad
        ctx.written = written
        ctx.K = input_2d.shape[1]
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = fp8_linear_bwd_into(
            grad_output, *ctx.saved_tensors, ctx.K, ctx.main_grad, ctx.written
        )
        return grad_input, None, None, None, None, None, None, None, None


class FP8SwiGLUDownProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, weight, quantized_weight, main_grad, written):
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        activation_fp8, scale, activation_t_fp8, scale_t = fp8_quant_swiglu(gate, up)
        output = fp8_linear_fwd_prequantized(
            gate,
            weight,
            weight_fp8,
            scale_weight,
            weight_t_fp8,
            scale_weight_t,
            activation_fp8,
            scale,
            activation_t_fp8,
            scale_t,
        )
        ctx.save_for_backward(gate, up, weight_t_fp8, scale_weight_t, activation_t_fp8, scale_t)
        ctx.main_grad = main_grad
        ctx.written = written
        return output

    @staticmethod
    def backward(ctx, grad_output):
        gate, up, *operands = ctx.saved_tensors
        if ctx.main_grad is None:
            grad_activation, grad_weight = fp8_linear_bwd(grad_output, *operands, gate.shape[1])
        else:
            grad_activation = fp8_linear_bwd_into(
                grad_output, *operands, gate.shape[1], ctx.main_grad, ctx.written
            )
            grad_weight = None
        grad_gate, grad_up = silu_mul_bwd(grad_activation, gate, up)
        return grad_gate, grad_up, grad_weight, None, None, None


@torch.library.custom_op(
    "pithtrain::fp8_delta_gate_bwd_into", mutates_args=("main_grad", "written")
)
def fp8_delta_gate_bwd_into(
    grad: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    inv_rms: torch.Tensor,
    weight_t_fp8: torch.Tensor,
    scale_weight_t: torch.Tensor,
    input_t_fp8: torch.Tensor,
    scale_input_t: torch.Tensor,
    K: int,
    main_grad: torch.Tensor,
    written: torch.Tensor | None = None,
) -> torch.Tensor:
    quantized_grad = fp8_quant_delta_grad(grad, value, gate, weight, inv_rms)
    return fp8_backward_into_main_grad(
        grad,
        weight_t_fp8,
        scale_weight_t,
        input_t_fp8,
        scale_input_t,
        K,
        main_grad,
        written,
        quantized_grad,
    )


@fp8_delta_gate_bwd_into.register_fake
def _(
    grad,
    value,
    gate,
    weight,
    inv_rms,
    weight_t_fp8,
    scale_weight_t,
    input_t_fp8,
    scale_input_t,
    K,
    main_grad,
    written=None,
):
    return grad.new_empty((grad.shape[0], K))


class FP8DeltaGateProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value,
        hidden,
        input_operands,
        norm_weight,
        eps,
        weight,
        quantized_weight,
        main_grad,
        written,
        z_weight,
        z_quantized_weight,
        z_main_grad,
        z_written,
    ):
        gate = fp8_linear_fwd_prequantized(hidden, z_weight, *z_quantized_weight, *input_operands)
        width = norm_weight.numel()
        inv_rms = torch.rsqrt(value.view(-1, width).float().pow(2).mean(-1, keepdim=True) + eps)
        act_fp8, scale, act_t_fp8, scale_t = fp8_quantize_delta_gate(
            value, gate, norm_weight.float(), inv_rms.flatten()
        )
        output = fp8_linear_fwd_prequantized(
            value, weight, *quantized_weight, act_fp8, scale, act_t_fp8, scale_t
        )
        _, _, weight_t_fp8, scale_weight_t = quantized_weight
        ctx.save_for_backward(
            value,
            gate,
            norm_weight,
            inv_rms,
            weight_t_fp8,
            scale_weight_t,
            act_t_fp8,
            scale_t,
            *z_quantized_weight[2:],
            *input_operands[2:],
        )
        ctx.main_grad = main_grad
        ctx.written = written
        ctx.z_main_grad = z_main_grad
        ctx.z_written = z_written
        ctx.hidden_width = hidden.shape[1]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, gate, norm_weight, inv_rms, *operands = ctx.saved_tensors
        out_operands, z_operands = operands[:4], operands[4:]
        grad_activation = fp8_linear_bwd_into(
            grad_output, *out_operands, value.shape[1], ctx.main_grad, ctx.written
        )
        width = norm_weight.numel()
        x, z, grad = (tensor.view(-1, width).float() for tensor in (value, gate, grad_activation))
        weight = norm_weight.float()
        grad_gated = grad * F.silu(z)
        grad_affine = grad_gated * weight
        grad_inv_rms = (grad_affine * x).sum(-1, keepdim=True)
        grad_mean = (grad_inv_rms * -0.5) * inv_rms.pow(3)
        grad_value = (grad_affine * inv_rms) + (grad_mean / width) * (x * 2.0)
        grad_norm_weight = (grad_gated * (x * inv_rms)).sum(0)
        grad_hidden = fp8_delta_gate_bwd_into(
            grad_activation,
            value,
            gate,
            weight,
            inv_rms.flatten(),
            *z_operands,
            ctx.hidden_width,
            ctx.z_main_grad,
            ctx.z_written,
        )
        return (
            grad_value.to(value.dtype).view_as(value),
            grad_hidden,
            None,
            grad_norm_weight.to(norm_weight.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FP8Linear(nn.Linear):
    """
    Outside torch.compile, the bf16 weight is quantized once per step and reused by every
    micro-batch of it; with fp8_param_gather set, the weight arrives from the
    all-gather already quantized.
    """

    fp8_param_gather: bool = False
    main_grad: Optional[torch.Tensor] = None
    main_grad_written: Optional[torch.Tensor] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized_weight_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # fmt: skip
        self.quantized_weight_version: int = -1

    def forward_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Project silu(gate) * up with fused activation quantization in eager execution."""
        gate_2d, up_2d = gate.flatten(0, -2), up.flatten(0, -2)
        output = FP8SwiGLUDownProjection.apply(
            gate_2d,
            up_2d,
            self.weight,
            self.get_quantized_weight(),
            self.main_grad,
            self.main_grad_written,
        ).view(*gate.shape[:-1], self.out_features)
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward_delta_gate(
        self,
        value: torch.Tensor,
        hidden: torch.Tensor,
        input_operands: tuple[torch.Tensor, ...],
        norm_weight: torch.Tensor,
        eps: float,
        z_proj,
    ) -> torch.Tensor:
        output = FP8DeltaGateProjection.apply(
            value.flatten(0, -2),
            hidden.flatten(0, -2),
            input_operands,
            norm_weight,
            eps,
            self.weight,
            self.get_quantized_weight(),
            self.main_grad,
            self.main_grad_written,
            z_proj.weight,
            z_proj.get_quantized_weight(),
            z_proj.main_grad,
            z_proj.main_grad_written,
        ).view(*value.shape[:-1], self.out_features)
        return output if self.bias is None else output + self.bias

    def get_quantized_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fp8_param_gather:
            return self.weight_fp8, self.weight_scale, self.weight_t_fp8, self.weight_scale_t
        if torch.compiler.is_compiling():
            return fp8_quant_weight(self.weight)
        ver = FP8WeightCacheControl.version
        if self.quantized_weight_version == ver:
            return self.quantized_weight_cache
        result = fp8_quant_weight(self.weight)
        self.quantized_weight_cache = result
        self.quantized_weight_version = ver
        return result

    @staticmethod
    def quantize_input(x: torch.Tensor) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Quantize an activation once for several FP8Linear layers that read it, to pass to each as
        quantized_input. The bytes are the same as each layer quantizing it itself.
        """
        if x.numel() == 0:
            return None
        # Not differentiable: each layer still takes x itself, which carries the gradient.
        return fp8_quantize_activation(x.detach().flatten(0, -2))

    def forward(
        self, input: torch.Tensor, quantized_input: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        if input.numel() == 0:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        quantized_weight = self.get_quantized_weight()
        weight_fp8, scale_weight, weight_t_fp8, scale_weight_t = quantized_weight
        input_2d = input.flatten(0, -2)
        if self.main_grad is not None and torch.is_grad_enabled():
            output_2d = FP8LinearMainGradFunc.apply(
                input_2d,
                self.weight,
                *quantized_weight,
                self.main_grad,
                quantized_input,
                self.main_grad_written,
            )
        elif quantized_input is not None:
            output_2d = fp8_linear_fwd_prequantized(
                input_2d,
                self.weight,
                weight_fp8,
                scale_weight,
                weight_t_fp8,
                scale_weight_t,
                *quantized_input,
            )
        else:
            output_2d, _, _ = fp8_linear_fwd(
                input_2d, self.weight, weight_fp8, scale_weight, weight_t_fp8, scale_weight_t
            )
        output = output_2d.view(*input.shape[:-1], self.weight.shape[0])
        if self.bias is not None:
            output = output + self.bias
        return output


class FP8SwiGLUMainGrad(torch.autograd.Function):
    """Quantize both rounded SwiGLU adjoints before the gate and up projection backpropagation."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight, qweights, main_grads, written, replay):
        gate_quantized, up_quantized, down_quantized = qweights
        quantized_x = fp8_quantize_activation(x)
        gate = fp8_linear_fwd_prequantized(x, gate_weight, *gate_quantized, *quantized_x)
        up = fp8_linear_fwd_prequantized(x, up_weight, *up_quantized, *quantized_x)
        quantized_hidden = fp8_quant_swiglu(gate, up)
        output = fp8_linear_fwd_prequantized(gate, down_weight, *down_quantized, *quantized_hidden)
        ctx.replay = replay
        if replay:
            ctx.save_for_backward(*quantized_x, *gate_quantized, *up_quantized, *down_quantized[2:])
        else:
            ctx.save_for_backward(
                gate,
                up,
                *quantized_x[2:],
                *gate_quantized[2:],
                *up_quantized[2:],
                *down_quantized[2:],
                *quantized_hidden[2:],
            )
        ctx.main_grads = main_grads
        ctx.written = written
        ctx.input_features = x.shape[1]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.replay:
            (
                input_fp8,
                input_scale,
                input_t,
                input_scale_t,
                gate_fp8,
                gate_scale,
                weight_gate_t,
                scale_gate_t,
                up_fp8,
                up_scale,
                weight_up_t,
                scale_up_t,
                weight_down_t,
                scale_down_t,
            ) = ctx.saved_tensors
            # The down projection's output is unused by backward. Replay only its inputs.
            M = input_fp8.shape[0]
            gate = fp8_gemm_nt(
                input_fp8,
                input_scale,
                gate_fp8,
                gate_scale,
                M,
                gate_fp8.shape[0],
                input_fp8.device,
                grad_output.dtype,
            )
            up = fp8_gemm_nt(
                input_fp8,
                input_scale,
                up_fp8,
                up_scale,
                M,
                up_fp8.shape[0],
                input_fp8.device,
                grad_output.dtype,
            )
            _, _, hidden_t, hidden_scale_t = fp8_quant_swiglu(gate, up)
        else:
            (
                gate,
                up,
                input_t,
                input_scale_t,
                weight_gate_t,
                scale_gate_t,
                weight_up_t,
                scale_up_t,
                weight_down_t,
                scale_down_t,
                hidden_t,
                hidden_scale_t,
            ) = ctx.saved_tensors
        gate_main_grad, up_main_grad, down_main_grad = ctx.main_grads
        gate_written, up_written, down_written = ctx.written
        M, N = gate.shape
        grad_hidden = fp8_linear_bwd_into(
            grad_output,
            weight_down_t,
            scale_down_t,
            hidden_t,
            hidden_scale_t,
            N,
            down_main_grad,
            down_written,
        )
        offsets = torch.full((1,), M, dtype=torch.int32, device=gate.device)
        grad_gate, grad_up = fp8_silu_mul_bwd(grad_hidden, gate, up, offsets)

        def project(grad, activation, weight_t, scale_t, main_grad, written):
            grad_fp8, grad_scale, grad_ch_fp8, grad_ch_scale = grad
            grad_t = fp8_channels_as_rows(grad_ch_fp8, grad_ch_scale, M, N)
            return fp8_backward_into_main_grad(
                activation,
                weight_t,
                scale_t,
                input_t,
                input_scale_t,
                ctx.input_features,
                main_grad,
                written,
                (grad_fp8, grad_scale, *grad_t),
            )

        grad_x_up = project(grad_up, up, weight_up_t, scale_up_t, up_main_grad, up_written)
        grad_x_gate = project(
            grad_gate, gate, weight_gate_t, scale_gate_t, gate_main_grad, gate_written
        )
        return grad_x_up + grad_x_gate, None, None, None, None, None, None, None


def fp8_swiglu_main_grad(x, gate_proj, up_proj, down_proj, *, replay=False):
    """Run an eager, bias-free FP8 SwiGLU MLP with an FP32 main_grad for each projection."""
    projections = gate_proj, up_proj, down_proj
    quantized_weights = tuple(layer.get_quantized_weight() for layer in projections)
    main_grads = tuple(layer.main_grad for layer in projections)
    written = tuple(layer.main_grad_written for layer in projections)
    output = FP8SwiGLUMainGrad.apply(
        x.flatten(0, -2),
        gate_proj.weight,
        up_proj.weight,
        down_proj.weight,
        quantized_weights,
        main_grads,
        written,
        replay,
    )
    return output.view(*x.shape[:-1], down_proj.out_features)


# Inductor has no single-GEMM lowering for addmm with out_dtype, so the fused GEMM is its own op.
wgrad_accumulate_library = torch.library.Library("pithtrain", "FRAGMENT")
wgrad_accumulate_library.define(
    "wgrad_accumulate_(Tensor(a!) main_grad, Tensor a, Tensor b, bool overwrite=False) -> ()"
)


def wgrad_accumulate_cuda(
    main_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor, overwrite: bool = False
) -> None:
    torch.addmm(main_grad, a, b, beta=0 if overwrite else 1, out_dtype=torch.float32, out=main_grad)


def wgrad_accumulate_meta(
    main_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor, overwrite: bool = False
) -> None:
    return None


wgrad_accumulate_library.impl("wgrad_accumulate_", wgrad_accumulate_cuda, "CUDA")
wgrad_accumulate_library.impl("wgrad_accumulate_", wgrad_accumulate_meta, "Meta")
addmm_dtype_out_fallback = fallback_handler(torch.ops.aten.addmm.dtype_out, add_to_fallback_set=False)  # fmt: skip


@register_lowering(torch.ops.pithtrain.wgrad_accumulate_.default, type_promotion_kind=None)
def lower_wgrad_accumulate(main_grad: Any, a: Any, b: Any, overwrite: bool = False) -> None:
    addmm_dtype_out_fallback(
        main_grad, a, b, torch.float32, beta=0 if overwrite else 1, alpha=1, out=main_grad
    )


@torch.library.custom_op("pithtrain::bf16_wgrad_into", mutates_args=("main_grad", "written"))
def bf16_wgrad_into(
    main_grad: torch.Tensor, a: torch.Tensor, b: torch.Tensor, written: torch.Tensor
) -> None:
    accumulate = written.item()
    wgrad_accumulate_cuda(main_grad, a, b, overwrite=not accumulate)
    if not accumulate:
        written.fill_(True)


@bf16_wgrad_into.register_fake
def _(main_grad, a, b, written):
    return None


class BF16LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        main_grad: torch.Tensor,
        overwrite: bool | torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        # Kept off the saved tensors: other micro-batches update main_grad in place meanwhile.
        ctx.main_grad = main_grad
        ctx.overwrite = overwrite
        return F.linear(input, weight)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_input = (grad_output_2d @ weight).view(*input.shape)
        if isinstance(ctx.overwrite, torch.Tensor):
            bf16_wgrad_into(main_grad, grad_output_2d.t(), input_2d, ctx.overwrite)
        else:
            torch.ops.pithtrain.wgrad_accumulate_(
                main_grad, grad_output_2d.t(), input_2d, ctx.overwrite
            )
        return grad_input, None, None, None


class BF16Linear(nn.Linear):
    """
    With main_grad set, the weight gradient accumulates into it in fp32 instead of a .grad.
    """

    main_grad: Optional[torch.Tensor] = None
    main_grad_written: Optional[torch.Tensor] = None
    overwrite_main_grad: bool = False

    @staticmethod
    def quantize_input(x: torch.Tensor) -> None:
        return None

    def forward(self, input: torch.Tensor, quantized_input: None = None) -> torch.Tensor:
        if self.main_grad is None or not torch.is_grad_enabled():
            return F.linear(input, self.weight, self.bias)
        overwrite = (
            self.overwrite_main_grad if self.main_grad_written is None else self.main_grad_written
        )
        output = BF16LinearFunc.apply(input, self.weight, self.main_grad, overwrite)
        return output if self.bias is None else output + self.bias
