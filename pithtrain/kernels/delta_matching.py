"""
Select the matched causal FP8 attention implementation for Hopper.
"""

import math

import torch

from pithtrain.kernels.arch import ARCH_MAJOR

__all__ = ["delta_matching_supported", "delta_matching_fwd", "delta_matching_bwd_prepare", "delta_matching_bwd"]


def delta_matching_supported(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """
    Whether SM90 Delta-Matching supports these CUDA BF16 BSHD inputs on the same device.
    Self-attention requires D in {128, 256}, equal sequence lengths divisible by 32768 / D,
    and a positive number of query heads divisible by the number of key/value heads.
    Batch times query heads must be at most 65535.
    """
    if ARCH_MAJOR != 9 or any(tensor.ndim != 4 or tensor.dtype != torch.bfloat16 or tensor.device != query.device for tensor in (query, key, value)) or not query.is_cuda:
        return False
    B, N, num_heads, D = query.shape
    num_key_heads = key.shape[2]
    return D in (128, 256) and B > 0 and N > 0 and N % (32768 // D) == 0 and 0 < B * num_heads <= 65535 and num_key_heads > 0 and num_heads % num_key_heads == 0 and key.shape == value.shape == (B, N, num_key_heads, D)


def delta_matching_fwd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, softmax_scale: float) -> tuple[torch.Tensor, ...]:
    """
    Causal grouped-query attention on SM90 with BF16 BSHD inputs and a positive softmax scale.
    Returns a fresh BF16 BSHD output followed by the backward state: FP32 logsumexp of shape
    (batch times query heads, sequence), the E4M3 query and key codes, their FP32 scales and
    transposed codes, and the E4M3 value codes and FP32 scales. Codes are (batch times heads,
    sequence, channels), transposed codes swap the last two axes, and scales are (batch times
    heads, sequence blocks). Blocks cover 16384 / D query rows and 8192 / D key or value rows.
    The query scale includes the softmax scale. Keys are centered per sequence and KV head before
    quantization. Returns newly allocated tensors. See delta_matching_supported for geometry.
    """
    match ARCH_MAJOR:
        case 9:
            if not delta_matching_supported(query, key, value):
                raise ValueError("Delta-Matching requires BF16 CUDA BSHD self-attention with divisible GQA heads, D in {128, 256}, and positive sequence length divisible by 32768 / D")
            if not math.isfinite(softmax_scale) or softmax_scale <= 0:
                raise ValueError("Delta-Matching requires a positive, finite softmax scale")
            from pithtrain.kernels.delta_matching_fwd_sm90 import delta_matching_fwd_sm90
            return delta_matching_fwd_sm90(query, key, value, softmax_scale)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x" if ARCH_MAJOR != 10 else "Delta-Matching requires SM90")


def delta_matching_bwd_prepare(grad_out: torch.Tensor, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """
    Quantize a BF16 BSHD output gradient and form the matched FP32 correction from the backward
    pre-cast probabilities and FP8 dP product. The state is delta_matching_fwd's outputs after
    the BF16 attention output. Returns E4M3 gradient codes (batch times heads, sequence,
    channels), FP32 scales (batch times heads, sequence blocks of 16384 / D rows), transposed
    codes and the FP32 correction (batch times heads, sequence). Accepts strided or expanded
    gradients and returns newly allocated tensors. SM90 only.
    """
    match ARCH_MAJOR:
        case 9:
            if grad_out.ndim != 4 or grad_out.dtype != torch.bfloat16 or grad_out.device != state[0].device:
                raise ValueError("Delta-Matching backward requires a BF16 BSHD output gradient on the forward device")
            B, N, H, D = grad_out.shape
            if state[1].shape != (B * H, N, D):
                raise ValueError("Delta-Matching output gradient shape must match the forward output")
            from pithtrain.kernels.delta_matching_bwd_sm90 import delta_matching_bwd_prepare_sm90
            return delta_matching_bwd_prepare_sm90(grad_out, state)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x" if ARCH_MAJOR != 10 else "Delta-Matching requires SM90")


def delta_matching_bwd(grad_out: torch.Tensor, state: tuple[torch.Tensor, ...], backward_state: tuple[torch.Tensor, ...], softmax_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the Delta-Matching backward on SM90. Takes the BF16 BSHD output gradient, the saved
    state from delta_matching_fwd, delta_matching_bwd_prepare's result and the forward softmax
    scale. Returns fresh BF16 BSHD gradients matching the original query, key and value shapes.
    Key and value gradients are summed within each GQA group in FP32 before the BF16 cast.
    """
    match ARCH_MAJOR:
        case 9:
            from pithtrain.kernels.delta_matching_bwd_sm90 import delta_matching_bwd_sm90
            return delta_matching_bwd_sm90(grad_out, state, backward_state, softmax_scale)
        case _:
            raise NotImplementedError(f"pithtrain.kernels supports SM90 and SM100, not compute capability {ARCH_MAJOR}.x" if ARCH_MAJOR != 10 else "Delta-Matching requires SM90")
