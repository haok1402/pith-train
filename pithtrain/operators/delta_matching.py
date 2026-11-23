"""
Causal Delta-Matching FP8 attention as a torch.compile-compatible training operator.
"""

import torch

from pithtrain.kernels import delta_matching_bwd, delta_matching_bwd_prepare, delta_matching_fwd

__all__ = ["delta_matching_attn_func"]


@torch.library.custom_op("pithtrain::delta_matching_attn_fwd", mutates_args=(), device_types="cuda")
def attention_fwd(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, softmax_scale: float
) -> list[torch.Tensor]:
    return list(delta_matching_fwd(query, key, value, softmax_scale))


@attention_fwd.register_fake
def fake_attention_fwd(query, key, value, softmax_scale):
    batch, sequence, num_heads, dim = query.shape
    num_key_heads = key.shape[2]
    group_keys = 8192 // dim

    def operand(heads, block_rows):
        return [
            query.new_empty((batch * heads, sequence, dim), dtype=torch.float8_e4m3fn),
            query.new_empty((batch * heads, sequence // block_rows), dtype=torch.float32),
        ]

    return [
        query.new_empty(query.shape),
        query.new_empty((batch * num_heads, sequence), dtype=torch.float32),
        *operand(num_heads, 2 * group_keys),
        query.new_empty((batch * num_heads, dim, sequence), dtype=torch.float8_e4m3fn),
        *operand(num_key_heads, group_keys),
        query.new_empty((batch * num_key_heads, dim, sequence), dtype=torch.float8_e4m3fn),
        *operand(num_key_heads, group_keys),
    ]


@torch.library.custom_op("pithtrain::delta_matching_attn_bwd", mutates_args=(), device_types="cuda")
def attention_bwd(
    grad_out: torch.Tensor,
    state: list[torch.Tensor],
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backward_state = delta_matching_bwd_prepare(grad_out, tuple(state))
    return delta_matching_bwd(grad_out, tuple(state), backward_state, softmax_scale)


@attention_bwd.register_fake
def fake_attention_bwd(grad_out, state, softmax_scale):
    batch, sequence, num_heads, dim = grad_out.shape
    num_key_heads = state[4].shape[0] // batch

    def gradient(heads):
        return grad_out.new_empty((batch, heads, sequence, dim)).transpose(1, 2)

    return gradient(num_heads), gradient(num_key_heads), gradient(num_key_heads)


def setup_context(ctx, inputs, output):
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*output[1:])
    ctx.softmax_scale = inputs[3]


def backward(ctx, gradients):
    grad_query, grad_key, grad_value = attention_bwd(
        gradients[0], list(ctx.saved_tensors), ctx.softmax_scale
    )
    return grad_query, grad_key, grad_value, None


attention_fwd.register_autograd(backward, setup_context=setup_context)


def delta_matching_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    Causal SM90 attention for BF16 BSHD Q, K and V, with D in {128, 256} and grouped KV heads.
    The sequence length must be divisible by 32768 / D. Returns BF16 BSHD attention output.
    """
    if not causal:
        raise ValueError("Delta-Matching attention requires causal=True")
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return attention_fwd(query, key, value, softmax_scale)[0]
