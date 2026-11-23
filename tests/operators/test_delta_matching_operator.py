"""Verify the Delta-Matching training operator and its torch.compile boundary."""

import pytest
import torch

from pithtrain.kernels import delta_matching_supported
from pithtrain.operators.delta_matching import delta_matching_attn_func


@pytest.mark.parametrize("head_dim", [128, 256])
def test_delta_matching_compiled_backward(head_dim):
    generator = torch.Generator(device="cuda").manual_seed(77)
    inputs = tuple(
        torch.randn(
            (1, 256, heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        for heads in (6, 1, 1)
    )
    if not delta_matching_supported(*inputs):
        pytest.skip("Delta-Matching requires SM90")

    compiled = torch.compile(delta_matching_attn_func, backend="aot_eager", fullgraph=True)
    query = inputs[0]
    gate = torch.randn(
        (1, 6, 256, head_dim), dtype=query.dtype, device=query.device, generator=generator
    ).sigmoid()
    grad = torch.randn(gate.shape, dtype=query.dtype, device=query.device, generator=generator)
    gate_scaled = (grad * gate).transpose(1, 2)
    expanded = torch.ones((), device=query.device, dtype=query.dtype).expand_as(query)
    for grad_out, is_zero in (
        (gate_scaled, False),
        (expanded, False),
        (torch.zeros_like(query), True),
    ):
        eager = delta_matching_attn_func(*inputs)
        actual = compiled(*inputs)
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual, inputs, grad_out)
        eager_grad = torch.autograd.grad(eager, inputs, grad_out)
        for actual_tensor, eager_tensor in zip(actual_grad, eager_grad, strict=True):
            torch.testing.assert_close(actual_tensor, eager_tensor, rtol=0, atol=0)
            if is_zero:
                assert torch.count_nonzero(actual_tensor).item() == 0


def test_delta_matching_operator_validates_causal_and_scale():
    query = torch.empty((1, 128, 2, 256), device="cuda", dtype=torch.bfloat16)
    key = torch.empty((1, 128, 1, 256), device="cuda", dtype=torch.bfloat16)
    if not delta_matching_supported(query, key, key):
        pytest.skip("Delta-Matching requires SM90")
    with pytest.raises(ValueError, match="causal"):
        delta_matching_attn_func(query, key, key, causal=False)
    with pytest.raises(ValueError, match="positive, finite"):
        delta_matching_attn_func(query, key, key, softmax_scale=0.0)


def test_delta_matching_strided_projections():
    generator = torch.Generator(device="cuda").manual_seed(88)

    def operand(heads, channels):
        return torch.randn(
            (2, 128, heads, channels), device="cuda", dtype=torch.bfloat16, generator=generator
        )

    query = operand(6, 512)[..., :256].requires_grad_()
    key = operand(2, 256)[:, :, 1:].requires_grad_()
    value = operand(1, 512)[..., ::2].requires_grad_()
    inputs = query, key, value
    if not delta_matching_supported(*inputs):
        pytest.skip("Delta-Matching requires SM90")

    contiguous = tuple(x.detach().contiguous().requires_grad_() for x in inputs)
    grad_out = operand(6, 512)[..., 256:]
    actual = delta_matching_attn_func(*inputs)
    expected = delta_matching_attn_func(*contiguous)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, inputs, grad_out),
        torch.autograd.grad(expected, contiguous, grad_out.contiguous()),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
