"""Verify matched causal FP8 attention, including its pre-cast score-gradient invariant."""

import pytest
import torch

from pithtrain.kernels import (
    delta_matching_bwd,
    delta_matching_bwd_prepare,
    delta_matching_fwd,
    delta_matching_supported,
)

FP32_BOUNDS = (0.08, 0.12, 0.12, 0.08)


def attention_inputs(dim, num_heads, num_key_heads, sequence, batch=1):
    generator = torch.Generator(device="cuda").manual_seed(17)

    def sample(heads):
        return torch.randn(
            (batch, sequence, heads, dim),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    return sample(num_heads), sample(num_key_heads), sample(num_key_heads), sample(num_heads)


def require_delta_matching(query, key, value):
    if not delta_matching_supported(query, key, value):
        pytest.skip("Delta-Matching requires SM90")


def run_attention(query, key, value, grad_output):
    scale = query.shape[-1] ** -0.5
    out, *state = delta_matching_fwd(query, key, value, scale)
    backward_state = delta_matching_bwd_prepare(grad_output, tuple(state))
    gradients = delta_matching_bwd(grad_output, tuple(state), backward_state, scale)
    return out, *gradients


def fp32_attention(query, key, value, grad_output):
    query, key, value = (
        tensor.transpose(1, 2).float().requires_grad_() for tensor in (query, key, value)
    )
    repeats = query.shape[1] // key.shape[1]
    scores = query @ key.repeat_interleave(repeats, dim=1).transpose(-1, -2)
    scores = scores * (query.shape[-1] ** -0.5)
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
    out = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ value.repeat_interleave(
        repeats, dim=1
    )
    gradients = torch.autograd.grad(out, (query, key, value), grad_output.transpose(1, 2).float())
    return tuple(tensor.transpose(1, 2) for tensor in (out, *gradients))


def relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("num_key_heads", [1, 2, 4])
def test_delta_matching_fp32_and_determinism(dim, num_key_heads):
    query, key, value, grad_output = attention_inputs(dim, 4, num_key_heads, 512, batch=2)
    require_delta_matching(query, key, value)
    # Mimic the varying gradient magnitude produced by a gate outside the attention core.
    grad_output = (grad_output.float() * grad_output.float().sigmoid()).to(torch.bfloat16)
    grad_output = grad_output.transpose(1, 2).contiguous().transpose(1, 2)
    actual = run_attention(query, key, value, grad_output)
    repeated = run_attention(query, key, value, grad_output)
    expected = fp32_attention(query, key, value, grad_output)
    for name, tensor, repeat, reference, bound in zip(
        ("O", "dQ", "dK", "dV"), actual, repeated, expected, FP32_BOUNDS, strict=True
    ):
        assert tensor.dtype == torch.bfloat16 and tensor.shape == reference.shape
        assert torch.equal(tensor, repeat), f"{name} is nondeterministic"
        assert (error := relative_error(tensor, reference)) < bound, f"{name}: {error:.5f}"


@pytest.mark.parametrize("dim", [128, 256])
def test_delta_matching_rowsum(dim):
    query, key, value, grad_output = (tensor.sign() for tensor in attention_inputs(dim, 4, 2, 256))
    # Centering leaves these key codes unchanged. The resulting FP8 code products and their
    # sums are exact even in Hopper's narrower tensor-core accumulator.
    key = torch.cat((key[:, :128], -key[:, :128]), dim=1)
    require_delta_matching(query, key, value)
    out, *state = delta_matching_fwd(query, key, value, dim**-0.5)
    lse, query_codes, query_scale, _, key_codes, key_scale, _, value_codes, value_scale = state
    grad_codes, grad_scale, _, delta = delta_matching_bwd_prepare(grad_output, tuple(state))
    group_keys = 8192 // dim

    def scales(block_scales, block_rows, repeat=1):
        return block_scales.repeat_interleave(block_rows, dim=-1).repeat_interleave(repeat, dim=0)

    def code_product(left, right):
        return (left.double() @ right.repeat_interleave(2, dim=0).double().mT).float()

    scores = code_product(query_codes, key_codes)
    scores *= scales(query_scale, 2 * group_keys).unsqueeze(-1) * scales(
        key_scale, group_keys, 2
    ).unsqueeze(-2)
    mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
    probability = (scores - lse.unsqueeze(-1)).exp().masked_fill(~mask, 0)
    grad_probability = code_product(grad_codes, value_codes)
    grad_probability *= scales(grad_scale, 2 * group_keys).unsqueeze(-1) * scales(
        value_scale, group_keys, 2
    ).unsqueeze(-2)
    score_gradient = probability * (grad_probability - delta.unsqueeze(-1))
    # Normalize by the terms before subtraction so rows with one available key have
    # a round-off bound even when their exact gradient is zero.
    magnitude = (probability * (grad_probability.abs() + delta.abs().unsqueeze(-1))).sum(-1)
    residual = score_gradient.sum(-1).abs()
    roundoff_bound = (16 * torch.finfo(torch.float32).eps) * magnitude
    excess = (residual - roundoff_bound).max().item()
    assert excess <= 0, f"a pre-cast row sum exceeded its FP32 round-off bound by {excess:.3g}"

    stale = (out.float() * grad_output.float()).transpose(1, 2).sum(-1).reshape_as(delta)
    stale_residual = (probability * (grad_probability - stale.unsqueeze(-1))).sum(-1)
    assert (stale_residual.abs() / magnitude.clamp_min(1e-6)).median() > 1e-3


@pytest.mark.parametrize("dim", [128, 256])
def test_delta_matching_reference_library(dim):
    query, key, value, grad_output = attention_inputs(dim, 6, 2, 512)
    require_delta_matching(query, key, value)
    reference = pytest.importorskip("delta_matching")
    actual = run_attention(query, key, value, grad_output)
    inputs = [tensor.detach().requires_grad_() for tensor in (query, key, value)]
    output = reference.dm_attn_func(*inputs, causal=True, mode="matched")
    expected = (output, *torch.autograd.grad(output, inputs, grad_output))
    for name, tensor, target in zip(("O", "dQ", "dK", "dV"), actual, expected, strict=True):
        assert (error := relative_error(tensor, target)) < 0.035, f"{name}: {error:.5f}"
