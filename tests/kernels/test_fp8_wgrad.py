"""Verify fresh FP32 dense wgrad on Hopper."""

import deep_gemm
import pytest
import torch

from pithtrain.kernels import fp8_quant_act, fp8_quant_grad, fp8_wgrad, fp8_wgrad_supported

pytestmark = pytest.mark.skipif(not fp8_wgrad_supported(), reason="requires Hopper")

RECIPE_SHAPES = [
    (1024, 5120),
    (6144, 5120),
    (10240, 5120),
    (12288, 5120),
    (5120, 6144),
    (17408, 5120),
    (5120, 17408),
]


def assert_same_bits(actual, expected):
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("rowwise", [False, True])
@pytest.mark.parametrize("shape", [(3, 40, 144), (71, 280, 384), (128, 128, 0), (0, 128, 128)])
def test_fp8_wgrad_pytorch(shape, rowwise):
    M, N, K = shape
    generator = torch.Generator(device="cuda").manual_seed(17)
    grad, activation = (
        torch.randint(-8, 9, (rows, K), generator=generator, device="cuda").to(torch.float8_e4m3fn)
        for rows in (M, N)
    )
    num_blocks = (K + 127) // 128
    scale_rows = N if rowwise else (N + 127) // 128
    grad_scale, activation_scale = (
        torch.exp2(torch.randint(-4, 5, (rows, num_blocks), generator=generator, device="cuda").float())
        for rows in (M, scale_rows)
    )
    out = torch.full((M, N), torch.nan, device="cuda")
    expected = torch.zeros_like(out)
    channel_scales = activation_scale if rowwise else activation_scale.repeat_interleave(128, 0)[:N]
    for block in range(num_blocks):
        span = slice(block * 128, (block + 1) * 128)
        product = grad[:, span].float() @ activation[:, span].float().T
        scale = grad_scale[:, block, None] * channel_scales[None, :, block]
        expected += scale * product
    fp8_wgrad(grad, grad_scale, activation, activation_scale, out)
    assert_same_bits(out, expected)


@pytest.mark.parametrize("value", [0.0, -1.0, 1.0])
def test_fp8_wgrad_signed_zero_and_subnormals(value):
    grad = torch.ones((64, 128), device="cuda").to(torch.float8_e4m3fn)
    activation = torch.full((128, 128), value, device="cuda").to(torch.float8_e4m3fn)
    grad_scale = torch.full((64, 1), 2.0**-70, device="cuda")
    activation_scale = torch.full((1, 1), 2.0**-70, device="cuda")
    out = torch.full((64, 128), torch.nan, device="cuda")
    fp8_wgrad(grad, grad_scale, activation, activation_scale, out)
    assert_same_bits(out, torch.full_like(out, value * 2.0**-133))


@pytest.mark.parametrize("shape", RECIPE_SHAPES)
def test_fp8_wgrad_recipe_bits(shape):
    M, N = shape
    K = 16384
    generator = torch.Generator(device="cuda").manual_seed(2315)
    grad = torch.randn((K, M), generator=generator, device="cuda", dtype=torch.bfloat16)
    activation = torch.randn((K, N), generator=generator, device="cuda", dtype=torch.bfloat16)
    _, _, grad_t, grad_scale = fp8_quant_grad(grad)
    _, _, activation_t, activation_scale = fp8_quant_act(activation)
    out = torch.full((M, N), torch.nan, device="cuda")
    expected = torch.zeros_like(out)
    # Stock DeepGEMM's reduce-add into zero also defines the fresh FP32 bits.
    channel_scales = activation_scale.repeat_interleave(128, dim=0)[:N]
    deep_gemm.fp8_fp4_gemm_nt(
        (grad_t, grad_scale), (activation_t, channel_scales), expected, c=expected, recipe=(1, 1, 128)
    )
    fp8_wgrad(grad_t, grad_scale, activation_t, activation_scale, out)
    assert_same_bits(out, expected)
