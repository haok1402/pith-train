"""Verify fresh FP32 dense wgrad on Hopper."""

from types import SimpleNamespace

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
@pytest.mark.parametrize(
    "shape,padded",
    [
        ((3, 40, 144), False),
        ((71, 280, 384), False),
        ((1, 8, 145), True),
        ((17, 136, 145), True),
        ((17, 136, 145), False),
        ((3, 13, 144), False),
        ((128, 128, 0), False),
        ((0, 128, 128), False),
    ],
)
def test_fp8_wgrad_pytorch(shape, rowwise, padded):
    operands, out, expected = wgrad_operands(*shape, rowwise=rowwise, padded=padded)
    fp8_wgrad(*operands, out)
    assert_same_bits(out, expected)


def wgrad_operands(M, N, K, *, rowwise=False, padded=False):
    generator = torch.Generator(device="cuda").manual_seed(17)
    pitch = (K + 15) // 16 * 16 if padded else K
    grad, activation = (
        torch.randint(-8, 9, (rows, pitch), generator=generator, device="cuda").to(
            torch.float8_e4m3fn
        )[:, :K]
        for rows in (M, N)
    )
    num_blocks = (K + 127) // 128
    scale_rows = N if rowwise else (N + 127) // 128
    grad_scale, activation_scale = (
        torch.exp2(
            torch.randint(-4, 5, (rows, num_blocks), generator=generator, device="cuda").float()
        )
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
    return (grad, grad_scale, activation, activation_scale), out, expected


@pytest.mark.parametrize("num_sms", range(1, 133))
def test_fp8_wgrad_visible_sm_budget(num_sms, monkeypatch):
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if num_sms > device_sms:
        pytest.skip("requires at least this many visible SMs")
    operands, out, expected = wgrad_operands(17, 24, 144)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=num_sms),
    )
    fp8_wgrad(*operands, out)
    assert_same_bits(out, expected)


@pytest.mark.parametrize("rowwise", [False, True])
def test_fp8_wgrad_offset_scales(rowwise):
    (grad, grad_scale, activation, activation_scale), out, expected = wgrad_operands(
        17, 136, 144, rowwise=rowwise
    )

    def offset_scale(scale):
        rows, blocks = scale.shape
        pitch = (rows + 3) // 4 * 4
        storage = torch.empty(1 + blocks * pitch, device=scale.device, dtype=scale.dtype)
        view = storage[1:].as_strided(scale.shape, (1, pitch))
        assert view.data_ptr() % 16 == 4
        return view.copy_(scale)

    grad_offset = offset_scale(grad_scale)
    if rowwise:
        act_offset = offset_scale(activation_scale)
    else:
        act_offset = activation_scale
    fp8_wgrad(grad, grad_scale, activation, activation_scale, out)
    baseline = out.clone()
    fp8_wgrad(grad, grad_offset, activation, act_offset, out)
    assert_same_bits(out, expected)
    assert_same_bits(out, baseline)


@pytest.mark.parametrize(
    "layout", ["fp8_offset", "fp8_inner_stride", "output_offset", "output_pitch"]
)
def test_fp8_wgrad_ineligible_physical_layout(layout):
    (grad, grad_scale, activation, activation_scale), out, expected = wgrad_operands(17, 24, 144)
    if layout == "fp8_offset":
        grad = grad.new_empty(1 + grad.numel())[1:].view_as(grad).copy_(grad)
        assert grad.data_ptr() % 16 == 1
    elif layout == "fp8_inner_stride":
        grad = grad.new_empty((17, 288))[:, ::2].copy_(grad)
    else:
        offset = int(layout == "output_offset")
        width = 28 if offset else 25
        backing = torch.full((17, width), torch.nan, device=grad.device)
        out = backing[:, offset : 24 + offset]
    fp8_wgrad(grad, grad_scale, activation, activation_scale, out)
    assert_same_bits(out, expected)
    if layout.startswith("output_"):
        assert backing[:, :offset].isnan().all()
        assert backing[:, 24 + offset :].isnan().all()


def test_fp8_wgrad_fallback_stays_fp32_under_autocast():
    generator = torch.Generator(device="cuda").manual_seed(23)
    grad, activation = (
        torch.randn((rows, 145), generator=generator, device="cuda").to(torch.float8_e4m3fn)
        for rows in (7, 13)
    )
    grad_scale, activation_scale = (
        torch.exp2(torch.randint(-2, 3, (rows, 2), device="cuda", generator=generator).float())
        for rows in (7, 1)
    )
    grad_ref = grad.double() * grad_scale.double().repeat_interleave(128, dim=1)[:, :145]
    activation_ref = (
        activation.double() * activation_scale.double().repeat_interleave(128, dim=1)[:, :145]
    )
    out = torch.full((7, 13), torch.nan, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fp8_wgrad(grad, grad_scale, activation, activation_scale, out)
    torch.testing.assert_close(out.double(), grad_ref @ activation_ref.T, atol=1e-5, rtol=1e-6)


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
        (grad_t, grad_scale),
        (activation_t, channel_scales),
        expected,
        c=expected,
        recipe=(1, 1, 128),
    )
    fp8_wgrad(grad_t, grad_scale, activation_t, activation_scale, out)
    assert_same_bits(out, expected)
