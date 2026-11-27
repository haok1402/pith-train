import importlib

import pytest
import torch

from pithtrain.kernels import fp8_gemm_bwd, fp8_wgrad_supported
from pithtrain.operators.linear import FP8Linear, fp8_linear_bwd_into

pytestmark = pytest.mark.skipif(not fp8_wgrad_supported(), reason="requires Hopper")


def test_dense_main_grad_routes_to_fp32_wgrad(monkeypatch):
    gemm = importlib.import_module(fp8_gemm_bwd.__module__)
    calls = []

    def wgrad(grad, grad_scale, activation, activation_scale, out):
        assert activation_scale.stride() == (2, 1)
        calls.append(out)
        out.fill_(1)

    monkeypatch.setattr(gemm, "fp8_wgrad", wgrad)
    layer = FP8Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    x = torch.ones_like(layer.weight, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weight.grad is not None
    assert not calls

    layer.weight.grad = None
    layer.main_grad = torch.full((256, 256), torch.nan, device="cuda")
    layer(x).sum().backward()
    assert len(calls) == 1 and calls[0] is layer.main_grad
    assert layer.weight.grad is None
    assert torch.all(layer.main_grad == 1)


def test_dense_wgrad_rejects_wrong_activation_scale_stride():
    grad = torch.empty((256, 256), device="cuda", dtype=torch.bfloat16)
    codes = grad.new_empty(grad.shape, dtype=torch.float8_e4m3fn)
    scale = grad.new_empty((2, 2), dtype=torch.float32)
    with pytest.raises(AssertionError, match="row-major transposed activation scales"):
        fp8_linear_bwd_into(grad, codes, scale, codes, scale.T, 256, grad.float())
