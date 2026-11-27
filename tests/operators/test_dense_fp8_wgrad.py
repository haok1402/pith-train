import pytest
import torch

from pithtrain.kernels import fp8_quant_act, fp8_wgrad_supported
from pithtrain.operators import linear

pytestmark = pytest.mark.skipif(not fp8_wgrad_supported(), reason="requires Hopper")


@pytest.fixture
def projection_spy(monkeypatch):
    """Use the real activation producers and observe each layer's backward dispatch."""
    calls = []

    def gemm_nt(a, sa, b, sb, m, n, device, dtype):
        return torch.zeros((m, n), device=device, dtype=dtype)

    def gemm_fwd(x, w, w_scale):
        x_fp8, x_scale, x_t, x_t_scale = fp8_quant_act(x)
        y = gemm_nt(x_fp8, x_scale, w, w_scale, x.shape[0], w.shape[0], x.device, x.dtype)
        return y, x_t, x_t_scale

    def wgrad(grad, grad_scale, xt, xt_scale, main_grad):
        calls.append(main_grad)
        assert xt_scale.shape == (2, 2)
        assert xt_scale.stride() == (2, 1)
        main_grad.fill_(1)

    def quantized_grad(dy, *args):
        codes = dy.new_empty(dy.shape, dtype=torch.float8_e4m3fn)
        scales = dy.new_empty((dy.shape[0], 2), dtype=torch.float32)
        return codes, scales, codes.T, scales

    def quantized_weight(layer):
        w = layer.weight.new_empty(layer.weight.shape, dtype=torch.float8_e4m3fn)
        scale = w.new_empty((2, 2), dtype=torch.float32)
        return w, scale, w.T, scale

    def swiglu_adjoint(dh, *args):
        codes, scales, *_ = quantized_grad(dh)
        return (codes, scales, codes.flatten(), scales), (codes, scales, codes.flatten(), scales)

    gemm_module = linear.fp8_gemm_bwd.__module__
    monkeypatch.setattr(f"{gemm_module}.fp8_gemm_nt", gemm_nt)
    monkeypatch.setattr(f"{gemm_module}.fp8_quant_grad", quantized_grad)
    monkeypatch.setattr(f"{gemm_module}.fp8_wgrad", wgrad)
    monkeypatch.setattr(linear, "fp8_gemm_nt", gemm_nt)
    monkeypatch.setattr(linear, "fp8_gemm_fwd", gemm_fwd)
    monkeypatch.setattr(linear.FP8Linear, "get_quantized_weight", quantized_weight)
    monkeypatch.setattr(linear, "fp8_silu_mul_bwd", swiglu_adjoint)
    monkeypatch.setattr(linear, "silu_mul_bwd", lambda grad, *_: (grad, grad))
    monkeypatch.setattr(linear, "fp8_quant_delta_grad", quantized_grad)
    return calls


@pytest.mark.parametrize(
    "path",
    ["plain", "shared", "swiglu_down", "swiglu_saved", "swiglu_replay", "delta_gate", "ordinary"],
)
def test_dense_projection_main_grad_routes(path, projection_spy):
    def projection(main_grad=True):
        layer = linear.FP8Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        if main_grad:
            layer.main_grad = torch.full_like(layer.weight, torch.nan, dtype=torch.float32)
        return layer

    x = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    p = projection(main_grad=path != "ordinary")
    if path == "shared":
        result = p(x, p.quantize_input(x))
        projections = [p]
    elif path == "swiglu_down":
        result = p.forward_swiglu(x, x)
        projections = [p]
    elif path in ("swiglu_saved", "swiglu_replay"):
        gate, up = projection(), projection()
        result = linear.fp8_swiglu_main_grad(x, gate, up, p, replay=path == "swiglu_replay")
        projections = [p, up, gate]
    elif path == "delta_gate":
        z = projection()
        result = p.forward_delta_gate(x, x, p.quantize_input(x), x.new_ones(128), 1e-6, z)
        projections = [p, z]
    else:
        result = p(x)
        projections = [p]

    result.sum().backward()
    if path == "ordinary":
        assert not projection_spy
        assert p.weight.grad is not None
        return
    assert len(projection_spy) == len(projections)
    for main_grad, layer in zip(projection_spy, projections, strict=True):
        assert main_grad is layer.main_grad
        assert layer.weight.grad is None
        assert torch.all(main_grad == 1)


def test_dense_wgrad_rejects_wrong_activation_scale_stride(projection_spy):
    x = torch.empty((256, 256), device="cuda", dtype=torch.bfloat16)
    xt = x.new_empty(x.T.shape, dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 2), device=x.device)
    with pytest.raises(AssertionError, match="row-major transposed activation scales"):
        linear.fp8_linear_bwd_into(x, xt, scale, xt, scale.T, 256, x.float())
    assert not projection_spy
