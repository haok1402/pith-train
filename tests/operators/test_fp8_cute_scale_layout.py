"""
The SM90 quantization kernels hand DeepGEMM 1x128 scales already MN-major and TMA-aligned
(``fp8_cute/scale_layout.py``): the GEMMs launch no ``transpose_fp32`` and their outputs are
bit-identical to the same GEMMs fed row-major copies of the scales (which DeepGEMM transposes).
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from pithtrain.operators.deepgemm_quantize import (
    fp8cast_blockwise_transpose,
    fp8cast_rowwise_blockwise_transpose,
    fp8cast_rowwise_transpose,
)
from pithtrain.operators.fp8_cute.scale_layout import is_mn_major, mn_major_scale, to_mn_major
from pithtrain.operators.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.operators.linear import ARCH_MAJOR, FP8Linear, _fp8_gemm_nt

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="SM90 scale layout")

# (M, K, N); every dim a multiple of 16 (DeepGEMM TMA row alignment), several not of 128.
SHAPES = [(4096, 2048, 4096), (4096, 2048, 512), (1008, 2000, 1040), (144, 400, 272), (16, 128, 16)]


def _transposes(fn) -> int:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(
        "transpose_fp32" in e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    )


@pytest.mark.parametrize("mn,sf_k", [(1, 1), (5, 1), (4096, 16), (1001, 3), (2, 7)])
def test_helpers(mn: int, sf_k: int) -> None:
    s = mn_major_scale(mn, sf_k, torch.device("cuda"))
    assert s.shape == (mn, sf_k) and s.stride() == (1, (mn + 3) // 4 * 4) and is_mn_major(s)
    x = torch.randn(mn, sf_k, device="cuda")
    y = to_mn_major(x)
    assert is_mn_major(y) and torch.equal(x, y) and to_mn_major(y) is y


@pytest.mark.parametrize("M,K,N", SHAPES)
def test_dense_gemms_no_transpose_bit_identical(M: int, K: int, N: int) -> None:
    torch.manual_seed(M + K + N)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    x_fp8, sx, x_t_fp8, sx_t = fp8cast_rowwise_blockwise_transpose(x)
    dy_fp8, sdy, dy_t_fp8, sdy_t = fp8cast_rowwise_transpose(dy)
    w_fp8, sw, w_t_fp8, sw_t = fp8cast_blockwise_transpose(w)
    assert is_mn_major(sx) and is_mn_major(sdy) and is_mn_major(sdy_t)
    bf16, dev = torch.bfloat16, x.device

    def gemms(rowmajor: bool) -> list[torch.Tensor]:
        # new_empty + copy_ rather than .contiguous(): a (mn, 1) MN-major scale already counts
        # as contiguous to torch, but DeepGEMM checks the strides literally.
        f = (lambda s: s.new_empty(s.shape).copy_(s)) if rowmajor else (lambda s: s)
        return [
            _fp8_gemm_nt(x_fp8, f(sx), w_fp8, sw, M, N, dev, bf16),
            _fp8_gemm_nt(dy_fp8, f(sdy), w_t_fp8, sw_t, M, K, dev, bf16),
            _fp8_gemm_nt(dy_t_fp8, f(sdy_t), x_t_fp8, sx_t, N, K, dev, bf16),
        ]

    ref, out = gemms(True), gemms(False)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    assert _transposes(lambda: gemms(False)) == 0
    assert _transposes(lambda: gemms(True)) == 3  # the row-major control still pays


def test_fp8_linear_fwd_bwd_no_transpose() -> None:
    torch.manual_seed(0)
    lins = [FP8Linear(2048, n, bias=False).cuda().to(torch.bfloat16) for n in (4096, 512, 512)]
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def step() -> None:
        FP8WeightCacheControl.step()
        q = FP8Linear.quantize_input(x)
        ys = [lin(x, quantized_input=q) for lin in lins] + [lins[0](x)]
        sum(y.float().sum() for y in ys).backward()

    step()
    assert _transposes(step) == 0
