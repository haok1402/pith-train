"""
The gpt-oss raw-expert FP8 wgrad into fp32 main_grad (fp8_gptoss._fp8_wgrad) must keep the exact
temporary-then-add semantics, bit for bit: the k-grouped GEMM accumulates into a zeroed fp32
temporary and main_grad.add_ adds it once. Accumulating straight into main_grad through DeepGEMM's
TMA reduce-add is not equivalent for every bit pattern (it may flush subnormal sums), so main_grad
here starts with signed zeros and subnormals next to normal values, some experts receive no rows,
and results are compared as raw int32 bit patterns.

    pytest tests/test_fp8_wgrad_exact.py -v
"""

import pytest
import torch

from pithtrain.operators.deepgemm_quantize import fp8cast_rowwise_kmajor
from pithtrain.operators.fp8_gptoss import _fp8_wgrad, _kgrouped_safe
from pithtrain.operators.grouped_linear import _k_grouped_fp8_gemm_nt, _pad_k_major_channels
from pithtrain.operators.linear import ARCH_MAJOR
from pithtrain.pipeline.execution import WeightGradStore

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper k-grouped FP8 wgrad")

H, INTER = 2880, 2880
# name -> (N, K, which operand carries padded channels), as in FP8ClampedSwiGLUExpertsFunc.backward
LAYOUTS = {"gate_up_proj": (2 * INTER, H, "a"), "down_proj": (H, INTER, "b")}


def operands(name, counts, seed):
    N, K, padded = LAYOUTS[name]
    g = torch.Generator(device="cuda").manual_seed(seed)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    M = sum(counts)
    a = fp8cast_rowwise_kmajor(torch.randn(M, N, generator=g, device="cuda").to(torch.bfloat16), offs)[2:]  # fmt: skip
    b = fp8cast_rowwise_kmajor(torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16), offs)[2:]  # fmt: skip
    if padded == "a":
        a = _pad_k_major_channels(a, counts, _kgrouped_safe(N))
    else:
        b = _pad_k_major_channels(b, counts, _kgrouped_safe(K))
    return a, b


def adversarial_main_grad(G, N, K, seed):
    """Normal values mixed with +-0.0, the smallest subnormals and values near the normal limit."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    mg = torch.randn(G, N, K, generator=g, device="cuda")
    specials = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1e-40, -1e-40, 1.1754942e-38, -1.1754942e-38], device="cuda")  # fmt: skip
    pick = torch.randint(0, 4, (G, N, K), generator=g, device="cuda") == 0
    idx = torch.randint(0, specials.numel(), (G, N, K), generator=g, device="cuda")
    return torch.where(pick, specials[idx], mg)


def reference(a, b, counts, kt, main_grad):
    """The exact semantics: zeroed fp32 temporary, k-grouped GEMM into it, one add into main_grad."""
    G, N, K = main_grad.shape
    NA, KB = a[1].shape[0], b[1].shape[0]
    if NA % 256 != 0 and KB % 256 == 0:
        c = torch.zeros((G, KB, NA), device="cuda", dtype=torch.float32)
        _k_grouped_fp8_gemm_nt(b, a, c, counts, kt, c=c)
        main_grad.add_(c[:, :K, :N].transpose(1, 2))
    else:
        c = torch.zeros((G, NA, KB), device="cuda", dtype=torch.float32)
        _k_grouped_fp8_gemm_nt(a, b, c, counts, kt, c=c)
        main_grad.add_(c[:, :N, :K])


@pytest.mark.parametrize("name", list(LAYOUTS))
@pytest.mark.parametrize("counts", [[2048] * 4, [0, 128, 256, 0, 640, 128, 1024, 0], [66048, 128]])
def test_wgrad_bit_patterns(name, counts, monkeypatch):
    monkeypatch.setattr(WeightGradStore, "enabled", False)
    G = len(counts)
    N, K, _ = LAYOUTS[name]
    kt = torch.tensor(counts, device="cuda", dtype=torch.int32)
    w = torch.nn.Parameter(torch.empty(G, N, K, device="cuda", dtype=torch.bfloat16))
    got = adversarial_main_grad(G, N, K, seed=0)
    want = got.clone()
    for mb in range(2):
        a, b = operands(name, counts, seed=100 + mb)
        _fp8_wgrad(a, b, counts, kt, w, got)
        reference(a, b, counts, kt, want)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), f"micro-batch {mb}"
    assert w.grad is None
