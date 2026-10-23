"""Bit-exactness of the fused clamped SwiGLU + FP8 quantization kernels (gpt-oss experts)."""

import pytest
import torch

from pithtrain.operators.clamped_swiglu import clamped_swiglu
from pithtrain.operators.deepgemm_quantize import (
    _fp8cast_rowwise_kmajor_triton,
    fp8cast_rowwise_kmajor,
)
from pithtrain.operators.fp8_cute.scale_layout import to_mn_major
from pithtrain.operators.indexed_bias_add import _segment_sum_along_rows, indexed_bias_add

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="CuTe-DSL FP8 kernels are SM90 only",
)

ALPHA, LIMIT = 1.702, 7.0


def _offs(M: int, G: int, gen: torch.Generator) -> torch.Tensor:
    """Random 128-aligned cumulative group ends (some groups empty), last entry M."""
    blocks = M // 128
    cuts = torch.sort(torch.randint(0, blocks + 1, (G - 1,), generator=gen)).values
    ends = torch.cat([cuts, torch.tensor([blocks])]) * 128
    return ends.to(torch.int32).cuda()


def _inputs(M: int, N: int, G: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    gu = torch.randn(M, 2 * N, generator=gen) * 4
    gu[torch.rand(M, 2 * N, generator=gen) < 0.01] *= 100  # outliers past the clamp
    gu[torch.rand(M, 2 * N, generator=gen) < 0.01] = 0
    if M >= 256:
        gu[128:256] = 0  # an all-zero row block
    bias = torch.randn(G, 2 * N, generator=gen)
    dh = torch.randn(M, N, generator=gen)
    dh[torch.rand(M, N, generator=gen) < 0.01] *= 1000
    bf = dict(dtype=torch.bfloat16, device="cuda")
    return gu.to(**bf), bias.to(**bf), dh.to(**bf), _offs(M, G, gen)


def _group_ids(offs: torch.Tensor, M: int) -> torch.Tensor:
    ids = torch.arange(M, device=offs.device)
    return torch.searchsorted(offs.long(), ids, right=True).clamp_(max=offs.shape[0] - 1)


def _assert_quant_equal(a, b):
    for x, y, name in zip(a, b, ("tok", "tok_scale", "kmajor", "kmajor_scale")):
        assert x.shape == y.shape and x.stride() == y.stride(), (name, x.shape, y.shape)
        if x.dtype == torch.float8_e4m3fn:
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), name
        else:
            assert torch.equal(x, y), name


SHAPES = [
    (128, 2880, 1),
    (1024, 2880, 4),
    (4096, 2880, 32),
    (2048, 2880, 16),
    (512, 64, 3),
    (640, 136, 5),  # N and 2N not multiples of 128
    (256, 8, 2),
]


@pytest.mark.parametrize("M,N,G", SHAPES)
def test_forward_bit_exact(M: int, N: int, G: int) -> None:
    from pithtrain.operators.fp8_cute.clamped_swiglu_quant import clamped_swiglu_quant_fwd

    gu, bias, _, offs = _inputs(M, N, G, seed=M + N + G)
    biased = indexed_bias_add(gu, bias, _group_ids(offs, M), offs)
    ref = fp8cast_rowwise_kmajor(clamped_swiglu(biased, ALPHA, LIMIT), offs)
    out = clamped_swiglu_quant_fwd(gu, bias, offs, ALPHA, LIMIT)
    _assert_quant_equal(out, ref)


@pytest.mark.parametrize("M,N,G", SHAPES)
def test_backward_bit_exact(M: int, N: int, G: int) -> None:
    from pithtrain.operators.fp8_cute.clamped_swiglu_quant import clamped_swiglu_bwd_quant

    gu, bias, dh, offs = _inputs(M, N, G, seed=7 * M + N + G)
    biased = indexed_bias_add(gu, bias, _group_ids(offs, M), offs).detach().requires_grad_()
    clamped_swiglu(biased, ALPHA, LIMIT).backward(dh)
    dgu = biased.grad
    ref = fp8cast_rowwise_kmajor(dgu, offs)
    ref_bias = _segment_sum_along_rows(dgu, offs, torch.float32)
    out, bias_grad = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT)
    _assert_quant_equal(out, ref)
    assert bias_grad.dtype == bias.dtype and bias_grad.shape == bias.shape
    torch.testing.assert_close(bias_grad.float(), ref_bias.to(bias.dtype).float(), rtol=1e-2, atol=1e-2)  # fmt: skip
    # Exactly the bf16 rounding of the fp32 block sums taken in row order.
    blocks = dgu.float().view(M // 128, 128, 2 * N).sum(1)
    exp = torch.zeros(G, 2 * N, device="cuda")
    start = 0
    for g, end in enumerate((offs // 128).tolist()):
        for b in range(start, end):
            exp[g] += blocks[b]
        start = end
    torch.testing.assert_close(bias_grad.float(), exp.to(bias.dtype).float(), rtol=1e-2, atol=1e-2)  # fmt: skip


@pytest.mark.parametrize("backward", [False, True])
def test_every_bf16_gate_and_up(backward: bool) -> None:
    """Every BF16 bit pattern (NaN included) as gate and, shuffled, as up, with a random bias.

    Compared with the Triton quantizer: on rows holding a NaN the CuTe fp8cast_rowwise_kmajor
    takes a NaN amax (integer max of the magnitudes) where Triton's max.f32 ignores it; this
    kernel follows Triton, as silu_mul_quant does."""
    from pithtrain.operators.fp8_cute.clamped_swiglu_quant import (
        clamped_swiglu_bwd_quant,
        clamped_swiglu_quant_fwd,
    )

    M, N, G = 512, 128, 2
    gen = torch.Generator().manual_seed(3)
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16)
    up = bits[torch.randperm(65536, generator=gen)].view(torch.bfloat16)
    gu = torch.stack([gate, up], dim=-1).view(M, 2 * N).cuda()
    bias = (torch.randn(G, 2 * N, generator=gen) * 0.1).to(torch.bfloat16).cuda()
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    biased = indexed_bias_add(gu, bias, _group_ids(offs, M), offs).detach().requires_grad_()
    h = clamped_swiglu(biased, ALPHA, LIMIT)

    def kmajor(x: torch.Tensor) -> tuple:
        tok, s_tok, km, s_km = _fp8cast_rowwise_kmajor_triton(x, offs)
        return tok, to_mn_major(s_tok), km, to_mn_major(s_km)

    if not backward:
        ref = kmajor(h)
        _assert_quant_equal(clamped_swiglu_quant_fwd(gu, bias, offs, ALPHA, LIMIT), ref)
        return
    dh = torch.randn(M, N, generator=gen).to(torch.bfloat16).cuda()
    h.backward(dh)
    ref = kmajor(biased.grad)
    out, _ = clamped_swiglu_bwd_quant(dh, gu, bias, offs, ALPHA, LIMIT)
    _assert_quant_equal(out, ref)


def test_padded_kmajor_channels() -> None:
    """kmajor_channels lays the K-major pair out with more channels; the first N match."""
    from pithtrain.operators.fp8_cute.clamped_swiglu_quant import clamped_swiglu_quant_fwd

    M, N, G, KP = 2048, 2880, 5, 2944
    gu, bias, _, offs = _inputs(M, N, G, seed=11)
    ref = clamped_swiglu_quant_fwd(gu, bias, offs, ALPHA, LIMIT)
    out = clamped_swiglu_quant_fwd(gu, bias, offs, ALPHA, LIMIT, kmajor_channels=KP)
    _assert_quant_equal(out[:2], ref[:2])
    assert out[2].numel() == M * KP and out[3].shape == (KP, M // 128)
    assert torch.equal(out[3][:N], ref[3])
    start = 0
    for end in offs.tolist():
        k = end - start
        a = out[2][start * KP : end * KP].view(KP, k)[:N].view(torch.uint8)
        b = ref[2][start * N : end * N].view(N, k).view(torch.uint8)
        assert torch.equal(a, b)
        start = end


@pytest.mark.parametrize("M,N,G", [(2048, 2880, 5), (640, 136, 5)])
def test_split_rowwise_kmajor(M: int, N: int, G: int) -> None:
    """The rowwise-only and K-major-only launches give the fused forward's bytes."""
    from pithtrain.operators.fp8_cute.clamped_swiglu_quant import (
        clamped_swiglu_quant_fwd,
        clamped_swiglu_quant_kmajor,
        clamped_swiglu_quant_rowwise,
    )

    gu, bias, _, offs = _inputs(M, N, G, seed=13)
    KP = -(-N // 256) * 256
    full = clamped_swiglu_quant_fwd(gu, bias, offs, ALPHA, LIMIT, kmajor_channels=KP)
    row = clamped_swiglu_quant_rowwise(gu, bias, offs, ALPHA, LIMIT)
    km = clamped_swiglu_quant_kmajor(gu, bias, offs, ALPHA, LIMIT, kmajor_channels=KP)
    _assert_quant_equal(row, full[:2])
    assert torch.equal(km[1][:N], full[3][:N])
    start = 0
    for end in offs.tolist():
        a = km[0][start * KP : end * KP].view(KP, end - start)[:N].view(torch.uint8)
        b = full[2][start * KP : end * KP].view(KP, end - start)[:N].view(torch.uint8)
        assert torch.equal(a, b)
        start = end
