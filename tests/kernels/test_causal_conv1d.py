"""
Verifier of ``causal_conv1d_fwd`` / ``causal_conv1d_bwd`` / ``causal_conv1d_out_like`` (the
channels-last causal depthwise conv of the Qwen3.5 Gated DeltaNet) against plain PyTorch.

Contract: ``x`` (B, offset + S, C) contiguous, ``w`` (C, 1, K); output t reads
``x[t + offset - (K - 1) + k]``, k = 0 .. K - 1, skipping negative indices, i.e.
``conv1d(x.transpose(1, 2), w, groups=C, padding=K - 1)[..., offset:offset + S]``. Forward and
input gradient are bit-identical to ATen's depthwise conv (taps accumulated in fp32 in order,
rounded once); the weight gradient is an fp32 reduction in a different order, so it agrees with
the exact sum to fp32 rounding (then one bf16 rounding). The output is (B, S, C) stored
channels-first, the layout ``causal_conv1d_out_like`` returns.

    pytest tests/kernels/test_causal_conv1d.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import causal_conv1d_bwd, causal_conv1d_fwd, causal_conv1d_out_like


def reference(x, w, offset, g):
    """ATen forward, input and weight gradients (autograd through F.conv1d)."""
    K, S = w.shape[-1], x.shape[1] - offset
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    y = F.conv1d(xr.transpose(1, 2), wr, groups=w.shape[0], padding=K - 1)
    y = y[..., offset : offset + S].transpose(1, 2)
    y.backward(g)
    return y.detach(), xr.grad, wr.grad


def exact_dw(x, w, offset, g):
    """float64 weight gradient and the sum of |terms| (for the fp32 rounding bound)."""
    B, L, C = x.shape
    K, S = w.shape[-1], L - offset
    xp = F.pad(x.double(), (0, 0, K - 1, 0))  # K - 1 zero rows of history
    dw = torch.zeros(C, K, dtype=torch.float64, device=x.device)
    mag = torch.zeros_like(dw)
    for k in range(K):
        xs = xp[:, offset + k : offset + k + S]  # x[t + offset - (K - 1) + k]
        prod = g.double() * xs
        dw[:, k] = prod.sum((0, 1))
        mag[:, k] = prod.abs().sum((0, 1))
    return dw.view(w.shape), mag.view(w.shape)


def make(B, S, C, offset, K=4, seed=0, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(B, S + offset, C, generator=gen, device="cuda") * scale).to(torch.bfloat16)
    w = (torch.randn(C, 1, K, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    g = torch.randn(B, S, C, generator=gen, device="cuda").to(torch.bfloat16)
    return x, w, g


def check(x, w, g, offset):
    B, L, C = x.shape
    S = L - offset
    y0, dx0, dw0 = reference(x, w, offset, g)

    y = causal_conv1d_fwd(x, w, offset)
    like = causal_conv1d_out_like(x, offset)
    assert y.shape == like.shape == (B, S, C)
    assert y.stride() == like.stride() == (C * S, 1, S)  # (B, C, S) dense, transposed
    assert y.dtype == like.dtype == x.dtype
    assert torch.equal(y, y0)

    dx, dw = causal_conv1d_bwd(g, x, w, offset)
    assert dx.shape == x.shape and dx.dtype == x.dtype
    assert dw.shape == w.shape and dw.dtype == w.dtype
    assert torch.equal(dx, dx0)
    want, mag = exact_dw(x, w, offset, g)
    # fp32 accumulation error, then one bf16 rounding.
    bound = 4 * torch.finfo(torch.float32).eps * mag + torch.finfo(w.dtype).eps * want.abs()
    assert ((dw.double() - want).abs() <= bound + 1e-30).all()
    # The gradient in the output's own (channels-first) layout gives the same result.
    g_like = causal_conv1d_out_like(x, offset).copy_(g)
    dx2, dw2 = causal_conv1d_bwd(g_like, x, w, offset)
    assert torch.equal(dx2, dx) and torch.equal(dw2, dw)
    # Deterministic.
    dx3, dw3 = causal_conv1d_bwd(g, x, w, offset)
    assert torch.equal(dx3, dx) and torch.equal(dw3, dw)


@pytest.mark.parametrize(
    "B,S,C,offset",
    [
        (1, 4096, 8192, 0),  # qwen3.5-35b-a3b: C = 2 * key_dim + value_dim = 8192
        (2, 1000, 520, 0),
        (1, 2051, 640, 3),  # with K - 1 rows of conv state prepended (context parallel)
        (2, 777, 8192, 3),
        (3, 100, 100, 1),  # C not a multiple of the channel block, offset < K - 1
        (1, 300, 130, 7),  # offset > K - 1
    ],
)
def test_matches_conv1d(B, S, C, offset):
    x, w, g = make(B, S, C, offset)
    check(x, w, g, offset)


@pytest.mark.parametrize("S,offset", [(1, 0), (2, 0), (3, 3), (1, 3)])
def test_short_sequences(S, offset):
    """Fewer outputs than taps."""
    x, w, g = make(2, S, 256, offset, seed=1)
    check(x, w, g, offset)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_kernel_widths(K):
    x, w, g = make(1, 513, 384, K - 1, K=K, seed=2)
    check(x, w, g, K - 1)


def test_large_negative_and_zero():
    x, w, g = make(1, 1024, 512, 3, seed=3, scale=1e3)
    x[:, :100] = 0
    w[:16] = 0
    w[16:32] = -w[16:32] * 64
    check(x, w, g, 3)


def test_history_only_changes_the_first_outputs():
    """offset rows of history feed exactly the first K - 1 outputs."""
    x, w, _ = make(1, 512, 256, 3, seed=4)
    y = causal_conv1d_fwd(x, w, 3)
    y_cold = causal_conv1d_fwd(x[:, 3:].contiguous(), w, 0)
    assert torch.equal(y[:, 3:], y_cold[:, 3:])
    assert not torch.equal(y[:, :3], y_cold[:, :3])
