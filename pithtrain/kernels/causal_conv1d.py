"""
Causal depthwise conv1d of the Qwen3.5 Gated DeltaNet, on the channels-last (B, S, C) layout.

``Qwen35MoeGatedDeltaNet`` convolves ``mixed_qkv`` (B, S, C = 2 * key_dim + value_dim) with a
per-channel kernel of width K = 4. Through ``nn.Conv1d`` on ``mixed_qkv.transpose(1, 2)`` it costs a
transpose copy in, ATen's generic ``conv_depthwise2d`` kernels (not specialized for a 1 x 4 kernel
or a channel-strided input) and K - 1 padded outputs that are then sliced away; backward pays the
same again plus a separate grad-weight reduction. Here one Triton pass reads the (B, S, C) input
where the projection GEMM wrote it and writes the output directly in the (B, C, S) storage that
the sliced conv output had (so the compiled consumers are unchanged); one more pass computes the
input gradient and per-block partials of the weight gradient. Measured on H200 at (1, 4096, 8192),
compiled with the SiLU: forward 420 -> 133 us, forward + backward 1495 -> 620 us.

Forward is bit-identical to ``conv1d(x.transpose(1, 2), w, padding=K - 1)[..., offset:offset + S]``:
ATen's ``conv_depthwise2d_forward_kernel_generic`` accumulates the taps in order k = 0 .. K - 1 as
fp32 ``acc += w * x`` (contracted to fma by nvcc; taps falling on the zero padding are skipped,
which ``fma(w, 0, acc) == acc`` makes equivalent) and rounds once to bf16; this kernel does the
same with explicit ``tl.fma``. The input gradient follows the same rule; the weight gradient is a
fp32 reduction over B * S in a different (deterministic) order than ATen's, so it agrees to fp32
rounding, not bit for bit.

``offset`` is the number of leading positions of ``x`` that are history rather than output (the
K - 1 tokens of conv state prepended under context parallelism): output t reads
``x[t + offset - (K - 1) + k]``, k = 0 .. K - 1, skipping negative indices.

The custom ops and autograd glue are ``pithtrain.operators.causal_conv1d``. Both architectures run
these kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "BLOCK_C",
    "BLOCK_T",
    "BWD",
    "causal_conv1d_bwd",
    "causal_conv1d_fwd",
    "causal_conv1d_out_like",
]


@triton.jit
def _conv_fwd_kernel(
    x_ptr, w_ptr, y_ptr, L, S, C, offset, sy_b, sy_t, sy_c,
    K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):  # fmt: skip
    pid_t, pid_c, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = c < C
    x_base = x_ptr + b.to(tl.int64) * L * C
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(K):
        w = tl.load(w_ptr + c * K + k, mask=c_ok, other=0.0).to(tl.float32)
        src = t + offset - (K - 1) + k
        ok = (src >= 0) & (t < S)
        xv = tl.load(x_base + src[:, None] * C + c[None, :], mask=ok[:, None] & c_ok[None, :], other=0.0)  # fmt: skip
        acc = tl.fma(w[None, :], xv.to(tl.float32), acc)
    y = y_ptr + b.to(tl.int64) * sy_b + t[:, None] * sy_t + c[None, :] * sy_c
    tl.store(y, acc.to(tl.bfloat16), mask=(t < S)[:, None] & c_ok[None, :])


@triton.jit
def _conv_bwd_kernel(
    g_ptr, x_ptr, w_ptr, dx_ptr, dw_ptr, L, S, C, offset, n_t, sg_b, sg_t, sg_c,
    K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):  # fmt: skip
    """dx over input rows [pid_t * BLOCK_T, +BLOCK_T) of x, and the partial dw of the outputs
    t in the same row range (one fp32 partial per (b, t block))."""
    pid_t, pid_c, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # input row (dx) / output row (dw)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = c < C
    g_base = g_ptr + b.to(tl.int64) * sg_b
    x_base = x_ptr + b.to(tl.int64) * L * C
    # dx[s] = sum_k w[k] * g[s - offset + (K - 1) - k], in the order of ATen's backward kernel
    # (taps k = 0 .. K - 1), over the outputs that exist.
    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(K):
        w = tl.load(w_ptr + c * K + k, mask=c_ok, other=0.0).to(tl.float32)
        t = r - offset + (K - 1) - k
        ok = (t >= 0) & (t < S)
        gv = tl.load(g_base + t[:, None] * sg_t + c[None, :] * sg_c, mask=ok[:, None] & c_ok[None, :], other=0.0)  # fmt: skip
        acc = tl.fma(w[None, :], gv.to(tl.float32), acc)
    tl.store(
        dx_ptr + b.to(tl.int64) * L * C + r[:, None] * C + c[None, :],
        acc.to(tl.bfloat16),
        mask=(r < L)[:, None] & c_ok[None, :],
    )
    # dw[c, k] partial = sum over outputs t of this block of g[t] * x[t + offset - (K - 1) + k].
    t_ok = r < S
    gv = tl.load(g_base + r[:, None] * sg_t + c[None, :] * sg_c, mask=t_ok[:, None] & c_ok[None, :], other=0.0)  # fmt: skip
    gv = gv.to(tl.float32)
    part = dw_ptr + (b * n_t + pid_t).to(tl.int64) * C * K
    for k in tl.static_range(K):
        src = r + offset - (K - 1) + k
        ok = t_ok & (src >= 0)
        xv = tl.load(x_base + src[:, None] * C + c[None, :], mask=ok[:, None] & c_ok[None, :], other=0.0)  # fmt: skip
        tl.store(part + c * K + k, tl.sum(gv * xv.to(tl.float32), axis=0), mask=c_ok)


BLOCK_T, BLOCK_C = 64, 128
BWD = dict(BLOCK_T=128, BLOCK_C=64, num_warps=4)  # tuned on H200, (1, 4096, 8192)


def causal_conv1d_out_like(x: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Uninitialized output of :func:`causal_conv1d_fwd` for ``x`` ``(B, offset + S, C)`` (any
    device, ``meta`` included): logical shape ``(B, S, C)`` in ``x``'s dtype, stored channels-first
    (a transposed view of a dense ``(B, C, S)`` tensor, strides ``(C * S, 1, S)``), the layout
    ``nn.Conv1d``'s output has after the slice, so compiled consumers (SiLU, split, L2 norm) see the
    strides they always saw. For fake / meta implementations.
    """
    B, L, C = x.shape
    return x.new_empty((B, C, L - offset)).transpose(1, 2)


def causal_conv1d_fwd(x: torch.Tensor, w: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Causal depthwise conv ``y[b, t, c] = sum_k w[c, k] * x[b, t + offset - (K - 1) + k, c]`` over
    ``k = 0 .. K - 1``, taps at negative positions skipped.

    - ``x``: bf16 ``(B, offset + S, C)`` contiguous; its first ``offset`` rows are history (conv
      state), not outputs. ``w``: ``(C, 1, K)`` or ``(C, K)`` contiguous (any float dtype, read as
      fp32). Neither is modified.
    - Returns a fresh bf16 ``(B, S, C)`` in the layout of :func:`causal_conv1d_out_like`.

    Bit-identical to ATen's CUDA ``conv1d(x.transpose(1, 2), w, padding=K - 1, groups=C)[...,
    offset:offset + S].transpose(1, 2)``: taps accumulated in order as fp32 fma, rounded once to
    bf16. Deterministic; same Triton kernel on both architectures.
    """
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    y = causal_conv1d_out_like(x, offset)
    grid = (triton.cdiv(S, BLOCK_T), triton.cdiv(C, BLOCK_C), B)
    _conv_fwd_kernel[grid](x, w, y, L, S, C, offset, *y.stride(), K=K, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C)  # fmt: skip
    return y


def causal_conv1d_bwd(
    g: torch.Tensor, x: torch.Tensor, w: torch.Tensor, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradients of :func:`causal_conv1d_fwd` for the output gradient ``g``: ``(dx, dw)``.

    - ``g``: bf16 ``(B, S, C)``, any strides; ``x``, ``w``, ``offset``: as in the forward. None is
      modified.
    - ``dx``: fresh bf16 ``(B, offset + S, C)`` contiguous, the gradient of every row of ``x``
      (the ``offset`` history rows included); bit-identical to ATen's depthwise conv backward
      (taps in order as fp32 fma, rounded once).
    - ``dw``: fresh, ``w``'s shape and dtype: fp32 partial sums per (batch, 128-row block) added by
      a fixed-order reduction, rounded once. Deterministic, but agrees with ATen only to fp32
      rounding (a different summation order).

    Same Triton kernel on both architectures.
    """
    B, L, C = x.shape
    K = w.shape[-1]
    S = L - offset
    n_t = triton.cdiv(L, BWD["BLOCK_T"])
    dx = torch.empty_like(x)
    part = torch.empty((B * n_t, C, K), dtype=torch.float32, device=x.device)
    grid = (n_t, triton.cdiv(C, BWD["BLOCK_C"]), B)
    _conv_bwd_kernel[grid](g, x, w, dx, part, L, S, C, offset, n_t, *g.stride(), K=K, **BWD)
    dw = part.sum(0).to(w.dtype).view(w.shape)
    return dx, dw
