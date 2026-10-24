# Adapted from NVIDIA TransformerEngine (Apache 2.0)
# https://github.com/NVIDIA/TransformerEngine
#   transformer_engine/common/triton/cross_entropy.py  (Triton kernels)
#   transformer_engine/pytorch/triton/cross_entropy.py  (autograd wrapper)
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications: fused online-softmax + cross-entropy into a single kernel,
# removed tensor-parallelism / label-smoothing / distributed support.

import torch
import triton
import triton.language as tl

__all__ = ["cross_entropy_fwd", "cross_entropy_fwd_inplace"]


@triton.jit
def cross_entropy_fwd(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore_ptr,
    ignore_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy forward: computes per-row loss and overwrites
    the logit tensor in-place with mean-reduced gradients.

    Each program processes one row (token position). The kernel performs
    two passes over the vocabulary dimension:
      1. Online softmax: numerically stable max (m) and sum-of-exp (d).
      2. Gradient write: softmax(x_i) / N for all i, then correct the
         target position to (softmax(x_y) - 1) / N.

    The per-row loss is stored separately for later summation.
    """
    row = tl.program_id(0).to(tl.int64)
    X_ptr += row * X_stride
    y = tl.load(Y_ptr + row * Y_stride)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        tl.store(loss_ptr + row * loss_stride, 0.0)
        return

    n_non_ignore: tl.float32 = tl.load(n_non_ignore_ptr).to(tl.float32)

    m: tl.float32 = float("-inf")
    d: tl.float32 = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(
            tl.float32
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    ori_X_y = tl.load(X_ptr + y).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
        grad_dtype = X_block.dtype
        X_block = (tl.exp(X_block.to(tl.float32) - m) / d) / n_non_ignore
        X_block = tl.where(X_offsets == y, X_block - 1.0 / n_non_ignore, X_block)
        tl.store(X_ptr + X_offsets, X_block.to(grad_dtype), mask=X_offsets < n_cols)

    loss = -(ori_X_y - m - tl.log(d))
    tl.store(loss_ptr + row * loss_stride, loss)


def cross_entropy_fwd_inplace(
    inp: torch.Tensor, target: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cross-entropy of the ``(N, V)`` logits ``inp`` against ``target``, overwriting the logits
    with the gradient of that mean. Returns ``(loss, grad)``: ``grad`` is ``inp`` itself, or a
    contiguous copy of it when ``inp`` is not row-contiguous (the in-place saving is then lost).
    """
    n_rows, n_cols = inp.shape

    if inp.stride(-1) != 1 or inp.stride(-2) != n_cols:
        inp = inp.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    n_non_ignore = (target != ignore_index).sum(dtype=torch.int64).view(1)
    n_non_ignore.clamp_(min=1)

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=inp.device)
    BLOCK_SIZE = min(65536 // 2, triton.next_power_of_2(n_cols))

    cross_entropy_fwd[(n_rows,)](
        X_ptr=inp,
        X_stride=inp.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_cols=n_cols,
        n_non_ignore_ptr=n_non_ignore,
        ignore_idx=ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    loss = loss_1d.sum() / n_non_ignore.float().squeeze()
    return loss, inp
