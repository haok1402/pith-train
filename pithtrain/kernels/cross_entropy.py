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
def cross_entropy_fwd(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride, n_cols, n_non_ignore_ptr, ignore_idx, BLOCK_SIZE: tl.constexpr):
    """
    One program per row: an online softmax over the vocabulary (max m, sum of exp d), then the row
    is overwritten with (softmax(x) - onehot(y)) / n_non_ignore and its loss stored. An ignored row
    is zeroed with loss 0.
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
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(tl.float32)
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


def cross_entropy_fwd_inplace(inp: torch.Tensor, target: torch.Tensor, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cross-entropy of the logits inp against target, overwriting the logits with the gradient
    of that mean. Returns (loss, grad).

    inp: (N, V) bf16 or fp32 logits, overwritten in place when row-contiguous (otherwise a
    contiguous copy is overwritten and inp is left alone). target: integer (N,), each in [0, V) or
    ignore_index (any other value reads out of bounds); not modified.

    loss: fp32 scalar, the sum over non-ignored rows of logsumexp(x) - x[target] divided by
    n = max(non-ignored rows, 1). grad: inp itself (or the copy), (softmax(x) - onehot(target)) / n
    per non-ignored row computed in fp32 and rounded once, zero rows where target == ignore_index.
    Matches F.cross_entropy(inp.float(), target, ignore_index=ignore_index) and its gradient to fp32
    rounding; deterministic.
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
        X_ptr=inp, X_stride=inp.stride(-2), Y_ptr=target, Y_stride=target.stride(-1), loss_ptr=loss_1d, loss_stride=loss_1d.stride(-1), n_cols=n_cols, n_non_ignore_ptr=n_non_ignore, ignore_idx=ignore_index, BLOCK_SIZE=BLOCK_SIZE, num_warps=32
    )

    loss = loss_1d.sum() / n_non_ignore.float().squeeze()
    return loss, inp
