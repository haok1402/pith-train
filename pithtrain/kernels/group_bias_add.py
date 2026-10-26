"""
In-place y += bias[group_indices] over grouped rows (the gpt-oss down-projection bias), as a Triton
kernel. Both architectures run it.
"""

import torch
import triton
import triton.language as tl

__all__ = ["group_bias_add_"]


@triton.jit
def _group_bias_add_kernel(y_ptr, bias_ptr, gi_ptr, H, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols[None, :] < H
    g = tl.load(gi_ptr + rows).to(tl.int64)
    offs = rows.to(tl.int64)[:, None] * H + cols[None, :]
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(bias_ptr + g[:, None] * H + cols[None, :], mask=mask).to(tl.float32)
    tl.store(y_ptr + offs, (y + b).to(y_ptr.dtype.element_ty), mask=mask)


def group_bias_add_(y: torch.Tensor, bias: torch.Tensor, group_indices: torch.Tensor) -> None:
    """
    In place y[r] += bias[group_indices[r]] for every row, without materializing
    bias[group_indices]. Returns None.

    y is (M, H), bf16 (any float dtype); bias is (G, H); group_indices is integer (M,) with values
    in [0, G). Each sum is computed in fp32 and rounded once to y's dtype, bit-identical to
    y.copy_(y + bias[group_indices]). The kernel runs when y and bias share a dtype, are contiguous
    and M is a multiple of 128; other inputs take that PyTorch expression. Deterministic.
    """
    M, H = y.shape
    if y.dtype != bias.dtype or M % 128 != 0 or not y.is_contiguous() or not bias.is_contiguous():
        y.copy_(y + bias[group_indices])
        return
    _group_bias_add_kernel[(M // 32, triton.cdiv(H, 256))](y, bias, group_indices, H, BLOCK_M=32, BLOCK_H=256)
