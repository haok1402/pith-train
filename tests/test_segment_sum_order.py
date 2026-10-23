"""
The expert bias gradient (indexed_bias_add._segment_sum_along_rows) is computed tile-parallel, yet
must keep every column's summation order: torch.equal to the one-block-per-group kernel it
replaced, copied below as the reference.

    pytest tests/test_segment_sum_order.py -v
"""

import pytest
import torch
import triton
import triton.language as tl

from pithtrain.operators.indexed_bias_add import _segment_sum_along_rows


@triton.jit
def _reference_kernel(x_ptr, offs_padded_ptr, out_ptr, H: tl.constexpr, HEAD_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr):  # fmt: skip
    g = tl.program_id(0)
    h_blk = tl.program_id(1)
    start = tl.load(offs_padded_ptr + g)
    end = tl.load(offs_padded_ptr + g + 1)
    h_offs = h_blk * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    h_mask = h_offs < H
    acc = tl.zeros((HEAD_BLOCK,), dtype=tl.float32)
    for m_start in tl.range(start, end, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < end
        ptrs = x_ptr + m_offs[:, None] * H + h_offs[None, :]
        x = tl.load(ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + g * H + h_offs, acc, mask=h_mask)


def reference(x, offs, out_dtype):
    M, H = x.shape
    E = offs.shape[0]
    offs_padded = torch.zeros(E + 1, dtype=torch.int32, device=x.device)
    offs_padded[1:].copy_(offs.to(torch.int32).clamp(max=M))
    out = torch.empty((E, H), dtype=torch.float32, device=x.device)
    _reference_kernel[(E, triton.cdiv(H, 128))](x, offs_padded, out, H=H, HEAD_BLOCK=128, BLOCK_M=128)  # fmt: skip
    return out.to(out_dtype) if out_dtype is not torch.float32 else out


def offsets(counts, device="cuda"):
    return torch.tensor(counts, device=device).cumsum(0).to(torch.int32)


def check(counts, H, dtype, out_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    M = sum(counts)
    x = (torch.randn(M, H, generator=g, device="cuda") * 3).to(dtype)
    offs = offsets(counts)
    got = _segment_sum_along_rows(x, offs, out_dtype)
    want = reference(x, offs, out_dtype)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts", [4, 8, 16, 32])
def test_balanced_128_aligned(dtype, num_experts):
    # EP8 / EP4 / EP2 / EP1 of gpt-oss-20b, grouped rows padded to 128.
    check([2048] * num_experts, 2880, dtype, torch.bfloat16 if dtype == torch.bfloat16 else dtype)


@pytest.mark.parametrize("H", [2880, 5760, 100])
def test_uneven_and_empty_segments(H):
    check([0, 1, 127, 128, 129, 0, 5000, 3, 0], H, torch.bfloat16, torch.bfloat16, seed=1)


def test_long_segment():
    check([70000, 128], 2880, torch.bfloat16, torch.float32, seed=2)


def test_all_empty_and_trailing_padding():
    x = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    for counts in ([0, 0, 0], [100, 0]):
        offs = offsets(counts)
        assert torch.equal(_segment_sum_along_rows(x, offs, torch.float32), reference(x, offs, torch.float32))  # fmt: skip
    # Offsets past M are clamped out of the reduction.
    offs = torch.tensor([128, 512], device="cuda", dtype=torch.int32)
    assert torch.equal(_segment_sum_along_rows(x, offs, torch.float32), reference(x, offs, torch.float32))  # fmt: skip


def test_reordered_sum_is_detected():
    """A split-row variant (sum the two halves of a group separately) changes the bits."""
    counts = [8192] * 4
    x = (torch.randn(sum(counts), 2880, device="cuda") * 3).to(torch.bfloat16)
    offs = offsets(counts)
    want = reference(x, offs, torch.float32)
    halves = [reference(x[i * 8192 : i * 8192 + 4096], offsets([4096]), torch.float32)
              + reference(x[i * 8192 + 4096 : (i + 1) * 8192], offsets([4096]), torch.float32)
              for i in range(4)]  # fmt: skip
    assert not torch.equal(torch.cat(halves), want)
