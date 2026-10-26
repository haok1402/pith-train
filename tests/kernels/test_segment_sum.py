"""
Verifier of ``segment_sum_along_rows`` against plain PyTorch.

Contract: rows of ``x`` (M, H) are sorted by group; group g spans rows
``[offs[g-1], offs[g])`` (``offs[-1] := 0``, offsets past M clamped to M); the result (G, H) holds
each group's column sums accumulated in fp32 (cast to ``out_dtype``). The summation order is fixed
(128-row tiles from the group's own start, tile sums added in row order) whatever the number of
groups, so the result is deterministic and a group's sum does not depend on the other groups:
checked run to run and against the same group summed alone, plus closeness to a float64 sum.

Moved from tests/operators/test_segment_sum.py, which compared bitwise against a copy of the
one-block-per-group Triton kernel; here the kernel's own two internal paths (many groups: one
block per group; few groups: tile partials then an ordered pass) are compared against each other
through the public call (the same group summed among many groups and alone).

    pytest tests/kernels/test_segment_sum.py -v
"""

import pytest
import torch

from pithtrain.kernels import segment_sum_along_rows


def offsets(counts):
    return torch.tensor(counts, device="cuda", dtype=torch.int64).cumsum(0).to(torch.int32)


def reference(x, offs):
    """float64 per-group column sums (offsets clamped to M)."""
    M, H = x.shape
    ends = offs.long().clamp(max=M).tolist()
    out, start = torch.zeros(len(ends), H, dtype=torch.float64, device=x.device), 0
    for g, end in enumerate(ends):
        if end > start:
            out[g] = x[start:end].double().sum(0)
        start = max(start, end)
    return out


def tolerance(x, offs):
    """fp32 accumulation error bound: a few eps32 of each group's sum of |x|."""
    return 4 * torch.finfo(torch.float32).eps * reference(x.abs(), offs) + 1e-30


def check(counts, H, dtype, out_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    M = sum(counts)
    x = (torch.randn(M, H, generator=gen, device="cuda") * 3).to(dtype)
    offs = offsets(counts)
    got = segment_sum_along_rows(x, offs, out_dtype)
    assert got.dtype == out_dtype and got.shape == (len(counts), H)
    # Deterministic: run to run.
    assert torch.equal(got, segment_sum_along_rows(x, offs, out_dtype))
    # Close to the exact sum (fp32 accumulation, then one rounding to out_dtype).
    got32 = segment_sum_along_rows(x, offs, torch.float32)
    want = reference(x, offs)
    assert ((got32.double() - want).abs() <= tolerance(x, offs)).all()
    assert torch.equal(got, got32.to(out_dtype))
    # Same bits as each group summed alone (the other internal path when there are many groups).
    start = 0
    for g, c in enumerate(counts):
        # (a fresh copy: a sliced base pointer can change Triton's specialization)
        alone = segment_sum_along_rows(x[start : start + c].clone(), offsets([c]), torch.float32)
        assert torch.equal(alone[0], got32[g]), f"group {g} differs when summed alone"
        start += c
    return got32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_experts", [4, 8, 16, 32])
def test_balanced_128_aligned(dtype, num_experts):
    # EP8 / EP4 / EP2 / EP1 of gpt-oss-20b, grouped rows padded to 128.
    check([2048] * num_experts, 2880, dtype, dtype)


@pytest.mark.parametrize("H", [2880, 5760, 100, 2048])
def test_uneven_and_empty_segments(H):
    check([0, 1, 127, 128, 129, 0, 5000, 3, 0], H, torch.bfloat16, torch.bfloat16, seed=1)


@pytest.mark.parametrize("H", [2048, 2880])
def test_many_ragged_groups(H):
    # 64 / 128 local experts with ragged counts, many empty (deepseek / qwen sizes).
    gen = torch.Generator().manual_seed(7)
    for G in (64, 128):
        counts = torch.randint(0, 300, (G,), generator=gen)
        counts[torch.randperm(G, generator=gen)[: G // 4]] = 0
        check(counts.tolist(), H, torch.bfloat16, torch.float32, seed=G)


def test_long_segment():
    check([70000, 128], 2880, torch.bfloat16, torch.float32, seed=2)


def test_float16_input():
    check([300, 0, 129], 1000, torch.float16, torch.float16, seed=3)


def test_all_empty_and_trailing_padding():
    x = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    for counts in ([0, 0, 0], [100, 0]):
        offs = offsets(counts)
        got = check_tensor(x, counts)
        if counts == [0, 0, 0]:
            assert torch.all(got == 0)
        else:
            assert torch.all(got[1] == 0)  # rows 100..255 belong to no group
    # Offsets past M are clamped out of the reduction: group 1 is rows 128..255.
    offs = torch.tensor([128, 512], device="cuda", dtype=torch.int32)
    got = segment_sum_along_rows(x, offs, torch.float32)
    assert ((got.double() - reference(x, offs)).abs() <= tolerance(x, offs)).all()
    alone = segment_sum_along_rows(x[128:].clone(), offsets([128]), torch.float32)
    assert torch.equal(got[1], alone[0])


def test_zero_rows():
    x = torch.empty(0, 2880, device="cuda", dtype=torch.bfloat16)
    got = segment_sum_along_rows(x, offsets([0, 0, 0, 0]), torch.bfloat16)
    assert got.shape == (4, 2880) and torch.all(got == 0)


def test_large_and_cancelling_values():
    gen = torch.Generator(device="cuda").manual_seed(8)
    x = (torch.randn(1000, 300, generator=gen, device="cuda") * 1e4).to(torch.bfloat16)
    x[500:] = -x[:500]  # group 0 sums to zero: only the fp32 rounding bound is left
    got = check_tensor(x, [1000, 0])
    assert torch.all(got[1] == 0)


def check_tensor(x, counts):
    offs = offsets(counts)
    got = segment_sum_along_rows(x, offs, torch.float32)
    assert ((got.double() - reference(x, offs)).abs() <= tolerance(x, offs)).all()
    return got


def test_order_is_tile_order():
    """The documented order (128-row tiles from the group start, tile sums added in row order)
    is what the kernel does: summing the halves of a group separately changes the bits, and a
    group split at a tile boundary into its tiles' sums added in order reproduces them."""
    counts = [8192] * 4
    gen = torch.Generator(device="cuda").manual_seed(9)
    x = (torch.randn(sum(counts), 2880, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    got = segment_sum_along_rows(x, offsets(counts), torch.float32)
    tiles = segment_sum_along_rows(x[:8192].clone(), offsets([128] * 64), torch.float32)
    acc = torch.zeros(2880, device="cuda")
    for t in tiles:
        acc = acc + t
    assert torch.equal(acc, got[0])
    halves = torch.stack(
        [
            segment_sum_along_rows(x[i * 8192 : i * 8192 + 4096], offsets([4096]), torch.float32)[0]
            + segment_sum_along_rows(
                x[i * 8192 + 4096 : (i + 1) * 8192], offsets([4096]), torch.float32
            )[0]  # fmt: skip
            for i in range(4)
        ]
    )
    assert not torch.equal(halves, got)
