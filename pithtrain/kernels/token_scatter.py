"""
Triton scatter of MoE tokens into the expert-grouped, zero-padded layout of a grouped GEMM
(scatter_by_expert). Rows within an expert land in atomic_add order. Both architectures run it.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.kernels.host_transfer import get_pinned_buffer

__all__ = ["GEMM_ALLOC_ALIGNMENT", "scatter_by_expert"]

GEMM_ALLOC_ALIGNMENT = 1024
"""
Row alignment of the scattered output's allocation (the tail rows are zeroed).
"""

# The padded sizes are copied to the host on a side stream so the copy overlaps the scatter kernel.
shared_copy_stream: Optional[torch.cuda.Stream] = None
shared_copy_event: Optional[torch.cuda.Event] = None


@triton.jit
def compute_group_offsets_kernel(
    expert_idxs_ptr,  # [m] int64
    group_counts_ptr,  # [num_groups] int64, pre-zeroed
    grouped_mm_offs_ptr,  # [num_groups] int32
    ks_ptr,  # [num_groups] int32, padded sizes
    completion_counter_ptr,  # [1] int64, pre-zeroed
    m,
    num_ctas,
    PADDING_ALIGNMENT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # bincount over all CTAs
    for start in range(pid * BLOCK, m, num_ctas * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < m
        eids = tl.load(expert_idxs_ptr + offs, mask=mask, other=0)
        tl.atomic_add(group_counts_ptr + eids, 1, mask=mask)

    # the last CTA to finish its counts does the pad + prefix sum
    is_last = tl.atomic_add(completion_counter_ptr, 1) == num_ctas - 1
    if is_last:
        running_sum: tl.int64 = 0
        for g in tl.static_range(NUM_GROUPS):
            count = tl.load(group_counts_ptr + g)
            padded = ((count + PADDING_ALIGNMENT - 1) // PADDING_ALIGNMENT) * PADDING_ALIGNMENT
            running_sum, off_mm = running_sum + padded, g + tl.arange(0, 1)
            tl.store(grouped_mm_offs_ptr + off_mm, running_sum.to(tl.int32))
            tl.store(ks_ptr + off_mm, padded.to(tl.int32))


@triton.jit
def scatter_for_grouped_gemm_kernel(
    input_tokens_ptr,  # [m, hidden_size]
    expert_idxs_ptr,  # [m] int64
    grouped_mm_offs_ptr,  # [num_groups] int32, padded cumulative offsets
    group_counters_ptr,  # [num_groups] int64, pre-zeroed (atomic row positions)
    actual_counts_ptr,  # [num_groups] int64, from the prep kernel
    output_tokens_ptr,  # [m_padded, hidden_size]
    reverse_shuffle_idxs_ptr,  # [m] int64
    hidden_size,
    stride_src_m,
    stride_dst_m,
    m,
    num_groups,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    GEMM_ALLOC_ALIGNMENT: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    expert_id = tl.load(expert_idxs_ptr + pid)
    pos_in_group = tl.atomic_add(group_counters_ptr + expert_id, 1)
    safe_idx = tl.maximum(expert_id - 1, 0)
    base_offset = tl.load(grouped_mm_offs_ptr + safe_idx, mask=expert_id > 0, other=0)
    global_pos = pos_in_group + base_offset.to(tl.int64)
    tl.store(reverse_shuffle_idxs_ptr + pid, global_pos)

    src_base, dst_base = pid * stride_src_m, global_pos * stride_dst_m
    for i in tl.static_range(NUM_H_BLOCKS):
        h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_size
        vals = tl.load(input_tokens_ptr + src_base + h_offs, mask=mask)
        tl.store(output_tokens_ptr + dst_base + h_offs, vals, mask=mask)

    # this group's padding rows, split among the group's programs
    actual_count = tl.load(actual_counts_ptr + expert_id)
    end_offset = tl.load(grouped_mm_offs_ptr + expert_id).to(tl.int64)
    padded_count = end_offset - base_offset.to(tl.int64)
    pad_rows = padded_count - actual_count

    if pad_rows > 0:
        rows_per = (pad_rows + actual_count - 1) // actual_count
        my_start = pos_in_group * rows_per
        my_end = tl.minimum(my_start + rows_per, pad_rows)

        pad_base = base_offset.to(tl.int64) + actual_count
        for row_off in range(my_start, my_end):
            row_base = (pad_base + row_off) * stride_dst_m
            for i in tl.static_range(NUM_H_BLOCKS):
                h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
                mask = h_offs < hidden_size
                tl.store(output_tokens_ptr + row_base + h_offs, tl.zeros([BLOCK_H], dtype=input_tokens_ptr.dtype.element_ty), mask=mask)

    # rows [actual_M, M_rounded) of the allocation, split among all programs
    last_group_end = tl.load(grouped_mm_offs_ptr + num_groups - 1).to(tl.int64)
    M_rounded = ((last_group_end + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT)
    align_pad = M_rounded - last_group_end

    if align_pad > 0:
        rows_per = (align_pad + m - 1) // m
        my_start = pid * rows_per
        my_end = tl.minimum(my_start + rows_per, align_pad)
        for row_off in range(my_start, my_end):
            row_base = (last_group_end + row_off) * stride_dst_m
            for i in tl.static_range(NUM_H_BLOCKS):
                h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
                mask = h_offs < hidden_size
                tl.store(output_tokens_ptr + row_base + h_offs, tl.zeros([BLOCK_H], dtype=input_tokens_ptr.dtype.element_ty), mask=mask)


def get_copy_stream_and_event(device):
    global shared_copy_stream, shared_copy_event
    if shared_copy_stream is None:
        shared_copy_stream = torch.cuda.Stream(device=device)
        shared_copy_event = torch.cuda.Event()
    return shared_copy_stream, shared_copy_event


def scatter_by_expert(input_tokens: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, padding_alignment: int = 128, group_counts: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Reorder input_tokens by expert_idxs into expert-major groups, each padded with zero rows to
    a multiple of padding_alignment. Returns (output_tokens, reverse_shuffle_idxs,
    grouped_mm_offs, ks_tensor, ks).

    - input_tokens: (m, hidden) with unit inner stride (any row stride, any dtype). expert_idxs:
      int64 (m,) contiguous, values in [0, num_groups); a length other than m raises ValueError.
      Inputs are not modified.
    - group_counts: the true rows per group on the host, if known; ks then follows without the
      device-to-host copy (and host sync), and must match expert_idxs.
    - output_tokens: (sum(ks), hidden) row-major in input_tokens' dtype, a prefix view of an
      allocation whose rows past sum(ks), up to a multiple of 1024, are zero too. Group g occupies
      rows [offs[g - 1], offs[g]), its tokens first, then zero rows.
    - reverse_shuffle_idxs: int64 (m,), the output row of each input token.
    - grouped_mm_offs: int32 (G,) cumulative padded group ends. ks_tensor: int32 (G,) padded group
      sizes (zero for an empty group). ks: the same as a host list.
    - m == 0 returns empty tensors, grouped_mm_offs and ks_tensor of shape (0,), and ks == [].

    Rows are copied exactly. Placement within a group is atomic_add order, not deterministic.
    """
    m = input_tokens.shape[0]
    hidden_size = input_tokens.shape[1]
    device = input_tokens.device

    if m == 0:
        output_tokens = torch.empty((0, hidden_size), device=device, dtype=input_tokens.dtype)
        reverse_shuffle_idxs = torch.empty((0,), device=device, dtype=torch.int64)
        grouped_mm_offs = torch.empty((0,), device=device, dtype=torch.int32)
        ks_tensor = torch.empty((0,), device=device, dtype=torch.int32)
        return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, []

    # upper bound on sum(ks) without a device sync
    max_m_padded = m + num_groups * (padding_alignment - 1)
    max_m_padded = ((max_m_padded + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT)

    # the scatter kernel zeroes the padding and tail rows
    output_tokens = torch.empty((max_m_padded, hidden_size), device=device, dtype=input_tokens.dtype)
    reverse_shuffle_idxs = torch.empty((m,), device=device, dtype=torch.int64)
    grouped_mm_offs = torch.empty((num_groups,), device=device, dtype=torch.int32)
    ks_tensor = torch.empty((num_groups,), device=device, dtype=torch.int32)
    all_zeros = torch.zeros((num_groups * 2 + 1,), device=device, dtype=torch.int64)
    group_counts_1, group_counts_2, completion_counter = all_zeros[:num_groups], all_zeros[num_groups : num_groups * 2], all_zeros[num_groups * 2 :]

    # The prep kernel masks its expert_idxs loads with m: a shorter expert_idxs would be read out
    # of bounds and the garbage would index group_counts.
    if expert_idxs.shape[0] != m:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {m=}")

    BLOCK = 1024
    num_ctas = min(triton.cdiv(m, BLOCK), 32)
    compute_group_offsets_kernel[(num_ctas,)](expert_idxs, group_counts_1, grouped_mm_offs, ks_tensor, completion_counter, m, num_ctas, PADDING_ALIGNMENT=padding_alignment, NUM_GROUPS=num_groups, BLOCK=BLOCK)

    # ks_tensor is final after the prep kernel: its copy to the host overlaps the scatter kernel
    copy_stream, copy_event = get_copy_stream_and_event(device)
    copy_event.record()

    BLOCK_H = 256
    NUM_H_BLOCKS = triton.cdiv(hidden_size, BLOCK_H)
    scatter_for_grouped_gemm_kernel[(m,)](input_tokens, expert_idxs, grouped_mm_offs, group_counts_2, group_counts_1, output_tokens, reverse_shuffle_idxs, hidden_size, input_tokens.stride(0), output_tokens.stride(0), m, num_groups, BLOCK_H=BLOCK_H, NUM_H_BLOCKS=NUM_H_BLOCKS, GEMM_ALLOC_ALIGNMENT=GEMM_ALLOC_ALIGNMENT)

    if group_counts is not None:
        a = padding_alignment
        ks = [(c + a - 1) // a * a for c in group_counts]
    else:
        ks_cpu = get_pinned_buffer("ks", num_groups, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_cpu.copy_(ks_tensor, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_cpu.tolist()
    # trimmed to sum(ks) exactly: DeepGEMM's k_grouped_fp8_gemm_tn_contiguous asserts it
    actual_M = sum(ks)
    output_tokens = output_tokens[:actual_M]

    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks
