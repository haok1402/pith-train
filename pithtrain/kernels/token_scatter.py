"""
Scatter of MoE tokens into the expert-grouped, padded layout of a grouped GEMM, as Triton kernels:
:func:`scatter_by_expert`.

One prep kernel computes the per-expert counts, padded sizes and cumulative offsets (a fused
bincount + pad + prefix sum); one scatter kernel copies each token to its row and zero-fills the
padding rows and the tail up to the allocation alignment. Rows within an expert are placed in
``atomic_add`` order. Both architectures run them; ``pithtrain.operators.token_scatter`` wraps them
in an autograd function.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from pithtrain.kernels.host_transfer import get_pinned_buffer

__all__ = ["GEMM_ALLOC_ALIGNMENT", "scatter_by_expert"]

GEMM_ALLOC_ALIGNMENT = 1024
"""Row alignment of the scattered output's allocation (the tail rows are zeroed)."""

# The padded sizes are copied to the host on a side stream so the copy overlaps the scatter kernel.
_copy_stream: Optional[torch.cuda.Stream] = None
_copy_event: Optional[torch.cuda.Event] = None


@triton.jit
def _compute_group_offsets_kernel(
    expert_idxs_ptr,  # [m] int64
    group_counts_ptr,  # [num_groups] int64, pre-zeroed
    grouped_mm_offs_ptr,  # [num_groups] int32, output
    ks_ptr,  # [num_groups] int32, output: per-group padded sizes
    completion_counter_ptr,  # [1] int64, pre-zeroed
    m,
    num_ctas,
    PADDING_ALIGNMENT: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Phase 1: Bincount via atomic adds (parallel across CTAs)
    for start in range(pid * BLOCK, m, num_ctas * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < m
        eids = tl.load(expert_idxs_ptr + offs, mask=mask, other=0)
        tl.atomic_add(group_counts_ptr + eids, 1, mask=mask)

    # Barrier: last CTA to finish Phase 1 runs Phase 2
    is_last = tl.atomic_add(completion_counter_ptr, 1) == num_ctas - 1
    if is_last:
        # Phase 2: Pad + prefix-sum (sequential over num_groups, trivial)
        running_sum: tl.int64 = 0
        for g in tl.static_range(NUM_GROUPS):
            count = tl.load(group_counts_ptr + g)
            padded = ((count + PADDING_ALIGNMENT - 1) // PADDING_ALIGNMENT) * PADDING_ALIGNMENT
            running_sum = running_sum + padded
            off_mm = g + tl.arange(0, 1)
            tl.store(grouped_mm_offs_ptr + off_mm, running_sum.to(tl.int32))
            tl.store(ks_ptr + off_mm, padded.to(tl.int32))


@triton.jit
def _scatter_for_grouped_gemm_kernel(
    input_tokens_ptr,  # [m, hidden_size]
    expert_idxs_ptr,  # [m] int64
    grouped_mm_offs_ptr,  # [num_groups] int32, padded cumulative offsets
    group_counters_ptr,  # [num_groups] int64, zero-initialized (for atomic positioning)
    actual_counts_ptr,  # [num_groups] int64, actual counts from prep kernel
    output_tokens_ptr,  # [m_padded, hidden_size]
    reverse_shuffle_idxs_ptr,  # [m] int64
    hidden_size,
    stride_src_m,  # input_tokens.stride(0)
    stride_dst_m,  # output_tokens.stride(0)
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

    src_base = pid * stride_src_m
    dst_base = global_pos * stride_dst_m
    for i in tl.static_range(NUM_H_BLOCKS):
        h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h_offs < hidden_size
        vals = tl.load(input_tokens_ptr + src_base + h_offs, mask=mask)
        tl.store(output_tokens_ptr + dst_base + h_offs, vals, mask=mask)

    # --- Zero padding rows for this group (distributed across programs) ---
    actual_count = tl.load(actual_counts_ptr + expert_id)
    end_offset = tl.load(grouped_mm_offs_ptr + expert_id).to(tl.int64)
    padded_count = end_offset - base_offset.to(tl.int64)
    pad_rows = padded_count - actual_count

    if pad_rows > 0:
        # Distribute pad_rows among actual_count programs in this group
        rows_per = (pad_rows + actual_count - 1) // actual_count  # ceil div
        my_start = pos_in_group * rows_per
        my_end = tl.minimum(my_start + rows_per, pad_rows)

        pad_base = base_offset.to(tl.int64) + actual_count
        for row_off in range(my_start, my_end):
            row_base = (pad_base + row_off) * stride_dst_m
            for i in tl.static_range(NUM_H_BLOCKS):
                h_offs = i * BLOCK_H + tl.arange(0, BLOCK_H)
                mask = h_offs < hidden_size
                tl.store(
                    output_tokens_ptr + row_base + h_offs,
                    tl.zeros([BLOCK_H], dtype=input_tokens_ptr.dtype.element_ty),
                    mask=mask,
                )

    # --- Zero rows [actual_M, M_rounded) for GEMM allocation alignment ---
    last_group_end = tl.load(grouped_mm_offs_ptr + num_groups - 1).to(tl.int64)
    M_rounded = (
        (last_group_end + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
    )
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
                tl.store(
                    output_tokens_ptr + row_base + h_offs,
                    tl.zeros([BLOCK_H], dtype=input_tokens_ptr.dtype.element_ty),
                    mask=mask,
                )


def _get_copy_stream_and_event(device):
    global _copy_stream, _copy_event
    if _copy_stream is None:
        _copy_stream = torch.cuda.Stream(device=device)
        _copy_event = torch.cuda.Event()
    return _copy_stream, _copy_event


def scatter_by_expert(
    input_tokens: torch.Tensor,
    expert_idxs: torch.Tensor,
    num_groups: int,
    padding_alignment: int = 128,
    group_counts: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """
    Reorder ``input_tokens`` by ``expert_idxs`` into expert-major groups, each padded with zero rows
    to a multiple of ``padding_alignment``: ``(output_tokens, reverse_shuffle_idxs,
    grouped_mm_offs, ks_tensor, ks)``.

    - ``input_tokens``: 2-D ``(m, hidden)`` with unit inner stride (any row stride, any dtype);
      ``expert_idxs``: int64 ``(m,)`` contiguous, values in ``[0, num_groups)`` (a length other than
      m raises ValueError). Neither is modified.
    - ``group_counts``: the true rows per group on the host, if known; ``ks`` then follows without
      the device-to-host copy (and host sync) of the padded sizes, and must match ``expert_idxs``.
    - ``output_tokens``: ``(sum(ks), hidden)`` row-major in ``input_tokens``' dtype, a prefix view
      of a larger allocation: group ``g`` occupies rows ``[offs[g - 1], offs[g])``, its tokens
      first, then zero rows. The rows of the allocation past ``sum(ks)`` (up to a multiple of 1024)
      are zero too.
    - ``reverse_shuffle_idxs``: int64 ``(m,)``, the output row of each input token
      (``output_tokens[reverse_shuffle_idxs] == input_tokens``).
    - ``grouped_mm_offs``: int32 ``(G,)`` cumulative padded group ends; ``ks_tensor``: int32
      ``(G,)`` padded group sizes (zero for an empty group); ``ks``: the same as a host list.
    - ``m == 0`` returns empty tensors, ``grouped_mm_offs`` and ``ks_tensor`` of shape ``(0,)``
      and ``ks == []``.

    Rows are copied exactly. Placement within a group is ``atomic_add`` order, not deterministic;
    the multiset of rows per group, the offsets and the sizes are. Same kernels on both
    architectures.
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

    # Over-allocate: tight upper bound, no GPU data needed (no .item() sync)
    max_m_padded = m + num_groups * (padding_alignment - 1)
    max_m_padded = (
        (max_m_padded + GEMM_ALLOC_ALIGNMENT - 1) // GEMM_ALLOC_ALIGNMENT * GEMM_ALLOC_ALIGNMENT
    )

    # Over-allocated output - padding rows zeroed inside the scatter kernel
    # `output_tokens` will be zeroed inside the scatter kernel.
    output_tokens = torch.empty(
        (max_m_padded, hidden_size), device=device, dtype=input_tokens.dtype
    )
    # Small buffers for prep kernel
    reverse_shuffle_idxs = torch.empty((m,), device=device, dtype=torch.int64)
    grouped_mm_offs = torch.empty((num_groups,), device=device, dtype=torch.int32)
    ks_tensor = torch.empty((num_groups,), device=device, dtype=torch.int32)
    all_zeros = torch.zeros((num_groups * 2 + 1,), device=device, dtype=torch.int64)
    group_counts_1 = all_zeros[:num_groups]
    group_counts_2 = all_zeros[num_groups : num_groups * 2]
    completion_counter = all_zeros[num_groups * 2 :]

    # The prep kernel masks its load of expert_idxs with m, the row count of input_tokens,
    # not with the length of expert_idxs. A shorter one is read out of bounds and the
    # garbage indexes group_counts, so check the length here rather than fault later.
    if expert_idxs.shape[0] != m:
        raise ValueError(f"{expert_idxs.shape[0]=} must equal the input row count {m=}")

    # Prep kernel: fused bincount + pad + cumsum (1 launch)
    BLOCK = 1024
    num_ctas = min(triton.cdiv(m, BLOCK), 32)
    _compute_group_offsets_kernel[(num_ctas,)](
        expert_idxs,
        group_counts_1,
        grouped_mm_offs,
        ks_tensor,
        completion_counter,
        m,
        num_ctas,
        PADDING_ALIGNMENT=padding_alignment,
        NUM_GROUPS=num_groups,
        BLOCK=BLOCK,
    )

    # Async D-to-H: ks_tensor is done after the prep kernel; copy it on
    # a separate stream so the memcpy overlaps with the scatter kernel.
    copy_stream, copy_event = _get_copy_stream_and_event(device)
    copy_event.record()  # marks prep kernel completion on default stream

    # Scatter kernel (on default stream, overlaps with D-to-H copy)
    BLOCK_H = 256
    NUM_H_BLOCKS = triton.cdiv(hidden_size, BLOCK_H)
    _scatter_for_grouped_gemm_kernel[(m,)](
        input_tokens,
        expert_idxs,
        grouped_mm_offs,
        group_counts_2,
        group_counts_1,  # actual counts from prep kernel
        output_tokens,
        reverse_shuffle_idxs,
        hidden_size,
        input_tokens.stride(0),
        output_tokens.stride(0),
        m,
        num_groups,
        BLOCK_H=BLOCK_H,
        NUM_H_BLOCKS=NUM_H_BLOCKS,
        GEMM_ALLOC_ALIGNMENT=GEMM_ALLOC_ALIGNMENT,
    )

    # Narrow to actual padded size (removes over-allocated tail).
    # ``actual_M = sum(ks)`` is already a multiple of ``padding_alignment``
    # (each group is rounded up to that alignment inside the prep kernel),
    # so this size is sufficient for BF16 grouped GEMM AND for DeepGEMM's
    # FP8 ``k_grouped_fp8_gemm_tn_contiguous`` which asserts the data row
    # count equals ``sum(ks)`` exactly.
    if group_counts is not None:
        # The caller knows the counts on the host (EP dispatch synced them already), so the
        # padded sizes follow without a device-to-host round trip.
        a = padding_alignment
        ks = [(c + a - 1) // a * a for c in group_counts]
    else:
        ks_cpu = get_pinned_buffer("ks", num_groups, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_cpu.copy_(ks_tensor, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_cpu.tolist()
    actual_M = sum(ks)
    output_tokens = output_tokens[:actual_M]

    return output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks_tensor, ks
