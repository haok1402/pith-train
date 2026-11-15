"""
Causal sliding-window attention for context parallelism.

A query attends to itself and the window_left tokens before it. Under the zigzag layout (rank r
holds blocks r and 2 * cp_size - r - 1 of 2 * cp_size, see ring_attention), the keys a block needs
from outside itself are the window_left tokens just before it: the tail of the previous rank's
front block for the front block, and the tail of the next rank's back block for the back block
(on the last rank, the tail of its own front block). So a sliding-window layer needs no ring: each
rank swaps these halos with its two neighbours and computes both blocks in one varlen flash call
over [halo, block] pairs, which flash aligns bottom-right so that the causal window lands on the
global positions. For cp_size = 4:

    chunk:  0  1  2  3  4  5  6  7
    rank:   0  1  2  3  3  2  1  0
    front halo of rank r: tail of chunk r - 1          (none on rank 0)
    back halo of rank r:  tail of chunk 2 * cp_size - r - 2

The backward reruns that call and sends each halo's key and value gradient back to the rank that
owns the tokens.

The halo covers one neighbouring block only, so it needs window_left <= block. A packed sequence
breaks that whenever a document's block is shorter than the window, so the packed form gathers K
and V from the whole CP group instead and indexes, for every block of every document, the keys of
its window; a dense sequence whose blocks are too short takes that path too, one document per
sample. Attention sinks enter the one flash call directly.
"""

from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather_into_tensor,
    batch_isend_irecv,
    get_global_rank,
    get_rank,
    get_world_size,
    irecv,
    isend,
    reduce_scatter_tensor,
)
from torch.distributed.distributed_c10d import _resolve_process_group

__all__ = ["sliding_window_attention_func", "sliding_window_attention_varlen_func"]


def grad_sink_or_empty(grad_sink: list, q: torch.Tensor) -> torch.Tensor:
    return grad_sink[0] if grad_sink else q.new_empty(0)


def empty_grad_sink_like(learnable_sink: Optional[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(learnable_sink) if learnable_sink is not None else q.new_empty(0)


def swap_halos(
    to_next: Optional[torch.Tensor],
    to_prev: Optional[torch.Tensor],
    from_next: Optional[torch.Tensor],
    from_prev: Optional[torch.Tensor],
    group: ProcessGroup,
) -> None:
    """
    Send to_next to rank + 1 and to_prev to rank - 1, receiving from_next from rank + 1 and
    from_prev from rank - 1; None where that neighbour does not exist.
    """
    cp_rank = get_rank(group)
    ops: List[P2POp] = []
    if to_next is not None:
        ops.append(P2POp(isend, to_next, get_global_rank(group, cp_rank + 1), group))
        ops.append(P2POp(irecv, from_next, get_global_rank(group, cp_rank + 1), group))
    if to_prev is not None:
        ops.append(P2POp(isend, to_prev, get_global_rank(group, cp_rank - 1), group))
        ops.append(P2POp(irecv, from_prev, get_global_rank(group, cp_rank - 1), group))
    for work in batch_isend_irecv(ops):
        work.wait()


@lru_cache(maxsize=None)
def halo_cu_seqlens(
    batch: int, block: int, front_halo: int, back_halo: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Varlen boundaries of the 2 * batch [front, back] query blocks and their [halo, block] keys.
    """
    q_lens = torch.full((2 * batch,), block, dtype=torch.int32)
    k_lens = torch.tensor([front_halo + block, back_halo + block] * batch, dtype=torch.int32)
    zero = torch.zeros(1, dtype=torch.int32)
    cu_q, cu_k = torch.cat([zero, q_lens.cumsum(0)]), torch.cat([zero, k_lens.cumsum(0)])
    return cu_q.to(device, torch.int32), cu_k.to(device, torch.int32)


def halo_keys(x: torch.Tensor, front_halo: torch.Tensor, back_halo: torch.Tensor) -> torch.Tensor:
    """
    [front halo, front block, back halo, back block] per sample, flattened to (tokens, H, D).
    """
    block = x.shape[1] // 2
    keys = torch.cat([front_halo, x[:, :block], back_halo, x[:, block:]], dim=1)
    return keys.flatten(0, 1)


def halo_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group: ProcessGroup,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    cp_rank, cp_size = get_rank(group), get_world_size(group)
    (B, S, H, _), block, h = q.shape, q.shape[1] // 2, window_left
    # Keys and values travel together, (2, B, h, Hkv, D) per halo.
    front_tail = torch.stack([k[:, block - h : block], v[:, block - h : block]])
    back_tail = torch.stack([k[:, S - h :], v[:, S - h :]])
    front_halo = front_tail.new_empty(front_tail[:, :, : h if cp_rank > 0 else 0].shape)
    last = cp_rank + 1 == cp_size
    back_halo = front_tail if last else torch.empty_like(back_tail)
    to_next, from_next = (None, None) if last else (front_tail, back_halo)
    to_prev, from_prev = (back_tail, front_halo) if cp_rank > 0 else (None, None)
    swap_halos(to_next, to_prev, from_next, from_prev, group)

    cu_q, cu_k = halo_cu_seqlens(B, block, front_halo.shape[2], h, q.device)
    out, lse, *_ = _flash_attn_fwd(
        q.flatten(0, 1),
        halo_keys(k, front_halo[0], back_halo[0]),
        halo_keys(v, front_halo[1], back_halo[1]),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=block,
        max_seqlen_k=h + block,
        softmax_scale=sm_scale,
        causal=True,
        window_size_left=h,
        window_size_right=0,
        learnable_sink=learnable_sink,
        return_lse=True,
    )
    return out.view(B, S, H, -1), lse, front_halo, back_halo


def halo_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    front_halo: torch.Tensor,
    back_halo: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group: ProcessGroup,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    cp_rank, cp_size = get_rank(group), get_world_size(group)
    (B, S, *_), block, h = q.shape, q.shape[1] // 2, window_left
    hf = front_halo.shape[2]
    cu_q, cu_k = halo_cu_seqlens(B, block, hf, h, q.device)
    dq, dk, dv, *grad_sink = _flash_attn_bwd(
        q.flatten(0, 1),
        halo_keys(k, front_halo[0], back_halo[0]),
        halo_keys(v, front_halo[1], back_halo[1]),
        out.flatten(0, 1),
        dout.contiguous().flatten(0, 1),
        lse,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=block,
        max_seqlen_k=h + block,
        softmax_scale=sm_scale,
        causal=True,
        window_size_left=h,
        window_size_right=0,
        learnable_sink=learnable_sink,
    )
    # Split [front halo, front block, back halo, back block] per sample back apart.
    dk, dv = (x.view(B, hf + h + S, *k.shape[2:]).split([hf, block, h, block], dim=1) for x in (dk, dv))  # fmt: skip
    d_front_halo, d_back_halo = torch.stack([dk[0], dv[0]]), torch.stack([dk[2], dv[2]])
    dk, dv = torch.cat([dk[1], dk[3]], dim=1), torch.cat([dv[1], dv[3]], dim=1)

    # Each halo's gradient belongs to the tokens the neighbour sent; take back our tails' share.
    last = cp_rank + 1 == cp_size
    front_tail_grad = d_back_halo if last else torch.empty_like(d_back_halo)
    back_tail_grad = torch.empty_like(d_back_halo) if cp_rank > 0 else None
    to_next, from_next = (None, None) if last else (d_back_halo, front_tail_grad)
    to_prev, from_prev = (d_front_halo, back_tail_grad) if cp_rank > 0 else (None, None)
    swap_halos(to_next, to_prev, from_next, from_prev, group)
    # A bf16 add accumulates in fp32 and rounds once, as an fp32 buffer would.
    dk[:, block - h : block] += front_tail_grad[0]
    dv[:, block - h : block] += front_tail_grad[1]
    if back_tail_grad is not None:
        dk[:, S - h :] += back_tail_grad[0]
        dv[:, S - h :] += back_tail_grad[1]
    return dq.view_as(q), dk, dv, grad_sink_or_empty(grad_sink, q)


@torch.library.custom_op("pithtrain::sliding_window_halo_fwd", mutates_args=())
def sliding_window_halo_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group_name: str,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    return halo_forward(q, k, v, sm_scale, window_left, group, learnable_sink)


@sliding_window_halo_fwd.register_fake
def _(q, k, v, sm_scale, window_left, group_name, learnable_sink):
    (B, S, H, _), h = q.shape, window_left
    hf = h if get_rank(_resolve_process_group(group_name)) > 0 else 0
    out = q.new_empty((B, S, H, v.shape[-1]))
    lse = torch.empty((H, B * S), dtype=torch.float32, device=q.device)
    front_halo = k.new_empty((2, B, hf, *k.shape[2:]))
    back_halo = k.new_empty((2, B, h, *k.shape[2:]))
    return out, lse, front_halo, back_halo


@torch.library.custom_op("pithtrain::sliding_window_halo_bwd", mutates_args=())
def sliding_window_halo_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    front_halo: torch.Tensor,
    back_halo: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group_name: str,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    args = (dout, q, k, v, out, lse, front_halo, back_halo, sm_scale, window_left, group)
    return halo_backward(*args, learnable_sink)


@sliding_window_halo_bwd.register_fake
def _(dout, q, k, v, out, lse, front_halo, back_halo, sm_scale, window_left, group_name, learnable_sink):  # fmt: skip
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), empty_grad_sink_like(learnable_sink, q)  # fmt: skip


def sliding_window_halo_setup_context(ctx, inputs, output):
    q, k, v, sm_scale, window_left, group_name, learnable_sink = inputs
    out, lse, front_halo, back_halo = output
    ctx.save_for_backward(q, k, v, out, lse, front_halo, back_halo, learnable_sink)
    ctx.sm_scale, ctx.window_left, ctx.group_name = sm_scale, window_left, group_name


def sliding_window_halo_backward(ctx, grad_out, grad_lse, grad_front_halo, grad_back_halo):
    q, k, v, out, lse, front_halo, back_halo, learnable_sink = ctx.saved_tensors
    args = (q, k, v, out, lse, front_halo, back_halo, ctx.sm_scale, ctx.window_left)
    dq, dk, dv, dsink = sliding_window_halo_bwd(grad_out, *args, ctx.group_name, learnable_sink)
    return dq, dk, dv, None, None, None, dsink if learnable_sink is not None else None


sliding_window_halo_fwd.register_autograd(
    sliding_window_halo_backward, setup_context=sliding_window_halo_setup_context
)


def window_key_index(
    cp_rank: int, cp_size: int, cu_seqlens: torch.Tensor, local_tokens: int, window_left: int
) -> Tuple[torch.Tensor, ...]:
    """
    Where the keys of every local query block live in the CP group, for the packed form.

    The local tokens are 2 * num_docs blocks, each document's front block then its back block
    (zigzag_varlen_index). Block j of a document reads the min(window_left, tokens before it) keys
    before it and its own. Returns the varlen boundaries cu_q and cu_k of those blocks and, for
    each of the num_keys = local_tokens + 2 * num_docs * window_left key slots, the rank and
    local index that holds its token, and whether the slot is used (cu_k[-1] of them are).
    Computed on the device, without reading cu_seqlens back.
    """
    cu = cu_seqlens.to(torch.int64)
    starts, lengths = cu[:-1], cu.diff()
    block = lengths // (2 * cp_size)
    local_start = starts // cp_size
    # Interleave each document's front and back block (2 * num_docs blocks).
    first = torch.stack([cp_rank * block, (2 * cp_size - 1 - cp_rank) * block], 1).flatten()
    block = block.repeat_interleave(2)
    halo = torch.minimum(first, torch.full_like(first, window_left))
    q_start = (torch.stack([local_start, local_start], 1) + torch.stack([torch.zeros_like(block[::2]), block[::2]], 1)).flatten()  # fmt: skip
    zero = cu.new_zeros(1)
    cu_q = torch.cat([q_start, cu.new_full((1,), local_tokens)])
    cu_k = torch.cat([zero, (halo + block).cumsum(0)])

    num_keys = local_tokens + (cu_seqlens.shape[0] - 1) * 2 * window_left
    slot = torch.arange(num_keys, device=cu.device)
    seg = (torch.searchsorted(cu_k, slot, right=True) - 1).clamp(max=block.shape[0] - 1)
    valid = slot < cu_k[-1]
    doc = seg // 2
    token = (first - halo)[seg] + slot - cu_k[seg]
    # The rank and local index of a document's token (the inverse of zigzag_varlen_index).
    doc_block = (cu[doc + 1] - cu[doc]) // (2 * cp_size)
    chunk = token // doc_block.clamp(min=1)
    back = chunk >= cp_size
    src_rank = torch.where(back, 2 * cp_size - 1 - chunk, chunk)
    src_index = cu[doc] // cp_size + back * doc_block + token - chunk * doc_block
    src_rank = torch.where(valid, src_rank, 0)
    src_index = torch.where(valid, src_index, 0)
    return cu_q.to(torch.int32), cu_k.to(torch.int32), src_rank, src_index, valid


def gather_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    src_rank: torch.Tensor,
    src_index: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group: ProcessGroup,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    cp_size = get_world_size(group)
    gathered = k.new_empty((cp_size, 2, *k.shape))
    all_gather_into_tensor(gathered, torch.stack([k, v]), group=group)
    k_keys, v_keys = gathered[src_rank, 0, src_index], gathered[src_rank, 1, src_index]
    out, lse, *_ = _flash_attn_fwd(
        q,
        k_keys,
        v_keys,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q.shape[0] // 2,
        max_seqlen_k=q.shape[0] // 2 + window_left,
        softmax_scale=sm_scale,
        causal=True,
        window_size_left=window_left,
        window_size_right=0,
        learnable_sink=learnable_sink,
        return_lse=True,
    )
    return out, lse, k_keys, v_keys


def gather_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k_keys: torch.Tensor,
    v_keys: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    src_rank: torch.Tensor,
    src_index: torch.Tensor,
    valid: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group: ProcessGroup,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    cp_size = get_world_size(group)
    dq, dk, dv, *grad_sink = _flash_attn_bwd(
        q,
        k_keys,
        v_keys,
        out,
        dout.contiguous(),
        lse,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q.shape[0] // 2,
        max_seqlen_k=q.shape[0] // 2 + window_left,
        softmax_scale=sm_scale,
        causal=True,
        window_size_left=window_left,
        window_size_right=0,
        learnable_sink=learnable_sink,
    )
    # Rows past cu_k[-1] are padding the kernel never wrote.
    mask = valid[:, None, None]
    local_tokens = q.shape[0]
    d_gathered = dk.new_zeros((cp_size, 2, local_tokens, *dk.shape[1:]), dtype=torch.float32)
    zeros = torch.zeros_like(src_rank)
    d_gathered.index_put_((src_rank, zeros, src_index), torch.where(mask, dk, 0).float(), accumulate=True)  # fmt: skip
    d_gathered.index_put_((src_rank, zeros + 1, src_index), torch.where(mask, dv, 0).float(), accumulate=True)  # fmt: skip
    d_local = d_gathered.new_empty(d_gathered.shape[1:])
    reduce_scatter_tensor(d_local, d_gathered, group=group)
    dk, dv = (d_local[i].to(q.dtype, copy=True) for i in range(2))
    return dq, dk, dv, grad_sink_or_empty(grad_sink, q)


@torch.library.custom_op("pithtrain::sliding_window_gather_fwd", mutates_args=())
def sliding_window_gather_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    src_rank: torch.Tensor,
    src_index: torch.Tensor,
    valid: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group_name: str,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    args = (q, k, v, cu_q, cu_k, src_rank, src_index, sm_scale, window_left, group)
    return gather_forward(*args, learnable_sink)


@sliding_window_gather_fwd.register_fake
def _(q, k, v, cu_q, cu_k, src_rank, src_index, valid, sm_scale, window_left, group_name, learnable_sink):  # fmt: skip
    (t, h, _), num_keys = q.shape, src_rank.shape[0]
    out = q.new_empty((t, h, v.shape[-1]))
    lse = torch.empty((h, t), dtype=torch.float32, device=q.device)
    return out, lse, k.new_empty((num_keys, *k.shape[1:])), v.new_empty((num_keys, *v.shape[1:]))


@torch.library.custom_op("pithtrain::sliding_window_gather_bwd", mutates_args=())
def sliding_window_gather_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k_keys: torch.Tensor,
    v_keys: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    src_rank: torch.Tensor,
    src_index: torch.Tensor,
    valid: torch.Tensor,
    sm_scale: float,
    window_left: int,
    group_name: str,
    learnable_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group = _resolve_process_group(group_name)
    args = (dout, q, k_keys, v_keys, out, lse, cu_q, cu_k, src_rank, src_index, valid)
    return gather_backward(*args, sm_scale, window_left, group, learnable_sink)


@sliding_window_gather_bwd.register_fake
def _(dout, q, k_keys, v_keys, out, lse, cu_q, cu_k, src_rank, src_index, valid, sm_scale, window_left, group_name, learnable_sink):  # fmt: skip
    t = q.shape[0]
    dk = k_keys.new_empty((t, *k_keys.shape[1:]))
    dv = v_keys.new_empty((t, *v_keys.shape[1:]))
    return torch.empty_like(q), dk, dv, empty_grad_sink_like(learnable_sink, q)


def sliding_window_gather_setup_context(ctx, inputs, output):
    q, k, v, cu_q, cu_k, src_rank, src_index, valid, sm_scale, window_left, group_name, learnable_sink = inputs  # fmt: skip
    out, lse, k_keys, v_keys = output
    ctx.save_for_backward(q, k_keys, v_keys, out, lse, cu_q, cu_k, src_rank, src_index, valid, learnable_sink)  # fmt: skip
    ctx.sm_scale, ctx.window_left, ctx.group_name = sm_scale, window_left, group_name


def sliding_window_gather_backward(ctx, grad_out, grad_lse, grad_k_keys, grad_v_keys):
    *saved, learnable_sink = ctx.saved_tensors
    args = (*saved, ctx.sm_scale, ctx.window_left, ctx.group_name, learnable_sink)
    dq, dk, dv, dsink = sliding_window_gather_bwd(grad_out, *args)
    return dq, dk, dv, *(None,) * 8, dsink if learnable_sink is not None else None


sliding_window_gather_fwd.register_autograd(
    sliding_window_gather_backward, setup_context=sliding_window_gather_setup_context
)


def sliding_window_attention_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sm_scale: float,
    window_left: int,
    cp_group: ProcessGroup,
    learnable_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Causal sliding-window attention over a packed sequence under context parallelism, each query
    attending to itself and the window_left tokens before it in its document. q (S_local, Hq, D),
    k and v (S_local, Hkv, D) hold this rank's tokens in the order zigzag_varlen_index gives, and
    cu_seqlens the int32 boundaries of the whole packed sequence, each document a multiple of
    2 * cp_size tokens (as for ring_attention_varlen_func). learnable_sink (Hq,), if given, is one
    logit per head added to every softmax denominator. Returns (S_local, Hq, Dv) in q.dtype.
    """
    cp_rank, cp_size = get_rank(cp_group), get_world_size(cp_group)
    index = window_key_index(cp_rank, cp_size, cu_seqlens, q.shape[0], window_left)
    args = (q, k, v, *index, sm_scale, window_left, cp_group.group_name, learnable_sink)
    out, *_ = sliding_window_gather_fwd(*args)
    return out


@lru_cache(maxsize=None)
def sample_cu_seqlens(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)


def sliding_window_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    window_left: int,
    cp_group: ProcessGroup,
    learnable_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Causal sliding-window attention under context parallelism, each query attending to itself and
    the window_left tokens before it. q (B, S_local, Hq, D), k and v (B, S_local, Hkv, D) are in
    the zigzag layout, S_local being S / cp_size; learnable_sink as for the packed form. Returns
    (B, S_local, Hq, Dv) in q.dtype, in the same layout.
    """
    B, S_local = q.shape[:2]
    if window_left <= S_local // 2:
        args = (q, k, v, sm_scale, window_left, cp_group.group_name, learnable_sink)
        out, *_ = sliding_window_halo_fwd(*args)
        return out
    # Each sample is one document, and its local [front, back] blocks are the packed order.
    cu_seqlens = sample_cu_seqlens(B, S_local * get_world_size(cp_group), q.device)
    flat = [x.flatten(0, 1) for x in (q, k, v)]
    out = sliding_window_attention_varlen_func(*flat, cu_seqlens, sm_scale, window_left, cp_group, learnable_sink)  # fmt: skip
    return out.view(B, S_local, *out.shape[1:])
