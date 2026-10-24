"""
Index-only grouping of the received token slots for the BF16 CuTe experts, and the gather-sum that
takes the grouped input gradient back to the received tokens.

Stage 3 receives ``gathered`` (one row per deduplicated token) plus, per token slot (one per
token and local expert), its expert and its row of ``gathered`` (``expand_idx``). The experts'
GEMMs read their rows straight from ``gathered`` through an index (gather-A), so instead of the
expanded and scattered copies of the tokens this module only produces maps:

- ``a_idx[r]``: row of ``gathered`` read by grouped row ``r``;
- ``row_of[i]``: grouped row of slot ``i`` (the expert output of slot ``i`` is row ``row_of[i]``);
- ``slot_of[r]``: slot of grouped row ``r`` (the inverse of ``row_of``);
- per token ``t``, a bitmask of its local experts and the grouped row it has in each, which the
  backward walks in expert order to sum the token's gradient rows in fp32.

Grouped rows are expert-major with no padding (``cu_seqlens``) and slots placed stably within
their expert (slot order), so every map is deterministic, and so is the gather-sum.

:func:`group_slots` is one launch: one CTA per chunk of slots ranks them stably within their
expert (the warp-level ranking of kernels/fp8_scatter_quant_sm90.py's prep kernel) and counts the experts
of all slots itself, so it needs neither the host's counts nor the other CTAs.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.bf16_expert_support import MAX_GROUPS
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import (
    bf16x2_to_f32,
    ctz64,
    f32_bits,
    f32x2_to_bf16x2,
    lanemask_lt,
    ld_b32,
    ld_b64,
    ld_v4,
    lds_b32,
    match_any,
    red_add_shared,
    red_or_u64,
    st_b32,
    st_b64,
    st_v4,
    sts_b32,
)

__all__ = ["SlotMaps", "group_slots", "gather_rows", "gather_sum", "expert_slot_segment_sum"]

_THREADS = 512
_WARPS = _THREADS // 32
_CHUNK = 2048  # slots per grouping CTA

_SUM_THREADS = 128
_SUM_TOKENS = 4  # tokens per gather-sum CTA


class _Hist:
    """
    Expert histogram of each ``_CHUNK``-slot chunk into ``hist[c * G + g]``, one CTA per chunk;
    the CTAs also zero the token bitmask the grouping launch ORs into.
    """

    def __init__(self, num_groups: int) -> None:
        self.G = num_groups

    @cute.jit
    def __call__(self, mE, mHist, mMask, stream) -> None:
        C = (mE.shape[0] + _CHUNK - 1) // _CHUNK
        self.kernel(mE, mHist, mMask).launch(grid=[C, 1, 1], block=[_THREADS, 1, 1], stream=stream)  # fmt: skip

    @cute.kernel
    def kernel(self, mE, mHist, mMask) -> None:
        G = self.G
        tid, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        m = mE.shape[0]
        C = (m + _CHUNK - 1) // _CHUNK
        T_ = mMask.shape[0]
        for k in cutlass.range((T_ + C * _THREADS - 1) // (C * _THREADS), unroll=1):
            t = (k * C + c) * _THREADS + tid
            st_b64(mMask.iterator.toint() + Int64(t) * 8, Int64(0), Int32(t < T_))
        smem = cutlass.memory.SmemAllocator()
        s_h = smem.allocate(G * 4, byte_alignment=16).toint()
        sts_b32(s_h + tid * 4, Uint32(0), Int32(tid < G))
        cute.arch.sync_threads()
        # one shared add per distinct expert in a warp: lanes of the same expert vote first
        lt_mask = lanemask_lt()
        for r in cutlass.range_constexpr(_CHUNK // _THREADS):
            i = c * _CHUNK + r * _THREADS + tid
            ok = Int32(i < m)
            e = ld_b32(mE.iterator.toint() + Int64(i) * 8, ok, Int32(-1))
            peers = match_any(e)
            first = ok & Int32(cute.arch.popc(peers & lt_mask) == 0)
            red_add_shared(s_h + e * 4, Int32(cute.arch.popc(peers)), first)
        cute.arch.sync_threads()
        dst = mHist.iterator.toint() + (Int64(c) * G + Int64(tid)) * 4
        st_b32(dst, lds_b32(s_h + tid * 4), Int32(tid < G))


class _Group:
    """Grouping launch: one CTA per ``_CHUNK`` slots, ``_THREADS`` threads.

    Each CTA sums the chunk histograms (_Hist, the launch before): all of them give every expert's
    first row, those below its chunk the chunk's base within each expert, so CTAs never wait for
    each other and the host never sends the counts; CTA 0 writes ``cu_seqlens``.
    """

    def __init__(self, num_groups: int, has_expand: bool) -> None:
        assert 0 < num_groups <= MAX_GROUPS
        self.G = num_groups
        self.has_expand = has_expand

    @cute.jit
    def __call__(self, mE, mExp, mHist, mCu, mAIdx, mRowOf, mSlotOf, mMask, mTab, stream) -> None:
        C = (mE.shape[0] + _CHUNK - 1) // _CHUNK
        args = (mE, mExp, mHist, mCu, mAIdx, mRowOf, mSlotOf, mMask, mTab)
        self.kernel(*args).launch(grid=[C, 1, 1], block=[_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mE, mExp, mHist, mCu, mAIdx, mRowOf, mSlotOf, mMask, mTab) -> None:
        G = self.G
        NT = _THREADS
        NW = _WARPS
        tid, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        warp = tid // 32
        m = mE.shape[0]
        e_base = mE.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        s_tot = smem.allocate(G * 4, byte_alignment=16).toint()  # slots per expert
        s_pre = smem.allocate(G * 4, byte_alignment=16).toint()  # slots per expert before chunk c
        s_run = smem.allocate(G * 4, byte_alignment=16).toint()  # running count within the chunk
        s_wc = smem.allocate(NW * G * 4, byte_alignment=16).toint()  # per-warp counts
        s_ex = smem.allocate(NW * G * 4, byte_alignment=16).toint()  # per-warp exclusive base

        for j in cutlass.range_constexpr((NW * G + NT - 1) // NT):
            idx = j * NT + tid
            sts_b32(s_wc + idx * 4, Uint32(0), Int32(idx < NW * G))
        sts_b32(s_tot + tid * 4, Uint32(0), Int32(tid < G))
        sts_b32(s_pre + tid * 4, Uint32(0), Int32(tid < G))
        sts_b32(s_run + tid * 4, Uint32(0), Int32(tid < G))
        cute.arch.sync_threads()

        # ---- expert totals and counts below this chunk, from the chunk histograms ----
        C = (m + _CHUNK - 1) // _CHUNK
        lo = c * _CHUNK
        if tid < G:
            tot = Int32(0)
            pre = Int32(0)
            for j in cutlass.range(C, unroll=1):
                v = ld_b32(mHist.iterator.toint() + (Int64(j) * G + Int64(tid)) * 4, Int32(1), Int32(0))  # fmt: skip
                tot = tot + v
                pre = pre + v * Int32(j < c)
            sts_b32(s_tot + tid * 4, Uint32(tot), Int32(1))
            sts_b32(s_pre + tid * 4, Uint32(pre), Int32(1))
        lt_mask = lanemask_lt()

        # ---- stable rank of each slot within its expert, inside this chunk ----
        srcs = []
        ids = []
        eids = []
        oks = []
        rls = []
        es_in = []
        for r in cutlass.range_constexpr(_CHUNK // NT):  # all loads in flight at once
            i = lo + r * NT + tid
            ok = Int32(i < m)
            e = ld_b32(e_base + Int64(i) * 8, ok, Int32(-1))
            src = i
            if cutlass.const_expr(self.has_expand):
                src = ld_b32(mExp.iterator.toint() + Int64(i) * 8, ok, Int32(0))
            es_in.append(e)
            srcs.append(src)
            ids.append(i)
            oks.append(ok)
        for r in cutlass.range_constexpr(_CHUNK // NT):
            e, ok = es_in[r], oks[r]
            peers = match_any(e)
            lt = cute.arch.popc(peers & lt_mask)
            cnt = cute.arch.popc(peers)
            es = e * ok  # safe index for invalid lanes (their stores/loads are masked)
            sts_b32(s_wc + (warp * G + es) * 4, Uint32(cnt), ok & Int32(lt == 0))
            cute.arch.sync_threads()
            # exclusive scan over warps per expert; clears the per-warp counts for the next round
            if tid < G:
                run = Int32(lds_b32(s_run + tid * 4))
                for w in cutlass.range_constexpr(NW):
                    a = s_wc + (w * G + tid) * 4
                    v = Int32(lds_b32(a))
                    sts_b32(a, Uint32(0), Int32(1))
                    sts_b32(s_ex + (w * G + tid) * 4, Uint32(run), Int32(1))
                    run = run + v
                sts_b32(s_run + tid * 4, Uint32(run), Int32(1))
            cute.arch.sync_threads()
            rl = Int32(lds_b32(s_ex + (warp * G + es) * 4)) + lt
            eids.append(es)
            rls.append(rl)

        # ---- base row of each expert for this chunk: its first row plus the lower chunks ----
        if tid < G:
            first = Int32(0)
            for g in cutlass.range(tid, unroll=1):
                first = first + Int32(lds_b32(s_tot + g * 4))
            sts_b32(s_run + tid * 4, Uint32(first + Int32(lds_b32(s_pre + tid * 4))), Int32(1))
            if c == 0:
                st_b32(mCu.iterator.toint() + Int64(tid) * 4, Uint32(first), Int32(1))
                if tid == G - 1:
                    st_b32(mCu.iterator.toint() + Int64(G) * 4, Uint32(Int32(m)), Int32(1))
        cute.arch.sync_threads()

        # ---- the maps ----
        for r in cutlass.range_constexpr(_CHUNK // NT):
            row = Int32(lds_b32(s_run + eids[r] * 4)) + rls[r]
            ok = oks[r]
            st_b32(mAIdx.iterator.toint() + Int64(row) * 4, Uint32(srcs[r]), ok)
            st_b32(mSlotOf.iterator.toint() + Int64(row) * 4, Uint32(ids[r]), ok)
            st_b32(mRowOf.iterator.toint() + Int64(ids[r]) * 4, Uint32(row), ok)
            if cutlass.const_expr(self.has_expand):
                src = Int64(srcs[r])
                red_or_u64(mMask.iterator.toint() + src * 8, Int64(1) << Int64(eids[r]), ok)
                st_b32(mTab.iterator.toint() + (src * G + Int64(eids[r])) * 4, Uint32(row), ok)


class _GatherSum:
    """``out[t] = sum over the token's local experts e (ascending) of grad[tab[t, e]]`` in fp32."""

    def __init__(self, hidden: int, num_groups: int, max_slots: int) -> None:
        assert hidden % 8 == 0
        self.H, self.G, self.K = hidden, num_groups, max_slots

    @cute.jit
    def __call__(self, mGrad, mMask, mTab, mOut, stream) -> None:
        n = (mOut.shape[0] + _SUM_TOKENS - 1) // _SUM_TOKENS
        self.kernel(mGrad, mMask, mTab, mOut).launch(
            grid=[n, 1, 1], block=[_SUM_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mGrad, mMask, mTab, mOut) -> None:
        H, G, K = self.H, self.G, self.K
        V = H // 8  # 16-byte vectors per row
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        T_ = mOut.shape[0]
        g_base = mGrad.iterator.toint()
        o_base = mOut.iterator.toint()
        for s in cutlass.range_constexpr(_SUM_TOKENS):
            t = bid * _SUM_TOKENS + s
            t_ok = Int32(t < T_)
            mask = ld_b64(mMask.iterator.toint() + Int64(t) * 8, t_ok)
            rows = []
            valid = []
            for j in cutlass.range_constexpr(K):
                ok = Int32(mask != Int64(0))
                e = ctz64(mask)
                rows.append(
                    ld_b32(mTab.iterator.toint() + (Int64(t) * G + Int64(e)) * 4, ok, Int32(0))
                )
                valid.append(ok)
                mask = mask & (mask - Int64(1))
            for q in cutlass.range_constexpr((V + _SUM_THREADS - 1) // _SUM_THREADS):
                v = q * _SUM_THREADS + tid
                v_ok = t_ok
                if cutlass.const_expr(V % _SUM_THREADS != 0):
                    v_ok = v_ok & Int32(v < V)
                acc = [cutlass.Float32(0.0)] * 8
                for j in cutlass.range_constexpr(K):
                    w = ld_v4(g_base + (Int64(rows[j]) * H + Int64(v) * 8) * 2, valid[j] & v_ok)
                    for x in cutlass.range_constexpr(4):
                        lo, hi = bf16x2_to_f32(w[x])
                        acc[2 * x] = acc[2 * x] + lo
                        acc[2 * x + 1] = acc[2 * x + 1] + hi
                out = tuple(f32x2_to_bf16x2(acc[2 * x], acc[2 * x + 1]) for x in range(4))
                st_v4(o_base + (Int64(t) * H + Int64(v) * 8) * 2, out, v_ok)


class SlotMaps(NamedTuple):
    """Index maps of one stage-3 call; see the module docstring."""

    cu_seqlens: torch.Tensor  # (G + 1,) int32
    a_idx: torch.Tensor  # (S,) int32
    row_of: torch.Tensor  # (S,) int32
    slot_of: torch.Tensor  # (S,) int32
    mask: torch.Tensor | None  # (T,) int64, None without expand_idx
    table: torch.Tensor | None  # (T * G,) int32, entries only where mask has the bit


_GROUP: dict = {}
_SUM: dict = {}


def _vec(dtype: type, align: int) -> object:
    return fake(dtype, (cute.sym_int(),), (1,), align)


def group_slots(
    expert_idxs: torch.Tensor,
    expand_idx: torch.Tensor | None,
    num_groups: int,
    num_tokens: int,
) -> SlotMaps:
    """
    Maps of the grouped expert input for slots with local experts ``expert_idxs`` ((S,) int64,
    values below ``num_groups``) reading rows ``expand_idx`` ((S,) int64, None for the identity)
    of ``num_tokens`` received tokens.
    """
    G = num_groups
    S = expert_idxs.shape[0]
    dev = expert_idxs.device
    assert 0 < G <= MAX_GROUPS
    # one allocation: the token bitmask (int64) first, then each int32 map 16-byte aligned
    has_expand = expand_idx is not None
    T4 = (num_tokens * 2 + 3) // 4 * 4 if has_expand else 0  # int32 words of the mask
    s4, g4 = (S + 3) // 4 * 4, (G + 4) // 4 * 4
    chunks = (S + _CHUNK - 1) // _CHUNK
    tab = (num_tokens * G + 3) // 4 * 4 if has_expand else 0
    buf = torch.empty(T4 + g4 + 3 * s4 + tab + chunks * G, dtype=torch.int32, device=dev)
    maps = buf[T4:]
    cu = maps[: G + 1]
    a_idx, row_of, slot_of = (maps[g4 + j * s4 : g4 + j * s4 + S] for j in range(3))
    table = maps[g4 + 3 * s4 : g4 + 3 * s4 + num_tokens * G] if has_expand else None
    hist = maps[g4 + 3 * s4 + tab :]
    mask = buf[: 2 * num_tokens].view(torch.int64) if has_expand else None
    if S == 0:
        cu.zero_()
        if mask is not None:
            mask.zero_()
        return SlotMaps(cu, a_idx, row_of, slot_of, mask, table)
    fns = _GROUP.get((G, has_expand))
    if fns is None:
        i32 = lambda: _vec(Int32, 4)  # noqa: E731
        fns = _GROUP[(G, has_expand)] = (
            cute.compile(
                _Hist(G), _vec(Int64, 8), i32(), _vec(Int64, 8), fake_stream(),
                options="--enable-tvm-ffi",
            ),
            cute.compile(
                _Group(G, has_expand), _vec(Int64, 8), _vec(Int64, 8), i32(), i32(), i32(),
                i32(), i32(), _vec(Int64, 8), i32(), fake_stream(), options="--enable-tvm-ffi",
            ),
        )  # fmt: skip
    fns[0](expert_idxs, hist, mask if has_expand else expert_idxs[:0])
    if has_expand:
        fns[1](expert_idxs, expand_idx, hist, cu, a_idx, row_of, slot_of, mask, table)
    else:
        fns[1](expert_idxs, expert_idxs, hist, cu, a_idx, row_of, slot_of, expert_idxs, a_idx)
    return SlotMaps(cu, a_idx, row_of, slot_of, mask, table)


def gather_sum(grad: torch.Tensor, maps: SlotMaps, num_groups: int, max_slots: int) -> torch.Tensor:
    """
    Gradient of the received tokens from the grouped input gradient ``grad`` ((S, H) bf16): each
    token's rows summed in fp32 in expert order, rounded once. ``max_slots`` bounds the local
    experts of one token (min(top-k, G)).
    """
    T_ = maps.mask.shape[0]
    H = grad.shape[1]
    out = grad.new_empty(T_, H)
    if T_ == 0:
        return out
    key = (H, num_groups, max_slots)
    fn = _SUM.get(key)
    if fn is None:
        bf = cutlass.BFloat16
        fn = _SUM[key] = cute.compile(
            _GatherSum(H, num_groups, max_slots), fake(bf, (cute.sym_int(), H), (H, 1)),
            _vec(Int64, 8), _vec(Int32, 4), fake(bf, (cute.sym_int(), H), (H, 1)), fake_stream(),
            options="--enable-tvm-ffi",
        )  # fmt: skip
    fn(grad, maps.mask, maps.table, out)
    return out


_SEG_THREADS = 128
_SEG_ROWS = 256  # rows per partial sum


class _SegmentSum:
    """
    Partial sums of ``_SEG_ROWS``-row chunks of each segment of rows ``x[idx[r]]`` (or ``x[r]``),
    ``r`` in ``[cu[g], cu[g + 1])``, in fp32 and row order; CTA (chunk, column block).
    """

    def __init__(self, n: int, gather: bool) -> None:
        assert n % 8 == 0
        self.N, self.gather = n, gather

    @cute.jit
    def __call__(self, mX, mIdx, mCu, mChunk, mPart, stream) -> None:
        nchunks = mPart.shape[0]
        cb = (self.N // 8 + _SEG_THREADS - 1) // _SEG_THREADS
        self.kernel(mX, mIdx, mCu, mChunk, mPart).launch(
            grid=[nchunks, cb, 1], block=[_SEG_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mX, mIdx, mCu, mChunk, mPart) -> None:
        N = self.N
        tid, _, _ = cute.arch.thread_idx()
        q, b, _ = cute.arch.block_idx()
        v = b * _SEG_THREADS + tid
        v_ok = Int32(v < N // 8)
        # chunk q covers rows [start, end) of its segment
        g = ld_b32(mChunk.iterator.toint() + Int64(q) * 8, Int32(1), Int32(0))
        k = ld_b32(mChunk.iterator.toint() + Int64(q) * 8 + 4, Int32(1), Int32(0))
        seg_end = ld_b32(mCu.iterator.toint() + Int64(g + 1) * 4, Int32(1), Int32(0))
        start = ld_b32(mCu.iterator.toint() + Int64(g) * 4, Int32(1), Int32(0)) + k * _SEG_ROWS
        n = cutlass.min(seg_end - start, Int32(_SEG_ROWS))
        acc = [cutlass.Float32(0.0)] * 8
        for r in cutlass.range(n, unroll=4):
            row = start + r
            if cutlass.const_expr(self.gather):
                row = ld_b32(mIdx.iterator.toint() + Int64(row) * 4, Int32(1), Int32(0))
            w = ld_v4(mX.iterator.toint() + (Int64(row) * N + Int64(v) * 8) * 2, v_ok)
            for x in cutlass.range_constexpr(4):
                lo, hi = bf16x2_to_f32(w[x])
                acc[2 * x] = acc[2 * x] + lo
                acc[2 * x + 1] = acc[2 * x + 1] + hi
        base = mPart.iterator.toint() + (Int64(q) * N + Int64(v) * 8) * 4
        for x in cutlass.range_constexpr(8):
            st_b32(base + x * 4, Uint32(f32_bits(acc[x])), v_ok)


_SEG: dict = {}


def expert_slot_segment_sum(
    x: torch.Tensor, cu_host: list[int], cu: torch.Tensor, idx: torch.Tensor | None = None
) -> torch.Tensor:
    """
    (G, N) fp32 sums of the rows of each segment ``[cu[g], cu[g + 1])`` of ``x`` ((rows, N) bf16),
    read as ``x[idx[r]]`` when ``idx`` is given; deterministic (fixed chunks, summed in order).
    ``cu_host`` is ``cu`` on the host (it sizes the chunk list).
    """
    G, N = len(cu_host) - 1, x.shape[1]
    lengths = [-(-(cu_host[g + 1] - cu_host[g]) // _SEG_ROWS) for g in range(G)]
    chunks = [(g, k) for g in range(G) for k in range(lengths[g])]
    if not chunks:
        return torch.zeros(G, N, dtype=torch.float32, device=x.device)
    meta = torch.tensor([v for c in chunks for v in c] + lengths, dtype=torch.int32)
    meta = meta.to(x.device, non_blocking=True)
    chunk, lengths_t = meta[: 2 * len(chunks)].view(-1, 2), meta[2 * len(chunks) :]
    part = torch.empty(len(chunks), N, dtype=torch.float32, device=x.device)
    key = (N, idx is not None)
    fn = _SEG.get(key)
    if fn is None:
        fn = _SEG[key] = cute.compile(
            _SegmentSum(N, idx is not None), fake(cutlass.BFloat16, (cute.sym_int(), N), (N, 1)),
            _vec(Int32, 4), _vec(Int32, 4), fake(Int32, (cute.sym_int(), 2), (2, 1), 8),
            fake(cutlass.Float32, (cute.sym_int(), N), (N, 1)), fake_stream(),
            options="--enable-tvm-ffi",
        )  # fmt: skip
    fn(x, idx if idx is not None else cu, cu, chunk, part)
    # the chunks of a segment are consecutive; sum them in order (an empty segment sums to 0)
    return torch.segment_reduce(part, "sum", lengths=lengths_t, initial=0.0)


_ROW_THREADS = 128
_ROWS_PER_CTA = 4


class _GatherRows:
    """``out[i] = x[idx[i]]`` for bf16 rows, 16-byte vectors."""

    def __init__(self, n: int) -> None:
        assert n % 8 == 0
        self.N = n

    @cute.jit
    def __call__(self, mX, mIdx, mOut, stream) -> None:
        grid = (mOut.shape[0] + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA
        self.kernel(mX, mIdx, mOut).launch(grid=[grid, 1, 1], block=[_ROW_THREADS, 1, 1], stream=stream)  # fmt: skip

    @cute.kernel
    def kernel(self, mX, mIdx, mOut) -> None:
        N, V = self.N, self.N // 8
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        rows = mOut.shape[0]
        for s in cutlass.range_constexpr(_ROWS_PER_CTA):
            i = bid * _ROWS_PER_CTA + s
            ok = Int32(i < rows)
            src = ld_b32(mIdx.iterator.toint() + Int64(i) * 4, ok, Int32(0))
            for q in cutlass.range_constexpr((V + _ROW_THREADS - 1) // _ROW_THREADS):
                v = q * _ROW_THREADS + tid
                v_ok = ok
                if cutlass.const_expr(V % _ROW_THREADS != 0):
                    v_ok = v_ok & Int32(v < V)
                w = ld_v4(mX.iterator.toint() + (Int64(src) * N + Int64(v) * 8) * 2, v_ok)
                st_v4(mOut.iterator.toint() + (Int64(i) * N + Int64(v) * 8) * 2, w, v_ok)


_ROWS: dict = {}


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``x[idx]`` for a contiguous (rows, N) bf16 ``x`` and an int32 ``idx``."""
    N = x.shape[1]
    out = x.new_empty(idx.shape[0], N)
    if idx.shape[0] == 0:
        return out
    fn = _ROWS.get(N)
    if fn is None:
        bf = cutlass.BFloat16
        fn = _ROWS[N] = cute.compile(
            _GatherRows(N), fake(bf, (cute.sym_int(), N), (N, 1)), _vec(Int32, 4),
            fake(bf, (cute.sym_int(), N), (N, 1)), fake_stream(), options="--enable-tvm-ffi",
        )  # fmt: skip
    fn(x, idx, out)
    return out
