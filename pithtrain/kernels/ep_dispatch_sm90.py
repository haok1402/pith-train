"""
CuTe-DSL kernels for the index math of expert-parallel dispatch (SM90), launched through TVM-FFI.

:func:`dispatch_index` (sender side) replaces the three Triton kernels behind
``fused_dedup_prepare_dispatch`` with one launch (``_DispatchIndex``): each CTA builds the
histograms of experts and of first visits per EP rank of its chunks of tokens in shared memory and
publishes them; after a grid-wide barrier (the grid never exceeds the SM count, so all CTAs are
resident) every CTA scans the totals into the expert / EP-rank starts and ranks its tokens within
their expert and within the EP ranks they visit *stably* (token order) with per-warp lane
bitmasks, writing ``idxs``, ``expand_idx``, ``dispatch_token_idxs`` and ``inv_idxs`` (the inverse
permutation of ``idxs``).

The placement is the one ``argsort(stable=True)`` / ``nonzero`` give, so every output is
deterministic; the Triton kernels place in ``atomic_add`` order instead. A token's top-k experts
must be distinct (as ``torch.topk`` returns them): the lane bitmasks count each expert once.

:func:`expand_index` (receiver side, after the expand_idx all-to-all): per received token
slot, its local expert and its expand_idx made global. Each CTA scans the received per-(sender,
expert) segment counts and per-sender dedup counts in shared memory; each slot then finds its
segment by a binary search instead of the Triton kernel's unrolled pass over every segment.

The feature entries :func:`fused_dedup_prepare_dispatch_sm90` and :func:`expert_and_expand_idx_sm90`
run these kernels for the inputs they cover and the Triton kernels of ``ep_dispatch_sm100``
otherwise. They call ``dispatch_index`` / ``expand_index`` through this module's globals.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.arch import sm_count
from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import (
    atom_add_u64,
    atom_add_u64_pred,
    fence_gpu,
    lanemask_lt,
    ld_b32,
    ld_relaxed,
    lds_b32,
    red_add_shared,
    red_or_shared,
    spin_epoch,
    st_b32,
    st_b64,
    sts_b32,
)
from pithtrain.kernels.ep_dispatch_sm100 import (
    expert_and_expand_idx_sm100,
    fused_dedup_prepare_dispatch_sm100,
)

__all__ = [
    "dispatch_index",
    "expand_index",
    "expert_and_expand_idx_sm90",
    "fused_dedup_prepare_dispatch_sm90",
    "supports_expand",
    "supports_index",
]

_THREADS = 256
_WARPS = _THREADS // 32
_ROUNDS = 2
_CHUNK = _ROUNDS * _THREADS  # tokens per CTA of the index kernels
_MAX_BINS = 1024  # experts + EP ranks the kernel's lane bitmasks cover (shared memory < 48 KB)
_SLOTS_PER_THREAD = 4
_MAX_SEGMENTS = 8 * _THREADS


def _exclusive_scan(s_in: Int32, s_out: Int32, n: int, s_tmp: Int32, tid: Int32) -> None:
    """Exclusive prefix sum of ``n`` int32 in shared memory (``s_in`` -> ``s_out``, which may be
    the same) by the whole CTA: serial sums over consecutive runs per thread, then a Hillis-Steele
    scan of the per-thread sums in ``s_tmp`` (``2 * _THREADS`` words). Ends with a barrier."""
    spt = (n + _THREADS - 1) // _THREADS
    vals = []
    part = Int32(0)
    for i in range(spt):
        idx = tid * spt + i
        ok = Int32(idx < n)
        v = Int32(lds_b32(s_in + cutlass.min(idx, n - 1) * 4)) * ok
        vals.append(v)
        part = part + v
    sts_b32(s_tmp + tid * 4, Uint32(part), Int32(1))
    cute.arch.sync_threads()
    src = 0
    for step in range(_THREADS.bit_length() - 1):
        off = 1 << step
        has = Int32(tid >= off)
        v = Int32(lds_b32(s_tmp + (src * _THREADS + tid) * 4))
        w = Int32(lds_b32(s_tmp + (src * _THREADS + tid - off * has) * 4))
        sts_b32(s_tmp + ((1 - src) * _THREADS + tid) * 4, Uint32(v + w * has), Int32(1))
        cute.arch.sync_threads()
        src = 1 - src
    run = Int32(lds_b32(s_tmp + (src * _THREADS + tid) * 4)) - part
    for i in range(spt):
        idx = tid * spt + i
        sts_b32(s_out + idx * 4, Uint32(run), Int32(idx < n))
        run = run + vals[i]
    cute.arch.sync_threads()


# ---------------------------------------------------------------------------
# Sender side: dispatch index (count kernel + placement kernel)
# ---------------------------------------------------------------------------


class _DispatchIndex:
    """One launch. Bins ``[0, NE)`` count slots per expert, bins ``[NE, NE + EP)`` count first
    visits (unique tokens) per EP rank.

    The launch is cooperative (and the grid capped at the SM count), so every CTA is resident at
    once, which the barrier below requires; each CTA owns chunks
    ``bid, bid + G, ...`` of ``_CHUNK`` tokens. Phase 1 publishes every owned chunk's histogram,
    then a grid-wide barrier (an ``epoch << 32 | arrivals`` counter that the last arrival moves to
    the next epoch, so it never needs resetting). Phase 2: every CTA sums the histograms into the
    totals and scans them into the expert / EP-rank starts, and for each owned chunk adds the
    histograms of the lower chunks to get the chunk's bases, then ranks and places its tokens.

    Output layout (one int64 tensor, ``mk = m * k``): ``idxs`` ``[0, mk)``, ``expand_idx``
    ``[mk, 2mk)``, ``inv_idxs`` ``[2mk, 3mk)``, ``dispatch_token_idxs`` ``[3mk, 3mk + m * EP)``,
    ``send_meta`` after it. Scratch: ``hist`` (``C * NB`` int32), ``ctr`` (int64).
    """

    def __init__(self, k: int, ep_size: int, experts_per_rank: int, max_ctas: int) -> None:
        self.K, self.EP, self.EPR = k, ep_size, experts_per_rank
        self.NE = ep_size * experts_per_rank
        self.NB = self.NE + ep_size
        self.max_ctas = max_ctas
        assert self.NB <= _MAX_BINS and ep_size <= 32

    @cute.jit
    def __call__(
        self,
        mE: cute.Tensor,
        mOut: cute.Tensor,
        mCounts: cute.Tensor,
        mHist: cute.Tensor,
        mCtr: cute.Tensor,
        stream,
    ) -> None:
        C = (mE.shape[0] + _CHUNK - 1) // _CHUNK
        G = cutlass.min(C, self.max_ctas)
        # cooperative: the runtime guarantees all G CTAs are co-resident (the grid barrier needs it)
        self.kernel(mE, mOut, mCounts, mHist, mCtr).launch(
            grid=[G, 1, 1], block=[_THREADS, 1, 1], stream=stream, cooperative=True
        )

    def _load_ids(self, mE: cute.Tensor, c: Int32, tid: Int32) -> tuple:
        """Expert ids of this thread's tokens (``_ROUNDS`` x ``K``) and the first-visit flags."""
        m = mE.shape[0]
        stride = Int64(mE.stride[0])
        base = mE.iterator.toint()
        ids, firsts, oks, toks = [], [], [], []
        for r in range(_ROUNDS):
            t = c * _CHUNK + r * _THREADS + tid
            ok = Int32(t < m)
            row = []
            for j in range(self.K):
                row.append(ld_b32(base + (Int64(t) * stride + j) * 8, ok, Int32(0)))
            visited = Int32(0)
            first = []
            for j in range(self.K):
                bit = Int32(1) << (row[j] // self.EPR)
                first.append(ok & Int32((visited & bit) == 0))
                visited = visited | bit
            ids.append(row)
            firsts.append(first)
            oks.append(ok)
            toks.append(t)
        return ids, firsts, oks, toks

    @cute.kernel
    def kernel(
        self,
        mE: cute.Tensor,
        mOut: cute.Tensor,
        mCounts: cute.Tensor,
        mHist: cute.Tensor,
        mCtr: cute.Tensor,
    ) -> None:
        NE, NB, EP, EPR, K = self.NE, self.NB, self.EP, self.EPR, self.K
        NBJ = (NB + _THREADS - 1) // _THREADS
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        G, _, _ = cute.arch.grid_dim()
        warp = tid // 32
        lane = tid - warp * 32
        m = mE.shape[0]
        C = (m + _CHUNK - 1) // _CHUNK
        mk = Int64(m) * K
        out = mOut.iterator.toint()
        idxs_p, expand_p, inv_p = out, out + mk * 8, out + 2 * mk * 8
        dti_p = out + 3 * mk * 8
        hist = mHist.iterator.toint()
        ctr = mCtr.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        s_a = smem.allocate(NB * 4, byte_alignment=16).toint()  # chunk histogram, then totals
        s_b = smem.allocate(NB * 4, byte_alignment=16).toint()  # exclusive starts of the totals
        s_run = smem.allocate(NB * 4, byte_alignment=16).toint()  # next free position per bin
        s_bits = smem.allocate(NB * _WARPS * 4, byte_alignment=16).toint()  # lane masks
        s_tmp = smem.allocate(2 * _THREADS * 4, byte_alignment=16).toint()

        # ---- phase 1: histograms of the owned chunks ----
        for c in cutlass.range(bid, C, G, unroll=1):
            for j in cutlass.range_constexpr(NBJ):
                b = j * _THREADS + tid
                sts_b32(s_a + b * 4, Uint32(0), Int32(b < NB))
            cute.arch.sync_threads()
            ids, firsts, oks, _ = self._load_ids(mE, c, tid)
            for r in cutlass.range_constexpr(_ROUNDS):
                for j in cutlass.range_constexpr(K):
                    e = ids[r][j]
                    red_add_shared(s_a + e * 4, Int32(1), oks[r])
                    red_add_shared(s_a + (NE + e // EPR) * 4, Int32(1), firsts[r][j])
            cute.arch.sync_threads()
            for j in cutlass.range_constexpr(NBJ):
                b = j * _THREADS + tid
                hv = lds_b32(s_a + cutlass.min(b, NB - 1) * 4)
                st_b32(hist + (Int64(c) * NB + b) * 4, hv, Int32(b < NB))
            cute.arch.sync_threads()

        # ---- grid barrier: every chunk's histogram is published ----
        fence_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            old = atom_add_u64(ctr, Int64(1))
            arrived = Int32(old & 0xFFFFFFFF)
            epoch = Int32(old >> 32)
            last = Int32(arrived == G - 1)
            # the last arrival starts the next epoch and zeroes the arrivals; the others wait for it
            atom_add_u64_pred(ctr, (Int64(1) << 32) - Int64(G), last)
            spin_epoch(ctr, epoch, Int32(1) - last)
        cute.arch.sync_threads()
        fence_gpu()

        # ---- totals and their exclusive starts (each CTA) ----
        for j in cutlass.range_constexpr(NBJ):
            b = j * _THREADS + tid
            ok = Int32(b < NB)
            tot = Int32(0)
            for cc in cutlass.range(C, unroll=4):
                tot = tot + ld_relaxed(hist + (Int64(cc) * NB + b) * 4, ok)
            sts_b32(s_a + cutlass.min(b, NB - 1) * 4, Uint32(tot), ok)
        cute.arch.sync_threads()
        _exclusive_scan(s_a, s_b, NB, s_tmp, tid)

        # tokens_per_ep_rank, dedup_tokens_per_gpu, send_meta: CTA 0
        if bid == 0:
            cnt = mCounts.iterator.toint()
            if tid < EP:
                lo = Int32(lds_b32(s_b + tid * EPR * 4))
                hi = Int32(lds_b32(s_b + (tid + 1) * EPR * 4))
                st_b64(cnt + Int64(tid) * 8, Int64(hi - lo), Int32(1))
                d = Int32(lds_b32(s_a + (NE + tid) * 4))
                st_b64(cnt + Int64(EP + tid) * 8, Int64(d), Int32(1))
            meta = out + (3 * mk + Int64(m) * EP) * 8
            for j in cutlass.range_constexpr((EP * (EPR + 1) + _THREADS - 1) // _THREADS):
                i = j * _THREADS + tid
                ok_i = Int32(i < EP * (EPR + 1))
                g = i // (EPR + 1)
                e = i - g * (EPR + 1)
                bb = g * EPR + e + (NE + g - g * EPR - e) * Int32(e == EPR)
                cnt_b = Int32(lds_b32(s_a + cutlass.min(bb, NB - 1) * 4))
                st_b64(meta + Int64(i) * 8, Int64(cnt_b), ok_i)

        # ---- phase 2: place the owned chunks ----
        lt = lanemask_lt()
        bit = Uint32(1) << Uint32(lane)
        d0 = Int32(lds_b32(s_b + NE * 4))  # start of the EP-rank bins in the scan
        for c in cutlass.range(bid, C, G, unroll=1):
            # chunk bases: the expert's global start (EP-rank bins: 0, expand_idx is local to the
            # rank's dedup chunk) plus everything the lower chunks put in the bin
            for j in cutlass.range_constexpr(NBJ):
                b = j * _THREADS + tid
                ok = Int32(b < NB)
                bs = cutlass.min(b, NB - 1)
                run = Int32(lds_b32(s_b + bs * 4)) * Int32(b < NE)
                for cc in cutlass.range(c, unroll=4):
                    run = run + ld_relaxed(hist + (Int64(cc) * NB + b) * 4, ok)
                sts_b32(s_run + bs * 4, Uint32(run), ok)
            for j in cutlass.range_constexpr((NB * _WARPS + _THREADS - 1) // _THREADS):
                w = j * _THREADS + tid
                sts_b32(s_bits + w * 4, Uint32(0), Int32(w < NB * _WARPS))
            cute.arch.sync_threads()
            ids, firsts, oks, toks = self._load_ids(mE, c, tid)
            for r in cutlass.range_constexpr(_ROUNDS):
                for j in cutlass.range_constexpr(K):
                    e = ids[r][j]
                    red_or_shared(s_bits + (e * _WARPS + warp) * 4, bit, oks[r])
                    red_or_shared(s_bits + ((NE + e // EPR) * _WARPS + warp) * 4, bit, firsts[r][j])  # noqa: E501
                cute.arch.sync_threads()
                t = toks[r]
                for j in cutlass.range_constexpr(K):
                    e = ids[r][j]
                    b_e = e * _WARPS
                    b_g = (NE + e // EPR) * _WARPS
                    re = Int32(0)
                    rg = Int32(0)
                    for w in cutlass.range_constexpr(_WARPS):
                        below = Int32(w < warp)
                        re = re + Int32(cute.arch.popc(lds_b32(s_bits + (b_e + w) * 4))) * below
                        rg = rg + Int32(cute.arch.popc(lds_b32(s_bits + (b_g + w) * 4))) * below
                    re = re + Int32(cute.arch.popc(lds_b32(s_bits + (b_e + warp) * 4) & lt))
                    rg = rg + Int32(cute.arch.popc(lds_b32(s_bits + (b_g + warp) * 4) & lt))
                    p = Int32(lds_b32(s_run + e * 4)) + re
                    dr = Int32(lds_b32(s_run + (NE + e // EPR) * 4)) + rg
                    gs = Int32(lds_b32(s_b + (NE + e // EPR) * 4)) - d0
                    flat = Int64(t) * K + j
                    ok = oks[r]
                    st_b64(idxs_p + Int64(p) * 8, flat, ok)
                    st_b64(expand_p + Int64(p) * 8, Int64(dr), ok)
                    st_b64(inv_p + flat * 8, Int64(p), ok)
                    st_b64(dti_p + Int64(gs + dr) * 8, Int64(t), firsts[r][j])
                cute.arch.sync_threads()
                # advance the running positions by this round's counts and clear the masks
                for j in cutlass.range_constexpr(NBJ):
                    b = j * _THREADS + tid
                    if b < NB:
                        n = Int32(0)
                        for w in cutlass.range_constexpr(_WARPS):
                            n = n + Int32(cute.arch.popc(lds_b32(s_bits + (b * _WARPS + w) * 4)))
                            sts_b32(s_bits + (b * _WARPS + w) * 4, Uint32(0), Int32(1))
                        sts_b32(s_run + b * 4, Uint32(Int32(lds_b32(s_run + b * 4)) + n), Int32(1))  # noqa: E501
                cute.arch.sync_threads()


# ---------------------------------------------------------------------------
# Receiver side: expert and global expand index
# ---------------------------------------------------------------------------


class _ExpertAndExpandIdx:
    """One CTA per ``_THREADS * _SLOTS_PER_THREAD`` received slots."""

    def __init__(self, ep_size: int, experts_per_rank: int) -> None:
        self.ep = ep_size
        self.epr = experts_per_rank
        self.nseg = ep_size * experts_per_rank
        assert 0 < self.nseg <= _MAX_SEGMENTS and ep_size <= _THREADS

    @cute.jit
    def __call__(self, mMeta: cute.Tensor, mRecv: cute.Tensor, mOut: cute.Tensor, stream) -> None:
        per_cta = _THREADS * _SLOTS_PER_THREAD
        grid = (mRecv.shape[0] + per_cta - 1) // per_cta
        self.kernel(mMeta, mRecv, mOut).launch(
            grid=[grid, 1, 1], block=[_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mMeta: cute.Tensor, mRecv: cute.Tensor, mOut: cute.Tensor) -> None:
        EP, EPR, NSEG = self.ep, self.epr, self.nseg
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        total = mRecv.shape[0]
        meta = mMeta.iterator.toint()
        recv = mRecv.iterator.toint()
        out = mOut.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        s_end = smem.allocate(NSEG * 4, byte_alignment=16).toint()  # segment starts, then ends
        s_dst = smem.allocate(EP * 4, byte_alignment=16).toint()  # exclusive dedup starts
        s_tmp = smem.allocate(2 * _THREADS * 4, byte_alignment=16).toint()

        for j in cutlass.range_constexpr((NSEG + _THREADS - 1) // _THREADS):
            seg = j * _THREADS + tid
            ok = Int32(seg < NSEG)
            sender = seg // EPR
            v = ld_b32(meta + Int64(seg + sender) * 8, ok, Int32(0))  # (EPR + 1) per sender
            sts_b32(s_end + cutlass.min(seg, NSEG - 1) * 4, Uint32(v), ok)
        if tid < EP:
            d = Int32(0)
            for g in cutlass.range(tid, unroll=1):
                d = d + ld_b32(meta + Int64(g * (EPR + 1) + EPR) * 8, Int32(1), Int32(0))
            sts_b32(s_dst + tid * 4, Uint32(d), Int32(1))
        cute.arch.sync_threads()
        _exclusive_scan(s_end, s_end, NSEG, s_tmp, tid)

        # ---- each slot: segment by binary search over the starts, then expert + global index ----
        top = 1 << (NSEG.bit_length() - 1)
        for i in cutlass.range_constexpr(_SLOTS_PER_THREAD):
            s = (bid * _SLOTS_PER_THREAD + i) * _THREADS + tid
            ok = Int32(s < total)
            pos = Int32(0)  # last segment whose start is <= s (empty segments share starts)
            b = top
            while b > 0:
                nxt = pos + b
                inb = Int32(nxt < NSEG)
                st = Int32(lds_b32(s_end + cutlass.min(nxt, NSEG - 1) * 4))
                pos = pos + b * (inb & Int32(st <= s))
                b //= 2
            sender = pos // EPR
            expert = pos - sender * EPR
            val = ld_b32(recv + Int64(s) * 8, ok, Int32(0))
            glob = val + Int32(lds_b32(s_dst + sender * 4))
            st_b64(out + Int64(s) * 8, Int64(expert), ok)
            st_b64(out + (Int64(total) + Int64(s)) * 8, Int64(glob), ok)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------

_INDEX: dict = {}
_EXPAND: dict = {}


class _Scratch:
    """Per-device scratch of the index kernel: chunk histograms and the barrier counter, whose
    arrivals return to 0 at every launch (the epoch in its high word only grows)."""

    def __init__(self, device: torch.device, chunks: int, bins: int) -> None:
        self.chunks, self.bins = chunks, bins
        self.hist = torch.empty(chunks * bins, dtype=torch.int32, device=device)
        self.ctr = torch.zeros(1, dtype=torch.int64, device=device)


_SCRATCH: dict = {}


def _scratch(device: torch.device, chunks: int, bins: int) -> _Scratch:
    s = _SCRATCH.get(device.index)
    if s is None or s.chunks < chunks or s.bins < bins:
        chunks = max(chunks, s.chunks if s is not None else 0, 64)
        bins = max(bins, s.bins if s is not None else 0)
        s = _SCRATCH[device.index] = _Scratch(device, chunks, bins)
    return s


def _compile_index(k: int, ep_size: int, experts_per_rank: int, max_ctas: int) -> Callable:
    i32, i64 = Int32, Int64
    return cute.compile(
        _DispatchIndex(k, ep_size, experts_per_rank, max_ctas),
        fake(i64, (cute.sym_int(), k), (cute.sym_int64(), 1), 8),
        fake(i64, (cute.sym_int(),), (1,), 8),
        fake(i64, (2 * ep_size,), (1,), 8),
        fake(i32, (cute.sym_int(),), (1,), 4),
        fake(i64, (1,), (1,), 8),
        fake_stream(),
        options="--enable-tvm-ffi",
    )


def supports_index(topk_ids: torch.Tensor, ep_size: int, experts_per_rank: int) -> bool:
    return (
        topk_ids.dtype == torch.int64
        and topk_ids.dim() == 2
        and topk_ids.stride(1) == 1
        and ep_size <= 32
        and ep_size * (experts_per_rank + 1) <= _MAX_BINS
        and topk_ids.shape[0] * max(topk_ids.shape[1], ep_size) < 2**31
    )


def dispatch_index(
    topk_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    meta_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    The contract of :func:`pithtrain.kernels.ep_dispatch.fused_dedup_prepare_dispatch` with
    stable placement, plus ``inv_idxs`` ``(m * k,)`` (``inv_idxs[idxs[p]] == p``) as a seventh
    output. All index outputs share one int64 allocation; ``m == 0`` launches nothing.
    """
    m, k = topk_ids.shape
    dev = topk_ids.device
    assert num_experts == ep_size * experts_per_rank
    if meta_out is None:
        meta_out = torch.empty(2 * ep_size, dtype=torch.int64, device=dev)
    counts = meta_out[: 2 * ep_size]
    nmeta = ep_size * (experts_per_rank + 1)
    mk = m * k
    out = torch.empty(3 * mk + m * ep_size + nmeta, dtype=torch.int64, device=dev)
    send_meta = out[3 * mk + m * ep_size :]
    if m == 0:
        counts.zero_()
        send_meta.zero_()
    else:
        key = (k, ep_size, experts_per_rank, sm_count(dev))
        fn = _INDEX.get(key)
        if fn is None:
            fn = _INDEX[key] = _compile_index(*key)
        sc = _scratch(dev, (m + _CHUNK - 1) // _CHUNK, num_experts + ep_size)
        fn(topk_ids, out, counts, sc.hist, sc.ctr)
    return (
        counts[:ep_size],
        counts[ep_size:],
        out[3 * mk : 3 * mk + m * ep_size],
        out[:mk],
        out[mk : 2 * mk],
        send_meta,
        out[2 * mk : 3 * mk],
    )


def _compile_expand(ep_size: int, experts_per_rank: int) -> Callable:
    i64 = Int64
    return cute.compile(
        _ExpertAndExpandIdx(ep_size, experts_per_rank),
        fake(i64, (ep_size * (experts_per_rank + 1),), (1,), 8),
        fake(i64, (cute.sym_int(),), (1,), 8),
        fake(i64, (cute.sym_int(),), (1,), 8),
        fake_stream(),
        options="--enable-tvm-ffi",
    )


def supports_expand(received_expand_idx: torch.Tensor, ep_size: int, experts_per_rank: int) -> bool:
    return (
        0 < ep_size * experts_per_rank <= _MAX_SEGMENTS
        and ep_size <= 32
        and received_expand_idx.shape[0] < 2**31
        and received_expand_idx.dtype == torch.int64
        and received_expand_idx.is_contiguous()
    )


def expand_index(
    recv_meta: torch.Tensor,
    received_expand_idx: torch.Tensor,
    ep_size: int,
    experts_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (expert_idxs, expand_idx) of the received token slots; the contract of
    :func:`pithtrain.kernels.ep_dispatch.expert_and_expand_idx`. Both outputs share one int64
    allocation. With no slots nothing is launched.
    """
    total = received_expand_idx.shape[0]
    out = received_expand_idx.new_empty(2 * total)
    if total == 0:
        return out[:0], out[0:0]
    key = (ep_size, experts_per_rank)
    fn = _EXPAND.get(key)
    if fn is None:
        fn = _EXPAND[key] = _compile_expand(ep_size, experts_per_rank)
    fn(recv_meta, received_expand_idx, out)
    return out[:total], out[total:]


def fused_dedup_prepare_dispatch_sm90(
    topk_ids: torch.Tensor,
    num_experts: int,
    ep_size: int,
    experts_per_rank: int,
    meta_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """
    The outputs of ``fused_dedup_prepare_dispatch_sm100``, computed by one CuTe launch with stable
    placement (token order within each bucket); inputs the CuTe kernel does not cover
    (:func:`supports_index`) take the Triton kernels.
    """
    if supports_index(topk_ids, ep_size, experts_per_rank):
        return dispatch_index(topk_ids, num_experts, ep_size, experts_per_rank, meta_out)[:6]
    return fused_dedup_prepare_dispatch_sm100(
        topk_ids, num_experts, ep_size, experts_per_rank, meta_out
    )


def expert_and_expand_idx_sm90(
    recv_meta: torch.Tensor,
    received_expand_idx: torch.Tensor,
    ep_size: int,
    experts_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (expert_idxs, expand_idx) of the received token slots by the CuTe kernel (:func:`expand_index`);
    inputs it does not cover (:func:`supports_expand`) take ``expert_and_expand_idx_sm100``.
    """
    if supports_expand(received_expand_idx, ep_size, experts_per_rank):
        return expand_index(recv_meta, received_expand_idx, ep_size, experts_per_rank)
    return expert_and_expand_idx_sm100(recv_meta, received_expand_idx, ep_size, experts_per_rank)
