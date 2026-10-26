"""
Fused MoE stage-3 input path for FP8 experts on SM90: dedup expansion + scatter by expert +
rowwise / K-major FP8 quantization, from the received tokens straight to the quantized grouped
layout.

Today stage 3 runs three memory passes before the first expert GEMM::

    x = padded_index_gather(gathered, expand_idx)                    # bf16 m x N write
    out, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(x, eidx, G)  # bf16 read + M_pad x N write
    xq = fp8cast_rowwise_kmajor(out, offs)                           # bf16 read, fp8 writes

:func:`scatter_quant_sm90` produces the same ``xq``, ``rsi``, ``offs``, ``ks`` and ``ks_t`` with two
launches and one read of the received tokens:

1. a prep kernel (one CTA per 2048 tokens) that ranks tokens stably within their expert, writes
   ``grouped_mm_offs`` / ``ks_tensor``, each token's rank, and for every expert ``g`` and rank
   ``k`` the source row that fills row ``k`` of the group (``inv[g * m + k] = expand_idx[token]``);
2. the ``rowwise_kmajor`` tile kernel (subclassed; its ``_tile`` does the quantization and
   writes both scales MN-major, as ``fp8cast_rowwise_kmajor`` returns them) whose row loads go
   through that map, so each 128 x 128 output tile reads its rows directly from the received tokens; padding
   rows load zeros. Column-block-0 CTAs also write ``reverse_shuffle_idxs`` token-side. It is
   launched as a programmatic dependent (PDL) of the prep kernel.

Under FP8 the bf16 grouped input is only ever read to be quantized, and its backward
(``scatter_quant_backward``, next to the autograd op that calls this) needs only
``reverse_shuffle_idxs``, ``expand_idx`` and the row count, so it never has to exist.

Inputs the fused kernels do not cover (:func:`supports_scatter_quant`) run the three-step path
with the Triton scatter (``kernels.token_scatter.scatter_by_expert``) and the SM90
``fp8cast_rowwise_kmajor_sm90``; ``fp8_scatter_quant_sm100`` is the same path with the Triton
quantization.

Placement. ``scatter_for_grouped_gemm`` assigns rows within a group in ``atomic_add`` order, which
is not deterministic, and the K-major per-channel scale of a 128-row block depends on which tokens
share the block. This kernel places tokens *stably* (token order within each group), so its output
is deterministic and bit-identical to the three-step path run with the same placement; the
placement-independent outputs (offs, ks, the rowwise FP8 of each token) match it outright.

The prep kernel orders chunks without a grid barrier: each CTA takes a ticket (chunk id in dispatch
order), publishes its chunk histogram behind a release flag, and waits only for lower chunks, which
already hold their tickets and never wait on it, so forward progress needs no co-residency. The
highest chunk, having waited for all others, writes the group offsets.
"""

from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32

from pithtrain.kernels.cute_compile_sm90 import fake, fake_stream
from pithtrain.kernels.cute_ptx_sm90 import (atom_add_u64, fence_gpu, lanemask_lt, ld_b32, ld_relaxed, ld_v4, lds_b32, match_any, red_add_shared, spin_until, st_b32, st_b64, st_release, sts_b32)
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import THREADS, TILE, RowwiseKmajor, fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.host_transfer import get_pinned_buffer
from pithtrain.kernels.token_scatter import scatter_by_expert

__all__ = ["Prep", "ScatterQuant", "scatter_quant_sm90", "supports_scatter_quant"]

_PREP_THREADS = 512
_PREP_WARPS = _PREP_THREADS // 32
_PREP_CHUNK = 2048  # tokens per prep CTA
_MAX_GROUPS = _PREP_THREADS
_ALIGN = 128

# ---------------------------------------------------------------------------
# Prep kernel: stable per-expert ranking, group offsets, inverse map
# ---------------------------------------------------------------------------


class Prep:
    """One CTA per chunk of ``_PREP_CHUNK`` tokens; ``_PREP_THREADS`` threads."""

    def __init__(self, num_groups: int, has_expand: bool) -> None:
        assert 0 < num_groups <= _MAX_GROUPS
        self.G = num_groups
        self.has_expand = has_expand

    @cute.jit
    def __call__(self, mE: cute.Tensor, mExp: cute.Tensor, mInv: cute.Tensor, mRank: cute.Tensor, mOffs: cute.Tensor, mKs: cute.Tensor, mCnt: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor, stream) -> None:
        C = (mE.shape[0] + _PREP_CHUNK - 1) // _PREP_CHUNK
        args = (mE, mExp, mInv, mRank, mOffs, mKs, mCnt, mFlags, mHist, mCtr)
        self.kernel(*args).launch(grid=[C, 1, 1], block=[_PREP_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mE: cute.Tensor, mExp: cute.Tensor, mInv: cute.Tensor, mRank: cute.Tensor, mOffs: cute.Tensor, mKs: cute.Tensor, mCnt: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor) -> None:
        G = self.G
        NT = _PREP_THREADS
        NW = _PREP_WARPS
        tid, _, _ = cute.arch.thread_idx()
        warp = tid // 32
        m = mE.shape[0]
        C = (m + _PREP_CHUNK - 1) // _PREP_CHUNK
        e_base = mE.iterator.toint()
        inv_base = mInv.iterator.toint()
        flags_base = mFlags.iterator.toint()
        hist_base = mHist.iterator.toint()
        ctr_base = mCtr.iterator.toint()

        smem = cutlass.memory.SmemAllocator()
        s_run = smem.allocate(G * 4, byte_alignment=16).toint()  # running per-expert count
        s_pre = smem.allocate(G * 4, byte_alignment=16).toint()  # counts in lower chunks
        s_wc = smem.allocate(NW * G * 4, byte_alignment=16).toint()  # per-warp counts
        s_ex = smem.allocate(NW * G * 4, byte_alignment=16).toint()  # per-warp exclusive base
        s_misc = smem.allocate(16, byte_alignment=16).toint()
        cute.arch.griddepcontrol_launch_dependents()  # let the main kernel's CTAs get resident

        for j in cutlass.range_constexpr((NW * G + NT - 1) // NT):
            idx = j * NT + tid
            sts_b32(s_wc + idx * 4, Uint32(0), Int32(idx < NW * G))
        sts_b32(s_run + tid * 4, Uint32(0), Int32(tid < G))
        sts_b32(s_pre + tid * 4, Uint32(0), Int32(tid < G))
        if tid == 0:
            # ctr packs (epoch << 32) | tickets taken; the old value gives this CTA both
            t = atom_add_u64(ctr_base, Int64(1))
            sts_b32(s_misc, Uint32(Int32(t & 0xFFFFFFFF)), Int32(1))
            sts_b32(s_misc + 4, Uint32(Int32(t >> 32)), Int32(1))
        cute.arch.sync_threads()
        c = Int32(lds_b32(s_misc))
        epoch = Int32(lds_b32(s_misc + 4))

        # ---- stable rank of each token within its expert, inside this chunk ----
        srcs = []
        ids = []
        eids = []
        oks = []
        rls = []
        lt_mask = lanemask_lt()
        es_in = []
        for r in cutlass.range_constexpr(_PREP_CHUNK // NT):  # all loads in flight at once
            i = c * _PREP_CHUNK + r * NT + tid
            ok = Int32(i < m)
            e = ld_b32(e_base + Int64(i) * 8, ok, Int32(-1))
            src = i
            if cutlass.const_expr(self.has_expand):
                src = ld_b32(mExp.iterator.toint() + Int64(i) * 8, ok, Int32(0))
            es_in.append(e)
            srcs.append(src)
            ids.append(i)
            oks.append(ok)
        for r in cutlass.range_constexpr(_PREP_CHUNK // NT):
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

        # ---- publish this chunk's histogram, then wait for every lower chunk ----
        if tid < G:
            st_b32(hist_base + (Int64(c) * G + tid) * 4, lds_b32(s_run + tid * 4), Int32(1))
        fence_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            st_release(flags_base + Int64(c) * 4, epoch)
        for k in cutlass.range((c + NT - 1) // NT, unroll=1):
            j = k * NT + tid
            spin_until(flags_base + Int64(j) * 4, epoch, Int32(j < c))
        cute.arch.sync_threads()
        fence_gpu()
        n_lo = c * G
        for k in cutlass.range((n_lo + NT - 1) // NT, unroll=1):
            j = k * NT + tid
            ok = Int32(j < n_lo)
            v = ld_relaxed(hist_base + Int64(j) * 4, ok)
            red_add_shared(s_pre + (j % G) * 4, v, ok)
        cute.arch.sync_threads()

        # ---- inverse map inv[g * m + rank] = source row, and each token's rank ----
        for r in cutlass.range_constexpr(_PREP_CHUNK // NT):
            rank = Int32(lds_b32(s_pre + eids[r] * 4)) + rls[r]
            addr = inv_base + (Int64(eids[r]) * m + rank) * 4
            st_b32(addr, Uint32(srcs[r]), oks[r])
            st_b32(mRank.iterator.toint() + Int64(ids[r]) * 4, Uint32(rank), oks[r])

        # ---- the last chunk has waited for all others: counts, offsets, padded sizes ----
        if c == C - 1:
            if tid < G:
                off = Int32(0)
                for g in cutlass.range(tid + 1, unroll=1):
                    cg = Int32(lds_b32(s_pre + g * 4)) + Int32(lds_b32(s_run + g * 4))
                    off = off + (cg + (_ALIGN - 1)) // _ALIGN * _ALIGN
                own = Int32(lds_b32(s_pre + tid * 4)) + Int32(lds_b32(s_run + tid * 4))
                pad = (own + (_ALIGN - 1)) // _ALIGN * _ALIGN
                st_b32(mOffs.iterator.toint() + Int64(tid) * 4, Uint32(off), Int32(1))
                st_b32(mKs.iterator.toint() + Int64(tid) * 4, Uint32(pad), Int32(1))
                st_b32(mCnt.iterator.toint() + Int64(tid) * 4, Uint32(own), Int32(1))
            if tid == 0:
                # every CTA holds its ticket (this one took the last): next epoch, zero tickets
                atom_add_u64(ctr_base, (Int64(1) << 32) - Int64(C))


# ---------------------------------------------------------------------------
# Main kernel: indirect-load rowwise_kmajor
# ---------------------------------------------------------------------------


class ScatterQuant(RowwiseKmajor):
    """``RowwiseKmajor`` whose row ``k`` of group ``g`` is ``gathered[inv[g * m + k]]``.

    With ``pdl`` it is launched as a programmatic dependent of the prep kernel: its CTAs become
    resident while the prep runs and wait in ``griddepcontrol.wait`` for the prep's results.
    """

    def __init__(self, n: int, num_groups: int, pdl: bool) -> None:
        super().__init__(n, num_groups)
        self.pdl = pdl

    @cute.jit
    def __call__(self, mX: cute.Tensor, mE: cute.Tensor, mRank: cute.Tensor, mInv: cute.Tensor, mCnt: cute.Tensor, mOffs: cute.Tensor, mRsi: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKm: cute.Tensor, mKmScale: cute.Tensor, stream) -> None:
        mb = mTok.shape[0] // TILE
        args = (mX, mE, mRank, mInv, mCnt, mOffs, mRsi, mTok, mTokScale, mKm, mKmScale)
        self.kernel(*args).launch(grid=[self.nb, mb, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mE: cute.Tensor, mRank: cute.Tensor, mInv: cute.Tensor, mCnt: cute.Tensor, mOffs: cute.Tensor, mRsi: cute.Tensor, mTok: cute.Tensor, mTokScale: cute.Tensor, mKm: cute.Tensor, mKmScale: cute.Tensor) -> None:
        N, G = self.n, self.num_groups
        tidx, _, _ = cute.arch.thread_idx()
        pid_col, pid_row, _ = cute.arch.block_idx()
        m = mRsi.shape[0]
        x_base = mX.iterator.toint()
        offs_base = mOffs.iterator.toint()
        gp = (x_base, offs_base, mTok.iterator.toint(), mTokScale.iterator.toint(), mKm.iterator.toint(), mKmScale.iterator.toint())
        smem = cutlass.memory.SmemAllocator()
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()

        total = Int64(ld_b32(offs_base + (G - 1) * 4, Int32(1), Int32(0)))
        row_start = Int64(pid_row) * TILE
        if row_start < total:
            lane, warp, cv, rb = self._coords(tidx)
            cnt = Int32(0)
            for c in range((G + 31) // 32):
                idx = lane + c * 32
                o = ld_b32(offs_base + Int64(idx) * 4, Int32(idx < G), Int32(0x7FFFFFFF))
                cnt = cnt + cute.arch.popc(cute.arch.vote_ballot_sync(Int64(o) <= row_start))
            gid = cutlass.min(cnt, G - 1)
            g_start = Int64(ld_b32(offs_base + Int64(gid - 1) * 4, Int32(gid > 0), Int32(0)))
            n_g = Int64(ld_b32(mCnt.iterator.toint() + Int64(gid) * 4, Int32(1), Int32(0)))
            col0 = pid_col * TILE + cv * 8
            col_ok = Int32(col0 < N)
            inv_row = mInv.iterator.toint() + Int64(gid) * m * 4
            v = []
            for i in cutlass.range_constexpr(8):
                k = row_start + rb * 8 + i - g_start
                ok = Int32(k < n_g)
                src = ld_b32(inv_row + k * 4, ok, Int32(0))
                v.append(ld_v4(x_base + (Int64(src) * N + col0) * 2, ok & col_ok))
            # reverse_shuffle_idxs, token side: rsi[i] = start of expert(i) + rank(i)
            t = Int64(pid_row) * TILE + tidx
            t_ok = Int32(pid_col == 0) & Int32(tidx < TILE) & Int32(t < m)
            e = ld_b32(mE.iterator.toint() + t * 8, t_ok, Int32(0))
            rk = ld_b32(mRank.iterator.toint() + t * 4, t_ok, Int32(0))
            es = ld_b32(offs_base + Int64(e - 1) * 4, t_ok & Int32(e > 0), Int32(0))
            st_b64(mRsi.iterator.toint() + t * 8, Int64(es) + Int64(rk), t_ok)
            # M = total rows: the column stride of the MN-major rowwise scale (M % 128 == 0)
            self._tile(v, pid_row, pid_col, tidx, total, gp, s_red.iterator.toint())


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

_PREP: dict = {}
_MAIN: dict = {}


class _Scratch:
    """Per-device prep scratch: a ticket/epoch counter, epoch flags and chunk histograms.

    ``ctr`` packs ``(epoch << 32) | tickets``; the last chunk of each launch advances the epoch and
    zeroes the tickets, and a chunk's flag holds the epoch it was published in. The state lives on
    the device (no host-side epoch), so launches may be captured in a CUDA graph. Launches must not
    run concurrently on two streams (the same assumption as token_scatter's pinned buffers).
    """

    def __init__(self, device: torch.device, chunks: int, groups: int) -> None:
        self.chunks, self.groups = chunks, groups
        self.ctr = torch.full((1,), 1 << 32, dtype=torch.int64, device=device)  # epoch 1
        self.flags = torch.zeros(chunks, dtype=torch.int32, device=device)
        self.hist = torch.empty(chunks * groups, dtype=torch.int32, device=device)


_SCRATCH: dict = {}
_copy_stream: Optional[torch.cuda.Stream] = None
_copy_event: Optional[torch.cuda.Event] = None


def _scratch(device: torch.device, chunks: int, groups: int) -> _Scratch:
    s = _SCRATCH.get(device.index)
    if s is None or s.chunks < chunks or s.groups < groups:
        chunks = max(chunks, s.chunks if s is not None else 0, 64)
        groups = max(groups, s.groups if s is not None else 0)
        s = _SCRATCH[device.index] = _Scratch(device, chunks, groups)
    return s


def _compile_prep(G: int, has_expand: bool) -> Callable:
    i32 = Int32
    vec = lambda: fake(i32, (cute.sym_int(),), (1,), 4)  # noqa: E731
    return cute.compile(
        Prep(G, has_expand), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int64, (cute.sym_int(),), (1,), 8), vec(), vec(), fake(i32, (G,), (1,), 4), fake(i32, (G,), (1,), 4), fake(i32, (G,), (1,), 4), vec(), vec(), fake(Int64, (1,), (1,), 8),
        fake_stream(), options="--enable-tvm-ffi",
    )


def _compile_main(n: int, G: int, pdl: bool) -> Callable:
    mp = cute.sym_int()
    nb = (n + TILE - 1) // TILE
    fp8 = cutlass.Float8E4M3FN
    return cute.compile(
        ScatterQuant(n, G, pdl), fake(cutlass.BFloat16, (cute.sym_int(), n), (n, 1)), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int32, (G,), (1,), 4),
        fake(Int32, (G,), (1,), 4), fake(Int64, (cute.sym_int(),), (1,), 8), fake(fp8, (mp, n), (n, 1)), fake(cutlass.Float32, (cute.sym_int(), nb), (1, cute.sym_int64())), fake(fp8, (cute.sym_int(),), (1,)),
        fake(cutlass.Float32, (n, cute.sym_int()), (1, mn_pad(n))), fake_stream(), options="--enable-tvm-ffi",
    )


def _reference(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor], group_counts: Optional[list[int]]) -> tuple:
    """The three-step path, for inputs the fused kernels do not cover: the dedup expansion, the
    Triton scatter (``kernels.token_scatter``) and the SM90 K-major quantization."""
    x = gathered if expand_idx is None else gathered[expand_idx]
    out, rsi, offs, ks_t, ks = scatter_by_expert(x, expert_idxs, num_groups, group_counts=group_counts)
    return fp8cast_rowwise_kmajor_sm90(out, offs), rsi, offs, ks, ks_t


def supports_scatter_quant(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None) -> bool:
    """Whether the fused kernels cover these inputs (else :func:`scatter_quant_sm90` runs the
    three-step path)."""
    m, N = expert_idxs.shape[0], gathered.shape[1]
    return not (
        m == 0
        or N == 0
        or not 0 < num_groups <= _MAX_GROUPS
        or gathered.dtype != torch.bfloat16
        or N % 8 != 0
        or not gathered.is_contiguous()
        or gathered.data_ptr() % 16 != 0
        or expert_idxs.dtype != torch.int64
        or not expert_idxs.is_contiguous()
        or (expand_idx is not None and (expand_idx.dtype != torch.int64 or not expand_idx.is_contiguous()))
    )


def scatter_quant_sm90(
    gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None, padding_alignment: int = 128, group_counts: Optional[list[int]] = None, _pdl: bool = True
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """``fp8cast_rowwise_kmajor(scatter_for_grouped_gemm(gathered[expand_idx], ...))`` fused.

    Args:
        gathered: ``(m_src, N)`` bf16 received (dedup'd) tokens.
        expert_idxs: ``(m,)`` int64 local expert of each expanded token slot.
        num_groups: experts on this rank.
        expand_idx: ``(m,)`` int64 row of ``gathered`` for each slot; None means identity
            (``gathered`` already has ``m`` rows).
        padding_alignment: must be 128 (the K-major per-channel blocks).
        group_counts: tokens per expert on the host, if known; skips the device-to-host copy of
            the padded sizes and sizes the outputs exactly.

    Returns:
        ``(x_fp8, x_scale, x_ch, x_ch_scale)`` exactly as ``fp8cast_rowwise_kmajor`` returns them for
        the scattered ``(M, N)`` input, ``reverse_shuffle_idxs`` ``(m,)`` int64,
        ``grouped_mm_offs`` ``(G,)`` int32, ``ks`` list[int], ``ks_tensor`` ``(G,)`` int32 - the
        same order as :func:`scatter_for_grouped_gemm` with its bf16 output replaced by the
        quantized tuple. Rows are placed stably (token order within each expert).
    """
    global _copy_stream, _copy_event
    assert padding_alignment == _ALIGN, "the K-major FP8 layout needs 128-row groups"
    G = num_groups
    m = expert_idxs.shape[0]
    N = gathered.shape[1]
    dev = gathered.device
    if expand_idx is None and gathered.shape[0] != m:
        raise ValueError(f"{gathered.shape[0]=} must equal {m=} without expand_idx")
    if expand_idx is not None and expand_idx.shape[0] != m:
        raise ValueError(f"{expand_idx.shape[0]=} must equal {m=}")
    if not supports_scatter_quant(gathered, expert_idxs, G, expand_idx):
        return _reference(gathered, expert_idxs, G, expand_idx, group_counts)

    # ---- outputs: exact when the host knows the counts, else a tight upper bound ----
    if group_counts is not None:
        ks = [(c + _ALIGN - 1) // _ALIGN * _ALIGN for c in group_counts]
        M_alloc = sum(ks)
    else:
        ks = None
        M_alloc = (m + G * (_ALIGN - 1) + _ALIGN - 1) // _ALIGN * _ALIGN
    nb = (N + TILE - 1) // TILE
    # one int32 allocation: offs | ks | counts | rank (m) | inv (G * m)
    meta = torch.empty(3 * G + m + G * m, dtype=torch.int32, device=dev)
    offs, ks_t, cnt = meta[:G], meta[G : 2 * G], meta[2 * G : 3 * G]
    rank, inv = meta[3 * G : 3 * G + m], meta[3 * G + m :]
    rsi = torch.empty(m, dtype=torch.int64, device=dev)
    x_fp8 = torch.empty((M_alloc, N), dtype=torch.float8_e4m3fn, device=dev)
    # MN-major scales (strides (1, align(rows, 4))), as fp8cast_rowwise_kmajor returns them; the
    # kernel addresses the rowwise one with the device-side M, so it fits the upper bound too
    x_scale = mn_major_scale(M_alloc, nb, dev)
    x_ch = torch.empty(M_alloc * N, dtype=torch.float8_e4m3fn, device=dev)
    x_ch_scale = mn_major_scale(N, M_alloc // TILE, dev)

    # ---- prep ----
    chunks = (m + _PREP_CHUNK - 1) // _PREP_CHUNK
    sc = _scratch(dev, chunks, G)
    has_expand = expand_idx is not None
    prep = _PREP.get((G, has_expand))
    if prep is None:
        prep = _PREP[(G, has_expand)] = _compile_prep(G, has_expand)
    exp = expand_idx if has_expand else expert_idxs
    prep(expert_idxs, exp, inv, rank, offs, ks_t, cnt, sc.flags, sc.hist, sc.ctr)

    if ks is None:
        if _copy_stream is None:
            _copy_stream = torch.cuda.Stream(device=dev)
            _copy_event = torch.cuda.Event()
        _copy_event.record()

    # ---- main ----
    key = (N, G, _pdl)
    fn = _MAIN.get(key)
    if fn is None:
        fn = _MAIN[key] = _compile_main(N, G, _pdl)
    fn(gathered, expert_idxs, rank, inv, cnt, offs, rsi, x_fp8, x_scale, x_ch, x_ch_scale)

    if ks is None:
        ks_cpu = get_pinned_buffer("scatter_quant_ks", G, torch.int32)
        with torch.cuda.stream(_copy_stream):
            _copy_stream.wait_event(_copy_event)
            ks_cpu.copy_(ks_t, non_blocking=True)
        _copy_stream.synchronize()
        ks = ks_cpu.tolist()
        M = sum(ks)
        if M != M_alloc:
            x_fp8, x_ch = x_fp8[:M], x_ch[: M * N]
            x_scale = x_scale.as_strided((M, nb), (1, M))
            x_ch_scale = x_ch_scale[:, : M // TILE]
    return (x_fp8, x_scale, x_ch, x_ch_scale), rsi, offs, ks, ks_t
