"""
SM90 CuTe-DSL scatter_quant: dedup expansion + scatter by expert + K-major FP8 quantization in two
launches and one read of the received tokens; the grouped bf16 input never exists.

1. Prep (one CTA per 2048 tokens) ranks each token stably within its expert, writes grouped_mm_offs
   and ks_tensor, each token's rank, and the inverse map inv[g * m + k] = the source row of row k
   of group g.
2. ScatterQuant, the rowwise_kmajor tile kernel whose row loads go through inv (padding rows load
   zeros), quantizes and writes both scales MN-major; its column-block-0 CTAs also write
   reverse_shuffle_idxs. It is launched as a programmatic dependent (PDL) of the prep kernel.

Placement is stable (token order within each group), so the output is deterministic and
bit-identical to the unfused path run with that placement.

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
from pithtrain.kernels.cute_ptx_sm90 import atom_add_u64, fence_gpu, lanemask_lt, ld_b32, ld_relaxed, ld_v4, lds_b32, match_any, red_add_shared, spin_until, st_b32, st_b64, st_release, sts_b32
from pithtrain.kernels.fp8_rowwise_kmajor_sm90 import THREADS, TILE, RowwiseKmajor, fp8cast_rowwise_kmajor_sm90
from pithtrain.kernels.fp8_scale_layout import mn_major_scale, mn_pad
from pithtrain.kernels.host_transfer import get_pinned_buffer
from pithtrain.kernels.token_scatter import scatter_by_expert

__all__ = ["Prep", "ScatterQuant", "scatter_quant_sm90", "supports_scatter_quant"]

PREP_THREADS = 512
PREP_WARPS = PREP_THREADS // 32
PREP_CHUNK = 2048  # tokens per prep CTA
MAX_GROUPS = PREP_THREADS
ALIGN = 128


class Prep:
    """
    Stable per-expert ranking, group offsets and inverse map; one CTA per PREP_CHUNK tokens.
    """

    def __init__(self, num_groups: int, has_expand: bool) -> None:
        assert 0 < num_groups <= MAX_GROUPS
        self.G = num_groups
        self.has_expand = has_expand

    @cute.jit
    def __call__(self, mE: cute.Tensor, mExp: cute.Tensor, mInv: cute.Tensor, mRank: cute.Tensor, mOffs: cute.Tensor, mKs: cute.Tensor, mCnt: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor, stream) -> None:
        C = (mE.shape[0] + PREP_CHUNK - 1) // PREP_CHUNK
        args = (mE, mExp, mInv, mRank, mOffs, mKs, mCnt, mFlags, mHist, mCtr)
        self.kernel(*args).launch(grid=[C, 1, 1], block=[PREP_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mE: cute.Tensor, mExp: cute.Tensor, mInv: cute.Tensor, mRank: cute.Tensor, mOffs: cute.Tensor, mKs: cute.Tensor, mCnt: cute.Tensor, mFlags: cute.Tensor, mHist: cute.Tensor, mCtr: cute.Tensor) -> None:
        G = self.G
        NT, NW = PREP_THREADS, PREP_WARPS
        tid, _, _ = cute.arch.thread_idx()
        warp = tid // 32
        m = mE.shape[0]
        C = (m + PREP_CHUNK - 1) // PREP_CHUNK
        e_base, inv_base, flags_base = mE.iterator.toint(), mInv.iterator.toint(), mFlags.iterator.toint()
        hist_base, ctr_base = mHist.iterator.toint(), mCtr.iterator.toint()

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
        c, epoch = Int32(lds_b32(s_misc)), Int32(lds_b32(s_misc + 4))

        # stable rank of each token within its expert, inside this chunk
        srcs, ids, eids, oks = [], [], [], []
        rls = []
        lt_mask = lanemask_lt()
        es_in = []
        for r in cutlass.range_constexpr(PREP_CHUNK // NT):  # all loads in flight at once
            i = c * PREP_CHUNK + r * NT + tid
            ok = Int32(i < m)
            e = ld_b32(e_base + Int64(i) * 8, ok, Int32(-1))
            src = i
            if cutlass.const_expr(self.has_expand):
                src = ld_b32(mExp.iterator.toint() + Int64(i) * 8, ok, Int32(0))
            es_in.append(e)
            srcs.append(src)
            ids.append(i)
            oks.append(ok)
        for r in cutlass.range_constexpr(PREP_CHUNK // NT):
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

        # publish this chunk's histogram, then wait for every lower chunk
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

        # inverse map inv[g * m + rank] = source row, and each token's rank
        for r in cutlass.range_constexpr(PREP_CHUNK // NT):
            rank = Int32(lds_b32(s_pre + eids[r] * 4)) + rls[r]
            addr = inv_base + (Int64(eids[r]) * m + rank) * 4
            st_b32(addr, Uint32(srcs[r]), oks[r])
            st_b32(mRank.iterator.toint() + Int64(ids[r]) * 4, Uint32(rank), oks[r])

        # the last chunk has waited for all others: counts, offsets, padded sizes
        if c == C - 1:
            if tid < G:
                off = Int32(0)
                for g in cutlass.range(tid + 1, unroll=1):
                    cg = Int32(lds_b32(s_pre + g * 4)) + Int32(lds_b32(s_run + g * 4))
                    off = off + (cg + (ALIGN - 1)) // ALIGN * ALIGN
                own = Int32(lds_b32(s_pre + tid * 4)) + Int32(lds_b32(s_run + tid * 4))
                pad = (own + (ALIGN - 1)) // ALIGN * ALIGN
                st_b32(mOffs.iterator.toint() + Int64(tid) * 4, Uint32(off), Int32(1))
                st_b32(mKs.iterator.toint() + Int64(tid) * 4, Uint32(pad), Int32(1))
                st_b32(mCnt.iterator.toint() + Int64(tid) * 4, Uint32(own), Int32(1))
            if tid == 0:
                # every CTA holds its ticket (this one took the last): next epoch, zero tickets
                atom_add_u64(ctr_base, (Int64(1) << 32) - Int64(C))



class ScatterQuant(RowwiseKmajor):
    """
    RowwiseKmajor whose row k of group g is gathered[inv[g * m + k]]. With pdl its CTAs become
    resident while the prep kernel runs and wait in griddepcontrol.wait for its results.
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
        x_base, offs_base = mX.iterator.toint(), mOffs.iterator.toint()
        gp = (x_base, offs_base, mTok.iterator.toint(), mTokScale.iterator.toint(), mKm.iterator.toint(), mKmScale.iterator.toint())
        smem = cutlass.memory.SmemAllocator()
        s_red = smem.allocate_tensor(Uint32, cute.make_layout(self.RED_WORDS), 16)
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()

        total = Int64(ld_b32(offs_base + (G - 1) * 4, Int32(1), Int32(0)))
        row_start = Int64(pid_row) * TILE
        if row_start < total:
            lane, _, cv, rb = self.coords(tidx)
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
            e, rk = ld_b32(mE.iterator.toint() + t * 8, t_ok, Int32(0)), ld_b32(mRank.iterator.toint() + t * 4, t_ok, Int32(0))
            es = ld_b32(offs_base + Int64(e - 1) * 4, t_ok & Int32(e > 0), Int32(0))
            st_b64(mRsi.iterator.toint() + t * 8, Int64(es) + Int64(rk), t_ok)
            # M = total rows: the column stride of the MN-major rowwise scale (M % 128 == 0)
            self.tile(v, pid_row, pid_col, tidx, total, gp, s_red.iterator.toint())


PREP: dict = {}
MAIN: dict = {}


class Scratch:
    """
    Per-device prep scratch: a ticket/epoch counter, epoch flags and chunk histograms. ctr packs
    (epoch << 32) | tickets; the last chunk of each launch advances the epoch and zeroes the
    tickets, and a chunk's flag holds the epoch it was published in. The state lives on the device,
    so launches may be captured in a CUDA graph; they must not run concurrently on two streams.
    """

    def __init__(self, device: torch.device, chunks: int, groups: int) -> None:
        self.chunks, self.groups = chunks, groups
        self.ctr = torch.full((1,), 1 << 32, dtype=torch.int64, device=device)  # epoch 1
        self.flags = torch.zeros(chunks, dtype=torch.int32, device=device)
        self.hist = torch.empty(chunks * groups, dtype=torch.int32, device=device)


SCRATCH: dict = {}
copy_stream: Optional[torch.cuda.Stream] = None
copy_event: Optional[torch.cuda.Event] = None


def scratch(device: torch.device, chunks: int, groups: int) -> Scratch:
    s = SCRATCH.get(device.index)
    if s is None or s.chunks < chunks or s.groups < groups:
        chunks, groups = max(chunks, s.chunks if s is not None else 0, 64), max(groups, s.groups if s is not None else 0)
        s = SCRATCH[device.index] = Scratch(device, chunks, groups)
    return s


def compile_prep(G: int, has_expand: bool) -> Callable:
    i32 = Int32
    vec = lambda: fake(i32, (cute.sym_int(),), (1,), 4)
    return cute.compile(Prep(G, has_expand), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int64, (cute.sym_int(),), (1,), 8), vec(), vec(), fake(i32, (G,), (1,), 4), fake(i32, (G,), (1,), 4), fake(i32, (G,), (1,), 4), vec(), vec(), fake(Int64, (1,), (1,), 8), fake_stream(), options="--enable-tvm-ffi")


def compile_main(n: int, G: int, pdl: bool) -> Callable:
    mp = cute.sym_int()
    nb = (n + TILE - 1) // TILE
    fp8 = cutlass.Float8E4M3FN
    return cute.compile(ScatterQuant(n, G, pdl), fake(cutlass.BFloat16, (cute.sym_int(), n), (n, 1)), fake(Int64, (cute.sym_int(),), (1,), 8), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int32, (cute.sym_int(),), (1,), 4), fake(Int32, (G,), (1,), 4), fake(Int32, (G,), (1,), 4), fake(Int64, (cute.sym_int(),), (1,), 8), fake(fp8, (mp, n), (n, 1)), fake(cutlass.Float32, (cute.sym_int(), nb), (1, cute.sym_int64())), fake(fp8, (cute.sym_int(),), (1,)), fake(cutlass.Float32, (n, cute.sym_int()), (1, mn_pad(n))), fake_stream(), options="--enable-tvm-ffi")


def reference(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor], group_counts: Optional[list[int]]) -> tuple:
    """
    The unfused path for inputs the fused kernels do not cover (rows in atomic_add order).
    """
    x = gathered if expand_idx is None else gathered[expand_idx]
    out, rsi, offs, ks_t, ks = scatter_by_expert(x, expert_idxs, num_groups, group_counts=group_counts)
    return fp8cast_rowwise_kmajor_sm90(out, offs), rsi, offs, ks, ks_t


def supports_scatter_quant(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None) -> bool:
    """
    Whether the fused kernels cover these inputs (else scatter_quant_sm90 runs the unfused path).
    """
    m, N = expert_idxs.shape[0], gathered.shape[1]
    return not (m == 0 or N == 0 or not 0 < num_groups <= MAX_GROUPS or gathered.dtype != torch.bfloat16 or N % 8 != 0 or not gathered.is_contiguous() or gathered.data_ptr() % 16 != 0 or expert_idxs.dtype != torch.int64 or not expert_idxs.is_contiguous() or (expand_idx is not None and (expand_idx.dtype != torch.int64 or not expand_idx.is_contiguous())))


def scatter_quant_sm90(gathered: torch.Tensor, expert_idxs: torch.Tensor, num_groups: int, expand_idx: Optional[torch.Tensor] = None, padding_alignment: int = 128, group_counts: Optional[list[int]] = None, pdl: bool = True) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    scatter_quant on SM90 (see the picker for the contract): fused where supports_scatter_quant,
    else the unfused path. Scales are MN-major. pdl=False launches the main kernel without PDL.
    """
    global copy_stream, copy_event
    assert padding_alignment == ALIGN, "the K-major FP8 layout needs 128-row groups"
    G = num_groups
    m, N = expert_idxs.shape[0], gathered.shape[1]
    dev = gathered.device
    if expand_idx is None and gathered.shape[0] != m:
        raise ValueError(f"{gathered.shape[0]=} must equal {m=} without expand_idx")
    if expand_idx is not None and expand_idx.shape[0] != m:
        raise ValueError(f"{expand_idx.shape[0]=} must equal {m=}")
    if not supports_scatter_quant(gathered, expert_idxs, G, expand_idx):
        return reference(gathered, expert_idxs, G, expand_idx, group_counts)

    # outputs: exact when the host knows the counts, else an upper bound
    if group_counts is not None:
        ks = [(c + ALIGN - 1) // ALIGN * ALIGN for c in group_counts]
        M_alloc = sum(ks)
    else:
        ks = None
        M_alloc = (m + G * (ALIGN - 1) + ALIGN - 1) // ALIGN * ALIGN
    nb = (N + TILE - 1) // TILE
    # one int32 allocation: offs | ks | counts | rank (m) | inv (G * m)
    meta = torch.empty(3 * G + m + G * m, dtype=torch.int32, device=dev)
    offs, ks_t, cnt = meta[:G], meta[G : 2 * G], meta[2 * G : 3 * G]
    rank, inv = meta[3 * G : 3 * G + m], meta[3 * G + m :]
    rsi, x_fp8 = torch.empty(m, dtype=torch.int64, device=dev), torch.empty((M_alloc, N), dtype=torch.float8_e4m3fn, device=dev)
    # the kernel addresses the MN-major rowwise scale with the device-side M, so it fits the upper bound too
    x_scale = mn_major_scale(M_alloc, nb, dev)
    x_ch = torch.empty(M_alloc * N, dtype=torch.float8_e4m3fn, device=dev)
    x_ch_scale = mn_major_scale(N, M_alloc // TILE, dev)

    chunks = (m + PREP_CHUNK - 1) // PREP_CHUNK
    sc = scratch(dev, chunks, G)
    has_expand = expand_idx is not None
    prep = PREP.get((G, has_expand))
    if prep is None:
        prep = PREP[(G, has_expand)] = compile_prep(G, has_expand)
    exp = expand_idx if has_expand else expert_idxs
    prep(expert_idxs, exp, inv, rank, offs, ks_t, cnt, sc.flags, sc.hist, sc.ctr)

    if ks is None:
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=dev)
            copy_event = torch.cuda.Event()
        copy_event.record()

    key = (N, G, pdl)
    fn = MAIN.get(key)
    if fn is None:
        fn = MAIN[key] = compile_main(N, G, pdl)
    fn(gathered, expert_idxs, rank, inv, cnt, offs, rsi, x_fp8, x_scale, x_ch, x_ch_scale)

    if ks is None:
        ks_cpu = get_pinned_buffer("scatter_quant_ks", G, torch.int32)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(copy_event)
            ks_cpu.copy_(ks_t, non_blocking=True)
        copy_stream.synchronize()
        ks = ks_cpu.tolist()
        M = sum(ks)
        if M != M_alloc:
            x_fp8, x_ch = x_fp8[:M], x_ch[: M * N]
            x_scale = x_scale.as_strided((M, nb), (1, M))
            x_ch_scale = x_ch_scale[:, : M // TILE]
    return (x_fp8, x_scale, x_ch, x_ch_scale), rsi, offs, ks, ks_t
