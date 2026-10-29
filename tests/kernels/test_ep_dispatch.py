"""
Verifier of the ``ep_dispatch`` kernels (``fused_dedup_prepare_dispatch``,
``expert_and_expand_idx``) against plain PyTorch.

Sender side, per token deduplicated per EP rank (a token routed to several experts of one rank is
sent once):

- order-free outputs (``tokens_per_ep_rank``, ``dedup_tokens_per_gpu``, ``send_meta``) are
  compared exactly;
- order-carrying outputs (``idxs``, ``expand_idx``, ``dispatch_token_idxs``) are compared exactly
  against a stable reference (``argsort(stable=True)`` / ``nonzero``) on the inputs the H200 path
  covers (the CuTe kernel places stably, in token order within each bucket). Inputs it does not
  cover fall back to ``atomic_add`` placement, where these outputs are compared as per-bucket sets
  plus the ``expand_idx`` invariant ``dispatch_token_idxs[rank_start + expand_idx] == idxs // k``.

Receiver side, ``expert_and_expand_idx`` is order-free and compared exactly; an end-to-end test
emulates the all-to-alls of the operator in one process and checks that every received slot
resolves to its token and expert.

The routing contract: a token's top-k experts are distinct (as ``torch.topk`` returns them).
"""

import pytest
import torch

from pithtrain.kernels import expert_and_expand_idx, fused_dedup_prepare_dispatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# (num_experts, top-k) of the models: deepseek-v2-lite, qwen3-30b-a3b, qwen3.5-35b-a3b,
# gpt-oss-20b, gpt-oss-120b
MODELS = [(64, 6), (128, 8), (256, 8), (32, 4), (128, 4)]


# ---------------------------------------------------------------------------
# References and inputs
# ---------------------------------------------------------------------------


def stable_reference(topk_ids: torch.Tensor, ep: int, epr: int) -> dict:
    """Every sender-side output, placed stably (token order within each bucket)."""
    m, k = topk_ids.shape
    ne = ep * epr
    flat = topk_ids.reshape(-1).long()
    idxs = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ne)
    visit = torch.zeros(m, ep, dtype=torch.bool, device=topk_ids.device)
    if m:
        visit.scatter_(1, topk_ids.long() // epr, True)
    dedup = visit.sum(0)
    dti = visit.T.nonzero()[:, 1]  # rank-major, token order within a rank
    rank = visit.long().cumsum(0) - 1  # position of a token within its rank's dedup chunk
    expand = rank[idxs // k, flat[idxs] // epr] if m else idxs
    return {
        "tokens_per_ep_rank": counts.view(ep, epr).sum(1),
        "dedup": dedup,
        "dti": dti,
        "idxs": idxs,
        "expand": expand,
        "send_meta": torch.cat([counts.view(ep, epr), dedup[:, None]], 1).reshape(-1),
    }


def routing(m: int, k: int, ne: int, epr: int, kind: str = "random", seed: int = 0):
    """(m, k) int64 distinct expert ids per token."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.rand(m, ne, generator=g).argsort(1)[:, :k]
    if kind == "one_expert" and m:
        ids[:, 0] = 0  # every token picks expert 0 (its other picks stay distinct)
        ids[:, 1:] = (torch.rand(m, ne - 1, generator=g).argsort(1)[:, : k - 1]) + 1
    elif kind == "one_rank" and m:
        assert epr >= k
        ids = torch.rand(m, epr, generator=g).argsort(1)[:, :k]  # every pick on rank 0
    elif kind == "ragged" and m:
        # a few hot experts, every third expert and all of EP rank 1 empty: k distinct experts
        # with skewed weights
        w = torch.exp(torch.randn(ne, generator=g) * 3)
        w[1::3] = 0
        w[epr : 2 * epr] = 0
        assert int((w > 0).sum()) >= k
        ids = torch.multinomial(w.expand(m, ne), k, replacement=False, generator=g)
    return ids.cuda()


def check_shapes(outs, m: int, k: int, ep: int, epr: int) -> None:
    tper, dedup, dti, idxs, expand, send_meta = outs
    assert tper.shape == dedup.shape == (ep,)
    assert dti.shape == (m * ep,)  # over-allocated; the first sum(dedup) entries are valid
    assert idxs.shape == expand.shape == (m * k,)
    assert send_meta.shape == (ep * (epr + 1),)
    assert all(t.dtype == torch.int64 for t in outs)


def check_order_free(outs, ref) -> None:
    assert torch.equal(outs[0], ref["tokens_per_ep_rank"])
    assert torch.equal(outs[1], ref["dedup"])
    assert torch.equal(outs[5], ref["send_meta"])


def check_stable(outs, ref) -> None:
    """Exact: the H200 path places stably."""
    check_order_free(outs, ref)
    total = int(ref["dedup"].sum())
    assert torch.equal(outs[3], ref["idxs"])
    assert torch.equal(outs[4], ref["expand"])
    assert torch.equal(outs[2][:total], ref["dti"])


def buckets(values: torch.Tensor, counts: list[int]) -> list[list[int]]:
    v, out, s = values.cpu().tolist(), [], 0
    for c in counts:
        out.append(sorted(v[s : s + c]))
        s += c
    return out


def check_any_order(outs, ref, topk_ids: torch.Tensor, ep: int, epr: int) -> None:
    """Order-free outputs exact; order-carrying ones as per-bucket sets plus the invariant."""
    check_order_free(outs, ref)
    k = topk_ids.shape[1]
    _, dedup, dti, idxs, expand, _ = outs
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=ep * epr).tolist()
    assert buckets(idxs, counts) == buckets(ref["idxs"], counts)
    dcounts = dedup.tolist()
    total = sum(dcounts)
    assert buckets(dti[:total], dcounts) == buckets(ref["dti"], dcounts)
    gpu = topk_ids.reshape(-1).long()[idxs] // epr
    gstarts = torch.cumsum(dedup, 0) - dedup
    assert bool(((expand >= 0) & (expand < dedup[gpu])).all())
    assert torch.equal(dti[gstarts[gpu] + expand], idxs // k)


def run_dispatch(topk_ids: torch.Tensor, ep: int, epr: int, meta_out=None):
    outs = fused_dedup_prepare_dispatch(topk_ids, ep * epr, ep, epr, meta_out)
    check_shapes(outs, topk_ids.shape[0], topk_ids.shape[1], ep, epr)
    return outs


# ---------------------------------------------------------------------------
# fused_dedup_prepare_dispatch (sender side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [0, 1, 511, 513, 4133])  # the CuTe kernel's chunk is 512 tokens
@pytest.mark.parametrize("ep", [1, 2, 4, 8])
@pytest.mark.parametrize("ne,k", MODELS)
def test_dispatch_models(ne: int, k: int, ep: int, m: int) -> None:
    epr = ne // ep
    topk_ids = routing(m, k, ne, epr, seed=m + k + ep)
    check_stable(run_dispatch(topk_ids, ep, epr), stable_reference(topk_ids, ep, epr))


@pytest.mark.parametrize("k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("m", [4096, 65536, 70000])
def test_dispatch_odd_experts_per_rank(k: int, ep: int, m: int) -> None:
    """Odd experts per rank, up to 32 EP ranks, token counts across many CTAs."""
    epr = 5 if ep <= 4 else 3
    topk_ids = routing(m, k, ep * epr, epr, seed=m + k)
    check_stable(run_dispatch(topk_ids, ep, epr), stable_reference(topk_ids, ep, epr))


@pytest.mark.parametrize("kind", ["one_expert", "one_rank", "ragged"])
@pytest.mark.parametrize(
    "ep,epr,k", [(4, 32, 8), (8, 32, 8), (4, 64, 8), (8, 8, 6), (4, 8, 4), (8, 16, 4)]
)
def test_dispatch_skewed(kind: str, ep: int, epr: int, k: int) -> None:
    """Hot experts, every token on one rank, empty experts and ranks."""
    if kind == "one_rank" and epr < k:
        pytest.skip("needs k experts on one rank")
    topk_ids = routing(4096, k, ep * epr, epr, kind, seed=3)
    ref = stable_reference(topk_ids, ep, epr)
    if kind == "ragged":
        assert int((ref["send_meta"] == 0).sum()) > 0  # some empty experts
        assert int(ref["dedup"][1]) == 0  # an empty EP rank
    check_stable(run_dispatch(topk_ids, ep, epr), ref)


def test_dispatch_m_zero() -> None:
    for ep, epr, k in ((4, 4, 4), (1, 64, 6), (8, 32, 8)):
        topk_ids = torch.empty((0, k), dtype=torch.int64, device="cuda")
        outs = run_dispatch(topk_ids, ep, epr)
        assert int(outs[0].abs().sum()) == int(outs[1].abs().sum()) == 0
        assert int(outs[5].abs().sum()) == 0
        assert all(o.numel() == 0 for o in outs[2:5])


def test_dispatch_meta_out() -> None:
    """With meta_out, tokens_per_ep_rank and dedup_tokens_per_gpu are its first 2 * ep_size
    entries (the operator lays a larger metadata buffer out this way); the rest is untouched."""
    ep, epr, k = 4, 32, 8
    for m in (0, 4133):
        topk_ids = routing(m, k, ep * epr, epr, seed=9)
        meta = torch.full((2 * ep + ep * (epr + 1),), -7, dtype=torch.int64, device="cuda")
        outs = run_dispatch(topk_ids, ep, epr, meta)
        assert outs[0].data_ptr() == meta.data_ptr()
        assert outs[1].data_ptr() == meta.data_ptr() + ep * 8
        ref = stable_reference(topk_ids, ep, epr)
        assert torch.equal(meta[:ep], ref["tokens_per_ep_rank"])
        assert torch.equal(meta[ep : 2 * ep], ref["dedup"])
        assert bool((meta[2 * ep :] == -7).all())
        check_stable(outs, ref)


def test_dispatch_row_stride() -> None:
    """The models hand over a contiguous (m, k) topk; a wider row stride is honoured too."""
    ep, epr, k = 4, 32, 8
    wide = routing(4096, k + 3, ep * epr, epr, seed=5)
    topk_ids = wide[:, :k]
    assert topk_ids.stride(0) == k + 3
    check_stable(run_dispatch(topk_ids, ep, epr), stable_reference(topk_ids.contiguous(), ep, epr))


def test_dispatch_back_to_back() -> None:
    """Back-to-back calls with different m on one stream (reused scratch) stay exact, and the
    outputs of earlier calls are not overwritten by later ones."""
    ep, epr, k = 8, 16, 8
    cases = [routing(m, k, ep * epr, epr, seed=m) for m in (70000, 1, 4096, 33, 65536, 4096)]
    outs = [run_dispatch(t, ep, epr) for t in cases]
    for t, o in zip(cases, outs):
        check_stable(o, stable_reference(t, ep, epr))


def test_dispatch_grid_barrier_stress() -> None:
    """Repeated launches whose grid spans every SM (many 512-token chunks per CTA), alternating
    with small ones: each result stays exact."""
    ep, epr, k = 8, 16, 8
    big = routing(300_000, k, ep * epr, epr, kind="ragged", seed=21)
    small = routing(700, k, ep * epr, epr, seed=22)
    refs = {id(t): stable_reference(t, ep, epr) for t in (big, small)}
    for i in range(12):
        t = big if i % 2 == 0 else small
        check_stable(run_dispatch(t, ep, epr), refs[id(t)])


def test_dispatch_stability_is_enforced() -> None:
    """A placement atomics could produce (two rows of one bucket swapped) passes the per-bucket
    comparison but not the stable one: the exact comparison really checks order."""
    ep, epr, k = 4, 8, 4
    topk_ids = routing(1024, k, ep * epr, epr, seed=11)
    outs = list(run_dispatch(topk_ids, ep, epr))
    ref = stable_reference(topk_ids, ep, epr)
    assert int(outs[5][0]) >= 2
    swapped = outs[3].clone()
    swapped[[0, 1]] = swapped[[1, 0]]  # both in expert 0's bucket
    assert not torch.equal(swapped, ref["idxs"])
    counts = torch.bincount(topk_ids.reshape(-1), minlength=ep * epr).tolist()
    assert buckets(swapped, counts) == buckets(ref["idxs"], counts)


# Inputs the H200's CuTe kernel does not cover take the atomic-placement fallback.


def test_dispatch_fallback_int32_ids() -> None:
    for ep, epr, k in ((4, 32, 8), (8, 8, 6), (1, 64, 6)):
        topk_ids = routing(4133, k, ep * epr, epr, seed=2).int()
        outs = run_dispatch(topk_ids, ep, epr)
        check_any_order(outs, stable_reference(topk_ids, ep, epr), topk_ids, ep, epr)


def test_dispatch_fallback_many_bins() -> None:
    """ep_size * (experts_per_rank + 1) > 1024 bins, with the fewest experts that get there: the
    fallback unrolls a loop per expert, and 32 x 40 took minutes to compile."""
    ep, epr, k = 32, 32, 8
    for m in (513,):
        topk_ids = routing(m, k, ep * epr, epr, seed=m)
        outs = run_dispatch(topk_ids, ep, epr)
        check_any_order(outs, stable_reference(topk_ids, ep, epr), topk_ids, ep, epr)


# ---------------------------------------------------------------------------
# expert_and_expand_idx (receiver side)
# ---------------------------------------------------------------------------


def expand_reference(meta: torch.Tensor, rex: torch.Tensor, ep: int, epr: int):
    """Slots are sender-major, expert-minor: each slot's local expert, and its sender-local
    expand_idx plus the dedup tokens of the preceding senders."""
    m2 = meta.view(ep, epr + 1)
    seg = m2[:, :epr].reshape(-1)
    expert = torch.arange(epr, device=meta.device).repeat(ep).repeat_interleave(seg)
    dstart = m2[:, epr].cumsum(0) - m2[:, epr]
    off = dstart.repeat_interleave(m2[:, :epr].sum(1))
    return expert, rex + off


def recv_case(ep: int, epr: int, total_target: int, seed: int, zero_frac: float = 0.3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = ep * epr
    w = torch.rand(n, generator=g) * (torch.rand(n, generator=g) > zero_frac)
    seg = (w / w.sum().clamp_min(1e-9) * total_target).long()
    dedup = torch.randint(1, 4000, (ep,), generator=g)
    meta = torch.cat([seg.view(ep, epr), dedup[:, None]], 1).reshape(-1).contiguous().cuda()
    rex = torch.randint(0, 4000, (int(seg.sum()),), generator=g).cuda()
    return meta, rex


def check_expand(meta, rex, ep, epr, ref_rex=None) -> None:
    got = expert_and_expand_idx(meta, rex, ep, epr)
    ref = expand_reference(meta, rex if ref_rex is None else ref_rex, ep, epr)
    assert got[0].dtype == got[1].dtype == torch.int64
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


# (ep_size, experts_per_rank) up to 2048 segments: a larger count leaves the H200's CuTe kernel
# for a Triton kernel that unrolls every segment (minutes to compile), which no model reaches.
EXPAND_GEOMETRIES = [
    (ep, epr)
    for ep in (1, 2, 4, 8, 16, 32)
    for epr in (1, 3, 5, 8, 32, 64, 128, 256)
    if ep * epr <= 2048
]


@pytest.mark.parametrize("ep,epr", EXPAND_GEOMETRIES)
def test_expert_and_expand_idx(ep: int, epr: int) -> None:
    for seed, total in ((0, 3000), (1, 70000), (2, 1), (3, 1023), (4, 1025)):
        meta, rex = recv_case(ep, epr, total, seed)
        check_expand(meta, rex, ep, epr)


def test_expert_and_expand_idx_empty() -> None:
    for ep, epr in ((4, 8), (1, 64), (8, 32)):
        meta = torch.zeros(ep * (epr + 1), dtype=torch.int64, device="cuda")
        rex = torch.empty(0, dtype=torch.int64, device="cuda")
        e, x = expert_and_expand_idx(meta, rex, ep, epr)
        assert e.shape == x.shape == (0,) and e.dtype == x.dtype == torch.int64


def test_expert_and_expand_idx_sensitive() -> None:
    """One segment count off by one shifts later slots: the comparison notices."""
    meta, rex = recv_case(4, 32, 5000, 7, zero_frac=0.0)
    bad = meta.clone()
    bad[0] += 1
    bad[1] -= 1
    got = expert_and_expand_idx(bad, rex, 4, 32)
    assert not torch.equal(got[0], expand_reference(meta, rex, 4, 32)[0])


def test_expert_and_expand_idx_fallback_strided() -> None:
    """A non-contiguous received expand_idx (outside the CuTe kernel's cover)."""
    for ep, epr, total in ((4, 8, 300), (8, 32, 70000)):
        meta, rex = recv_case(ep, epr, total, 0)
        strided = torch.empty(2 * rex.numel(), dtype=torch.int64, device="cuda")[::2]
        strided.copy_(rex)
        assert not strided.is_contiguous()
        check_expand(meta, strided, ep, epr, ref_rex=rex)


# ---------------------------------------------------------------------------
# Both sides, with the operator's all-to-alls emulated in one process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ne,k", MODELS)
@pytest.mark.parametrize("ep", [2, 4, 8])
def test_dispatch_round_trip(ne: int, k: int, ep: int) -> None:
    """
    ep_size senders with their own routing (ragged token counts, one empty): each receiver
    gathers the senders' send_meta rows and expand_idx segments for it, as the operator's
    all-to-alls do, and every received slot must name its sender's token (through the
    concatenated dedup chunks and the global expand_idx) and that token's expert.
    """
    epr = ne // ep
    ms = [0 if s == 1 else 37 + 1000 * s for s in range(ep)]
    senders = []
    for s, m in enumerate(ms):
        topk_ids = routing(m, k, ne, epr, kind="ragged" if s % 2 else "random", seed=100 + s)
        outs = run_dispatch(topk_ids, ep, epr)
        check_stable(outs, stable_reference(topk_ids, ep, epr))
        senders.append((topk_ids, outs))
    for r in range(ep):
        recv_meta = torch.cat([o[5].view(ep, epr + 1)[r] for _, o in senders])
        rex, tokens, experts, dedup_tokens = [], [], [], []
        for s, (topk_ids, o) in enumerate(senders):
            tper, dedup, dti, idxs, expand, _ = o
            lo, hi = int(tper[:r].sum()), int(tper[: r + 1].sum())
            rex.append(expand[lo:hi])
            tokens.append(idxs[lo:hi] // k + s * 1_000_000)  # tag tokens with their sender
            experts.append(topk_ids.reshape(-1)[idxs[lo:hi]] - r * epr)
            dlo, dhi = int(dedup[:r].sum()), int(dedup[: r + 1].sum())
            dedup_tokens.append(dti[dlo:dhi] + s * 1_000_000)
        rex = torch.cat(rex)
        expert_idxs, expand_idx = expert_and_expand_idx(recv_meta, rex, ep, epr)
        assert torch.equal(expert_idxs, torch.cat(experts))
        assert torch.equal(torch.cat(dedup_tokens)[expand_idx], torch.cat(tokens))


# ---------------------------------------------------------------------------
# Cases absorbed from tests/test_ep_dedup_dispatch.py (single-process dispatch simulation)
# ---------------------------------------------------------------------------

# (tokens per sender, top-k, num_experts, ep_size)
SIM_CONFIGS = [
    (32, 8, 32, 8),
    (64, 4, 16, 4),
    (128, 8, 64, 8),
    (16, 2, 8, 4),
    (100, 8, 32, 4),
    (256, 8, 128, 16),
    (10, 4, 8, 2),
    (32, 1, 8, 4),  # k=1: no dedup
    (32, 4, 8, 2),  # many experts per rank
    (2, 2, 4, 2),  # tiny
    (1, 2, 8, 4),  # single token
    (2048, 8, 128, 2),  # Qwen3-30B-A3B: many experts, few EP ranks
]


def randperm_routing(ms: int, k: int, num_experts: int) -> torch.Tensor:
    return torch.stack([torch.randperm(num_experts, device="cuda")[:k] for _ in range(ms)])


@pytest.mark.parametrize("ms,k,num_experts,ep_size", SIM_CONFIGS)
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_fused_dedup_dispatch(ms: int, k: int, num_experts: int, ep_size: int, seed: int) -> None:
    """Every sender-side output against the stable PyTorch reference."""
    torch.manual_seed(seed)
    epr = num_experts // ep_size
    topk_ids = randperm_routing(ms, k, num_experts)
    check_stable(run_dispatch(topk_ids, ep_size, epr), stable_reference(topk_ids, ep_size, epr))


@pytest.mark.parametrize("ms,k,num_experts,ep_size", SIM_CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_fused_dedup_end_to_end(ms: int, k: int, num_experts: int, ep_size: int, seed: int) -> None:
    """
    Full sender -> receiver pipeline on hidden states: each receiver gathers the senders' dedup
    token chunks and expand_idx segments, makes expand_idx global through expert_and_expand_idx,
    and the expanded rows equal each slot's own token row.
    """
    torch.manual_seed(seed)
    epr = num_experts // ep_size
    senders = []
    for _ in range(ep_size):
        h = torch.randn(ms, 64, device="cuda")
        ids = randperm_routing(ms, k, num_experts)
        tper, dedup, dti, idxs, expand, send_meta = run_dispatch(ids, ep_size, epr)
        dedup_rows = h[dti[: int(dedup.sum())]]
        senders.append(
            {
                "meta": send_meta.view(ep_size, epr + 1),
                "dedup_chunks": dedup_rows.split(dedup.tolist()),
                "expand_chunks": expand.split(tper.tolist()),
                "ref_chunks": h[idxs // k].split(tper.tolist()),
            }
        )
    for g in range(ep_size):
        recv_meta = torch.cat([s["meta"][g] for s in senders])
        rex = torch.cat([s["expand_chunks"][g] for s in senders])
        _, expand_idx = expert_and_expand_idx(recv_meta, rex, ep_size, epr)
        gathered = torch.cat([s["dedup_chunks"][g] for s in senders])
        if expand_idx.numel():
            assert int(expand_idx.min()) >= 0 and int(expand_idx.max()) < gathered.shape[0]
        reference = torch.cat([s["ref_chunks"][g] for s in senders])
        assert torch.equal(gathered[expand_idx], reference), f"GPU {g}: expanded != reference"


def test_fused_dedup_dispatch_m_zero() -> None:
    topk_ids = torch.empty((0, 4), dtype=torch.int64, device="cuda")
    outs = run_dispatch(topk_ids, 4, 4)
    assert int(outs[5].sum()) == 0 and outs[3].numel() == outs[4].numel() == 0


def recv_meta_of(tpe: torch.Tensor, dedup: torch.Tensor) -> torch.Tensor:
    """Interleaved received metadata: per sender, its per-expert counts then its dedup count."""
    return torch.cat([tpe, dedup[:, None]], dim=1).reshape(-1).contiguous()


@pytest.mark.parametrize("ms,k,num_experts,ep_size", SIM_CONFIGS)
@pytest.mark.parametrize("seed", [0, 42])
def test_expert_and_expand_idx_sim(
    ms: int, k: int, num_experts: int, ep_size: int, seed: int
) -> None:
    """Random received counts: each slot's local expert (slots sender-major, expert-minor) and
    its expand_idx plus the dedup counts of the preceding senders."""
    torch.manual_seed(seed)
    epr = num_experts // ep_size
    tpe = torch.randint(0, max(ms, 2), (ep_size, epr), device="cuda")
    dedup = torch.randint(1, max(ms, 2) + 1, (ep_size,), device="cuda")
    splits = tpe.sum(dim=1)
    total = int(splits.sum())
    sender = torch.arange(ep_size, device="cuda").repeat_interleave(splits)
    rex = (torch.rand(total, device="cuda") * dedup[sender]).long()
    meta = recv_meta_of(tpe, dedup)
    expert_idxs, _ = expert_and_expand_idx(meta, rex, ep_size, epr)
    assert expert_idxs.numel() == total, "one entry per received token slot"
    check_expand(meta, rex, ep_size, epr)


def test_expert_and_expand_idx_slot_count_from_input() -> None:
    """
    The slot count is the received expand_idx's length, decided by the senders, even where the
    metadata's segment counts add up to more: the outputs have exactly that many entries, those
    entries are right, and a tail not a multiple of any CTA size is handled.
    """
    ep_size, epr = 8, 16
    total = 5501
    tpe = torch.full((ep_size, epr), total // (ep_size * epr), device="cuda")
    tpe.view(-1)[: total % (ep_size * epr)] += 1
    assert int(tpe.sum()) == total
    dedup = torch.full((ep_size,), 64, device="cuda")
    meta = recv_meta_of(tpe, dedup)
    rex = torch.randint(0, 64, (total,), device="cuda")
    check_expand(meta, rex, ep_size, epr)
    ref_e, ref_x = expand_reference(meta, rex, ep_size, epr)
    for short in (total - 1, 4096, 1000, 1):
        e, x = expert_and_expand_idx(meta, rex[:short], ep_size, epr)
        assert e.shape == x.shape == (short,)
        assert torch.equal(e, ref_e[:short]) and torch.equal(x, ref_x[:short])
