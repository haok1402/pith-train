"""CuTe dispatch index kernels against a stable PyTorch reference and the Triton kernels.

Parity rule (kda-task/CONTRACT.md): the Triton kernels place rows in ``atomic_add`` order, so
order-carrying outputs (``idxs``, ``expand_idx``, ``dispatch_token_idxs``) are compared with Triton
only after sorting within each bucket, plus exact semantic invariants; against the stable
reference (``argsort(stable=True)`` / ``nonzero``) every output is ``torch.equal``. Order-free
outputs are ``torch.equal`` to both.
"""

import pytest
import torch

from pithtrain.operators import ep_dispatch as epd
from pithtrain.operators.dispatch_cute import dispatch_index as cdi

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 CuTe kernels",
)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def stable_reference(topk_ids: torch.Tensor, ep: int, epr: int) -> dict:
    m, k = topk_ids.shape
    ne = ep * epr
    flat = topk_ids.reshape(-1)
    idxs = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ne)
    visit = torch.zeros(m, ep, dtype=torch.bool, device=topk_ids.device)
    if m:
        visit.scatter_(1, topk_ids // epr, True)
    dedup = visit.sum(0)
    dti = visit.T.nonzero()[:, 1]
    rank = visit.long().cumsum(0) - 1
    expand = rank[idxs // k, flat[idxs] // epr] if m else idxs
    inv = torch.empty_like(idxs)
    inv[idxs] = torch.arange(idxs.numel(), device=idxs.device)
    return {
        "tokens_per_ep_rank": counts.view(ep, epr).sum(1),
        "dedup": dedup,
        "dti": dti,
        "idxs": idxs,
        "expand": expand,
        "send_meta": torch.cat([counts.view(ep, epr), dedup[:, None]], 1).reshape(-1),
        "inv": inv,
    }


def routing(m: int, k: int, ep: int, epr: int, kind: str = "random", seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ne = ep * epr
    ids = torch.rand(m, ne, generator=g).argsort(1)[:, :k]
    if kind == "one_expert" and m:
        ids[:, 0] = 0  # every token picks expert 0 (its other picks stay distinct)
        ids[:, 1:] = (torch.rand(m, ne - 1, generator=g).argsort(1)[:, : k - 1]) + 1
    elif kind == "one_rank" and m:
        assert epr >= k
        ids = torch.rand(m, epr, generator=g).argsort(1)[:, :k]  # every pick on rank 0
    elif kind == "ragged" and m:
        # a few hot experts: sample k distinct experts with skewed weights
        w = torch.exp(torch.randn(ne, generator=g) * 3)
        ids = torch.multinomial(w.expand(m, ne), k, replacement=False, generator=g)
    return ids.cuda()


def check_vs_stable(outs, ref, total_dedup: int):
    tper, dedup, dti, idxs, expand, send_meta, *rest = outs
    assert torch.equal(tper, ref["tokens_per_ep_rank"])
    assert torch.equal(dedup, ref["dedup"])
    assert torch.equal(send_meta, ref["send_meta"])
    assert torch.equal(idxs, ref["idxs"])
    assert torch.equal(expand, ref["expand"])
    assert torch.equal(dti[:total_dedup], ref["dti"])
    if rest:
        assert torch.equal(rest[0], ref["inv"])


def canon(values: torch.Tensor, starts: list[int], counts: list[int]) -> list[list[int]]:
    v = values.cpu().tolist()
    return [sorted(v[s : s + c]) for s, c in zip(starts, counts)]


def check_vs_triton(outs, tri, topk_ids, ep, epr):
    """Order-free outputs equal; order-carrying ones equal per bucket after sorting; invariants."""
    m, k = topk_ids.shape
    for a, b in zip((0, 1, 5), (0, 1, 5)):
        assert torch.equal(outs[a], tri[b])
    counts = outs[5].view(ep, epr + 1)[:, :epr].reshape(-1).tolist()
    starts = [sum(counts[:e]) for e in range(len(counts))]
    assert canon(outs[3], starts, counts) == canon(tri[3], starts, counts)  # idxs per expert
    dedup = outs[1].tolist()
    gstarts = [sum(dedup[:g]) for g in range(ep)]
    total = sum(dedup)
    assert canon(outs[2][:total], gstarts, dedup) == canon(tri[2][:total], gstarts, dedup)
    # expand_idx: (expert bucket, token) -> dedup rank is a function of the placement; compare the
    # (token, rank) pairs it implies through each side's own dispatch_token_idxs instead
    for o in (outs, tri):
        idxs, expand, dti = o[3], o[4], o[2]
        gpu = topk_ids.reshape(-1)[idxs] // epr
        gs = torch.tensor(gstarts, device=idxs.device)
        assert torch.equal(dti[gs[gpu] + expand], idxs // k)  # expand_idx semantic invariant


# ---------------------------------------------------------------------------
# expert_and_expand_idx (receiver side)
# ---------------------------------------------------------------------------


def expand_reference(meta: torch.Tensor, rex: torch.Tensor, ep: int, epr: int):
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
    seg = (w / w.sum().clamp_min(1e-9) * total_target).long() if total_target else w.long() * 0
    dedup = torch.randint(1, 4000, (ep,), generator=g)
    meta = torch.cat([seg.view(ep, epr), dedup[:, None]], 1).reshape(-1).contiguous().cuda()
    rex = torch.randint(0, 4000, (int(seg.sum()),), generator=g).cuda()
    return meta, rex


@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("epr", [1, 3, 5, 32, 64])
def test_expert_and_expand_idx(ep, epr):
    for seed, total in ((0, 3000), (1, 70000), (2, 1)):
        meta, rex = recv_case(ep, epr, total, seed)
        got = cdi.expert_and_expand_idx(meta, rex, ep, epr)
        ref = expand_reference(meta, rex, ep, epr)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        assert torch.equal(epd.expert_and_expand_idx(meta, rex, ep, epr)[1], ref[1])
        tri = epd.expert_and_expand_idx_triton(meta, rex, ep, epr)
        assert torch.equal(got[0], tri[0]) and torch.equal(got[1], tri[1])


def test_expert_and_expand_idx_sensitive():
    """One segment count off by one shifts every later slot: the comparison must notice."""
    meta, rex = recv_case(4, 32, 5000, 7, zero_frac=0.0)
    bad = meta.clone()
    bad[0] += 1
    bad[1] -= 1
    got = cdi.expert_and_expand_idx(bad, rex, 4, 32)
    ref = expand_reference(meta, rex, 4, 32)
    assert not torch.equal(got[0], ref[0])


def test_expert_and_expand_idx_empty(monkeypatch):
    """No slots: correctly shaped empty outputs and no launch."""
    monkeypatch.setattr(cdi, "_EXPAND", {})
    monkeypatch.setattr(cdi, "_compile_expand", lambda *a: pytest.fail("compiled for 0 slots"))
    meta = torch.zeros(4 * 9, dtype=torch.int64, device="cuda")
    rex = torch.empty(0, dtype=torch.int64, device="cuda")
    e, x = cdi.expert_and_expand_idx(meta, rex, 4, 8)
    assert e.shape == (0,) and x.shape == (0,) and e.dtype == x.dtype == torch.int64


# ---------------------------------------------------------------------------
# dispatch index (sender side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("m", [0, 1, 4096, 65536, 70000])
def test_dispatch_index(k, ep, m):
    epr = 5 if ep <= 4 else 3  # odd experts per rank
    ne = ep * epr
    topk_ids = routing(m, k, ep, epr, seed=m + k)
    outs = cdi.dispatch_index(topk_ids, ne, ep, epr)
    ref = stable_reference(topk_ids, ep, epr)
    total = int(ref["dedup"].sum())
    check_vs_stable(outs, ref, total)
    assert outs[2].shape == (m * ep,)  # over-allocated as the Triton path
    if m:
        check_vs_triton(
            outs, epd.fused_dedup_prepare_dispatch_triton(topk_ids, ne, ep, epr), topk_ids, ep, epr
        )


@pytest.mark.parametrize("kind", ["one_expert", "one_rank", "ragged"])
@pytest.mark.parametrize("ep,epr,k", [(4, 32, 8), (8, 32, 8), (4, 64, 8), (8, 8, 6), (4, 8, 4)])
def test_dispatch_index_skewed(kind, ep, epr, k):
    if kind == "one_rank" and epr < k:
        pytest.skip("needs k experts on one rank")
    topk_ids = routing(4096, k, ep, epr, kind, seed=3)
    outs = cdi.dispatch_index(topk_ids, ep * epr, ep, epr)
    ref = stable_reference(topk_ids, ep, epr)
    check_vs_stable(outs, ref, int(ref["dedup"].sum()))
    check_vs_triton(
        outs,
        epd.fused_dedup_prepare_dispatch_triton(topk_ids, ep * epr, ep, epr),
        topk_ids,
        ep,
        epr,
    )


def test_dispatch_index_strides():
    """The models hand over a contiguous (m, k) topk; a wider row stride is honoured too."""
    ep, epr, k = 4, 32, 8
    wide = routing(4096, k + 3, ep, epr, seed=5)
    topk_ids = wide[:, :k]
    assert topk_ids.stride(0) == k + 3
    outs = cdi.dispatch_index(topk_ids, ep * epr, ep, epr)
    ref = stable_reference(topk_ids.contiguous(), ep, epr)
    check_vs_stable(outs, ref, int(ref["dedup"].sum()))


def test_dispatch_index_scratch_reuse():
    """Back-to-back calls with different m on one stream match fresh-scratch calls."""
    ep, epr, k = 8, 16, 8
    cases = [routing(m, k, ep, epr, seed=m) for m in (70000, 1, 4096, 33, 65536, 4096)]
    outs = [cdi.dispatch_index(t, ep * epr, ep, epr) for t in cases]
    for t, o in zip(cases, outs):
        ref = stable_reference(t, ep, epr)
        check_vs_stable(o, ref, int(ref["dedup"].sum()))


def test_dispatch_index_stability_enforced():
    """A placement atomics could produce (two rows of one bucket swapped) passes the per-bucket
    comparison but fails torch.equal against the stable reference: the test enforces order."""
    ep, epr, k = 4, 8, 4
    topk_ids = routing(1024, k, ep, epr, seed=11)
    outs = list(cdi.dispatch_index(topk_ids, ep * epr, ep, epr))
    ref = stable_reference(topk_ids, ep, epr)
    swapped = outs[3].clone()
    swapped[[0, 1]] = swapped[[1, 0]]  # both in expert 0's bucket
    assert int(outs[5][0]) >= 2
    assert not torch.equal(swapped, ref["idxs"])
    counts = outs[5].view(ep, epr + 1)[:, :epr].reshape(-1).tolist()
    starts = [sum(counts[:e]) for e in range(len(counts))]
    assert canon(swapped, starts, counts) == canon(ref["idxs"], starts, counts)


def test_dispatch_index_m_zero(monkeypatch):
    monkeypatch.setattr(cdi, "_compile_index", lambda *a: pytest.fail("compiled for m == 0"))
    monkeypatch.setattr(cdi, "_INDEX", {})
    topk_ids = torch.empty((0, 4), dtype=torch.int64, device="cuda")
    outs = cdi.dispatch_index(topk_ids, 16, 4, 4)
    assert outs[0].shape == (4,) and outs[1].shape == (4,) and outs[5].shape == (20,)
    assert int(outs[0].sum()) == int(outs[1].sum()) == int(outs[5].sum()) == 0
    assert all(o.numel() == 0 for o in (outs[2], outs[3], outs[4], outs[6]))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def test_entry_points_use_cute(monkeypatch):
    """On SM90 the ep_dispatch entry points run the CuTe kernels; order-free outputs equal the
    Triton path's."""
    topk_ids = routing(4096, 8, 4, 32, seed=1)
    calls = []
    real = cdi.dispatch_index
    monkeypatch.setattr(cdi, "dispatch_index", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    a = epd.fused_dedup_prepare_dispatch(topk_ids, 128, 4, 32)
    b = epd.fused_dedup_prepare_dispatch_triton(topk_ids, 128, 4, 32)
    assert calls == [1]
    for i in (0, 1, 5):
        assert torch.equal(a[i], b[i])
    meta, rex = recv_case(4, 32, 3000, 0)
    real_expand = cdi.expert_and_expand_idx
    monkeypatch.setattr(cdi, "expert_and_expand_idx", lambda *a: calls.append(2) or real_expand(*a))
    got = epd.expert_and_expand_idx(meta, rex, 4, 32)
    tri = epd.expert_and_expand_idx_triton(meta, rex, 4, 32)
    assert calls == [1, 2] and torch.equal(got[0], tri[0]) and torch.equal(got[1], tri[1])


def test_unsupported_falls_back(monkeypatch):
    """More bins than the placement kernel's masks cover, or int32 ids, take the Triton path."""
    monkeypatch.setattr(cdi, "dispatch_index", lambda *a, **kw: pytest.fail("reached CuTe"))
    big = routing(512, 8, 32, 40, seed=2)  # 32 * 41 bins > _MAX_BINS
    outs = epd.fused_dedup_prepare_dispatch(big, 32 * 40, 32, 40)
    ref = stable_reference(big, 32, 40)
    assert torch.equal(outs[1], ref["dedup"])
    small = routing(512, 4, 4, 8, seed=2).int()
    assert not cdi.supports_index(small, 4, 8)
    monkeypatch.setattr(cdi, "expert_and_expand_idx", lambda *a, **kw: pytest.fail("reached CuTe"))
    meta, rex = recv_case(4, 8, 300, 0)
    strided = torch.empty(2 * rex.numel(), dtype=torch.int64, device="cuda")[::2]
    strided.copy_(rex)
    assert not cdi.supports_expand(strided, 4, 8)
    got = epd.expert_and_expand_idx(meta, strided, 4, 8)  # non-contiguous: the Triton path
    ref = expand_reference(meta, rex, 4, 8)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_compile_cache_dynamic_extents(monkeypatch):
    """A second call with the same (k, ep, epr) and another m does not recompile."""
    compiled = []
    real = cdi._compile_index
    monkeypatch.setattr(cdi, "_compile_index", lambda *a: compiled.append(a) or real(*a))
    monkeypatch.setattr(cdi, "_INDEX", {})
    for m in (100, 5000, 70000):
        cdi.dispatch_index(routing(m, 4, 4, 5, seed=m), 20, 4, 5)
    assert len(compiled) == 1 and compiled[0][:3] == (4, 4, 5)


def test_dispatch_index_barrier_stress():
    """Repeated launches whose grid spans every SM (many chunks per CTA) and alternate with small
    ones keep the grid barrier and its epoch counter consistent: each result matches the stable
    reference, and the arrival count returns to 0 after every launch."""
    ep, epr, k = 8, 16, 8
    big = routing(300_000, k, ep, epr, kind="ragged", seed=21)  # > 512 * SM count tokens
    small = routing(700, k, ep, epr, seed=22)
    refs = {id(t): stable_reference(t, ep, epr) for t in (big, small)}
    for i in range(12):
        t = big if i % 2 == 0 else small
        outs = cdi.dispatch_index(t, ep * epr, ep, epr)
        ref = refs[id(t)]
        check_vs_stable(outs, ref, int(ref["dedup"].sum()))
        ctr = int(cdi._SCRATCH[t.device.index].ctr.item())
        assert ctr & 0xFFFFFFFF == 0  # every launch's arrivals were consumed
