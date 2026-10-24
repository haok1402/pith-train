"""
Index maps of the BF16 CuTe experts (kernels.bf16_expert_slots_sm90.group_slots) and the gather-sum of the
grouped input gradient back to the received tokens, against PyTorch references, on EP-dedup
routings laid out as ep_dispatch delivers them: slots sender-major, expert-minor, a token repeated
once per local expert it chose (repeated expand_idx), empty experts, and ep_size 1 (no expand).
"""

import pytest
import torch

from pithtrain.kernels.arch import is_sm90
from pithtrain.kernels.bf16_expert_support import QUACK_VERSION
from pithtrain.kernels.bf16_expert_support import quack_version as _quack_version

pytestmark = pytest.mark.skipif(
    not is_sm90() or _quack_version() != QUACK_VERSION,
    reason="needs SM90 and the validated quack-kernels",
)


def ep_routing(G: int, ep: int, tokens_per_sender: int, k: int, empty=(), seed=0):
    """(expert_idxs, expand_idx, T): per sender, each token picks up to k of the G local experts
    (none of ``empty``); slots sender-major, expert-minor; expand_idx indexes the senders'
    concatenated dedup'd tokens."""
    g = torch.Generator().manual_seed(seed)
    allowed = torch.tensor([e for e in range(G) if e not in empty])
    eids, exps, base = [], [], 0
    for _ in range(ep):
        choice = []
        for t in range(tokens_per_sender):
            n = int(torch.randint(1, k + 1, (1,), generator=g))
            for e in allowed[torch.randperm(len(allowed), generator=g)[:n]].tolist():
                choice.append((e, t))
        choice.sort()
        eids += [e for e, _ in choice]
        exps += [base + t for _, t in choice]
        base += tokens_per_sender
    return torch.tensor(eids).cuda(), torch.tensor(exps).cuda(), base


def check_maps(maps, expert_idxs, expand_idx, G):
    S = expert_idxs.shape[0]
    counts = torch.bincount(expert_idxs, minlength=G)
    cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    assert maps.cu_seqlens.tolist() == cu.tolist()
    rows = maps.row_of.long()
    assert torch.equal(torch.sort(rows).values, torch.arange(S, device="cuda"))
    assert torch.equal(maps.slot_of.long()[rows], torch.arange(S, device="cuda"))
    src = expand_idx if expand_idx is not None else torch.arange(S, device="cuda")
    assert torch.equal(maps.a_idx.long()[rows], src)
    # expert-major, and stable: the rows of an expert hold its slots in slot order
    order = torch.argsort(expert_idxs, stable=True)
    assert torch.equal(maps.slot_of.long(), order)


@pytest.mark.parametrize(
    "G,ep,tps,k,empty",
    [(32, 4, 2048, 8, ()), (16, 8, 1000, 6, (3, 7)), (64, 4, 1500, 8, (0, 63)), (8, 2, 3, 4, ())],
)
def test_group_slots_maps(G, ep, tps, k, empty):
    from pithtrain.kernels.bf16_expert_slots_sm90 import group_slots

    expert_idxs, expand_idx, T = ep_routing(G, ep, tps, k, empty)
    maps = group_slots(expert_idxs, expand_idx, G, T)
    check_maps(maps, expert_idxs, expand_idx, G)
    # deterministic: a second run gives the same maps
    again = group_slots(expert_idxs, expand_idx, G, T)
    for a, b in zip(maps[:5], again[:5]):
        assert torch.equal(a, b)


def test_group_slots_without_expand():
    from pithtrain.kernels.bf16_expert_slots_sm90 import group_slots

    g = torch.Generator().manual_seed(1)
    expert_idxs = torch.randint(0, 16, (5000,), generator=g).cuda()
    maps = group_slots(expert_idxs, None, 16, 5000)
    assert maps.mask is None
    check_maps(maps, expert_idxs, None, 16)


def gather_sum_reference(grad, expert_idxs, expand_idx, row_of, T):
    """fp32 sum per token of its grouped rows in ascending expert order, rounded once."""
    out = torch.zeros(T, grad.shape[1], dtype=torch.float32, device=grad.device)
    order = torch.argsort(expert_idxs * (T + 1) + expand_idx, stable=True)
    # visit slots by expert, so each token's rows are added in expert order
    for e in torch.unique(expert_idxs).tolist():
        sl = order[expert_idxs[order] == e]
        out.index_add_(0, expand_idx[sl], grad[row_of[sl].long()].float())
    return out.to(grad.dtype)


@pytest.mark.parametrize("G,ep,tps,k,H", [(32, 4, 2048, 8, 2048), (8, 4, 1500, 4, 2880), (64, 2, 700, 8, 2048)])  # fmt: skip
def test_gather_sum(G, ep, tps, k, H):
    from pithtrain.kernels.bf16_expert_slots_sm90 import gather_sum, group_slots

    expert_idxs, expand_idx, T = ep_routing(G, ep, tps, k, seed=2)
    maps = group_slots(expert_idxs, expand_idx, G, T)
    grad = torch.randn(expert_idxs.shape[0], H, device="cuda", dtype=torch.bfloat16)
    out = gather_sum(grad, maps, G, min(k, G))
    ref = gather_sum_reference(grad, expert_idxs, expand_idx, maps.row_of, T)
    # same order, same fp32 adds, one rounding: exact
    assert torch.equal(out, ref)
    assert torch.equal(out, gather_sum(grad, maps, G, min(k, G)))


def test_gather_sum_perturbed_map_fails():
    """A wrong row map is caught by the exact comparison."""
    from pithtrain.kernels.bf16_expert_slots_sm90 import gather_sum, group_slots

    expert_idxs, expand_idx, T = ep_routing(16, 4, 500, 4, seed=3)
    maps = group_slots(expert_idxs, expand_idx, 16, T)
    grad = torch.randn(expert_idxs.shape[0], 2048, device="cuda", dtype=torch.bfloat16)
    ref = gather_sum_reference(grad, expert_idxs, expand_idx, maps.row_of, T)
    # point every valid table entry at the next grouped row
    valid = (expand_idx * 16 + expert_idxs).long()
    maps.table[valid] = (maps.table[valid] + 1) % expert_idxs.shape[0]
    assert not torch.equal(gather_sum(grad, maps, 16, 4), ref)


@pytest.mark.parametrize("N", [2048, 2880, 8])
def test_gather_rows(N):
    from pithtrain.kernels.bf16_expert_slots_sm90 import gather_rows

    x = torch.randn(3000, N, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, 3000, (5001,), device="cuda", dtype=torch.int32)
    assert torch.equal(gather_rows(x, idx), x[idx.long()])
    assert gather_rows(x, idx[:0]).shape == (0, N)
