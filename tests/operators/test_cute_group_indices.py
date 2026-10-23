"""One-launch CuTe group_indices against precompute_group_indices' searchsorted formula."""

import pytest
import torch

from pithtrain.operators.dispatch_cute import group_indices as cgi

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 CuTe kernels",
)


def reference(offs: torch.Tensor, M: int) -> torch.Tensor:
    rows = torch.arange(M, device=offs.device)
    return torch.searchsorted(offs, rows, right=True).to(torch.int32).clamp_(max=offs.shape[0] - 1)


@pytest.mark.parametrize("G", [1, 3, 8, 16, 32, 64, 128, 1024])
@pytest.mark.parametrize("seed", [0, 1])
def test_group_indices(G, seed):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 3000, (G,), generator=g)
    counts[torch.rand(G, generator=g) < 0.3] = 0  # empty groups
    ks = (counts + 127) // 128 * 128
    offs = ks.cumsum(0).to(torch.int32).cuda()
    M = int(offs[-1])
    for rows in {M, max(M - 5, 0), M + 300}:  # exact, short, and past the last end (clamped)
        assert torch.equal(cgi.group_indices(offs, rows), reference(offs, rows))


def test_group_indices_empty():
    offs = torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
    assert cgi.group_indices(offs, 0).shape == (0,)
    assert torch.equal(cgi.group_indices(offs, 7), reference(offs, 7))


def test_precompute_uses_cute(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.operators.token_scatter import precompute_group_indices

    monkeypatch.setattr(training, "fp8", True, raising=False)
    offs = torch.tensor([128, 256, 256, 640], dtype=torch.int32, device="cuda")
    calls = []
    real = cgi.group_indices
    monkeypatch.setattr(cgi, "group_indices", lambda *a: calls.append(1) or real(*a))
    assert torch.equal(precompute_group_indices(offs, 640), reference(offs, 640))
    assert calls == [1]
