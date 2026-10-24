"""
Stage 3 of an FP8 MoE layer through fp8_scatter_quant (expand + scatter + quantize fused) against the
three-step path (padded_index_gather -> scatter_for_grouped_gemm -> experts). The forward is
row-wise, so each token's output must be bit-identical; gradients go through 128-row column-wise
quantization, whose blocks depend on where rows land in each expert group (atomic order before,
token order now), so they are compared with calc_diff. Hopper only.
"""

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.operators.fp8_experts import fp8_scatter_quant, fp8_swiglu_experts
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper-only fused path")


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def make(G: int, H: int, inter: int) -> tuple[FP8GroupedLinear, ...]:
    torch.manual_seed(0)
    mods = (
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, H, inter),
        FP8GroupedLinear(G, inter, H),
    )
    for m in mods:
        torch.nn.init.normal_(m.weight, std=0.02)
    return tuple(m.cuda().to(torch.bfloat16) for m in mods)


def stage3(mods, gathered, expert_idxs, expand_idx, counts, fused: bool):
    FP8WeightCacheControl.step()
    gathered = gathered.detach().requires_grad_()
    G = mods[0].num_groups
    if fused:
        x, rsi, offs, ks, ks_t, xq = fp8_scatter_quant(gathered, expert_idxs, G, expand_idx, counts)
    else:
        expanded = padded_index_gather(gathered, expand_idx)
        x, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(expanded, expert_idxs, G, group_counts=counts)  # fmt: skip
        xq = None
    gi = precompute_group_indices(offs, x.shape[0])
    y = fp8_swiglu_experts(*mods, x, offs, ks, ks_t, gi, xq)
    out = padded_index_gather(y, rsi)
    torch.manual_seed(2)
    out.backward(torch.randn_like(out) * 1e-2)
    return out.detach(), gathered.grad, [m.weight.grad for m in mods]


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("expansion", [1, 4])
def test_fused_stage3_matches_three_step(skew: bool, expansion: int, monkeypatch) -> None:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    G, H, inter, m = 16, 2048, 768, 16384
    torch.manual_seed(1)
    gathered = torch.randn(m // expansion, H, device="cuda", dtype=torch.bfloat16)
    expand_idx = torch.randint(0, m // expansion, (m,), device="cuda")
    probs = torch.rand(G, device="cuda") ** 4 if skew else torch.ones(G, device="cuda")
    probs[3] = 0  # an empty expert
    expert_idxs = torch.multinomial(probs, m, replacement=True)
    counts = torch.bincount(expert_idxs, minlength=G).tolist()
    results = [stage3(make(G, H, inter), gathered, expert_idxs, expand_idx, counts, f) for f in (False, True)]  # fmt: skip
    (o0, dx0, w0), (o1, dx1, w1) = results
    assert torch.equal(o0, o1)
    assert calc_diff(dx0, dx1) < 1e-3
    for a, b in zip(w0, w1):
        assert calc_diff(a, b) < 1e-3
