"""
SequenceLevelLoadBalanceLoss without context parallelism: a size-1 CP group gives the same loss and
gradient bits as no group, eager and compiled, and the compiled graph issues no collective for it
(a sum over one rank is the identity).
"""

import pytest
import torch
import torch.distributed as dist

from pithtrain.modules.load_balance import SequenceLevelLoadBalanceLoss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

S, E, TOP_K, B = 4096, 64, 6, 1


@pytest.fixture(scope="module")
def cp_group():
    if not dist.is_initialized():
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
    yield dist.new_group([0])


def loss_and_grad(fn, logits: torch.Tensor, topk_idx: torch.Tensor):
    logits = logits.detach().requires_grad_()
    loss = fn(logits.softmax(dim=-1, dtype=torch.float32), topk_idx)
    loss.backward()
    return loss.detach(), logits.grad


@pytest.mark.parametrize("compiled", [False, True])
def test_size1_group_matches_no_group(cp_group, compiled: bool) -> None:
    torch.manual_seed(0)
    logits = torch.randn(B * S, E, device="cuda", dtype=torch.float32)
    topk_idx = torch.topk(logits, TOP_K, dim=-1).indices
    out = []
    graphs: list[str] = []
    for group in (None, cp_group):
        lb = SequenceLevelLoadBalanceLoss(1e-3, S, group)
        fn = lambda scores, idx: lb(scores, idx, E, TOP_K)  # noqa: E731
        if compiled:
            torch._dynamo.reset()

            def backend(gm, example_inputs):
                graphs.append(str(gm.graph))
                return gm.forward

            fn = torch.compile(fn, fullgraph=True, backend=backend)
        out.append(loss_and_grad(fn, logits, topk_idx))
    (l0, g0), (l1, g1) = out
    assert torch.equal(l0, l1) and torch.equal(g0, g1)
    for graph in graphs:
        assert "all_reduce" not in graph and "wait_tensor" not in graph
