"""Correctness tests for the in-place fused cross-entropy operator."""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.cross_entropy import cross_entropy


@pytest.mark.parametrize("vocab", [1000, 32000, 151936])
def test_cross_entropy_loss_and_grad(vocab):
    torch.manual_seed(0)
    device = "cuda"
    tokens = 4096

    logits = (torch.randn(tokens, vocab, device=device) * 2).to(torch.bfloat16)
    target = torch.randint(0, vocab, (tokens,), device=device)
    target[::7] = -100  # ignored positions

    ref = logits.float().requires_grad_()
    loss_ref = F.cross_entropy(ref, target, ignore_index=-100)
    loss_ref.backward()

    inp = logits.clone().requires_grad_()
    loss = cross_entropy(inp, target)
    loss.backward()

    torch.testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(inp.grad.float(), ref.grad, atol=1e-6, rtol=2e-2)
    # The target entry is the one entry whose gradient is not a plain softmax term: it must hold
    # (p_y - 1) / N, never the stale logit (a race in the store loop once let it read one back).
    rows = torch.nonzero(target != -100).squeeze(1)
    torch.testing.assert_close(
        inp.grad.float()[rows, target[rows]], ref.grad[rows, target[rows]], atol=1e-6, rtol=2e-2
    )
