"""
Embedding with an fp32 main_grad: the forward matches nn.Embedding, no .grad is written, and main_grad
equals an fp32 index_add_ of the output gradients over micro-batches (repeated ids included).
"""

import pytest
import torch

from pithtrain.operators.embedding import Embedding


@pytest.mark.parametrize("vocab,hidden", [(1000, 64), (32000, 2048), (201088, 128)])
@pytest.mark.parametrize("microbatches", [1, 4])
def test_embedding_main_grad(vocab: int, hidden: int, microbatches: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    emb = Embedding(vocab, hidden, device=device, dtype=torch.bfloat16)
    ref = torch.nn.Embedding(vocab, hidden, device=device, dtype=torch.bfloat16)
    ref.weight.data.copy_(emb.weight.data)
    emb.main_grad = torch.zeros(vocab, hidden, device=device, dtype=torch.float32)
    expected = torch.zeros(vocab, hidden, device=device, dtype=torch.float32)
    for _ in range(microbatches):
        # Few distinct ids, so rows repeat within a micro-batch.
        ids = torch.randint(0, min(vocab, 512), (2, 300), device=device)
        grad = torch.randn(2, 300, hidden, device=device, dtype=torch.bfloat16)
        out = emb(ids)
        torch.testing.assert_close(out, ref(ids), rtol=0, atol=0)
        out.backward(grad)
        expected.index_add_(0, ids.reshape(-1), grad.reshape(-1, hidden).float())
    assert emb.weight.grad is None
    torch.testing.assert_close(emb.main_grad, expected, rtol=1e-6, atol=1e-5)


def test_embedding_without_main_grad_is_nn_embedding():
    torch.manual_seed(0)
    emb = Embedding(100, 16)
    ids = torch.randint(0, 100, (3, 7))
    emb(ids).sum().backward()
    ref = torch.nn.Embedding(100, 16)
    ref.weight.data.copy_(emb.weight.data)
    ref(ids).sum().backward()
    torch.testing.assert_close(emb.weight.grad, ref.weight.grad, rtol=0, atol=0)
