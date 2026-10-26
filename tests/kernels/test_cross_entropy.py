"""
Verifier of ``cross_entropy_fwd_inplace`` against plain PyTorch.

Contract: the mean cross-entropy of the (N, V) logits against ``target`` over the rows whose
target is not ``ignore_index`` (at least one counted, so all-ignored gives 0), computed in fp32;
the logits are overwritten with the gradient of that mean, ``(softmax(x) - onehot(y)) / n`` per
counted row and zero on ignored rows, rounded once to the logits' dtype. ``grad`` is ``inp``
itself when ``inp`` is row-contiguous, otherwise a contiguous copy.

    pytest tests/kernels/test_cross_entropy.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.kernels import cross_entropy_fwd_inplace

IGNORE = -100


def reference(logits, target, ignore_index):
    """(loss, grad) in fp32: F.cross_entropy and the closed-form gradient of the mean."""
    x = logits.float()
    keep = target != ignore_index
    n = max(int(keep.sum()), 1)
    loss = F.cross_entropy(x, target, ignore_index=ignore_index, reduction="sum") / n
    grad = torch.softmax(x, dim=-1) / n
    rows = torch.nonzero(keep).squeeze(1)
    grad[rows, target[rows]] -= 1.0 / n
    grad[~keep] = 0
    return loss, grad, n


def check(logits, target, ignore_index=IGNORE):
    want_loss, want_grad, n = reference(logits, target, ignore_index)
    inp = logits.clone()
    loss, grad = cross_entropy_fwd_inplace(inp, target, ignore_index)
    assert loss.dtype == torch.float32 and loss.dim() == 0
    assert grad.shape == logits.shape and grad.dtype == logits.dtype
    assert grad.data_ptr() == inp.data_ptr()  # in place on row-contiguous logits
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    rtol = 1e-5 if logits.dtype == torch.float32 else torch.finfo(logits.dtype).eps
    # (p_y - 1) / n cancels when p_y ~ 1: fp32 noise is absolute there, of order eps32 / n.
    # fp16 gradients of large vocabularies are subnormal: their spacing is absolute.
    atol = 1e-6 / n + torch.finfo(logits.dtype).smallest_normal * torch.finfo(logits.dtype).eps
    torch.testing.assert_close(grad.float(), want_grad, rtol=rtol, atol=atol)
    # The target entry must hold (p_y - 1) / n, never a stale logit, and ignored rows are zero.
    keep = target != ignore_index
    rows = torch.nonzero(keep).squeeze(1)
    tgt = grad.float()[rows, target[rows]]
    torch.testing.assert_close(tgt, want_grad[rows, target[rows]], rtol=rtol, atol=atol)
    assert torch.all(tgt <= 0)
    assert torch.all(grad[~keep] == 0)
    return loss, grad


# vocab sizes: small, not a power of two, and the four models' vocabularies
# (deepseek-v2-lite 102400, qwen3-30b-a3b 151936, gpt-oss-20b 201088, qwen3.5-35b-a3b 248320).
@pytest.mark.parametrize("vocab", [1000, 32000, 102400, 151936, 201088, 248320])
def test_loss_and_grad(vocab):
    gen = torch.Generator(device="cuda").manual_seed(0)
    tokens = 4096 if vocab <= 32000 else 1024
    logits = (torch.randn(tokens, vocab, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    target = torch.randint(0, vocab, (tokens,), generator=gen, device="cuda")
    target[::7] = IGNORE
    check(logits, target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtypes(dtype):
    gen = torch.Generator(device="cuda").manual_seed(1)
    logits = (torch.randn(777, 5003, generator=gen, device="cuda") * 3).to(dtype)
    target = torch.randint(0, 5003, (777,), generator=gen, device="cuda")
    target[5::11] = IGNORE
    check(logits, target)


def test_large_logits_stable():
    """Online softmax stays finite for large and very negative logits."""
    gen = torch.Generator(device="cuda").manual_seed(2)
    logits = (torch.randn(512, 151936, generator=gen, device="cuda") * 50).to(torch.bfloat16)
    logits[:8, :1000] = -1e4
    logits[8:16, 7] = 3e4  # one dominant class: p ~ 1, gradient cancels to ~0
    target = torch.randint(0, 151936, (512,), generator=gen, device="cuda")
    target[8:16] = 7
    loss, grad = check(logits, target)
    assert torch.isfinite(loss) and torch.isfinite(grad).all()


def test_target_at_edges_and_one_column():
    gen = torch.Generator(device="cuda").manual_seed(3)
    V = 70000  # more than one 32768-column block, with a ragged tail
    logits = torch.randn(64, V, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.tensor([0, V - 1, 32767, 32768, 65535, 65536] * 10 + [1, 2, 3, 4], device="cuda")
    check(logits, target)
    one = torch.randn(16, 1, device="cuda")
    loss, grad = check(one, torch.zeros(16, dtype=torch.long, device="cuda"))
    assert loss.item() == 0 and torch.all(grad == 0)


def test_all_ignored_gives_zero():
    logits = torch.randn(32, 1000, device="cuda").to(torch.bfloat16)
    target = torch.full((32,), IGNORE, device="cuda")
    loss, grad = cross_entropy_fwd_inplace(logits.clone(), target, IGNORE)
    assert loss.item() == 0.0
    assert torch.all(grad == 0)


def test_other_ignore_index():
    gen = torch.Generator(device="cuda").manual_seed(4)
    logits = torch.randn(1000, 4096, generator=gen, device="cuda").to(torch.bfloat16)
    target = torch.randint(0, 4096, (1000,), generator=gen, device="cuda")
    target[::3] = 5  # 5 is a real class elsewhere, here the ignore index
    check(logits, target, ignore_index=5)


def test_single_row_and_mean_over_counted_rows():
    logits = torch.randn(1, 151936, device="cuda").to(torch.bfloat16)
    check(logits, torch.tensor([123], device="cuda"))
    # The mean divides by the counted rows only: one counted row out of 100.
    logits = torch.randn(100, 1000, device="cuda").to(torch.bfloat16)
    target = torch.full((100,), IGNORE, device="cuda")
    target[42] = 9
    loss, _ = check(logits, target)
    torch.testing.assert_close(loss, F.cross_entropy(logits[42:43].float(), target[42:43]))


def test_non_contiguous_logits_return_a_copy():
    """A column slice of a wider buffer is not row-contiguous: the gradient comes back in a
    contiguous copy, still correct."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    wide = torch.randn(300, 5000, generator=gen, device="cuda").to(torch.bfloat16)
    logits = wide[:, :4000]
    target = torch.randint(0, 4000, (300,), generator=gen, device="cuda")
    target[::4] = IGNORE
    want_loss, want_grad, n = reference(logits, target, IGNORE)
    loss, grad = cross_entropy_fwd_inplace(logits, target, IGNORE)
    assert grad.is_contiguous() and grad.data_ptr() != wide.data_ptr()
    torch.testing.assert_close(loss, want_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad.float(), want_grad, rtol=2**-8, atol=1e-6 / n)
    # Strided target (every other element of a longer tensor).
    t2 = torch.stack([target, torch.zeros_like(target)], 1).reshape(-1)[::2]
    assert not t2.is_contiguous() or t2.stride(-1) != 1
    loss2, _ = cross_entropy_fwd_inplace(logits.contiguous(), t2, IGNORE)
    torch.testing.assert_close(loss2, want_loss, rtol=1e-5, atol=1e-5)
