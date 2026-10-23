"""Gradient of gpt-oss learnable attention sinks through the FA4 wrapper.

A sink logit s_h joins each query row's softmax denominator without contributing a value, so the
FA4 output must match softmax-with-sink attention, and dq/dk/dv/d(sink) must match fp32 autograd
of that reference, for the dense and the varlen (packed) entry points, causal with and without a
sliding window, as gpt-oss uses them.
"""

from typing import Optional

import pytest
import torch

from pithtrain.operators.flash_attn_v4 import flash_attn_func, flash_attn_varlen_func

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    window: Optional[int],
) -> torch.Tensor:
    """fp32 causal softmax-with-sink attention for one sequence, (S, H, D) layout, GQA."""
    S, H, _ = q.shape
    rep = H // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q, k) * scale
    i = torch.arange(S, device=q.device)
    mask = i[None, :] <= i[:, None]
    if window is not None:
        mask &= i[:, None] - i[None, :] <= window
    logits = logits.masked_fill(~mask, float("-inf"))
    sink_col = sink.view(H, 1, 1).expand(H, S, 1)
    probs = torch.softmax(torch.cat([logits, sink_col], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("hqk,khd->qhd", probs, v)


def _make(B: int, S: int, H: int, KV: int, D: int, sink_dtype: torch.dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", generator=g).bfloat16()
    k = torch.randn(B, S, KV, D, device="cuda", generator=g).bfloat16()
    v = torch.randn(B, S, KV, D, device="cuda", generator=g).bfloat16()
    # Sinks spread around the typical row LSE so that P_sink is neither ~0 nor ~1.
    sink = (torch.randn(H, device="cuda", generator=g) * 2 + 2).to(sink_dtype)
    do = torch.randn(B, S, H, D, device="cuda", generator=g).bfloat16()
    return q, k, v, sink, do


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _check(got: dict, ref: dict) -> None:
    for name in ("o", "dq", "dk", "dv"):
        assert _rel(got[name], ref[name]) < 2e-2, (name, _rel(got[name], ref[name]))
    # d(sink) sums over every query row; compare per head as well as in norm.
    assert _rel(got["dsink"], ref["dsink"]) < 1e-2, (got["dsink"], ref["dsink"])
    torch.testing.assert_close(got["dsink"].float(), ref["dsink"], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("window", [None, 128])
@pytest.mark.parametrize("sink_dtype", [torch.bfloat16, torch.float32])
def test_dense_sink_grad(window: Optional[int], sink_dtype: torch.dtype) -> None:
    B, S, H, KV, D = 2, 512, 8, 2, 64
    scale = D**-0.5
    q, k, v, sink, do = _make(B, S, H, KV, D, sink_dtype)
    ws = (window, 0) if window is not None else (None, None)

    t = [x.clone().requires_grad_() for x in (q, k, v, sink)]
    o = flash_attn_func(
        *t[:3], softmax_scale=scale, causal=True, window_size=ws, learnable_sink=t[3]
    )
    o.backward(do)
    assert t[3].grad is not None and t[3].grad.dtype == sink_dtype
    got = dict(o=o, dq=t[0].grad, dk=t[1].grad, dv=t[2].grad, dsink=t[3].grad)

    r = [x.float().clone().requires_grad_() for x in (q, k, v, sink)]
    ro = torch.stack([reference_attention(r[0][b], r[1][b], r[2][b], r[3], scale, window) for b in range(B)])  # fmt: skip
    ro.backward(do.float())
    ref = dict(o=ro, dq=r[0].grad, dk=r[1].grad, dv=r[2].grad, dsink=r[3].grad)
    _check(got, ref)


@pytest.mark.parametrize("window", [None, 128])
def test_varlen_sink_grad(window: Optional[int]) -> None:
    H, KV, D = 8, 2, 64
    lens = [300, 17, 512, 195]
    S = sum(lens)
    scale = D**-0.5
    q, k, v, sink, do = _make(1, S, H, KV, D, torch.bfloat16)
    q, k, v, do = (x.squeeze(0) for x in (q, k, v, do))
    cu = torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)
    ws = (window, 0) if window is not None else (None, None)

    t = [x.clone().requires_grad_() for x in (q, k, v, sink)]
    o = flash_attn_varlen_func(*t[:3], cu, S, softmax_scale=scale, causal=True, window_size=ws, learnable_sink=t[3])  # fmt: skip
    o.backward(do)
    got = dict(o=o, dq=t[0].grad, dk=t[1].grad, dv=t[2].grad, dsink=t[3].grad)

    r = [x.float().clone().requires_grad_() for x in (q, k, v, sink)]
    segs = zip(cu[:-1].tolist(), cu[1:].tolist())
    ro = torch.cat([reference_attention(r[0][a:b], r[1][a:b], r[2][a:b], r[3], scale, window) for a, b in segs])  # fmt: skip
    ro.backward(do.float())
    ref = dict(o=ro, dq=r[0].grad, dk=r[1].grad, dv=r[2].grad, dsink=r[3].grad)
    _check(got, ref)


def test_no_sink_unchanged() -> None:
    """Without a sink the wrapper still returns dq/dk/dv and no sink gradient."""
    B, S, H, KV, D = 1, 256, 4, 4, 64
    q, k, v, _, do = _make(B, S, H, KV, D, torch.bfloat16)
    t = [x.clone().requires_grad_() for x in (q, k, v)]
    o = flash_attn_func(*t, softmax_scale=D**-0.5, causal=True)
    o.backward(do)
    assert all(x.grad is not None for x in t)
