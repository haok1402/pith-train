"""The memoized FA4 call path launches exactly what FA4's own entry launches."""

import pytest
import torch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

import pithtrain.operators.flash_attn_v4 as fa
from pithtrain.kernels import flash_attn_memoize as memo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not memo.FAST_PATH, reason="needs a GPU and the pinned FA4"
)

# (batch, seq, q heads, kv heads, qk dim, v dim, causal, window, sink): the attention shapes of
# the four models, at a shorter sequence.
CASES = {
    "qwen3-gqa": (1, 1024, 32, 4, 128, 128, True, (None, None), False),
    "dsv2-mla": (1, 1024, 16, 16, 192, 128, True, (None, None), False),
    "gptoss-sliding-sink": (1, 1024, 64, 8, 64, 64, True, (127, 0), True),
    "gptoss-full-sink": (1, 1024, 64, 8, 64, 64, True, (None, None), True),
    "qwen35-full": (1, 1024, 16, 2, 256, 256, True, (None, None), False),
}


def make(case, seed=0):
    b, s, hq, hkv, d, dv, causal, window, sink = CASES[case]
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, s, hq, d, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, s, hkv, d, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, s, hkv, dv, generator=g, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(b, s, hq, dv, generator=g, device="cuda", dtype=torch.bfloat16)
    s_ = torch.randn(hq, generator=g, device="cuda", dtype=torch.float32) if sink else None
    return q, k, v, do, s_, d**-0.5, causal, window


def fwd_args(q, k, v, s_, scale, causal, window):
    return (q, k, v, scale, causal, window[0], window[1], s_)


def plan_for(kind, inputs, scalars):
    return memo.plans.get(memo.signature(kind, inputs, *scalars), memo.MISSING)


@pytest.mark.parametrize("case", list(CASES))
def test_forward_bitwise_and_plan(case):
    q, k, v, do, s_, scale, causal, window = make(case)
    args = fwd_args(q, k, v, s_, scale, causal, window)
    inputs = dict(q=q, k=k, v=v, learnable_sink=s_)
    scalars = (scale, causal, window[0], window[1])
    outs = [fa._mha_fwd(*args) for _ in range(4)]  # plain, recorded, replayed, replayed
    plan = plan_for("fwd", inputs, scalars)
    assert isinstance(plan, memo._Plan), "no plan was recorded for this signature"
    ref = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, window_size_left=window[0],
                          window_size_right=window[1], learnable_sink=s_, return_lse=True)  # fmt: skip
    for o, lse in outs:
        assert torch.equal(o, ref[0]) and torch.equal(lse, ref[1])
    # The plan recorded from a fresh call of FA4's entry is the same: same compiled kernels,
    # argument layout, scalars, allocation shapes / strides / dtypes and zero-filling.
    _, again = memo._record(lambda: tuple(fa._flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, window_size_left=window[0], window_size_right=window[1], learnable_sink=s_, return_lse=True)[:2]), inputs)  # fmt: skip
    assert again == plan


@pytest.mark.parametrize("case", list(CASES))
def test_backward_same_launches(case):
    q, k, v, do, s_, scale, causal, window = make(case)
    o, lse = fa._mha_fwd(*fwd_args(q, k, v, s_, scale, causal, window))
    bargs = (q, k, v, o, lse, do, scale, causal, window[0], window[1], s_)
    inputs = dict(q=q, k=k, v=v, o=o, lse=lse, do=do, learnable_sink=s_)
    scalars = (scale, causal, window[0], window[1])
    grads = [fa._mha_bwd(*bargs) for _ in range(4)]
    plan = plan_for("bwd", inputs, scalars)
    assert isinstance(plan, memo._Plan)

    def entry():
        return tuple(_flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=scale, causal=causal,
                                     window_size_left=window[0], window_size_right=window[1],
                                     learnable_sink=s_))  # fmt: skip

    _, again = memo._record(entry, inputs)
    assert again == plan
    # FA4 accumulates dQ (and dK/dV under GQA) with atomics, so its backward is not
    # bit-reproducible run to run:
    # beyond the identical launches above, the replayed gradients differ from FA4's entry only by
    # accumulation-order rounding, a few bf16 ulps, as two runs of the entry do.
    ref = entry()
    for got in grads[2:]:
        for x, r in zip(got, ref):
            if x.numel() == 0:
                continue
            ulp = torch.finfo(torch.bfloat16).eps * torch.maximum(x.float().abs(), r.float().abs())
            assert ((x.float() - r.float()).abs() <= 4 * ulp + 1e-6).all()


def test_back_to_back_calls_own_their_buffers():
    q, k, v, do, s_, scale, causal, window = make("qwen3-gqa")
    o, lse = fa._mha_fwd(*fwd_args(q, k, v, s_, scale, causal, window))
    bargs = (q, k, v, o, lse, do, scale, causal, window[0], window[1], s_)
    for _ in range(2):
        fa._mha_bwd(*bargs)
    first, second = fa._mha_bwd(*bargs), fa._mha_bwd(*bargs)
    ptrs = {t.data_ptr() for t in first[:3]}
    assert not ptrs & {t.data_ptr() for t in second[:3]}


def test_unseen_signature_and_varlen_fall_through():
    q, k, v, do, s_, scale, causal, window = make("qwen3-gqa")
    q2 = q[:, :512].contiguous()
    k2, v2 = k[:, :512].contiguous(), v[:, :512].contiguous()
    inputs = dict(q=q2, k=k2, v=v2, learnable_sink=None)
    scalars = (scale, causal, None, None)
    memo.plans.pop(memo.signature("fwd", inputs, *scalars), None)
    fa._mha_fwd(q2, k2, v2, scale, causal, None, None, None)
    assert plan_for("fwd", inputs, scalars) is memo._SEEN  # first call: FA4's entry only
    # Packed sequences take the varlen op, which never goes through the memo.
    cu = torch.tensor([0, 300, 1024], dtype=torch.int32, device="cuda")
    before = len(memo.plans)
    fa.flash_attn_varlen_func(q[0], k[0], v[0], cu, 1024, softmax_scale=scale, causal=True)
    assert len(memo.plans) == before
