"""
Fusing a model's stage 5 (previous layer) and stage 1 (next layer) into one compiled region (as
qwen3_moe.py does) does NOT give the same forward bits as the two separate regions, at any kind of
layer boundary of deepseek-v2, gpt-oss or qwen3.5, so these models keep separate regions. The fused region here is the model's own stage bodies composed under one
torch.compile with the model's compile options, i.e. exactly what a fused method would trace.
Deterministic algorithms (aten and Inductor) take the stage-5 scatter_add's atomics out of the
comparison. Recorded as the evidence for the C8 fusion decision of each model.
"""

import json

import pytest
import torch

from pithtrain.contexts import distributed, training
from pithtrain.kernels import fp8_fused_experts_supported
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.linear import FP8Linear

pytestmark = pytest.mark.skipif(
    not fp8_fused_experts_supported(), reason="needs the fused FP8 expert ops"
)

S = 1024


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(training, "fp8", True, raising=False)
    monkeypatch.setattr(training, "Linear", FP8Linear, raising=False)
    monkeypatch.setattr(training, "GroupedLinear", FP8GroupedLinear, raising=False)
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    monkeypatch.setattr(distributed, "ep_size", 2, raising=False)
    monkeypatch.setattr(torch._inductor.config, "deterministic", True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch._dynamo.reset()
    yield
    torch.use_deterministic_algorithms(False)
    torch._dynamo.reset()


def init(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else torch.nn.init.ones_(p)
    return module.cuda().to(torch.bfloat16)


def stage5_inputs(prev, hidden: int, top_k: int, moe: bool, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(1, S, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
    if not moe:
        moe_outs = torch.randn(1, S, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
        return moe_outs, None, None, residual
    moe_outs = torch.randn(S * top_k, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
    idxs = torch.randperm(S * top_k, generator=g, device="cuda")
    topk_weight = torch.rand(S, top_k, generator=g, device="cuda").to(torch.bfloat16)
    return moe_outs, idxs, topk_weight, residual


def raw(fn):
    """The Python function under a torch.compile-decorated method."""
    return getattr(fn, "_torchdynamo_orig_callable", None) or fn.__wrapped__


def compare(prev, nxt, args, posemb, options, stage1_extra=(None, None)):
    stage5, stage1 = raw(type(prev).forward_stage5), raw(type(nxt).forward_stage1_compute)

    def fused_fn(prev, nxt, args, posemb):
        return stage1(nxt, stage5(prev, *args), posemb, *stage1_extra)

    fused_region = torch.compile(fused_fn, fullgraph=True, options=options)
    with torch.no_grad():
        runs = []
        for _ in range(2):  # control: the separate regions reproduce themselves bit for bit
            hidden = prev.forward_stage5(*args)
            runs.append(nxt.forward_stage1_compute(hidden, posemb, *stage1_extra))
        assert_equal(*runs)
        fused = fused_region(prev, nxt, args, posemb)
    # The fused region does not: Inductor keeps the stage-5 sum in fp32 into the stage-1 norm
    # instead of rounding it to bf16 at the region boundary. C8 fusion is rejected on this.
    with pytest.raises(AssertionError):
        assert_equal(runs[0], fused)


def assert_equal(separate, fused):
    assert len(separate) == len(fused)
    for i, (a, b) in enumerate(zip(separate, fused)):
        if a is None:
            assert b is None
            continue
        assert a.dtype == b.dtype and torch.equal(a, b), (i, (a.float() - b.float()).abs().max())


def auto_config(model):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(f"examples/pretrain_lm/{model}")
    config.max_position_embeddings = 8192
    return config


def dsv2_config():
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

    with open("examples/pretrain_lm/deepseek-v2-lite/config.json") as f:
        return DeepseekV2Config(**json.load(f), max_position_embeddings=8192)


@pytest.mark.parametrize("prev_id,next_id", [(0, 1), (1, 2)])  # dense -> MoE, MoE -> MoE
def test_deepseek_v2(ctx, prev_id, next_id):
    from pithtrain.models.deepseek_v2 import DeepSeekV2DecoderLayer, DeepSeekV2RotaryEmbedding

    config = dsv2_config()
    prev = init(DeepSeekV2DecoderLayer(config, prev_id), seed=prev_id)
    nxt = init(DeepSeekV2DecoderLayer(config, next_id), seed=next_id)
    cos, sin = DeepSeekV2RotaryEmbedding(config).cuda()(S)
    posemb = (cos.unsqueeze(0), sin.unsqueeze(0))
    from pithtrain.models import deepseek_v2

    moe = prev_id >= config.first_k_dense_replace
    args = stage5_inputs(prev, config.hidden_size, config.num_experts_per_tok, moe, seed=7)
    compare(prev, nxt, args, posemb, deepseek_v2.COMPILE_OPTIONS, (None, None))


@pytest.mark.parametrize("prev_id,next_id", [(0, 1), (1, 2)])  # sliding <-> full attention
def test_gpt_oss(ctx, prev_id, next_id):
    from pithtrain.models import gpt_oss

    config = auto_config("gpt-oss-20b")
    prev = init(gpt_oss.GptOssDecoderLayer(config, prev_id), seed=prev_id)
    nxt = init(gpt_oss.GptOssDecoderLayer(config, next_id), seed=next_id)
    assert prev.self_attn.is_sliding != nxt.self_attn.is_sliding
    cos, sin = gpt_oss.GptOssRotaryEmbedding(config).cuda()(S)
    posemb = (cos.unsqueeze(0), sin.unsqueeze(0))
    args = stage5_inputs(prev, config.hidden_size, config.num_experts_per_tok, True, seed=7)
    compare(prev, nxt, args, posemb, gpt_oss.COMPILE_OPTIONS, (None,))


def qwen35_pairs():
    config = auto_config("qwen3.5-35b-a3b")
    kinds = config.layer_types
    want = {}
    for i in range(len(kinds) - 1):
        want.setdefault((kinds[i], kinds[i + 1]), (i, i + 1))
    return sorted(want.values())


@pytest.mark.parametrize("prev_id,next_id", qwen35_pairs())  # linear/full in every order
def test_qwen35(ctx, prev_id, next_id):
    from pithtrain.models import qwen35_moe

    config = auto_config("qwen3.5-35b-a3b")
    prev = init(qwen35_moe.Qwen35MoeDecoderLayer(config, prev_id), seed=prev_id)
    nxt = init(qwen35_moe.Qwen35MoeDecoderLayer(config, next_id), seed=next_id)
    cos, sin = qwen35_moe.Qwen35MoeRotaryEmbedding(config).cuda()(S)
    posemb = (cos.unsqueeze(0), sin.unsqueeze(0))
    args = stage5_inputs(prev, config.hidden_size, config.num_experts_per_tok, True, seed=7)
    compare(prev, nxt, args, posemb, {}, (None, None))
