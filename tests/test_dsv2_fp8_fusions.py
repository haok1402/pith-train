"""
DeepSeek-V2 FP8 fusions against their unfused paths, bit for bit (forward output, input gradient and
weight gradients): the shared activation quantization of the MLA q/kv_a projections and of the
shared-expert gate/up projections, and the fused routed-expert stage 3 (fp8_scatter_quant +
FP8SwiGLUExpertsFunc) against scatter_for_grouped_gemm -> FP8GroupedLinear -> silu_mul. Hopper only.
"""

import json

import pytest
import torch
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config

from pithtrain.contexts import distributed, training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.kernels.arch import ARCH_MAJOR
from pithtrain.operators.grouped_linear import FP8GroupedLinear
from pithtrain.operators.linear import FP8Linear
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

pytestmark = pytest.mark.skipif(ARCH_MAJOR != 9, reason="Hopper-only fused path")


@pytest.fixture
def fp8_ctx(monkeypatch: pytest.MonkeyPatch) -> DeepseekV2Config:
    monkeypatch.setattr(training, "fp8", True, raising=False)
    monkeypatch.setattr(training, "Linear", FP8Linear, raising=False)
    monkeypatch.setattr(training, "GroupedLinear", FP8GroupedLinear, raising=False)
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    monkeypatch.setattr(distributed, "ep_size", 1, raising=False)
    with open("examples/pretrain_lm/deepseek-v2-lite/config.json") as f:
        return DeepseekV2Config(**json.load(f), max_position_embeddings=8192)


def init(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else torch.nn.init.ones_(p)
    return module.cuda().to(torch.bfloat16)


def grads(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.grad for p in module.parameters() if p.grad is not None]


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()


def assert_same(a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
    assert len(a) == len(b) and len(a) > 0
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), (i, (x - y).abs().max())


def run_dense(module, args, dy, shared: bool, monkeypatch) -> list[torch.Tensor]:
    if not shared:
        monkeypatch.setattr(FP8Linear, "quantize_input", staticmethod(lambda x: None))
    FP8WeightCacheControl.step()
    x = args[0].detach().requires_grad_()
    y = module(x, *args[1:])
    y.backward(dy)
    return [y.detach(), x.grad, *grads(module)]


@pytest.mark.parametrize("S,q_lora", [(4096, None), (2048, None), (2048, 1536)])
def test_mla_shared_quant(fp8_ctx, S: int, q_lora, monkeypatch: pytest.MonkeyPatch) -> None:
    from pithtrain.models.deepseek_v2 import DeepSeekV2Attention, DeepSeekV2RotaryEmbedding

    config = fp8_ctx
    config.q_lora_rank = q_lora  # 1536: DeepSeek-V2 (full), q_a_proj shares kv_a's quantization
    rope = DeepSeekV2RotaryEmbedding(config).cuda()
    cos, sin = rope(S)
    posemb = (cos.unsqueeze(0), sin.unsqueeze(0))
    torch.manual_seed(1)
    x = torch.randn(1, S, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(1, S, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 1e-2
    out = []
    for shared in (False, False, True):
        attn = init(DeepSeekV2Attention(config), seed=0)
        with monkeypatch.context() as m:
            out.append(run_dense(attn, (x, posemb), dy, shared, m))
    if S > 2048:
        # FA4's backward is not run-to-run deterministic at this length (out[0] and out[1], both
        # unshared, already differ in the input gradient), so only the forward is compared.
        out = [o[:1] for o in out]
    assert_same(out[0], out[1])
    assert_same(out[0], out[2])


@pytest.mark.parametrize("inter", [None, 2816])
def test_mlp_shared_quant(fp8_ctx, inter: int | None, monkeypatch: pytest.MonkeyPatch) -> None:
    from pithtrain.models.deepseek_v2 import DeepSeekV2MLP

    config = fp8_ctx
    torch.manual_seed(1)
    x = torch.randn(1, 4096, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(1, 4096, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 1e-2
    out = []
    for shared in (False, True):
        mlp = init(DeepSeekV2MLP(config, intermediate_size=inter), seed=0)
        with monkeypatch.context() as m:
            m.setattr("pithtrain.models.deepseek_v2.fp8_mlp_fusable", lambda x: False)
            out.append(run_dense(mlp, (x,), dy, shared, m))
    assert_same(*out)


@pytest.mark.parametrize("dedup", [False, True])
def test_routed_experts_fused_stage3(fp8_ctx, dedup: bool) -> None:
    from pithtrain.models.deepseek_v2 import DeepSeekV2Experts

    config = fp8_ctx
    G, top_k, T, H = 8, config.num_experts_per_tok, 2048, config.hidden_size
    torch.manual_seed(2)
    M = T * top_k
    expert_idxs = torch.randint(0, G, (M,), device="cuda")
    src_rows = M // 2 if dedup else M
    gathered = torch.randn(src_rows, H, device="cuda", dtype=torch.bfloat16)
    expand_idx = torch.randint(0, src_rows, (M,), device="cuda") if dedup else None
    dy = torch.randn(M, H, device="cuda", dtype=torch.bfloat16) * 1e-2
    out = []
    for fused in (False, True):
        experts = init(DeepSeekV2Experts(config, G), seed=0)
        FP8WeightCacheControl.step()
        src = gathered.detach().requires_grad_()
        if fused:
            outs, rsi = experts.forward_from_tokens(src, expert_idxs, expand_idx, None)
        else:
            x = padded_index_gather(src, expand_idx) if dedup else src
            xs, rsi, offs, ks, ks_t = scatter_for_grouped_gemm(x, expert_idxs, G)
            gi = precompute_group_indices(offs, xs.shape[0])
            kw = dict(grouped_mm_offs=offs, ks=ks, ks_tensor=ks_t, group_indices=gi)
            h = silu_mul(experts.gate_proj(xs, **kw), experts.up_proj(xs, **kw))
            outs = experts.down_proj(h, **kw)
        y = padded_index_gather(outs, rsi)
        y.backward(dy)
        out.append([y.detach(), src.grad, *grads(experts)])
    (y0, *g0), (y1, *g1) = out
    # The forward is bit-identical. The gradients are sums whose order follows the row placement
    # within each expert (fp8_scatter_quant keeps token order, the sort need not), so they agree to
    # FP8 accuracy, as in test_fp8_scatter_quant_experts.
    assert torch.equal(y0, y1)
    for a, b in zip(g0, g1):
        assert calc_diff(a, b) < 1e-3


@pytest.mark.parametrize("inter", [None, 2816])
def test_mlp_fused_swiglu(fp8_ctx, inter: int | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused dense SwiGLU op (fp8_swiglu_mlp) against FP8Linear -> silu_mul -> FP8Linear: forward
    and weight gradients bit-identical, input gradient the same dx_gate + dx_up sum."""
    from pithtrain.models.deepseek_v2 import DeepSeekV2MLP

    config = fp8_ctx
    torch.manual_seed(1)
    x = torch.randn(1, 4096, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(1, 4096, config.hidden_size, device="cuda", dtype=torch.bfloat16) * 1e-2
    out = []
    for fused in (False, True):
        mlp = init(DeepSeekV2MLP(config, intermediate_size=inter), seed=0)
        with monkeypatch.context() as m:
            if not fused:
                m.setattr("pithtrain.models.deepseek_v2.fp8_mlp_fusable", lambda x: False)
            out.append(run_dense(mlp, (x,), dy, True, m))
    assert_same(*out)
