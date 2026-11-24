"""An outer checkpoint must keep the compiled Gated DeltaNet result until replay."""

from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pithtrain.contexts import distributed, training
from pithtrain.models.qwen35 import Qwen35TextModel
from pithtrain.operators.linear import BF16Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_gdn_selective_checkpoint_gradients(monkeypatch):
    monkeypatch.setattr(distributed, "cp_size", 1, raising=False)
    monkeypatch.setattr(training, "Linear", BF16Linear, raising=False)
    monkeypatch.setattr(training, "fp8", False, raising=False)
    monkeypatch.setattr(training, "fp8_attention", False, raising=False)
    # Like the 9B model, V heads × V head dimension equals the residual width.
    config = Qwen3_5TextConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        layer_types=["linear_attention", "linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        vocab_size=1024,
        max_position_embeddings=512,
        head_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        tie_word_embeddings=False,
    )
    torch.manual_seed(1234)
    model = Qwen35TextModel(config).to(device="cuda", dtype=torch.bfloat16)
    mixer_forwards = [layer.linear_attn.forward for layer in model.layers]
    tokens = torch.randint(config.vocab_size, (2, 512), device="cuda")
    save = [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.pithtrain.gated_delta_rule_dense_fwd.default,
    ]
    contexts = partial(create_selective_checkpoint_contexts, save)

    def gradients(selective):
        model.zero_grad(set_to_none=True)
        for layer, mixer_forward in zip(model.layers, mixer_forwards, strict=True):
            layer.linear_attn.forward = mixer_forward if selective else layer.linear_attn.block
            layer.forward = layer.forward_stock_stages
            if selective:
                layer.forward = partial(
                    checkpoint,
                    layer.forward,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    context_fn=contexts,
                )
        loss = model(tokens).float().square().mean()
        loss.backward()
        grads = {
            name: parameter.grad.detach().float() for name, parameter in model.named_parameters()
        }
        norm = torch.linalg.vector_norm(torch.stack([grad.norm() for grad in grads.values()]))
        return loss.detach(), norm, grads

    ref_loss, ref_norm, ref_grads = gradients(selective=False)
    loss, norm, grads = gradients(selective=True)
    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    torch.testing.assert_close(norm, ref_norm, rtol=5e-3, atol=1e-5)
    assert grads.keys() == ref_grads.keys()
    for name, grad in grads.items():
        relative_l2 = (grad - ref_grads[name]).norm() / ref_grads[name].norm().clamp_min(1e-8)
        assert relative_l2 <= 1e-2, (name, relative_l2.item())
