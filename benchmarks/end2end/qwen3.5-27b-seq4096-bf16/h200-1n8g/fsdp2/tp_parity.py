"""Audit stock BF16 and sequence-parallel training against unsharded FP32."""

import argparse
import copy
import logging as std_logging
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def recurrent_delta(q, k, v, decay, beta):
    # Same per-head FP32 recurrence as the pinned Gated DeltaNet kernel verifier.
    heads = v.size(2)
    q, k = (x.repeat_interleave(heads // x.size(2), dim=2) for x in (q, k))
    state = q.new_zeros(q.size(0), heads, k.size(-1), v.size(-1))
    output = []
    for t in range(q.size(1)):
        state = state * decay[:, t].exp()[..., None, None]
        key = k[:, t]
        prediction = torch.einsum("bhk,bhkv->bhv", key, state)
        update = beta[:, t, :, None] * (v[:, t] - prediction)
        state = state + key[..., :, None] * update[..., None, :]
        query = q[:, t] * q.size(-1) ** -0.5
        output.append(torch.einsum("bhk,bhkv->bhv", query, state))
    return torch.stack(output, dim=1)


def torch_conv(x, weight, offset=0):
    assert offset == 0
    return F.conv1d(x.transpose(1, 2), weight, padding=weight.size(-1) - 1, groups=x.size(-1))[
        ..., : x.size(1)
    ].transpose(1, 2)


def torch_attention(q, k, v, softmax_scale, causal):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=softmax_scale,
        is_causal=causal,
        enable_gqa=True,
    ).transpose(1, 2)


def fp32_reference(model, tokens):
    batch, seq = tokens.shape
    hidden = model.embed_tokens(tokens)
    rotary = tuple(t.unsqueeze(0) for t in model.rotary_emb(seq))
    for layer in model.layers:
        x = layer.input_layernorm(hidden)
        if layer.is_linear:
            delta = layer.linear_attn
            convolution = torch_conv(delta.in_proj_qkv(x), delta.conv1d.weight)
            q, k, v = delta.split_qkv(F.silu(convolution))
            beta = delta.in_proj_b(x).sigmoid()
            decay = -delta.A_log.exp() * F.softplus(delta.in_proj_a(x) + delta.dt_bias)
            out = recurrent_delta(q, k, v, decay, beta)
            z = delta.in_proj_z(x)
            out = delta.norm(out.reshape(-1, delta.head_v_dim), z.reshape(-1, delta.head_v_dim))
            x = delta.out_proj(out.reshape(batch, seq, -1))
        else:
            attn = layer.self_attn
            q, gate = (
                attn.q_proj(x).view(batch, seq, attn.num_heads, 2 * attn.head_dim).chunk(2, -1)
            )
            kv_shape = (batch, seq, attn.num_kv_heads, attn.head_dim)
            k, v = attn.k_proj(x).view(kv_shape), attn.v_proj(x).view(kv_shape)
            q, k = attn.apply_rotary_posemb(attn.q_norm(q), attn.k_norm(k), rotary)
            out = torch_attention(q, k, v, attn.scaling, True)
            x = attn.o_proj(out.reshape(batch, seq, -1) * gate.reshape(batch, seq, -1).sigmoid())
        hidden = hidden + x
        mlp, x = layer.mlp, layer.post_attention_layernorm(hidden)
        hidden = hidden + mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    return model.lm_head(model.norm(hidden)).flatten(0, 1)


def rel_l2(actual, expected):
    error = (actual.float() - expected.float()).norm()
    return (error / expected.float().norm().clamp_min(1e-6)).item()


def backward(model, tokens, labels, dp, tp=None):
    from pithtrain.tasks.pretrain_lm import objective

    losses = []
    for x, y in zip(tokens[:, dp.rank()], labels[:, dp.rank()], strict=True):
        logits = model(x)
        if tp is None:
            loss, _ = objective((logits,), (y,))
        else:
            loss = tp.loss(logits, y)
        loss.backward()
        losses.append(loss.detach())
    if tp is not None:
        tp.sync_replicated_grads(model)
    loss = torch.stack(losses).sum()
    dist.all_reduce(loss, group=dp)
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--async-tp", action="store_true")
    parser.add_argument("--tp-regather-inputs", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--blocks-per-group", type=int, default=2)
    parser.add_argument("--checkpoint", choices=("none", "selective", "full"), default="none")
    args = parser.parse_args()

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    from pithtrain.contexts import distributed, logging, training
    from pithtrain.models import qwen35_moe
    from pithtrain.models.qwen35 import Qwen35TextModel
    from pithtrain.modules import dense_tp
    from pithtrain.modules.dense_fsdp import DenseFSDPCfg, shard_dense_model
    from pithtrain.modules.training import init_weights
    from pithtrain.modules.wgrad_accumulation import install_wgrad_accumulation
    from pithtrain.operators.linear import BF16Linear

    training.fp8, training.Linear = False, BF16Linear
    distributed.cp_size = 1
    logging.stdout = std_logging.getLogger("tp-parity")
    std_logging.basicConfig(level=std_logging.INFO)
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        hidden_size=768,
        intermediate_size=1024,
        vocab_size=2048,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=512,
        head_dim=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    oracle = Qwen35TextModel(config).to(device)
    init_weights(oracle, config.num_hidden_layers, seed=17, pipeline_chunks=False)
    parallel, stock, fp32_tp = (copy.deepcopy(oracle) for _ in range(3))
    mesh = init_device_mesh(
        "cuda", (dist.get_world_size() // args.tp, args.tp), mesh_dim_names=("dp", "tp")
    )
    dp_mesh = mesh["dp"]
    dp = dp_mesh.get_group()
    layout = dense_tp.DenseTP(mesh, async_tp=args.async_tp, regather_inputs=args.tp_regather_inputs)
    layout.shard_model(parallel)
    config_fsdp = DenseFSDPCfg()
    config_fsdp.tensor_parallel_size = args.tp
    config_fsdp.compile_blocks = args.compile_blocks
    config_fsdp.activation_checkpoint = args.checkpoint
    config_fsdp.blocks_per_group = args.blocks_per_group
    config_fsdp.skip_embedding_backward_gather = True
    shard_dense_model(parallel, dp_mesh, config_fsdp)
    install_wgrad_accumulation(parallel)

    for layer in stock.layers:
        layer.forward = layer.forward_stock_stages
    stock_fsdp = DenseFSDPCfg()
    stock_fsdp.blocks_per_group = args.blocks_per_group
    shard_dense_model(stock, dp_mesh, stock_fsdp)
    install_wgrad_accumulation(stock)

    tokens = torch.randint(config.vocab_size, (2, dp.size(), 1, 512), device=device)
    labels = torch.randint(config.vocab_size, tokens.shape, device=device)
    labels[..., 3::11] = -100
    tp_loss = backward(parallel, tokens, labels, dp, layout)
    stock_loss = backward(stock, tokens, labels, dp)

    dense_tokens = tokens.reshape(-1, tokens.size(-1))
    dense_labels = labels.reshape(-1, labels.size(-1))
    oracle_loss = F.cross_entropy(
        fp32_reference(oracle, dense_tokens), dense_labels.flatten(), reduction="sum"
    )
    oracle_loss.backward()

    fp32_layout = dense_tp.DenseTP(mesh, async_tp=False)
    fp32_layout.shard_model(fp32_tp)
    with patch.object(training, "PARAM_DTYPE", torch.float32):
        shard_dense_model(fp32_tp, dp_mesh, DenseFSDPCfg())
    with (
        patch.multiple(
            dense_tp,
            gated_delta_rule=recurrent_delta,
            flash_attn_func=torch_attention,
            silu_mul=lambda gate, up: F.silu(gate) * up,
        ),
        patch.object(qwen35_moe, "causal_conv1d", torch_conv),
    ):
        fp32_loss = backward(fp32_tp, tokens, labels, dp, fp32_layout)

    rank = dist.get_rank()
    max_bf16_error = max_layout_error = 0.0
    for name, param in parallel.named_parameters():
        assert param.dtype == param.grad.dtype == torch.float32, name
        actual = param.grad.full_tensor()
        baseline = stock.get_parameter(name).grad.full_tensor()
        expected = oracle.get_parameter(name).grad
        fp32_actual = fp32_tp.get_parameter(name).grad.full_tensor()
        if spec := layout.shards.get(name):
            baseline, expected = spec.select(baseline), spec.select(expected)
        assert torch.isfinite(actual).all()
        bf16_error = rel_l2(actual, expected)
        layout_error = rel_l2(fp32_actual, expected)
        max_bf16_error = max(max_bf16_error, bf16_error)
        max_layout_error = max(max_layout_error, layout_error)
        print(
            f"rank={rank} grad={name} tp_to_fp32={bf16_error:.8f} "
            f"stock_to_fp32={rel_l2(baseline, expected):.8f} "
            f"tp_to_stock={rel_l2(actual, baseline):.8f} fp32_layout={layout_error:.8f}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-5, fused=True)
    before = parallel.lm_head.weight.to_local().detach().clone()
    optimizer.step()
    assert not torch.equal(before, parallel.lm_head.weight.to_local())
    for param in parallel.parameters():
        state = optimizer.state[param]
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    loss_rel = ((tp_loss - stock_loss).abs() / stock_loss).item()
    oracle_rel = ((fp32_loss - oracle_loss).abs() / oracle_loss).item()
    print(
        f"rank={rank} TP={layout.size} FSDP={dp.size()} async={args.async_tp} "
        f"compiled={args.compile_blocks} checkpoint={args.checkpoint} "
        f"stock_loss={stock_loss.item():.8f} tp_loss={tp_loss.item():.8f} "
        f"fp32_loss={oracle_loss.item():.8f} max_tp_to_fp32={max_bf16_error:.8f} "
        f"max_fp32_layout={max_layout_error:.8f} fp32_adamw=passed",
        flush=True,
    )
    assert loss_rel < 0.001
    assert oracle_rel < 1e-5
    assert max_layout_error < 2e-4
    assert max_bf16_error < 0.06
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
