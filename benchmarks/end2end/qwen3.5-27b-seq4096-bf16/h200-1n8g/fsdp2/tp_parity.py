"""Compare full and sequence-sharded BF16 Qwen3.5 loss and all parameter gradients."""

import argparse
import copy
import json
import logging as std_logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
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


def split_row_forward(projection):
    def forward(x, quantized_input=None):
        x0, x1 = x.chunk(2, dim=-1)
        w0, w1 = projection.weight.chunk(2, dim=-1)
        partial = F.linear(x0.contiguous(), w0.contiguous()) + F.linear(
            x1.contiguous(), w1.contiguous()
        )
        return partial if projection.bias is None else partial + projection.bias

    return forward


def unsharded_bf16_split_k(reference, tokens, labels):
    model = copy.deepcopy(reference)
    model.zero_grad(set_to_none=True)
    for layer in model.layers:
        out = layer.linear_attn.out_proj if layer.is_linear else layer.self_attn.o_proj
        for projection in (out, layer.mlp.down_proj):
            projection.forward = split_row_forward(projection)
    logits = model(tokens).flatten(0, 1)
    F.cross_entropy(logits.float(), labels.flatten(), reduction="sum").backward()
    return model


def fp32_tp(reference, tp, tokens, labels, shard_dense_model, fsdp_cfg):
    from pithtrain.contexts import training
    from pithtrain.models import qwen35_moe
    from pithtrain.modules import dense_tp

    model = copy.deepcopy(reference).float()
    model.zero_grad(set_to_none=True)
    layout = dense_tp.DenseTP(tp.mesh, async_tp=False)
    layout.shard_model(model)
    with patch.object(training, "PARAM_DTYPE", torch.float32):
        shard_dense_model(model, tp.mesh["dp"], fsdp_cfg)
    with (
        patch.multiple(
            dense_tp,
            gated_delta_rule=recurrent_delta,
            flash_attn_func=torch_attention,
            silu_mul=lambda gate, up: F.silu(gate) * up,
        ),
        patch.object(qwen35_moe, "causal_conv1d", torch_conv),
    ):
        logits = model(tokens)
        layout.loss(logits, labels).backward()
    layout.sync_replicated_grads(model)
    return model


def check_bf16_local_heads(reference, tp, tokens):
    from pithtrain.operators.causal_conv1d import causal_conv1d
    from pithtrain.operators.gated_delta_rule import gated_delta_rule

    layer = reference.layers[0]
    delta = layer.linear_attn
    conv_shard = tp.shards["layers.0.linear_attn.conv1d.weight"]
    with torch.no_grad():
        x = layer.input_layernorm(reference.embed_tokens(tokens))
        mixed = delta.in_proj_qkv(x)
        convolved = causal_conv1d(mixed, delta.conv1d.weight)
        local_mixed = torch.cat(
            [mixed[..., span.start : span.stop] for span in conv_shard.spans], -1
        )
        local_convolved = causal_conv1d(local_mixed, conv_shard.select(delta.conv1d.weight))
        expected_conv = torch.cat(
            [convolved[..., span.start : span.stop] for span in conv_shard.spans], -1
        )
        q, k, v = delta.split_qkv(F.silu(convolved))
        beta = delta.in_proj_b(x).sigmoid()
        advance = F.softplus(delta.in_proj_a(x).float() + delta.dt_bias.float())
        generator = torch.Generator(device=x.device).manual_seed(71)
        dout = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)

    alpha = delta.A_log.detach().requires_grad_(True)
    full = gated_delta_rule(q, k, v, -alpha.float().exp() * advance, beta)
    (full_grad,) = torch.autograd.grad(full, alpha, dout)
    local_alpha = alpha.detach().chunk(tp.size)[tp.rank].contiguous().requires_grad_(True)
    q_local, k_local, v_local, adv_local, beta_local, do_local = (
        value.chunk(tp.size, dim=2)[tp.rank].contiguous()
        for value in (q, k, v, advance, beta, dout)
    )
    local = gated_delta_rule(
        q_local, k_local, v_local, -local_alpha.float().exp() * adv_local, beta_local
    )
    (local_grad,) = torch.autograd.grad(local, local_alpha, do_local)
    values = (
        rel_l2(local_convolved, expected_conv),
        rel_l2(local, full.chunk(tp.size, dim=2)[tp.rank]),
        rel_l2(local_grad, full_grad.chunk(tp.size)[tp.rank]),
    )
    print(
        f"rank={tp.rank} same_input_bf16 conv_rel_l2={values[0]:.8f} "
        f"gdn_rel_l2={values[1]:.8f} A_log_rel_l2={values[2]:.8f}",
        flush=True,
    )


def check_bf16_fp32_partials(reference, tp, tokens, labels, shard_dense_model, fsdp_cfg, oracle):
    from pithtrain.modules import dense_tp

    def precise_row_forward(ctx, x, weight, size, name, overlap):
        ctx.save_for_backward(x, weight)
        ctx.size, ctx.name, ctx.overlap = size, name, False
        partial = torch.mm(x, weight.t(), out_dtype=torch.float32)
        return dense_tp._scatter_sum(partial, size, name).to(x.dtype)

    def precise_column_backward(ctx, *grads):
        x, *weights = ctx.saved_tensors
        grads = tuple(grad.contiguous() for grad in grads)
        partials = [
            torch.mm(grad, weight, out_dtype=torch.float32)
            for grad, weight in zip(grads, weights, strict=True)
        ]
        total = sum(partials[1:], partials[0])
        dx = dense_tp._scatter_sum(total, ctx.size, ctx.name).to(x.dtype)
        dw = tuple(grad.t() @ x for grad in grads)
        return dx, None, None, None, *dw

    for patch_backward in (False, True):
        model = copy.deepcopy(reference)
        model.zero_grad(set_to_none=True)
        layout = dense_tp.DenseTP(tp.mesh, async_tp=False)
        layout.shard_model(model)
        shard_dense_model(model, tp.mesh["dp"], fsdp_cfg)
        backward = precise_column_backward if patch_backward else dense_tp._ColumnParallel.backward
        with (
            patch.object(dense_tp._RowParallel, "forward", staticmethod(precise_row_forward)),
            patch.object(dense_tp._ColumnParallel, "backward", staticmethod(backward)),
        ):
            layout.loss(model(tokens), labels).backward()
        layout.sync_replicated_grads(model)
        for attr in ("A_log", "dt_bias"):
            name = "layers.0.linear_attn." + attr
            expected = layout.shards[name].select(oracle.get_parameter(name).grad)
            actual = model.get_parameter(name).grad.to_local()
            print(
                f"rank={tp.rank} fp32_partials {name} column_backward={patch_backward} "
                f"tp_to_fp32={rel_l2(actual, expected):.8f}",
                flush=True,
            )


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


def precision_check(
    reference, parallel, tp, tokens, labels, shard_dense_model, fsdp_cfg, check_partials=False
):
    # The two BF16 controls use the same batch, either one local B=2 backward or
    # two local B=1 backwards followed by FSDP2's cross-rank reduction.
    dp = copy.deepcopy(reference)
    dp.zero_grad(set_to_none=True)
    shard_dense_model(dp, tp.mesh["tp"], fsdp_cfg)
    dp_logits = dp(tokens[tp.rank : tp.rank + 1]).flatten(0, 1)
    dp_labels = labels[tp.rank].flatten()
    F.cross_entropy(dp_logits.float(), dp_labels, reduction="sum").backward()

    oracle = copy.deepcopy(reference).float()
    oracle.zero_grad(set_to_none=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    oracle_logits = fp32_reference(oracle, tokens)
    oracle_loss = F.cross_entropy(oracle_logits, labels.flatten(), reduction="sum")
    oracle_loss.backward()

    split_k = unsharded_bf16_split_k(reference, tokens, labels)
    precise_sharded = fp32_tp(reference, tp, tokens, labels, shard_dense_model, fsdp_cfg)
    original = dict(reference.named_parameters())
    reordered = dict(dp.named_parameters())
    sharded = dict(parallel.named_parameters())
    fp32 = dict(oracle.named_parameters())
    split_k_params = dict(split_k.named_parameters())
    precise_tp_params = dict(precise_sharded.named_parameters())
    for name in (
        "layers.0.linear_attn.A_log",
        "layers.0.linear_attn.dt_bias",
        "layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attn.in_proj_b.weight",
        "layers.0.linear_attn.norm.weight",
        "layers.1.self_attn.q_norm.weight",
    ):
        baseline = original[name].grad
        alternate = reordered[name].grad.full_tensor()
        precise = fp32[name].grad
        split = split_k_params[name].grad
        layout = tp.shards.get(name)
        if layout is not None:
            baseline, alternate, precise, split = map(
                layout.select, (baseline, alternate, precise, split)
            )
        actual = sharded[name].grad.to_local()
        precise_tp = precise_tp_params[name].grad.to_local()
        print(
            f"rank={tp.rank} precision {name} "
            f"bf16_to_fp32={rel_l2(baseline, precise):.6f} "
            f"reordered_bf16_to_fp32={rel_l2(alternate, precise):.6f} "
            f"tp_to_fp32={rel_l2(actual, precise):.6f} "
            f"bf16_reorder={rel_l2(alternate, baseline):.6f} "
            f"tp_to_bf16={rel_l2(actual, baseline):.6f} "
            f"split_k_to_bf16={rel_l2(split, baseline):.6f} "
            f"split_k_to_fp32={rel_l2(split, precise):.6f} "
            f"tp_to_split_k={rel_l2(actual, split):.6f} "
            f"fp32_tp_to_fp32={rel_l2(precise_tp, precise):.8f} "
            f"fp32_norm={precise.norm().item():.9f}",
            flush=True,
        )

    if check_partials:
        check_bf16_local_heads(reference, tp, tokens)
        check_bf16_fp32_partials(reference, tp, tokens, labels, shard_dense_model, fsdp_cfg, oracle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--async-tp", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--hf-roundtrip", action="store_true")
    parser.add_argument("--precision-reference", action="store_true")
    parser.add_argument("--check-tp-partials", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    if (args.precision_reference or args.check_tp_partials) and dist.get_world_size() != 2:
        raise ValueError("The two-sample precision comparison requires two GPUs")

    from pithtrain.contexts import distributed, logging, training
    from pithtrain.models.qwen35 import Qwen35TextModel
    from pithtrain.modules.dense_fsdp import DenseFSDPCfg, shard_dense_model
    from pithtrain.modules.dense_tp import DenseTP
    from pithtrain.modules.hf_loader import load_hf_into_model
    from pithtrain.modules.training import init_weights
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
    reference = Qwen35TextModel(config).to(device=device, dtype=torch.bfloat16)
    init_weights(reference, config.num_hidden_layers, seed=17, pipeline_chunks=False)
    parallel = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (1, dist.get_world_size()), mesh_dim_names=("dp", "tp"))
    tp = DenseTP(mesh, async_tp=args.async_tp)
    tp.shard_model(parallel)
    fsdp = DenseFSDPCfg()
    fsdp.compile_blocks = args.compile_blocks
    shard_dense_model(parallel, mesh["dp"], fsdp)
    if args.hf_roundtrip:
        fixture = [None]
        if dist.get_rank() == 0:
            scratch = TemporaryDirectory(dir=os.environ.get("TMPDIR", "."), prefix="tp-parity-")
            fixture[0] = scratch.name
            folder = Path(scratch.name)
            filename = "model.safetensors"
            weights = {
                "model.language_model." + name
                if name != "lm_head.weight"
                else name: param.detach().cpu()
                for name, param in reference.named_parameters()
            }
            save_file(weights, folder / filename)
            (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(weights, filename)})
            )
        dist.broadcast_object_list(fixture)
        with torch.no_grad():
            for param in parallel.parameters():
                param.to_local().fill_(float("nan"))
            load_hf_into_model(Path(fixture[0]), parallel)
        if dist.get_rank() == 0:
            scratch.cleanup()

    tokens = torch.randint(config.vocab_size, (2, 512), device=device)
    labels = torch.randint(config.vocab_size, tokens.shape, device=device)
    labels[:, 3::11] = -100

    reference_logits = reference(tokens).flatten(0, 1)
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.float(), labels.flatten(), reduction="sum"
    )
    reference_loss.backward()
    parallel_logits = parallel(tokens)
    parallel_loss = tp.loss(parallel_logits, labels)
    parallel_loss.backward()
    tp.sync_replicated_grads(parallel)

    expected_logits = reference_logits.chunk(tp.size, dim=-1)[tp.rank]
    logit_error = rel_l2(parallel_logits, expected_logits)
    loss_error = ((parallel_loss - reference_loss).abs() / reference_loss).item()
    worst = 0.0
    for (name, actual), (expected_name, expected) in zip(
        parallel.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == expected_name
        spec = tp.shards.get(name)
        expected_grad = spec.select(expected.grad) if spec else expected.grad
        error = rel_l2(actual.grad.to_local(), expected_grad)
        worst = max(worst, error)
        print(f"rank={tp.rank} grad {name} relative-L2={error:.6f}", flush=True)

    print(
        f"rank={tp.rank} TP={tp.size} async={args.async_tp} compiled={args.compile_blocks} "
        f"hf_roundtrip={args.hf_roundtrip} "
        f"reference_loss={reference_loss.item():.6f} tp_loss={parallel_loss.item():.6f} "
        f"loss_rel={loss_error:.6f} logits_rel_l2={logit_error:.6f} max_grad_rel_l2={worst:.6f}",
        flush=True,
    )
    if args.precision_reference or args.check_tp_partials:
        precision_check(
            reference,
            parallel,
            tp,
            tokens,
            labels,
            shard_dense_model,
            DenseFSDPCfg(),
            check_partials=args.check_tp_partials,
        )
    assert loss_error < 0.001
    assert logit_error < 0.025
    assert worst < 0.06
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
