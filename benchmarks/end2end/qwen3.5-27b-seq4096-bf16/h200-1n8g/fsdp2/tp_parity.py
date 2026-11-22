"""Audit stock BF16 and sequence-parallel training against unsharded FP32."""

import argparse
import copy
import gc
import logging as std_logging
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from statistics import geometric_mean, mean, stdev
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


def backward(model, tokens, labels, dp, tp=None, begin_microbatch=None):
    from pithtrain.tasks.pretrain_lm import objective

    losses = []
    retained = model.retained_fsdp_units if tokens.size(0) > 1 else ()
    for unit in retained:
        unit.set_reshard_after_backward(False, recurse=False)
    for i, (x, y) in enumerate(zip(tokens[:, dp.rank()], labels[:, dp.rank()], strict=True)):
        if i == tokens.size(0) - 1:
            for unit in retained:
                unit.set_reshard_after_backward(True, recurse=False)
        if begin_microbatch is not None:
            begin_microbatch(i)
        if tp is None:
            loss, _ = objective((model(x),), (y,))
        else:
            loss = model(x, y)
        loss.backward()
        losses.append(loss.detach())
    if tp is not None:
        tp.sync_replicated_grads(model)
    loss = torch.stack(losses).sum()
    dist.all_reduce(loss, group=dp)
    return loss


def capture_parameter_partials(model):
    """Capture gradient producers before FSDP overwrites or releases them."""
    from torch.distributed.fsdp import FSDPModule, fully_shard

    names = {id(module): name for name, module in model.named_modules()}
    partials = {}
    groups = {}
    for module in model.modules():
        if isinstance(module, FSDPModule):
            for group in fully_shard.state(module)._fsdp_param_groups:
                groups[group] = group.post_backward

    def wrap(group, post_backward):
        def capture():
            for fsdp_param in group.fsdp_params:
                info = fsdp_param._module_info
                name = f"{names[id(info.module)]}.{info.param_name}"
                param = fsdp_param.unsharded_param
                main_grad = getattr(info.module, "main_grad", None)
                main_grad = (
                    main_grad
                    if info.param_name == "weight"
                    and main_grad is not None
                    and main_grad.untyped_storage().size()
                    else None
                )
                if param.grad is None and main_grad is None:
                    continue
                for grad in (param.grad, main_grad):
                    if grad is None:
                        continue
                    assert grad.dtype == torch.float32, (name, grad.dtype)
                    if name in partials:
                        partials[name].add_(grad.detach())
                    else:
                        partials[name] = grad.detach().clone()
            return post_backward()

        return capture

    for group, post_backward in groups.items():
        group.post_backward = wrap(group, post_backward)
    return partials, groups


def check_parameter_partials(model, tp, partials, groups, case):
    for group, post_backward in groups.items():
        group.post_backward = post_backward
    for name, param in model.named_parameters():
        partial = partials[name]
        dist.all_reduce(partial, group=tp.dp_group)
        replicated = name not in tp.shards
        if replicated:
            dist.all_reduce(partial, group=tp.group)
        error = rel_l2(param.grad.full_tensor(), partial)
        assert error < 2e-6, (name, error)
        print(
            f"token_sums case={case} rank={dist.get_rank()} grad={name} before_fsdp={partial.dtype} "
            f"tp_reduce={'fp32' if replicated else 'head-or-vocab-sharded'} "
            f"reduction_error={error:.8f}",
            flush=True,
        )


def record_block_adjoints(layer, adjoints):
    def record_output(_layer, _inputs, output):
        def record_grad(grad):
            adjoints.append(grad.detach().clone())

        output.register_hook(record_grad)

    return layer.register_forward_hook(record_output)


def use_stock_block_adjoints(layer, adjoints, layout):
    next_adjoint = iter(adjoints)

    def repair_output(_layer, _inputs, output):
        expected = next(next_adjoint).flatten(0, 1).chunk(layout.size)[layout.rank].contiguous()

        def replace_grad(grad):
            print(f"rank={dist.get_rank()} first_block_adjoint_error={rel_l2(grad, expected):.8f}")
            return expected

        output.register_hook(replace_grad)

    return layer.register_forward_hook(repair_output)


@contextmanager
def cublas_fp32_reduction(case):
    if "-cublas-" not in case:
        yield
        return
    matmul = torch.backends.cuda.matmul
    previous = (
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction_split_k,
    )
    split_k = "nosplit" not in case
    backend = torch.backends.cuda.preferred_blas_library()
    if not split_k:
        torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        matmul.allow_bf16_reduced_precision_reduction = (False, split_k)
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous
        torch.backends.cuda.preferred_blas_library(backend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--bisect", action="store_true")
    parser.add_argument("--cases", nargs="+", help="Subset of cases for the bisection")
    parser.add_argument("--async-tp", action="store_true")
    parser.add_argument("--tp-regather-inputs", action="store_true")
    parser.add_argument("--tp-head-chunk-tokens", type=int, default=0)
    parser.add_argument("--tp-fp32-affine", action="store_true")
    parser.add_argument("--tp-fp32-gates", action="store_true")
    parser.add_argument("--tp-overwrite-wgrad", action="store_true")
    parser.add_argument("--tp-recompute-mlp-columns", action="store_true")
    parser.add_argument("--tp-recompute-mlp-once", action="store_true")
    parser.add_argument("--tp-bulk-row-reduce", action="store_true")
    parser.add_argument("--tp-no-delta-checkpoint", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--blocks-per-group", type=int, default=2)
    parser.add_argument("--reuse-single-weight-gather-buffers", action="store_true")
    parser.add_argument("--keep-head-gathered", action="store_true")
    parser.add_argument("--checkpoint", choices=("none", "selective", "full"), default="none")
    parser.add_argument("--full-checkpoint-blocks", type=int, default=0)
    parser.add_argument("--token-sums-audit", action="store_true")
    parser.add_argument("--attention-bias", action="store_true")
    parser.add_argument("--statistical", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 18, 19, 20, 21])
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()
    if args.tp_overwrite_wgrad and args.bisect:
        parser.error("TP weight-gradient overwrite requires the production TP forwards")
    if args.statistical and (
        args.bisect
        or args.cases
        or len(set(args.seeds)) != len(args.seeds)
        or len(args.seeds) < 5
        or args.batches < 2
    ):
        parser.error(
            "statistical parity needs one configuration, at least five unique seeds, and at least two batches"
        )

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    mesh = init_device_mesh(
        "cuda", (dist.get_world_size() // args.tp, args.tp), mesh_dim_names=("dp", "tp")
    )
    if args.statistical:
        passed = statistical_parity(args, device, mesh)
    else:
        passed, _ = run_trial(args, device, mesh, seed=17, batch_index=0)
    dist.destroy_process_group()
    assert passed, "TP failed the selected FP32-reference parity gate"


def error_ratio(tp_error, stock_error):
    if stock_error:
        return tp_error / stock_error
    return math.inf if tp_error else 1.0


def statistical_parity(args, device, mesh):
    results = defaultdict(lambda: defaultdict(list))
    for seed in args.seeds:
        for batch_index in range(args.batches):
            _, local = run_trial(args, device, mesh, seed, batch_index)
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            seen = set()
            for samples in gathered:
                for name, shard, stock_error, tp_error in samples:
                    assert (name, shard) not in seen, (name, shard)
                    seen.add((name, shard))
                    results[name][seed].append((stock_error, tp_error))
            gc.collect()

    ratios = []
    for name, per_seed in sorted(results.items()):
        assert len(per_seed) == len(args.seeds), name
        counts = {len(samples) for samples in per_seed.values()}
        assert counts in ({args.batches}, {args.batches * args.tp}), (name, counts)
        seed_means = [tuple(map(mean, zip(*samples, strict=True))) for samples in per_seed.values()]
        stock_error, tp_error = map(mean, zip(*seed_means, strict=True))
        ratio = error_ratio(tp_error, stock_error)
        spread = [error_ratio(tp, stock) for stock, tp in seed_means]
        seed_sd = stdev(spread) if all(map(math.isfinite, spread)) else math.inf
        ratios.append(ratio)
        if dist.get_rank() == 0:
            print(
                f"parity_parameter name={name} stock_mean={stock_error:.10g} "
                f"tp_mean={tp_error:.10g} ratio={ratio:.10g} "
                f"seed_min={min(spread):.10g} seed_max={max(spread):.10g} seed_sd={seed_sd:.10g} "
                f"observations={len(args.seeds) * next(iter(counts))}",
                flush=True,
            )
    geomean = math.inf if math.inf in ratios else 0.0 if 0.0 in ratios else geometric_mean(ratios)
    passed = max(ratios) <= 1.02 and geomean <= 1.0
    if dist.get_rank() == 0:
        print(
            f"parity_statistics seeds={','.join(map(str, args.seeds))} batches={args.batches} "
            f"parameters={len(ratios)} max_ratio={max(ratios):.10g} "
            f"geomean_ratio={geomean:.10g} passed={passed}",
            flush=True,
        )
    return passed


def run_trial(args, device, mesh, seed, batch_index):

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
    torch.manual_seed(seed)
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
        attention_bias=args.attention_bias,
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
    init_weights(oracle, config.num_hidden_layers, seed=seed, pipeline_chunks=False)
    initial, stock = copy.deepcopy(oracle), copy.deepcopy(oracle)
    fp32_tp = copy.deepcopy(oracle)
    dp_mesh = mesh["dp"]
    dp = dp_mesh.get_group()
    for layer in stock.layers:
        layer.forward = layer.forward_stock_stages
    stock_fsdp = DenseFSDPCfg()
    stock_fsdp.blocks_per_group = args.blocks_per_group
    shard_dense_model(stock, dp_mesh, stock_fsdp)
    install_wgrad_accumulation(stock)

    data_generator = None
    if args.statistical:
        data_generator = torch.Generator(device=device).manual_seed(seed * 1_000_003 + batch_index)
    tokens = torch.randint(
        config.vocab_size, (2, dp.size(), 1, 512), device=device, generator=data_generator
    )
    labels = torch.randint(config.vocab_size, tokens.shape, device=device, generator=data_generator)
    labels[..., 3::11] = -100
    stock_adjoints = []
    if args.cases and "stock-adjoint-compiled" in args.cases:
        capture = record_block_adjoints(stock.layers[0], stock_adjoints)
    from tp_delta_trace import (
        DECODER_TRACE_CASES,
        GQA_TRACE_CASES,
        TRACE_CASES,
        begin_microbatch,
        report,
        trace_attention,
        trace_stock_decoder,
    )

    if args.cases and any(case in TRACE_CASES for case in args.cases):
        if any(case in DECODER_TRACE_CASES for case in args.cases):
            trace_stock_decoder(stock)
        trace_gqa = any(case in GQA_TRACE_CASES for case in args.cases)
        with trace_attention(qwen35_moe, "stock", include_gqa=trace_gqa):
            stock_loss = backward(stock, tokens, labels, dp, begin_microbatch=begin_microbatch)
    else:
        stock_loss = backward(stock, tokens, labels, dp)
    if stock_adjoints:
        capture.remove()

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

    if args.bisect:
        from tp_precision_cases import (
            COMPILED_DECAY_CASES,
            MODULE_TP_CASES,
            PROJECTION_CASES,
            DecayProbeTP,
            ModuleTP,
            PrecisionTP,
            check_column_partials,
            check_row_partials,
            tp_staged,
        )

        precision_layout = dense_tp.DenseTP(mesh, async_tp=True)
        check_row_partials(precision_layout, device)
        check_column_partials(precision_layout, device)
        cases = args.cases or (
            "sync",
            "async",
            "sync-compiled",
            "async-compiled",
            "sync-staged",
            "async-staged",
            "stock-loss",
            "separate-dgrad",
            "stock-row",
            "fp32-norm-grads",
            "fp32-decay-grads",
            "fp32-decay-projection",
            *PROJECTION_CASES,
        )
    else:
        cases = ("requested",)

    rank = dist.get_rank()
    fp32_master_cases = (
        "fp32-affine-compiled",
        "fp32-gates-compiled",
        "fp32-affine-gates-compiled",
    )
    baseline_grads = {name: p.grad.full_tensor() for name, p in stock.named_parameters()}
    oracle_grads = {name: p.grad for name, p in oracle.named_parameters()}
    layout_grads = {name: p.grad.full_tensor() for name, p in fp32_tp.named_parameters()}
    oracle_rel = ((fp32_loss - oracle_loss).abs() / oracle_loss).item()
    assert oracle_rel < 1e-5
    passed, samples = [], []
    for case in cases:
        parallel = copy.deepcopy(initial)
        if case == "requested":
            overlap, compiled = args.async_tp, args.compile_blocks
            checkpoint = args.checkpoint
            layout = dense_tp.DenseTP(
                mesh,
                async_tp=overlap,
                regather_inputs=args.tp_regather_inputs,
                head_chunk_tokens=args.tp_head_chunk_tokens,
            )
        else:
            overlap, compiled = case.startswith("async"), case.endswith("-compiled")
            checkpoint = "selective" if compiled else "none"
            if case in (
                "sync",
                "async",
                "sync-compiled",
                "async-compiled",
                "sync-staged",
                "async-staged",
                "sync-cublas-fp32-compiled",
                "async-cublas-fp32-compiled",
                "sync-cublas-nosplit-compiled",
                "async-cublas-nosplit-compiled",
                *fp32_master_cases,
            ):
                layout = dense_tp.DenseTP(mesh, async_tp=overlap)
            elif case in COMPILED_DECAY_CASES:
                layout = DecayProbeTP(mesh, case)
            elif case in MODULE_TP_CASES:
                layout = ModuleTP(mesh)
            elif case in TRACE_CASES:
                layout = dense_tp.DenseTP(mesh, async_tp=False)
            else:
                layout = PrecisionTP(mesh, case, async_tp=overlap)
        layout.shard_model(parallel)
        if case.endswith("-staged"):
            for layer in parallel.layers:
                layer.forward = partial(tp_staged, layer)
        config_fsdp = DenseFSDPCfg()
        config_fsdp.tensor_parallel_size = args.tp
        config_fsdp.compile_blocks = compiled
        config_fsdp.tp_fp32_affine = args.tp_fp32_affine or case in (
            "fp32-affine-compiled",
            "fp32-affine-gates-compiled",
        )
        config_fsdp.tp_fp32_gates = args.tp_fp32_gates or case in (
            "fp32-gates-compiled",
            "fp32-affine-gates-compiled",
        )
        config_fsdp.tp_overwrite_wgrad = args.tp_overwrite_wgrad
        config_fsdp.tp_recompute_mlp_columns = args.tp_recompute_mlp_columns
        config_fsdp.tp_recompute_mlp_once = args.tp_recompute_mlp_once
        config_fsdp.tp_bulk_row_reduce = args.tp_bulk_row_reduce
        config_fsdp.tp_no_delta_checkpoint = args.tp_no_delta_checkpoint
        config_fsdp.activation_checkpoint = checkpoint
        config_fsdp.full_checkpoint_blocks = args.full_checkpoint_blocks
        config_fsdp.blocks_per_group = args.blocks_per_group
        config_fsdp.skip_embedding_backward_gather = True
        config_fsdp.reuse_single_weight_gather_buffers = args.reuse_single_weight_gather_buffers
        config_fsdp.keep_head_gathered = args.keep_head_gathered
        shard_dense_model(parallel, dp_mesh, config_fsdp)
        install_wgrad_accumulation(parallel, tp_overwrite=config_fsdp.tp_overwrite_wgrad)
        if args.token_sums_audit:
            partials, groups = capture_parameter_partials(parallel)
        if case == "stock-adjoint-compiled":
            override = use_stock_block_adjoints(parallel.layers[0], stock_adjoints, layout)
        if case in TRACE_CASES:
            with trace_attention(
                dense_tp,
                case,
                layout.rank,
                layout.size,
                include_gqa=case in GQA_TRACE_CASES,
            ):
                tp_loss = backward(parallel, tokens, labels, dp, layout, begin_microbatch)
            report(case, layout.rank, layout.size, rel_l2)
        else:
            with cublas_fp32_reduction(case):
                tp_loss = backward(parallel, tokens, labels, dp, layout)
        if case == "stock-adjoint-compiled":
            override.remove()
        if args.token_sums_audit:
            check_parameter_partials(parallel, layout, partials, groups, case)

        max_bf16_error = max_layout_error = max_excess = 0.0
        exceeding = 0
        for name, param in parallel.named_parameters():
            assert param.dtype == param.grad.dtype == torch.float32, name
            actual = param.grad.full_tensor()
            baseline, expected, fp32_actual = (
                baseline_grads[name],
                oracle_grads[name],
                layout_grads[name],
            )
            if spec := layout.shards.get(name):
                baseline, expected = spec.select(baseline), spec.select(expected)
            assert torch.isfinite(actual).all()
            bf16_error = rel_l2(actual, expected)
            stock_error = rel_l2(baseline, expected)
            assert math.isfinite(stock_error) and math.isfinite(bf16_error), name
            if args.statistical and dp.rank() == 0 and (spec or layout.rank == 0):
                samples.append((name, layout.rank, stock_error, bf16_error))
            layout_error = rel_l2(fp32_actual, expected)
            excess = bf16_error - stock_error
            exceeding += excess > 0
            max_excess = max(max_excess, excess)
            max_bf16_error = max(max_bf16_error, bf16_error)
            max_layout_error = max(max_layout_error, layout_error)
            print(
                f"case={case} rank={rank} grad={name} tp_to_fp32={bf16_error:.8f} "
                f"stock_to_fp32={stock_error:.8f} tp_to_stock={rel_l2(actual, baseline):.8f} "
                f"fp32_layout={layout_error:.8f} seed={seed} batch={batch_index}",
                flush=True,
            )

        loss_rel = ((tp_loss - stock_loss).abs() / stock_loss).item()
        print(
            f"case={case} rank={rank} TP={layout.size} FSDP={dp.size()} async={overlap} "
            f"compiled={compiled} checkpoint={checkpoint} stock_loss={stock_loss.item():.8f} "
            f"tp_loss={tp_loss.item():.8f} fp32_loss={oracle_loss.item():.8f} "
            f"max_tp_to_fp32={max_bf16_error:.8f} max_fp32_layout={max_layout_error:.8f} "
            f"exceeding_stock={exceeding} max_excess={max_excess:.8f} seed={seed} batch={batch_index}",
            flush=True,
        )
        if case == "requested":
            optimizer = torch.optim.AdamW(parallel.parameters(), lr=1e-5, fused=True)
            before = parallel.lm_head.weight.to_local().detach().clone()
            optimizer.step()
            assert not torch.equal(before, parallel.lm_head.weight.to_local())
            for param in parallel.parameters():
                state = optimizer.state[param]
                assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
            print(f"rank={rank} fp32_adamw=passed", flush=True)
        assert math.isfinite(loss_rel)
        assert max_layout_error < 2e-4
        if not args.statistical:
            assert loss_rel < 0.001
            bad = torch.tensor(exceeding > 0, dtype=torch.int, device=device)
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if not bad.item():
                passed.append(case)
        del parallel, layout

    return passed, samples


if __name__ == "__main__":
    main()
