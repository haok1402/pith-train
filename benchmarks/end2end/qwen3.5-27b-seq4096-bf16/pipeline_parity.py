"""Report gradient errors against an unsplit FP32 Qwen3.5 and stock DualPipeV."""

import argparse
import csv
import logging as std_logging
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def relative_error(actual, reference):
    return ((actual - reference).norm() / reference.norm().clamp_min(1e-6)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["1f1b-attention", "interleaved-attention"])
    parser.add_argument("--repeat-stock", action="store_true")
    parser.add_argument("--compute-stream-grad-buffers", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from pithtrain.contexts import distributed, logging, training
    from pithtrain.models.qwen35 import Qwen35Model
    from pithtrain.modules import training as train
    from pithtrain.modules.distributed import setup_distributed
    from pithtrain.modules.optimizer import clip_grad_norm, local_grads, scale_grads_
    from pithtrain.operators.linear import BF16Linear
    from pithtrain.pipeline import Microbatch
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, objective

    std_logging.basicConfig(level=std_logging.INFO)
    logging.stdout = std_logging.getLogger("pipeline-parity")
    cfg = PretrainLMCfg()
    cfg.distributed.pipeline_parallel_size = 2
    setup_distributed(cfg)
    device = distributed.device
    rank = dist.get_rank()
    cfg.training.model = args.output / "config"
    cfg.training.sequence_length = seq = 64
    config = Qwen3_5TextConfig(
        hidden_size=768,
        intermediate_size=1024,
        vocab_size=2048,
        num_hidden_layers=8,
        layer_types=["linear_attention", "full_attention"] * 4,
        max_position_embeddings=seq,
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
    if rank == 0:
        config.save_pretrained(cfg.training.model)
    dist.barrier()

    training.fp8, training.Linear = False, BF16Linear
    torch.manual_seed(17)
    cfg.training.seed = 17
    with torch.device(device):
        oracle = Qwen35Model(config, phase=-1)
    train.init_weights(nn.Sequential(oracle), config.num_hidden_layers, seed=17)
    weights = {name: param.detach() for name, param in oracle.named_parameters()}
    generator = torch.Generator(device=device).manual_seed(713)
    local_batches = 8
    shape = (local_batches, distributed.dp_size, 2, seq)
    tokens = torch.randint(32, shape, generator=generator, device=device)
    labels = torch.randint(config.vocab_size, shape, generator=generator, device=device)
    labels[..., -1] = -100
    num_tokens = (labels != -100).sum()
    batches = [
        Microbatch(model_inputs=(x,), cu_seqlens=None, objective_inputs=(y,))
        for x, y in zip(tokens[:, distributed.dp_rank], labels[:, distributed.dp_rank], strict=True)
    ]

    fp32_reference = runpy.run_path(str(args.oracle))["fp32_reference"]
    reference_model = SimpleNamespace(
        layers=oracle.layers.values(),
        embed_tokens=oracle.embed_tokens,
        rotary_emb=oracle.rotary_emb,
        norm=oracle.norm,
        lm_head=oracle.lm_head,
    )
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    reference_loss = F.cross_entropy(
        fp32_reference(reference_model, tokens.view(-1, seq)), labels.flatten(), ignore_index=-100
    )
    reference_loss.backward()
    torch.set_float32_matmul_precision(old_precision)
    expected = {name: param.grad for name, param in oracle.named_parameters()}

    stock_grads = {}
    contiguous_grads = {}
    controls = ["stock", "stock-repeat"] if args.repeat_stock else ["stock"]
    for arm in [*controls, *args.arms]:
        records = []
        schedule, compile_mode = (
            ("dualpipev", "attention") if arm in controls else arm.rsplit("-", 1)
        )
        replay_up = schedule.endswith("-up")
        schedule = schedule.removesuffix("-up")
        cfg.training.pipeline_schedule = schedule
        cfg.training.pipeline_virtual_stages = 2 if schedule in ("interleaved", "zbv") else 1
        cfg.training.pipeline_compile = compile_mode
        cfg.training.pipeline_compute_stream_grad_buffers = (
            args.compute_stream_grad_buffers and arm not in controls
        )
        cfg.training.pipeline_recompute_mlp = replay_up
        cfg.training.pipeline_recompute_mlp_up_only = replay_up
        with torch.device(device):
            train.setup_model(cfg.training, cfg.distributed)
        model = training.model
        params = {name.split(".", 2)[2]: param for name, param in model.named_parameters()}
        with torch.no_grad():
            for name, param in params.items():
                assert isinstance(param, DTensor)
                (placement,) = param.placements
                shard = weights[name].chunk(param.device_mesh.size(), dim=placement.dim)[
                    param.device_mesh.get_local_rank()
                ]
                local(param).copy_(shard)
        print(f"rank={rank} starting {arm}", flush=True)
        losses = model.step(batches, objective)
        loss = torch.stack(losses).sum() if losses else torch.zeros((), device=device)
        dist.all_reduce(loss)
        torch.testing.assert_close(loss / num_tokens, reference_loss, rtol=0.002, atol=0.002)
        scale_grads_(local_grads(model), num_tokens.reciprocal())
        gradients = {}
        for name, param in params.items():
            assert param.dtype == torch.float32, (arm, name, param.dtype)
            assert param.grad is not None and param.grad.dtype == torch.float32, (arm, name)
            gradient = param.grad.full_tensor()
            assert torch.isfinite(gradient).all(), (arm, name)
            gradients[name] = gradient

        # Share the controls with every pipeline stage since a parameter can change owners.
        target = None
        if arm == "stock":
            target = stock_grads
        elif schedule == "1f1b":
            target = contiguous_grads.setdefault(compile_mode, {})
        if target is not None:
            owners = [None] * distributed.pp_size
            group = distributed.pp_group
            dist.all_gather_object(owners, tuple(gradients), group=group)
            for owner, names in enumerate(owners):
                for name in names:
                    tensor = (
                        gradients[name] if name in gradients else torch.empty_like(expected[name])
                    )
                    dist.broadcast(tensor, group=group, group_src=owner)
                    target[name] = tensor
        contiguous = contiguous_grads.get(compile_mode)
        for name, gradient in gradients.items():
            if compile_mode == "attention":
                assert torch.equal(gradient, stock_grads[name]), (arm, name)
            if distributed.dp_rank == 0:
                records.append(
                    (
                        arm,
                        name,
                        relative_error(gradient, expected[name]),
                        relative_error(stock_grads[name], expected[name]),
                        relative_error(gradient, stock_grads[name]),
                        relative_error(gradient, contiguous[name]) if contiguous else None,
                    )
                )

        norm = clip_grad_norm(model, max_norm=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1, fused=True)
        first = next(iter(params.values()))
        before = local(first).detach().clone()
        optimizer.step()
        for state in optimizer.state.values():
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        assert not torch.equal(local(first), before), (arm, rank)
        if rank == 0:
            print(
                f"{arm} loss={loss.item() / num_tokens.item():.9f} "
                f"FP32 loss={reference_loss.item():.9f} grad_norm={norm.item():.9f} "
                "masters/grads/moments=FP32 AdamW=updated",
                flush=True,
            )
        all_records = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(records, all_records, dst=0)
        if rank == 0:
            rows = sorted(row for rank_records in all_records for row in rank_records)
            assert {row[1] for row in rows} == expected.keys() and len(rows) == len(expected)
            with (args.output / "gradients.csv").open(
                "w" if arm == "stock" else "a", newline=""
            ) as stream:
                writer = csv.writer(stream)
                if arm == "stock":
                    writer.writerow(
                        ("arm", "param", "vs_fp32", "stock_vs_fp32", "vs_stock", "vs_contiguous")
                    )
                writer.writerows(rows)
            worst = max(rows, key=lambda row: row[4])
            same_layout = max((row[5] for row in rows if row[5] is not None), default=None)
            excess = max(row[2] - row[3] for row in rows)
            print(
                f"{arm}: {len(rows)} parameters, max rel_L2 vs stock={worst[4]:.8g} "
                f"({worst[1]}), max vs contiguous={same_layout}, "
                f"max excess FP32 reference error={excess:.8g}",
                flush=True,
            )
        del model, optimizer, params, gradients, training.model
        dist.barrier()


if __name__ == "__main__":
    main()
