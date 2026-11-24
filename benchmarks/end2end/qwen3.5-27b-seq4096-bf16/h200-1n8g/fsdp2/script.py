"""Benchmark dense Qwen3.5 with FSDP2 on eight H200 GPUs."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from functools import partial
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35TextModel


def checkpoint_layers(
    model: Qwen35TextModel,
    *,
    mlp_tail: int | None,
    keep_tail: int = 0,
    fp8_mlp_recompute: bool = False,
    replay_mixers: str = "none",
    fp8: bool = False,
    fp8_attention: bool = False,
) -> None:
    """Keep the final blocks intact, with MLP-only checkpointing immediately before them."""
    retain_attention = fp8 and mlp_tail is None
    if mlp_tail is None:
        mlp_tail = 30 if fp8 else 12
    # Inductor reuses FP8 projection output storage inside compiled stages.
    # Recompute those projections so this eager checkpoint never retains their aliases.
    saved_ops = [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    ]
    if replay_mixers not in ("gated-delta", "both"):
        saved_ops.append(torch.ops.pithtrain.gated_delta_rule_dense_fwd.default)
    if replay_mixers not in ("full", "both"):
        saved_ops.append(torch.ops.pithtrain.flash_attn4_mha_fwd.default)
        if fp8_attention:
            saved_ops.append(torch.ops.pithtrain.delta_matching_attn_fwd.default)
    selective_checkpoint = partial(
        checkpoint,
        use_reentrant=False,
        preserve_rng_state=False,
        context_fn=partial(create_selective_checkpoint_contexts, saved_ops),
    )
    keep_start = len(model.layers) - keep_tail
    tail_start = keep_start - mlp_tail
    if min(keep_tail, mlp_tail, tail_start) < 0:
        raise ValueError("Checkpoint tails must fit within the decoder")
    extra_attention = (
        next((i for i in reversed(range(tail_start)) if not model.layers[i].is_linear), -1)
        if retain_attention
        else -1
    )
    for i, layer in enumerate(model.layers):
        layer.forward = layer.forward_stock_stages
        if i >= keep_start:
            mixer = layer.linear_attn if layer.is_linear else layer.self_attn
            # The mixer normally has an inner selective checkpoint inside the compiled stage.
            mixer.forward = mixer.block
            continue
        checkpoint_mixer = i < tail_start and i != extra_attention
        if fp8_mlp_recompute:
            layer.mlp.replay_fp8_gate_up = True
            if checkpoint_mixer:
                layer.forward_stage1 = partial(selective_checkpoint, layer.forward_stage1)
        elif checkpoint_mixer:
            layer.forward = partial(selective_checkpoint, layer.forward)
        else:
            layer.forward_stage3 = partial(selective_checkpoint, layer.forward_stage3)


def shard_model(
    model: Qwen35TextModel,
    mesh: DeviceMesh,
    *,
    mlp_tail: int | None = 12,
    keep_tail: int = 0,
    fp8_mlp_recompute: bool = False,
    replay_mixers: str = "none",
    fp8: bool = False,
    fp8_attention: bool = False,
    fsdp_group_size: int = 2,
) -> None:
    checkpoint_layers(
        model,
        mlp_tail=mlp_tail,
        keep_tail=keep_tail,
        fp8_mlp_recompute=fp8_mlp_recompute,
        replay_mixers=replay_mixers,
        fp8=fp8,
        fp8_attention=fp8_attention,
    )

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blocks in batched(model.layers, fsdp_group_size):
        fully_shard(list(blocks), mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        blocks[0].set_gradient_divide_factor(1.0)

    if model.lm_head is not None:
        fully_shard(model.embed_tokens, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        if fp8:
            # Embedding backward uses only the saved token indices.
            model.embed_tokens.set_unshard_in_backward(False)
        model.embed_tokens.set_gradient_divide_factor(1.0)

        fully_shard(model.lm_head, mesh=mesh, mp_policy=mp, reshard_after_forward=not fp8)
        model.lm_head.set_gradient_divide_factor(1.0)

    # The root owns the final norm and any tied embedding and drives implicit prefetch.
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
    model.set_gradient_divide_factor(1.0)


@contextmanager
def bulk_reduce_scatter_packing(enabled: bool):
    """Use bulk copies for eligible FSDP buckets during this benchmark."""
    if not enabled:
        yield
        return

    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    from pithtrain.kernels import pack_reduce_scatter

    original = _fsdp_collectives.foreach_reduce_scatter_copy_in

    def copy_in(grads, output, world_size):
        if not pack_reduce_scatter(grads, output, world_size):
            original(grads, output, world_size)

    _fsdp_collectives.foreach_reduce_scatter_copy_in = copy_in
    try:
        yield
    finally:
        _fsdp_collectives.foreach_reduce_scatter_copy_in = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("examples/pretrain_lm/qwen3.5-27b"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path, help="Tokenized corpus directory")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--fsdp-group-size", type=int, default=2)
    parser.add_argument(
        "--checkpoint-mlp-tail", type=int, help="Layers that checkpoint only their MLP"
    )
    parser.add_argument(
        "--checkpoint-keep-tail", type=int, default=0, help="Final blocks with no checkpoint"
    )
    parser.add_argument("--fp8-shard-cache-mib", type=int)
    parser.add_argument(
        "--fp8-mlp-recompute",
        action="store_true",
        help="Replay the gate and up projections from saved FP8 input codes",
    )
    parser.add_argument(
        "--checkpoint-replay-mixers",
        choices=("none", "gated-delta", "full", "both"),
        default="none",
    )
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument(
        "--fp8-attention", action="store_true", help="Use FP8 Delta-Matching attention"
    )
    args = parser.parse_args()
    if args.fsdp_group_size < 1:
        parser.error("--fsdp-group-size must be positive")
    if args.fp8_mlp_recompute and not args.fp8:
        parser.error("--fp8-mlp-recompute requires --fp8")
    if args.fp8_shard_cache_mib is not None and args.fp8_shard_cache_mib < 0:
        parser.error("--fp8-shard-cache-mib must be nonnegative")

    # PithTrain's kernel imports query the current GPU.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset if args.dataset is not None else Path(".")
    train = cfg.training
    train.model = args.model
    train.hf_import_path = args.weights
    train.benchmark = args.dataset is None
    train.sequence_length = args.seq
    train.global_batch_size = args.batch
    train.micro_batch_size = args.micro_batch
    train.max_steps = args.steps
    train.lr = 1e-5
    train.fp8 = args.fp8
    train.fp8_attention = args.fp8_attention
    mlp_tail = args.checkpoint_mlp_tail
    if args.fp8 and mlp_tail is None:
        train.fp8_shard_cache_bytes = 1 << 30
        if args.fp8_attention:
            # Free space for Delta-Matching's backward workspace at microbatch four.
            mlp_tail = 16 if args.fp8_mlp_recompute else 29
    if args.fp8_shard_cache_mib is not None:
        train.fp8_shard_cache_bytes = args.fp8_shard_cache_mib << 20
    train.optimizer = partial(make_adamw_optimizer, fused=True)
    train.scheduler = make_constant_scheduler
    shard = partial(
        shard_model,
        mlp_tail=mlp_tail,
        keep_tail=args.checkpoint_keep_tail,
        fp8_mlp_recompute=args.fp8_mlp_recompute,
        replay_mixers=args.checkpoint_replay_mixers,
        fp8=args.fp8,
        fp8_attention=args.fp8_attention,
        fsdp_group_size=args.fsdp_group_size,
    )
    with bulk_reduce_scatter_packing(args.fp8):
        launch(cfg, shard_dense_model=shard)
