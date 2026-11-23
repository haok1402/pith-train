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


def checkpoint_layers(model: Qwen35TextModel, *, mlp_tail: int) -> None:
    """Checkpoint stock decoder stages, replaying only the MLP in the tail."""
    # Inductor reuses FP8 projection output storage inside compiled stages.
    # Recompute those projections so this eager checkpoint never retains their aliases.
    saved_ops = [
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.pithtrain.gated_delta_rule_dense_fwd.default,
        torch.ops.pithtrain.flash_attn4_mha_fwd.default,
    ]
    selective_checkpoint = partial(
        checkpoint,
        use_reentrant=False,
        preserve_rng_state=False,
        context_fn=partial(create_selective_checkpoint_contexts, saved_ops),
    )
    for i, layer in enumerate(model.layers):
        layer.forward = layer.forward_stock_stages
        if i < len(model.layers) - mlp_tail:
            layer.forward = partial(selective_checkpoint, layer.forward)
        else:
            layer.forward_stage3 = partial(selective_checkpoint, layer.forward_stage3)


def shard_model(
    model: Qwen35TextModel, mesh: DeviceMesh, *, mlp_tail: int = 12, fp8: bool = False
) -> None:
    checkpoint_layers(model, mlp_tail=mlp_tail)

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blocks in batched(model.layers, 2):
        fully_shard(list(blocks), mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        blocks[0].set_gradient_divide_factor(1.0)

    fully_shard(model.embed_tokens, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
    if fp8:
        # Embedding backward uses only the saved token indices.
        model.embed_tokens.set_unshard_in_backward(False)
    model.embed_tokens.set_gradient_divide_factor(1.0)

    fully_shard(model.lm_head, mesh=mesh, mp_policy=mp, reshard_after_forward=not fp8)
    model.lm_head.set_gradient_divide_factor(1.0)

    # The root owns the final norm and drives FSDP's implicit prefetch.
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
    parser.add_argument(
        "--checkpoint-mlp-tail", type=int, help="Layers that checkpoint only their MLP"
    )
    parser.add_argument("--fp8", action="store_true")
    args = parser.parse_args()

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
    train.optimizer = partial(make_adamw_optimizer, fused=True)
    train.scheduler = make_constant_scheduler
    mlp_tail = args.checkpoint_mlp_tail
    if mlp_tail is None:
        mlp_tail = 29 if args.fp8 else 12
    with bulk_reduce_scatter_packing(args.fp8):
        launch(cfg, shard_dense_model=partial(shard_model, mlp_tail=mlp_tail, fp8=args.fp8))
