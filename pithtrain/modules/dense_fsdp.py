"""FSDP2 configuration and sharding for a dense text tower."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import batched, groupby
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from pithtrain.config import SlottedDefault
from pithtrain.contexts import logging, training
from pithtrain.kernels import stage_compile_options

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35TextModel

__all__ = ["DenseFSDPCfg", "shard_dense_model"]


@dataclass(init=False, slots=True)
class DenseFSDPCfg(SlottedDefault):
    tensor_parallel_size: int = 1

    async_tensor_parallel: bool = False
    """Overlap sequence collectives and tensor-core GEMMs via CUDA symmetric memory."""

    tp_regather_inputs: bool = False
    """Save only local column inputs and gather them again before the weight gradient."""

    tp_head_chunk_tokens: int = 0
    """Bound vocab-logit storage by rematerializing token chunks of this size in backward."""

    tp_fp32_affine: bool = False
    """Gather the small norm and decay parameters directly in their master dtype."""

    tp_fp32_gates: bool = False
    """Gather the small DeltaNet a and b projection weights in their master dtype."""

    tp_overwrite_wgrad: bool = False
    """Write each TP weight's first microbatch contribution directly into the FP32 buffer."""

    tp_recompute_mlp_columns: bool = False
    """Replay the two wide MLP column projections during selective checkpointing."""

    tp_recompute_mlp_once: bool = False
    """Use the MLP projection replay gather for the weight gradients too."""

    tp_bulk_row_reduce: bool = False
    """Keep async input gathers and use full FP32 GEMMs followed by NCCL reduce-scatter."""

    blocks_per_group: int = 1
    """Number of consecutive decoder blocks per communication bucket."""

    group_by_attention_type: bool = False
    """Start a new bucket at each boundary between DeltaNet and gated attention."""

    edge_sharding: Literal["separate", "root"] = "separate"
    """Shard embedding and LM head independently or put both in the root's bucket."""

    reshard_after_forward: bool | int = True
    """Reshard to the full mesh, retain gathered parameters, or reshard to a smaller group."""

    keep_last_groups: int = 0
    """Keep this many final decoder buckets materialized until backward."""

    keep_gathered_across_microbatches: bool = False
    """Keep the selected buckets materialized until the final backward of the step."""

    skip_embedding_backward_gather: bool = False
    """Skip gathering the embedding weight in backward when it is its own bucket."""

    reuse_single_weight_gather_buffers: bool = False
    """Use the all-gather destination directly for separately sharded edge weights."""

    prefetch: int = 0
    """Explicit lookahead in buckets in both directions. Zero uses FSDP's implicit prefetch."""

    reduce_dtype: Literal["fp32", "bf16"] = "fp32"

    bf16_sharded_gradients: bool = False
    """Accumulate reduced shards in BF16 and upcast once before AdamW."""

    reduce_at_step_end: bool = False
    """Accumulate unsharded gradients across microbatches and reduce them in the last backward."""

    reduce_scatter_buffers: int = 1
    """Maximum number of in-flight reduce-scatter staging buffers."""

    separate_reduce_scatter: bool = False
    """Use a dedicated communicator for reduce-scatter to allow backward all-gather overlap."""

    activation_checkpoint: Literal["none", "selective", "full"] = "none"

    checkpoint_blocks: int | None = None
    """Checkpoint this many leading decoder blocks, or all by default."""

    full_checkpoint_blocks: int = 0
    """Override selective checkpointing with full recompute for this decoder prefix."""

    compile_blocks: bool = False


def checkpoint_policy(ctx, op, *args, recompute_mlp_columns=False, **kwargs):
    name = op.name()
    if recompute_mlp_columns and name == "symm_mem::fused_all_gather_matmul" and len(args[1]) == 2:
        # The MLP is the only dense TP call that gathers for two projections.
        return CheckpointPolicy.MUST_RECOMPUTE
    # The combined TP operators contain GEMMs just like aten.mm.
    if name in {
        "symm_mem::fused_all_gather_matmul",
        "pithtrain::tp_fp32_matmul_reduce_scatter",
    }:
        return CheckpointPolicy.MUST_SAVE
    if op in {
        torch.ops.aten.mm.default,
        torch.ops.aten.mm.dtype,
        torch.ops.aten.addmm.default,
        torch.ops.pithtrain.gated_delta_rule_dense_fwd.default,
        torch.ops.pithtrain.flash_attn4_mha_fwd.default,
    }:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_checkpoint_contexts(recompute_mlp_columns=False):
    policy = checkpoint_policy
    if recompute_mlp_columns:
        policy = partial(checkpoint_policy, recompute_mlp_columns=True)
    return create_selective_checkpoint_contexts(policy)


def reuse_single_weight_gather_buffer(unit: nn.Module) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllGather
    from torch.distributed.fsdp._fully_shard._fsdp_param import alloc_storage

    (group,) = unit._get_fsdp_state()._fsdp_param_groups
    (param,) = group.fsdp_params

    class AllGatherIntoWeight(DefaultAllGather):
        def allocate(self, size, *, dtype, device):
            if not param.all_gather_outputs:
                param.all_gather_outputs = [super().allocate(size, dtype=dtype, device=device)]
            (output,) = param.all_gather_outputs
            if output.shape != size or output.dtype != dtype or output.device != device:
                raise ValueError("All-gather buffer metadata changed")
            alloc_storage(output)
            return output

    def reuse_output(_params, result, _world_size):
        result.all_gather_output.record_stream(torch.cuda.current_stream())

    unshard = group.unshard

    def unshard_into_weight(async_op: bool = False):
        # A backward gather restores a saved weight. Preserve its version just
        # as FSDP does when copying the all-gather output into that weight.
        if param.all_gather_outputs and not param.all_gather_outputs[0].is_inference():
            with torch.autograd._unsafe_preserve_version_counter(tuple(param.all_gather_outputs)):
                return unshard(async_op=async_op)
        return unshard(async_op=async_op)

    group.unshard = unshard_into_weight
    group._all_gather_output_fn = reuse_output
    unit.set_custom_all_gather(AllGatherIntoWeight())


def log_bucket_sizes(
    buckets: list[tuple[str, list[nn.Module]]],
    shard_size: int,
    reduce_dtype: torch.dtype,
    fp32_params: set[nn.Parameter],
) -> None:
    logger = getattr(logging, "stdout", None)
    if logger is None:
        return
    sizes = Counter()
    for name, modules in buckets:
        gather_bytes = reduce_bytes = 0
        for module in modules:
            for param in module.parameters():
                rows = -(-param.size(0) // shard_size) * shard_size
                numel = rows * (param.numel() // param.size(0))
                gather_dtype = param.dtype if param in fp32_params else training.PARAM_DTYPE
                gather_bytes += numel * gather_dtype.itemsize
                reduce_bytes += numel * reduce_dtype.itemsize
        sizes[name, gather_bytes, reduce_bytes] += 1
    payloads = (
        f"{count} {name} {gather / 2**20:.1f}/{reduce / 2**20:.1f}"
        for (name, gather, reduce), count in sizes.items()
    )
    logger.info("Dense FSDP buckets (AG/RS MiB) | %s", ", ".join(payloads))


def shard_dense_model(model: Qwen35TextModel, mesh: DeviceMesh, cfg: DenseFSDPCfg) -> None:
    if cfg.blocks_per_group < 1:
        raise ValueError("dense_fsdp.blocks_per_group must be positive")
    if cfg.keep_last_groups < 0:
        raise ValueError("dense_fsdp.keep_last_groups must be nonnegative")
    if cfg.edge_sharding not in ("separate", "root"):
        raise ValueError("dense_fsdp.edge_sharding must be separate or root")
    if cfg.edge_sharding == "root" and cfg.skip_embedding_backward_gather:
        raise ValueError("Skipping the embedding backward gather requires a separate embedding")
    if cfg.edge_sharding == "root" and cfg.reuse_single_weight_gather_buffers:
        raise ValueError("Reusing gathered weights requires separate edge buckets")
    if cfg.prefetch < 0:
        raise ValueError("dense_fsdp.prefetch must be nonnegative")
    if cfg.reduce_dtype not in ("fp32", "bf16"):
        raise ValueError("dense_fsdp.reduce_dtype must be fp32 or bf16")
    if cfg.activation_checkpoint not in ("none", "selective", "full"):
        raise ValueError("dense_fsdp.activation_checkpoint must be none, selective or full")
    tp = model.tensor_parallel
    if cfg.tp_recompute_mlp_once and (
        tp is None or not tp.async_tp or cfg.tp_recompute_mlp_columns
    ):
        raise ValueError("Single-gather MLP replay requires async TP and its own replay policy")
    if cfg.tp_bulk_row_reduce and (tp is None or not tp.async_tp):
        raise ValueError("The bulk row reduction override requires async TP")
    if tp is not None:
        tp.recompute_mlp_once = cfg.tp_recompute_mlp_once
        tp.bulk_row_reduce = cfg.tp_bulk_row_reduce
    if cfg.tp_recompute_mlp_columns and (
        cfg.activation_checkpoint != "selective"
        or model.tensor_parallel is None
        or not model.tensor_parallel.async_tp
    ):
        raise ValueError("Replaying TP MLP columns requires async TP and selective checkpointing")
    if cfg.tensor_parallel_size > 1 and (cfg.reduce_dtype != "fp32" or cfg.bf16_sharded_gradients):
        raise ValueError("Dense TP requires FP32 reduction and gradient shards")
    if cfg.tp_overwrite_wgrad and (model.tensor_parallel is None or cfg.reduce_at_step_end):
        raise ValueError("Overwriting TP weight gradients requires reduction on each microbatch")
    if cfg.tp_head_chunk_tokens < 0:
        raise ValueError("TP head chunk size must be nonnegative")
    if cfg.keep_gathered_across_microbatches and not cfg.keep_last_groups:
        raise ValueError("Keeping weights across microbatches requires --keep-last-groups")
    if cfg.checkpoint_blocks is not None and not (
        cfg.activation_checkpoint != "none" and 1 <= cfg.checkpoint_blocks <= len(model.layers)
    ):
        raise ValueError("Checkpoint prefix requires activation checkpointing and a valid depth")

    reduce_dtype = torch.float32 if cfg.reduce_dtype == "fp32" else torch.bfloat16
    if cfg.bf16_sharded_gradients:
        if reduce_dtype != torch.bfloat16:
            raise ValueError("BF16 sharded gradients require BF16 reduction")
        for param in model.parameters():
            param.grad_dtype = torch.bfloat16
    fp32_params = set()
    if model.tensor_parallel is not None and training.PARAM_DTYPE == torch.bfloat16:
        fp32_params.update(
            model.tensor_parallel.configure_fp32_token_sums(
                model, round_affine=not cfg.tp_fp32_affine
            )
        )
    elif cfg.tp_fp32_affine:
        fp32_params.update(param for param in model.parameters() if param.ndim == 1)
    if cfg.tp_fp32_gates:
        for layer in model.layers:
            if layer.is_linear:
                fp32_params.update(
                    (layer.linear_attn.in_proj_a.weight, layer.linear_attn.in_proj_b.weight)
                )
    mp = MixedPrecisionPolicy(
        param_dtype=training.PARAM_DTYPE,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
        param_dtype_override_fn=(lambda param: param.dtype if param in fp32_params else None)
        if fp32_params
        else None,
    )
    checkpoint_blocks = cfg.checkpoint_blocks or len(model.layers)
    if cfg.full_checkpoint_blocks and (
        cfg.activation_checkpoint != "selective"
        or not 1 <= cfg.full_checkpoint_blocks <= checkpoint_blocks
    ):
        raise ValueError(
            "Full checkpoint prefix requires selective checkpointing and a valid depth"
        )
    contexts = selective_checkpoint_contexts
    if cfg.tp_recompute_mlp_columns:
        contexts = partial(selective_checkpoint_contexts, recompute_mlp_columns=True)
    for i, layer in enumerate(model.layers):
        if i < checkpoint_blocks and cfg.activation_checkpoint != "none":
            ctx = (
                {"context_fn": contexts}
                if cfg.activation_checkpoint == "selective" and i >= cfg.full_checkpoint_blocks
                else {}
            )
            layer.forward = partial(
                checkpoint, layer.forward, use_reentrant=False, preserve_rng_state=False, **ctx
            )
        if cfg.compile_blocks:
            layer.forward = torch.compile(
                layer.forward,
                fullgraph=True,
                dynamic=False,
                options=stage_compile_options() or None,
            )

    decoder_groups = [
        list(group)
        for _, run in groupby(
            model.layers, key=lambda layer: cfg.group_by_attention_type and layer.is_linear
        )
        for group in batched(run, cfg.blocks_per_group)
    ]
    buckets = []
    for group in decoder_groups:
        kinds = {layer.layer_type for layer in group}
        label = next(iter(kinds)) if len(kinds) == 1 else "hybrid"
        buckets.append((label, group))
    if cfg.edge_sharding == "separate":
        buckets += [
            ("embedding", [model.embed_tokens]),
            ("head", [model.lm_head]),
            ("root", [model.norm]),
        ]
    else:
        buckets.append(("root", [model.embed_tokens, model.norm, model.lm_head]))
    log_bucket_sizes(buckets, mesh.size(mesh.ndim - 1), reduce_dtype, fp32_params)

    for i, group in enumerate(decoder_groups):
        keep = i >= len(decoder_groups) - cfg.keep_last_groups
        reshard = False if keep else cfg.reshard_after_forward
        fully_shard(
            group[0] if len(group) == 1 else group,
            mesh=mesh,
            reshard_after_forward=reshard,
            mp_policy=mp,
        )

    # One FSDP state per communication bucket, even when several blocks share it.
    order = [group[0] for group in decoder_groups]
    model.retained_fsdp_units = (
        tuple(order[-cfg.keep_last_groups :]) if cfg.keep_gathered_across_microbatches else ()
    )
    if cfg.edge_sharding == "separate":
        fully_shard(model.embed_tokens, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
        fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
        if cfg.reuse_single_weight_gather_buffers:
            reuse_single_weight_gather_buffer(model.embed_tokens)
            reuse_single_weight_gather_buffer(model.lm_head)
        if cfg.skip_embedding_backward_gather:
            model.embed_tokens.set_unshard_in_backward(False)
        order = [model.embed_tokens, *order, model.lm_head]
    fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp)

    if cfg.edge_sharding == "root" and cfg.prefetch:
        model.set_modules_to_forward_prefetch(order[: cfg.prefetch])
        model.set_modules_to_backward_prefetch(order[-cfg.prefetch :][::-1])
    for i, module in enumerate(order):
        module.set_gradient_divide_factor(1.0)
        if cfg.prefetch:
            module.set_modules_to_forward_prefetch(order[i + 1 : i + 1 + cfg.prefetch])
            module.set_modules_to_backward_prefetch(order[max(0, i - cfg.prefetch) : i][::-1])
    model.set_gradient_divide_factor(1.0)
    if cfg.reduce_scatter_buffers != 1:
        model.set_reduce_scatter_max_input_buffers(cfg.reduce_scatter_buffers)
    if cfg.separate_reduce_scatter:
        model.set_separate_reduce_scatter_group()
