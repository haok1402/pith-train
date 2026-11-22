"""Sequence and tensor parallelism for the standalone BF16 Qwen3.5 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import DeviceMesh

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35TextModel


@dataclass(frozen=True)
class DenseTPShard:
    """Map the rows or columns of a local parameter to its canonical checkpoint."""

    dim: int
    spans: tuple[range, ...]

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        pieces = [tensor.narrow(self.dim, span.start, len(span)) for span in self.spans]
        return torch.cat(pieces, dim=self.dim) if len(pieces) > 1 else pieces[0].clone()


class DenseTP:
    def __init__(self, mesh: DeviceMesh, *, async_tp: bool):
        self.mesh = mesh
        self.group = mesh.get_group("tp")
        self.dp_group = mesh.get_group("dp")
        self.rank = self.group.rank()
        self.size = self.group.size()
        self.name = self.group.group_name
        self.async_tp = async_tp
        self.shards: dict[str, DenseTPShard] = {}

    def shard_model(self, model: Qwen35TextModel) -> None:
        """Shard complete query/key groups and place their corresponding value heads together."""
        modules = dict(model.named_modules())

        def shard(name: str, dim: int = 0, segments: tuple[int, ...] = ()) -> None:
            mod_name, _, attr = name.rpartition(".")
            module = modules[mod_name]
            param = getattr(module, attr)
            if param is None:
                return
            offset, spans = 0, []
            for width in segments or (param.size(dim),):
                if width % self.size:
                    raise ValueError(f"{name} has width {width} indivisible by TP={self.size}")
                local_width = width // self.size
                start = offset + self.rank * local_width
                spans.append(range(start, start + local_width))
                offset += width
            spec = self.shards[name] = DenseTPShard(dim, tuple(spans))
            setattr(module, attr, nn.Parameter(spec.select(param.detach())))

        def linear(name: str, dim: int = 0, segments: tuple[int, ...] = ()) -> None:
            shard(name + ".weight", dim, segments)
            module = modules[name]
            if dim == 0:
                shard(name + ".bias")
                module.out_features //= self.size
            else:
                module.in_features //= self.size

        shard("embed_tokens.weight")
        linear("lm_head")
        model.embed_tokens.num_embeddings //= self.size

        for i, layer in enumerate(model.layers):
            prefix = f"layers.{i}."
            for projection in ("mlp.gate_proj", "mlp.up_proj"):
                linear(prefix + projection)
            linear(prefix + "mlp.down_proj", 1)
            if layer.is_linear:
                delta = layer.linear_attn
                if delta.num_k_heads % self.size or delta.num_v_heads % self.size:
                    raise ValueError("TP must shard DeltaNet key/value heads evenly")
                name = prefix + "linear_attn."
                qkv = (delta.key_dim, delta.key_dim, delta.value_dim)
                linear(name + "in_proj_qkv", segments=qkv)
                shard(name + "conv1d.weight", segments=qkv)
                for projection in ("in_proj_z", "in_proj_a", "in_proj_b"):
                    linear(name + projection)
                linear(name + "out_proj", 1)
                for attr in ("dt_bias", "A_log"):
                    shard(name + attr)
                delta.num_k_heads //= self.size
                delta.num_v_heads //= self.size
                delta.key_dim //= self.size
                delta.value_dim //= self.size
                delta.conv_dim //= self.size
                delta.conv1d.in_channels = delta.conv1d.out_channels = delta.conv1d.groups = (
                    delta.conv_dim
                )
            else:
                attn = layer.self_attn
                if attn.num_heads % self.size or attn.num_kv_heads % self.size:
                    raise ValueError("TP must shard gated attention query/KV heads evenly")
                name = prefix + "self_attn."
                for projection in ("q_proj", "k_proj", "v_proj"):
                    linear(name + projection)
                linear(name + "o_proj", 1)
                attn.num_heads //= self.size
                attn.num_kv_heads //= self.size
            layer.tensor_parallel = self

        model.tensor_parallel = self

    @torch.no_grad()
    def sync_replicated_grads(self, model: nn.Module) -> list[torch.Tensor]:
        """Sum the partial norm and row-bias gradients, then identify unique local shards."""
        sharded, replicated = [], []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.to_local()
                (sharded if name in self.shards else replicated).append(grad)
        for dtype in {grad.dtype for grad in replicated}:
            grads = [grad for grad in replicated if grad.dtype == dtype]
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, group=self.group)
            torch._foreach_copy_(grads, torch._utils._unflatten_dense_tensors(flat, grads))
        return sharded if self.rank else sharded + replicated
