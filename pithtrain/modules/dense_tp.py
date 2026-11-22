"""Sequence and tensor parallelism for the standalone BF16 Qwen3.5 trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import DeviceMesh

from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.gated_delta_rule import gated_delta_rule
from pithtrain.operators.silu_mul import silu_mul

if TYPE_CHECKING:
    from pithtrain.models.qwen35 import Qwen35DecoderLayer, Qwen35TextModel


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
        if async_tp:
            # Import registers the symmetric-memory operators.
            import torch.distributed._symmetric_memory  # noqa: F401
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

    def embed(self, module: nn.Embedding, tokens: torch.Tensor) -> torch.Tensor:
        shard_vocab = module.num_embeddings
        indices = tokens.flatten() - self.rank * shard_vocab
        owned = (indices >= 0) & (indices < shard_vocab)
        embedded = module(indices.clamp(0, shard_vocab - 1))
        return _ScatterSum.apply(embedded * owned[:, None], self.size, self.name)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        return _Gather.apply(x, self.size, self.name)

    def column_weights(self, x: torch.Tensor, *weights: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _ColumnParallel.apply(x, self.size, self.name, self.async_tp, *weights)

    def columns(self, x: torch.Tensor, *modules: nn.Linear) -> tuple[torch.Tensor, ...]:
        outputs = self.column_weights(x, *(module.weight for module in modules))
        return tuple(
            output if module.bias is None else output + module.bias
            for output, module in zip(outputs, modules)
        )

    def rows(self, x: torch.Tensor, module: nn.Linear) -> torch.Tensor:
        output = _RowParallel.apply(x, module.weight, self.size, self.name, self.async_tp)
        return output if module.bias is None else output + module.bias

    def decoder(
        self,
        layer: Qwen35DecoderLayer,
        hidden: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq = rotary[0].size(1)
        batch = hidden.size(0) * self.size // seq
        x = layer.input_layernorm(hidden)
        if layer.is_linear:
            delta = layer.linear_attn
            # Both gates use the same gathered input. A combined small GEMM also
            # combines their input gradients before rounding to BF16.
            gate_weights = torch.cat((delta.in_proj_b.weight, delta.in_proj_a.weight), dim=0)
            qkv, z, gates = self.column_weights(
                x, delta.in_proj_qkv.weight, delta.in_proj_z.weight, gate_weights
            )
            b, a = gates.chunk(2, dim=-1)
            q, k, v = delta.conv_qkv(qkv.view(batch, seq, -1), 0)
            beta = b.view(batch, seq, -1).sigmoid()
            decay = -delta.A_log.float().exp() * F.softplus(a.float() + delta.dt_bias.float())
            x = gated_delta_rule(q, k, v, decay.view(batch, seq, -1), beta)
            x = delta.norm(x.reshape(-1, delta.head_v_dim), z.reshape(-1, delta.head_v_dim))
            x = self.rows(x.view(batch * seq, -1), delta.out_proj)
        else:
            attn = layer.self_attn
            q, k, v = self.columns(x, attn.q_proj, attn.k_proj, attn.v_proj)
            q, gate = q.view(batch, seq, attn.num_heads, 2 * attn.head_dim).chunk(2, dim=-1)
            kv_shape = (batch, seq, attn.num_kv_heads, attn.head_dim)
            q, k = attn.apply_rotary_posemb(attn.q_norm(q), attn.k_norm(k.view(kv_shape)), rotary)
            x = flash_attn_func(q, k, v.view(kv_shape), softmax_scale=attn.scaling, causal=True)
            x = x.reshape(batch * seq, -1) * gate.reshape(batch * seq, -1).sigmoid()
            x = self.rows(x, attn.o_proj)
        hidden = hidden + x

        mlp = layer.mlp
        gate, up = self.columns(layer.post_attention_layernorm(hidden), mlp.gate_proj, mlp.up_proj)
        return hidden + self.rows(silu_mul(gate, up), mlp.down_proj)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return _VocabCrossEntropy.apply(
            logits, labels.flatten(), self.rank * logits.size(1), self.group
        )

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


def _gather(x: torch.Tensor, size: int, name: str) -> torch.Tensor:
    output = torch.ops._c10d_functional.all_gather_into_tensor(x.contiguous(), size, name)
    return torch.ops._c10d_functional.wait_tensor(output)


def _scatter_sum(x: torch.Tensor, size: int, name: str) -> torch.Tensor:
    output = torch.ops._c10d_functional.reduce_scatter_tensor(x.contiguous(), "sum", size, name)
    return torch.ops._c10d_functional.wait_tensor(output)


class _ColumnParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, name, overlap, *weights):
        if overlap:
            gathered, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x, [w.t() for w in weights], 0, name
            )
        else:
            gathered = _gather(x, size, name)
            outputs = tuple(F.linear(gathered, w) for w in weights)
        ctx.save_for_backward(gathered, *weights)
        ctx.size, ctx.name, ctx.overlap = size, name, overlap
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        x, *weights = ctx.saved_tensors
        grads = tuple(grad.contiguous() for grad in grads)
        if ctx.overlap and len(weights) == 1:
            dx = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                grads[0], weights[0], "sum", 0, ctx.name
            )
        else:
            contributions = (grad @ weight for grad, weight in zip(grads, weights))
            total = next(contributions)
            for contribution in contributions:
                total = total + contribution
            dx = _scatter_sum(total, ctx.size, ctx.name)
        dweights = tuple(grad.t() @ x for grad in grads)
        return dx, None, None, None, *dweights


class _RowParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, size, name, overlap):
        ctx.save_for_backward(x, weight)
        ctx.size, ctx.name, ctx.overlap = size, name, overlap
        if overlap:
            return torch.ops.symm_mem.fused_matmul_reduce_scatter(x, weight.t(), "sum", 0, name)
        return _scatter_sum(F.linear(x, weight), size, name)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        if ctx.overlap:
            grad, (dx,) = torch.ops.symm_mem.fused_all_gather_matmul(dout, [weight], 0, ctx.name)
        else:
            grad = _gather(dout, ctx.size, ctx.name)
            dx = grad @ weight
        return dx, grad.t() @ x, None, None, None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, name):
        ctx.size, ctx.name = size, name
        return _gather(x, size, name)

    @staticmethod
    def backward(ctx, grad):
        return _scatter_sum(grad, ctx.size, ctx.name), None, None


class _ScatterSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, name):
        ctx.size, ctx.name = size, name
        return _scatter_sum(x, size, name)

    @staticmethod
    def backward(ctx, grad):
        return _gather(grad, ctx.size, ctx.name), None, None


@torch.compile(fullgraph=True, dynamic=False)
def _local_loss_stats(logits, maximum, local_label, owned):
    selected = logits.gather(1, local_label[:, None]).squeeze(1).float()
    sumexp = (logits.float() - maximum[:, None]).exp().sum(1)
    return torch.stack((sumexp, torch.where(owned, selected, 0)))


@torch.compile(fullgraph=True, dynamic=False)
def _local_loss_gradient(logits, lse, local_label, owned, valid, grad):
    probs = (logits.float() - lse[:, None]).exp()
    columns = torch.arange(logits.size(1), device=logits.device)
    target = owned[:, None] & (columns == local_label[:, None])
    result = (probs - target.float()) * valid[:, None] * grad
    return result.to(logits.dtype)


class _VocabCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, offset, group):
        valid = labels != -100
        local_label = labels - offset
        owned = valid & (local_label >= 0) & (local_label < logits.size(1))
        local_label = local_label.clamp(0, logits.size(1) - 1)
        maximum = logits.amax(1).float()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        stats = _local_loss_stats(logits, maximum, local_label, owned)
        dist.all_reduce(stats, group=group)
        sumexp, selected = stats.unbind()
        lse = maximum + sumexp.log()
        ctx.save_for_backward(logits, lse, local_label, owned, valid)
        return torch.where(valid, lse - selected, 0).sum()

    @staticmethod
    def backward(ctx, grad):
        return _local_loss_gradient(*ctx.saved_tensors, grad), None, None, None
