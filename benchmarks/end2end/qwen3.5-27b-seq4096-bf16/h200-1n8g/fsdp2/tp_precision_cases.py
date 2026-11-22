"""Counterfactual projection and loss arithmetic for the reduced TP audit."""

from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tp_delta_trace import DECODER_TRACE_CASES, capture

from pithtrain.kernels import cross_entropy_fwd_bwd, stage_compile_options
from pithtrain.models.qwen35_moe import Qwen35MoeRMSNorm, Qwen35MoeRMSNormGated
from pithtrain.modules import dense_tp
from pithtrain.operators.causal_conv1d import causal_conv1d
from pithtrain.operators.parameter_rounding import stock_bf16_value

PROJECTION_CASES = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "lm_head",
)


@torch.compile(fullgraph=True, options=stage_compile_options() or None)
def tp_stage1(layer, hidden, rotary):
    hidden = layer.tensor_parallel.attention(layer, hidden, rotary)
    return layer.post_attention_layernorm(hidden), hidden


def tp_staged(layer, hidden, rotary):
    x, residual = tp_stage1(layer, hidden, rotary)
    x = layer.tensor_parallel.mlp(layer, x)
    return layer.forward_stage5(x, None, None, residual)


COMPILED_DECAY_CASES = (
    "stock-adjoint-compiled",
    "stock-loss-compiled",
    "fp32-decay-grad-compiled",
    "fp32-decay-master-compiled",
    "fp32-gate-compiled",
    "fp32-decay-combined-compiled",
    "separate-dgrad-compiled",
    "mlp-dgrad-compiled",
    "gqa-dgrad-compiled",
    "qk-round-silu-compiled",
    "qk-fp32-silu-compiled",
    "qk-fp32-silu-rounded-loss-compiled",
    "delta-stock-checkpoint-compiled",
    "qkv-out32-compiled",
    "gqa-k-out32-compiled",
    "mlp-gate-out32-compiled",
    "gqa-k-mlp-gate-out32-compiled",
    "residual-round-compiled",
    "residual-round-stock-loss-compiled",
    "residual-round-out32-compiled",
    "trace-decoder-compiled",
    "residual-round-trace-compiled",
    "residual-fp32-norm-compiled",
    "fp32-residual-stream-compiled",
    "fp32-residual-stream-stock-loss-compiled",
    "fp32-residual-stream-out32-compiled",
    "fp32-residual-stream-row-dgrad-compiled",
    "fp32-residual-stream-row-out32-compiled",
    "fp32-residual-stream-all-row-out32-compiled",
    "fp32-residual-stream-delta-adjoint-compiled",
    "fp32-residual-stream-row-adjoints-compiled",
    "fp32-residual-stream-bf16-forward-compiled",
    "fp32-residual-stream-bf16-forward-row-adjoints-compiled",
    "async-fp32-residual-stream-bf16-forward-compiled",
    "async-fp32-residual-stream-bf16-forward-mlp-gate-out32-compiled",
)

MODULE_TP_CASES = ("module-tp-compiled", "module-tp-staged")


class ModuleTP(dense_tp.DenseTP):
    """Run the stock attention and MLP modules with TP at projection boundaries."""

    def __init__(self, mesh):
        super().__init__(mesh, async_tp=False)

    def shard_model(self, model):
        super().shard_model(model)
        delta_columns = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
        gqa_columns = ("q_proj", "k_proj", "v_proj")
        for layer in model.layers:
            attn = layer.linear_attn if layer.is_linear else layer.self_attn
            for name in delta_columns if layer.is_linear else gqa_columns:
                module = getattr(attn, name)
                module.forward = partial(self.column, module)
            row = attn.out_proj if layer.is_linear else attn.o_proj
            row.forward = partial(self.row, row)
            for module in (layer.mlp.gate_proj, layer.mlp.up_proj):
                module.forward = partial(self.column, module)
            down = layer.mlp.down_proj
            down.forward = partial(self.row, down)

    def column(self, module, x, quantized_input=None):
        result = ModuleColumn.apply(x, module.weight, getattr(module, "main_grad", None))
        return result if module.bias is None else result + module.bias

    def row(self, module, x, quantized_input=None):
        return self.rows(x.flatten(0, -2), module)

    def token_mixing(self, layer, hidden, rotary):
        seq = rotary[0].size(1)
        batch = hidden.size(0) * self.size // seq
        x = self.attention_input(layer, hidden)
        x = ModuleGather.apply(x, self.size, self.name).view(batch, seq, -1)
        if layer.is_linear:
            return layer.linear_attn(x)
        return layer.self_attn(x, rotary)

    def mlp(self, layer, x):
        return layer.mlp(ModuleGather.apply(x, self.size, self.name))


class ModuleGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, name):
        ctx.size, ctx.name, ctx.dtype = size, name, x.dtype
        return dense_tp._gather(x, size, name).float()

    @staticmethod
    def backward(ctx, grad):
        return dense_tp._scatter_sum(grad, ctx.size, ctx.name).to(ctx.dtype), None, None


class ModuleColumn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, main_grad):
        operand = x.to(weight.dtype).flatten(0, -2)
        ctx.save_for_backward(operand, weight)
        ctx.input_shape, ctx.main_grad = x.shape, main_grad
        return F.linear(operand, weight).view(*x.shape[:-1], weight.size(0))

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.reshape(-1, grad.size(-1))
        dx = dense_tp._mm_fp32(grad, weight).view(ctx.input_shape)
        dw = dense_tp._weight_grad(ctx.main_grad, grad, x)
        return dx, dw, None


class DecayProbeTP(dense_tp.DenseTP):
    def __init__(self, mesh, case):
        super().__init__(mesh, async_tp=case.startswith("async-"))
        case = case.removeprefix("async-")
        self.case = case
        self.master = case in ("fp32-decay-master-compiled", "fp32-decay-combined-compiled")
        self.fp32_residual = case.startswith("fp32-residual-stream-")
        self.round_residual = case.startswith("fp32-residual-stream-bf16-forward-")

    def embed(self, module, tokens):
        hidden = super().embed(module, tokens)
        return hidden.float() if self.fp32_residual else hidden

    def attention_input(self, layer, hidden):
        x = super().attention_input(layer, hidden)
        return x.to(torch.bfloat16) if self.fp32_residual else x

    def head(self, module, x, labels=None):
        return super().head(module, x.to(torch.bfloat16) if self.fp32_residual else x, labels)

    def shard_model(self, model):
        super().shard_model(model)
        for layer in model.layers:
            if not layer.is_linear:
                continue
            delta = layer.linear_attn
            delta.out_proj.audit_delta_row = True
            if self.master:
                delta.audit_master_A_log = delta.A_log.detach().clone()
                delta.audit_master_dt_bias = delta.dt_bias.detach().clone()
            if self.case.startswith("qk-"):
                round_silu = self.case == "qk-round-silu-compiled"
                delta.conv_qkv = partial(probe_conv_qkv, delta, round_silu)

    def delta_decay(self, delta, a):
        if self.master:
            decay = ForwardMaster.apply(delta.A_log, delta.audit_master_A_log)
            bias = ForwardMaster.apply(delta.dt_bias, delta.audit_master_dt_bias)
        elif self.case == "fp32-decay-grad-compiled":
            decay, bias = map(FP32WeightGrad.apply, (delta.A_log, delta.dt_bias))
        else:
            return super().delta_decay(delta, a)
        return -decay.exp() * F.softplus(a.float() + bias)

    def attention(self, layer, hidden, rotary):
        residual = super().attention(layer, hidden, rotary)
        if self.round_residual:
            return stock_bf16_value(residual)
        if self.case.startswith("residual-round-"):
            return materialize(residual)
        return residual

    def decoder(self, layer, hidden, rotary):
        if self.fp32_residual:
            residual = self.attention(layer, hidden, rotary)
            mlp_input = layer.post_attention_layernorm(residual).to(torch.bfloat16)
            output = residual + self.mlp(layer, mlp_input)
            return stock_bf16_value(output) if self.round_residual else output
        if self.case != "residual-fp32-norm-compiled":
            return super().decoder(layer, hidden, rotary)
        contribution = self.token_mixing(layer, hidden, rotary)
        residual = hidden.float() + contribution.float()
        mlp_input = layer.post_attention_layernorm(residual).to(hidden.dtype)
        return residual.to(hidden.dtype) + self.mlp(layer, mlp_input)

    def delta_attention(self, delta, x, batch, seq):
        output = super().delta_attention(delta, x, batch, seq)
        if self.case in DECODER_TRACE_CASES:
            return capture(output, self.case, "gdn_row", self.rank, self.size)
        return output

    def mlp(self, layer, x):
        if self.case not in DECODER_TRACE_CASES or not layer.is_linear:
            return super().mlp(layer, x)
        x = capture(x, self.case, "mlp_input", self.rank, self.size)
        return capture(super().mlp(layer, x), self.case, "mlp_output", self.rank, self.size)

    def delta_project(self, delta, x):
        if self.case not in ("fp32-gate-compiled", "fp32-decay-combined-compiled"):
            return super().delta_project(delta, x)
        return ProbeColumns.apply(
            x,
            self.size,
            self.name,
            (False, False, False, True),
            False,
            True,
            False,
            (delta.in_proj_qkv.main_grad, delta.in_proj_z.main_grad, None, None),
            delta.in_proj_qkv.weight,
            delta.in_proj_z.weight,
            delta.in_proj_b.weight,
            delta.in_proj_a.weight,
        )

    def columns(self, x, *modules):
        count = len(modules)
        selected = (False,) * count
        if count == 4 and self.case == "qkv-out32-compiled":
            selected = (True, False, False, False)
        if count == 3 and self.case in (
            "gqa-k-out32-compiled",
            "gqa-k-mlp-gate-out32-compiled",
            "residual-round-out32-compiled",
            "fp32-residual-stream-out32-compiled",
        ):
            selected = (False, True, False)
        if count == 2 and self.case in (
            "mlp-gate-out32-compiled",
            "gqa-k-mlp-gate-out32-compiled",
            "residual-round-out32-compiled",
            "fp32-residual-stream-out32-compiled",
            "fp32-residual-stream-bf16-forward-mlp-gate-out32-compiled",
        ):
            selected = (True, False)
        split = (
            self.case == "separate-dgrad-compiled"
            and count > 1
            or self.case == "mlp-dgrad-compiled"
            and count == 2
            or self.case == "gqa-dgrad-compiled"
            and count == 3
        )
        if not split and not any(selected):
            return super().columns(x, *modules)
        return ProbeColumns.apply(
            x,
            self.size,
            self.name,
            selected,
            split,
            False,
            True,
            tuple(getattr(module, "main_grad", None) for module in modules),
            *(module.weight for module in modules),
        )

    def rows(self, x, module):
        if self.case in (
            "fp32-residual-stream-row-adjoints-compiled",
            "fp32-residual-stream-bf16-forward-row-adjoints-compiled",
        ) or (
            self.case == "fp32-residual-stream-delta-adjoint-compiled"
            and getattr(module, "audit_delta_row", False)
        ):
            return FP32RowAdjoint.apply(x, module.weight, module.main_grad, self.size, self.name)
        use_fp32_dgrad = (
            self.case == "fp32-residual-stream-all-row-out32-compiled"
            or self.case
            in (
                "fp32-residual-stream-row-dgrad-compiled",
                "fp32-residual-stream-row-out32-compiled",
            )
            and getattr(module, "audit_delta_row", False)
        )
        if not use_fp32_dgrad:
            return super().rows(x, module)
        return ProbeRow.apply(
            x,
            module.weight,
            module.main_grad,
            self.size,
            self.name,
            self.rank,
            False,
            True,
            self.case.endswith("-out32-compiled"),
        )

    def loss(self, logits, labels):
        if self.case in (
            "stock-loss-compiled",
            "qk-fp32-silu-rounded-loss-compiled",
            "residual-round-stock-loss-compiled",
            "fp32-residual-stream-stock-loss-compiled",
        ):
            return StockLoss.apply(logits, labels.flatten(), self.size, self.name, self.rank)
        return super().loss(logits, labels)


class ForwardMaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, master):
        return master.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


@torch.library.custom_op("pithtrain::tp_audit_materialize", mutates_args=())
def materialize(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@materialize.register_fake
def fake_materialize(x):
    return torch.empty_like(x)


def materialize_backward(ctx, grad):
    return grad


materialize.register_autograd(materialize_backward)


def probe_conv_qkv(delta, round_silu, qkv, offset):
    conv = causal_conv1d(qkv, delta.conv1d.weight, offset)
    if round_silu:
        return delta.split_qkv(materialize(F.silu(conv)))

    batch, seq, _ = conv.shape
    q, k, v = F.silu(conv.float()).split([delta.key_dim, delta.key_dim, delta.value_dim], dim=-1)
    q, k = (tensor.reshape(batch, seq, -1, delta.head_k_dim) for tensor in (q, k))
    q, k = (
        (tensor * torch.rsqrt(tensor.square().sum(-1, keepdim=True) + 1e-6)).to(conv.dtype)
        for tensor in (q, k)
    )
    return q, k, v.reshape(batch, seq, -1, delta.head_v_dim).to(conv.dtype).contiguous()


class PrecisionTP(dense_tp.DenseTP):
    def __init__(self, mesh, case, *, async_tp=False):
        super().__init__(mesh, async_tp=async_tp)
        self.case = case.removesuffix("-staged").removesuffix("-compiled")

    def shard_model(self, model):
        super().shard_model(model)
        for name, module in model.named_modules():
            module.audit_name = name
            if spec := self.shards.get(name + ".weight"):
                module.audit_spans = tuple(len(span) for span in spec.spans)
        if self.case == "fp32-norm-grads":
            for module in model.modules():
                if isinstance(module, (Qwen35MoeRMSNorm, Qwen35MoeRMSNormGated)):
                    module.forward = partial(norm_with_fp32_grad, module)

    def selects(self, module):
        return module.audit_name.endswith(self.case)

    def rows(self, x, module):
        stock_row = self.case in ("stock-row", "stock-linear-loss")
        fp32_dgrad = (
            self.case == "row-dgrad-fp32"
            or self.case.endswith("-dgrad-fp32")
            and module.audit_name.endswith(self.case.removesuffix("-dgrad-fp32"))
        )
        if self.selects(module) or stock_row or fp32_dgrad:
            return ProbeRow.apply(
                x,
                module.weight,
                module.main_grad,
                self.size,
                self.name,
                self.rank,
                stock_row,
                fp32_dgrad,
                False,
            )
        return super().rows(x, module)

    def columns(self, x, *modules):
        if self.case in ("stock-column-forward", "stock-column-backward", "stock-linear-loss"):
            return tuple(
                ReferenceColumn.apply(
                    x,
                    module.weight,
                    getattr(module, "main_grad", None),
                    self.size,
                    self.name,
                    self.rank,
                    module.audit_spans,
                    self.case != "stock-column-backward",
                    self.case != "stock-column-forward",
                )
                for module in modules
            )
        fp32_decay = self.case == "fp32-decay-projection"
        selected = tuple(
            self.selects(module)
            or fp32_decay
            and module.audit_name.endswith("linear_attn.in_proj_a")
            for module in modules
        )
        split_dgrad = self.case == "separate-dgrad"
        if not any(selected) and not split_dgrad:
            return super().columns(x, *modules)
        return ProbeColumns.apply(
            x,
            self.size,
            self.name,
            selected,
            split_dgrad,
            fp32_decay,
            False,
            tuple(getattr(module, "main_grad", None) for module in modules),
            *(module.weight for module in modules),
        )

    def loss(self, logits, labels):
        if self.case in ("stock-loss", "stock-linear-loss"):
            return StockLoss.apply(logits, labels.flatten(), self.size, self.name, self.rank)
        return super().loss(logits, labels)

    def delta_decay(self, delta, a):
        if self.case == "fp32-decay-grads":
            decay, bias = map(FP32WeightGrad.apply, (delta.A_log, delta.dt_bias))
            return -decay.exp() * F.softplus(a.float() + bias)
        return super().delta_decay(delta, a)


class FP32WeightGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return weight.float()

    @staticmethod
    def backward(ctx, grad):
        return grad


def norm_with_fp32_grad(module, x, gate=None):
    value = x.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + module.eps)
    weight = FP32WeightGrad.apply(module.weight)
    if gate is None:
        return (normalized * (1 + weight)).to(x.dtype)
    return (normalized * weight * F.silu(gate.float())).to(x.dtype)


class ProbeRow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, main_grad, size, name, rank, stock_row, fp32_dgrad, out32_dgrad):
        ctx.save_for_backward(x, weight)
        ctx.main_grad, ctx.size, ctx.name = main_grad, size, name
        ctx.fp32_dgrad = fp32_dgrad
        ctx.out32_dgrad = out32_dgrad
        if stock_row:
            full_x = dense_tp._gather(x.t().contiguous(), size, name).t()
            full_weight = dense_tp._gather(weight.t().contiguous(), size, name).t()
            return F.linear(full_x.contiguous(), full_weight.contiguous()).chunk(size)[rank].clone()
        if fp32_dgrad:
            return dense_tp._mm_scatter(x, weight.t(), size, name, False)
        partial = x.float() @ weight.float().t()
        return dense_tp._scatter_sum(partial, size, name).to(x.dtype)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        gathered = dense_tp._gather(dout, ctx.size, ctx.name)
        dw = dense_tp._weight_grad(ctx.main_grad, gathered, x)
        if ctx.out32_dgrad:
            dx = dense_tp._mm_fp32(gathered, weight).to(x.dtype)
        elif ctx.fp32_dgrad:
            dx = (gathered.float() @ weight.float()).to(x.dtype)
        else:
            dx = gathered @ weight
        return dx, dw, None, None, None, None, None, None, None


class FP32RowAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, main_grad, size, name):
        ctx.save_for_backward(x, weight)
        ctx.main_grad, ctx.size, ctx.name = main_grad, size, name
        # The sequence reduction still produces the stock BF16 activation.
        # Carry its adjoint in FP32 through the residual stream.
        return dense_tp._mm_scatter(x, weight.t(), size, name, False).float()

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        grad = dense_tp._gather(dout, ctx.size, ctx.name)
        dx = (grad @ weight.float()).to(x.dtype)
        ctx.main_grad.addmm_(grad.t(), x.float())
        return dx, None, None, None, None


class ProbeColumns(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, name, selected, split_dgrad, fp32_decay, out32, main_grads, *weights):
        gathered = dense_tp._gather(x, size, name)
        ctx.save_for_backward(gathered, *weights)
        ctx.size, ctx.name, ctx.selected = size, name, selected
        ctx.split_dgrad, ctx.main_grads = split_dgrad, main_grads
        ctx.out32 = out32
        return tuple(
            (
                dense_tp._mm_fp32(gathered, weight.t())
                if out32
                else gathered.float() @ weight.float().t()
            ).to(torch.float32 if fp32_decay else x.dtype)
            if precise
            else F.linear(gathered, weight)
            for precise, weight in zip(selected, weights, strict=True)
        )

    @staticmethod
    def backward(ctx, *grads):
        x, *weights = ctx.saved_tensors
        grads = tuple(grad.contiguous() for grad in grads)
        partials = [
            grad.float() @ weight.float()
            if precise and not ctx.out32
            else dense_tp._mm_fp32(grad, weight)
            for precise, grad, weight in zip(ctx.selected, grads, weights, strict=True)
        ]
        if ctx.split_dgrad:
            # Each complete projection rounds where stock BF16 Linear rounds its dgrad.
            # Sequence collectives still carry FP32 partials.
            pieces = [
                dense_tp._scatter_sum(partial, ctx.size, ctx.name).to(x.dtype)
                for partial in partials
            ]
        else:
            total = sum(partials[1:], partials[0])
            pieces = [dense_tp._scatter_sum(total, ctx.size, ctx.name).to(x.dtype)]
        dx = sum(pieces[-2::-1], pieces[-1])
        dw = tuple(
            grad.t() @ x.float()
            if grad.dtype == torch.float32
            else dense_tp._weight_grad(mg, grad, x)
            for mg, grad in zip(ctx.main_grads, grads, strict=True)
        )
        return dx, None, None, None, None, None, None, None, *dw


def gather_column_rows(x, spans, size, name):
    pieces = dense_tp._gather(x.contiguous(), size, name).unflatten(0, (size, x.size(0)))
    return torch.cat([part.flatten(0, 1) for part in pieces.split(spans, dim=1)])


class ReferenceColumn(torch.autograd.Function):
    """Diagnostic only. Each projection uses the canonical full GEMM as a control."""

    @staticmethod
    def forward(ctx, x, weight, main_grad, size, name, rank, spans, stock_fwd, stock_bwd):
        gathered = dense_tp._gather(x, size, name)
        ctx.save_for_backward(gathered, weight)
        ctx.main_grad, ctx.size, ctx.name, ctx.rank = main_grad, size, name, rank
        ctx.spans, ctx.stock_bwd = spans, stock_bwd
        if stock_fwd:
            full_weight = gather_column_rows(weight, spans, size, name)
            full = F.linear(gathered, full_weight)
            return torch.cat(
                [
                    segment.chunk(size, dim=1)[rank]
                    for segment in full.split(tuple(span * size for span in spans), dim=1)
                ],
                dim=1,
            )
        return F.linear(gathered, weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        grad = grad.contiguous()
        dw = dense_tp._weight_grad(ctx.main_grad, grad, x)
        if ctx.stock_bwd:
            full_grad = gather_column_rows(grad.t(), ctx.spans, ctx.size, ctx.name).t()
            full_weight = gather_column_rows(weight, ctx.spans, ctx.size, ctx.name)
            dx = (full_grad @ full_weight).chunk(ctx.size)[ctx.rank].clone()
        else:
            dx = dense_tp._mm_scatter(grad, weight, ctx.size, ctx.name, False)
        return dx, dw, None, None, None, None, None, None, None


class StockLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, size, name, rank):
        full_logits = dense_tp._gather(logits.t().contiguous(), size, name).t().contiguous()
        loss, full_grad = cross_entropy_fwd_bwd(full_logits, labels, -100)
        count = (labels != -100).sum()
        ctx.save_for_backward(full_grad.chunk(size, dim=-1)[rank].contiguous(), count)
        return loss * count

    @staticmethod
    def backward(ctx, grad):
        local_grad, count = ctx.saved_tensors
        return local_grad * (grad * count).to(local_grad.dtype), None, None, None, None


@torch.no_grad()
def check_row_partials(layout, device):
    """Use a cancellation fixture that distinguishes FP32 GEMM output from BF16 output."""
    rank, size, name = layout.rank, layout.size, layout.name
    # Rank 0's exact output is 1 + 2^-8, which would round to 1 in BF16.
    # Rank 1 contributes -1, exposing the lost bit when reduction is performed after rounding.
    x = torch.zeros(2 * size, 32, device=device, dtype=torch.bfloat16)
    weight = torch.zeros(32, 32, device=device, dtype=torch.bfloat16)
    x[:, 0] = 1
    weight[0, :] = 1 if rank == 0 else -1 if rank == 1 else 0
    if rank == 0:
        x[:, 1] = 1
        weight[1, :] = 2**-8
    for overlap in (False, True):
        actual = dense_tp._mm_scatter(x, weight, size, name, overlap)
        assert torch.equal(actual, torch.full_like(actual, 2**-8)), (rank, overlap, actual)
        print(
            f"row_probe rank={dist.get_rank()} async={overlap} fp32_cancellation=passed", flush=True
        )


def check_column_partials(layout, device):
    rank, size = layout.rank, layout.size
    first = torch.zeros(32, 32, dtype=torch.bfloat16, device=device)
    second = torch.zeros_like(first)
    first[0] = 1 if rank == 0 else -1 if rank == 1 else 0
    if rank == 0:
        second[0] = 2**-8
    grad = torch.zeros(2 * size, 32, dtype=torch.bfloat16, device=device)
    grad[:, 0] = 1
    for overlap in (False, True):
        x = torch.zeros_like(grad[:2], requires_grad=True)
        outputs = dense_tp._ColumnParallel.apply(
            x, size, layout.name, overlap, False, (None, None), first, second
        )
        torch.autograd.backward(outputs, (grad, grad))
        assert torch.equal(x.grad, torch.full_like(x, 2**-8)), (rank, overlap, x.grad)
        print(
            f"column_probe rank={dist.get_rank()} async={overlap} fp32_cancellation=passed",
            flush=True,
        )
