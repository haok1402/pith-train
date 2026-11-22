"""Counterfactual projection and loss arithmetic for the reduced TP audit."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from pithtrain.kernels import cross_entropy_fwd_bwd
from pithtrain.modules import dense_tp

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


class PrecisionTP(dense_tp.DenseTP):
    def __init__(self, mesh, case, *, async_tp=False):
        super().__init__(mesh, async_tp=async_tp)
        self.case = case
        self.names = {}

    def shard_model(self, model):
        super().shard_model(model)
        self.names = {id(module): name for name, module in model.named_modules()}

    def selects(self, module):
        return self.names[id(module)].endswith(self.case)

    def rows(self, x, module):
        if self.selects(module) or self.case == "stock-row":
            return ProbeRow.apply(
                x,
                module.weight,
                module.main_grad,
                self.size,
                self.name,
                self.rank,
                self.case == "stock-row",
            )
        return super().rows(x, module)

    def columns(self, x, *modules):
        selected = tuple(self.selects(module) for module in modules)
        split_dgrad = self.case == "separate-dgrad"
        if not any(selected) and not split_dgrad:
            return super().columns(x, *modules)
        return ProbeColumns.apply(
            x,
            self.size,
            self.name,
            selected,
            split_dgrad,
            tuple(getattr(module, "main_grad", None) for module in modules),
            *(module.weight for module in modules),
        )

    def loss(self, logits, labels):
        if self.case == "stock-loss":
            return StockLoss.apply(logits, labels.flatten(), self.size, self.name, self.rank)
        return super().loss(logits, labels)


class ProbeRow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, main_grad, size, name, rank, stock_row):
        ctx.save_for_backward(x, weight)
        ctx.main_grad, ctx.size, ctx.name = main_grad, size, name
        if stock_row:
            full_x = dense_tp._gather(x.t().contiguous(), size, name).t()
            full_weight = dense_tp._gather(weight.t().contiguous(), size, name).t()
            return F.linear(full_x.contiguous(), full_weight.contiguous()).chunk(size)[rank].clone()
        partial = x.float() @ weight.float().t()
        return dense_tp._scatter_sum(partial, size, name).to(x.dtype)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensors
        gathered = dense_tp._gather(dout, ctx.size, ctx.name)
        dw = dense_tp._weight_grad(ctx.main_grad, gathered, x)
        return gathered @ weight, dw, None, None, None, None, None


class ProbeColumns(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, name, selected, split_dgrad, main_grads, *weights):
        gathered = dense_tp._gather(x, size, name)
        ctx.save_for_backward(gathered, *weights)
        ctx.size, ctx.name, ctx.selected = size, name, selected
        ctx.split_dgrad, ctx.main_grads = split_dgrad, main_grads
        return tuple(
            (gathered.float() @ weight.float().t()).to(x.dtype)
            if precise
            else F.linear(gathered, weight)
            for precise, weight in zip(selected, weights, strict=True)
        )

    @staticmethod
    def backward(ctx, *grads):
        x, *weights = ctx.saved_tensors
        grads = tuple(grad.contiguous() for grad in grads)
        partials = [
            grad.float() @ weight.float() if precise else dense_tp._mm_fp32(grad, weight)
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
            dense_tp._weight_grad(mg, grad, x)
            for mg, grad in zip(ctx.main_grads, grads, strict=True)
        )
        return dx, None, None, None, None, None, *dw


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
    weight[:, 0] = 1 if rank == 0 else -1 if rank == 1 else 0
    if rank == 0:
        x[:, 1] = 1
        weight[:, 1] = 2**-8
    for overlap in (False, True):
        actual = dense_tp._mm_scatter(x, weight, size, name, overlap)
        assert torch.equal(actual, torch.full_like(actual, 2**-8)), (rank, overlap, actual)
        print(
            f"row_probe rank={dist.get_rank()} async={overlap} fp32_cancellation=passed", flush=True
        )
