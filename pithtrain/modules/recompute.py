"""
Selective recompute of the decoder layer's two RMSNorm outputs in compiled stage 1.

The input_layernorm output feeds q/k/v (and, for DeepSeek, the MLA projections) and the
post_attention_layernorm output feeds the router and the shared experts; each is saved by those
consumers for their weight gradients, 2 x B*S*hidden bf16 per layer and micro-batch kept from
forward to backward. Megatron-LM's ``recompute_modules=["layernorm"]`` drops them and runs the norm
again in backward. :func:`norm_recompute` does the same with a selective-checkpoint policy: only
``aten._fused_rms_norm`` on hidden-size inputs is recomputed (the q/k head norms and DeepSeek's
latent norms have other widths and stay saved); GEMMs, attention and the pithtrain custom ops must
be saved; everything else is left to the partitioner as before. BF16 only (``training.fp8``
unset); the forward is unchanged and the recomputed norms are the same kernels on the same saved
inputs.
"""

from functools import partial

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from pithtrain.contexts import training

_GEMM_OPS = frozenset(
    (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default)
)


def _policy(hidden_size: int, ctx, op, *args, **kwargs) -> CheckpointPolicy:
    x = args[0] if args and isinstance(args[0], torch.Tensor) else None
    if (
        op == torch.ops.aten._fused_rms_norm.default
        and x is not None
        and x.shape[-1] == hidden_size
    ):
        return CheckpointPolicy.PREFER_RECOMPUTE
    name = op.name() if hasattr(op, "name") else str(op)
    if op in _GEMM_OPS or getattr(op, "namespace", None) == "pithtrain" or "flash" in name:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_SAVE


def _context(hidden_size: int):
    return create_selective_checkpoint_contexts(partial(_policy, hidden_size))


def norm_recompute(fn, hidden_size: int, *args):
    """
    ``fn(*args)``, with the hidden-size RMSNorm outputs recomputed in backward instead of saved
    under BF16 (``training.fp8`` unset). Call it inside the compiled region.
    """
    if training.fp8:
        return fn(*args)
    return checkpoint(fn, *args, use_reentrant=False, context_fn=partial(_context, hidden_size))
