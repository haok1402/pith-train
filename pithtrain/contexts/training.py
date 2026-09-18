"""
Training runtime state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from pithtrain.operators import grouped_linear, linear
    from pithtrain.pipeline import DualPipeV

PARAM_DTYPE = torch.bfloat16
"""
Parameter compute and activation dtype.

Pipeline parallelism cuts at layer boundaries, so the tensors crossing between stages are layer
inputs and carry this dtype. DualPipeV allocates its receive buffers with it.
"""

fp8: bool
"""
Whether layers are built for FP8 compute, False for BF16.

This and the two class bindings below are read from module constructors and kernels that hold no
configuration object, so they live here rather than being passed down.
"""

Linear: type[nn.Linear | linear.FP8Linear]
"""
The dense linear class the models construct.
"""

GroupedLinear: type[grouped_linear.GroupedLinear | grouped_linear.FP8GroupedLinear]
"""
The grouped linear class the routed experts construct.
"""

model: DualPipeV
"""
The pipeline engine for this rank, wrapping the two V-shaped chunks it owns.
"""

optimizers: tuple[Optimizer, ...]
"""
The optimizers stepping this model, one per parameter class under Muon.
"""

schedulers: tuple[LRScheduler, ...]
"""
One learning-rate scheduler per optimizer, in the same order.
"""
