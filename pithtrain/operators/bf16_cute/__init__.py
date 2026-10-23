"""
BF16 routed experts on SM90 as fused CuTe-DSL grouped GEMMs built on QuACK's SM90 GEMM classes.

The kernels live in :mod:`.gemm` (GEMMs and their epilogues), :mod:`.index` (the index-only
grouping of the received tokens and the gather-sum of their gradients) and :mod:`.experts` (the
autograd node). QuACK is imported only by those modules, and only once :func:`enabled` said yes:
its import patches CuTe-DSL globally, and its GEMM classes are not a stable API, so the path is
taken only with the exact version the kernels were validated against.

BF16 training takes this path wherever it can run (SM90, the validated QuACK version); elsewhere it
keeps ``F.grouped_mm`` with the separate scatter / gather / SwiGLU kernels.
"""

import importlib.metadata
import warnings
from functools import lru_cache

import torch

from pithtrain.contexts import training

MAX_GROUPS = 64
"""Most local experts the fused path takes (one int64 bitmask per token in its backward)."""

QUACK_VERSION = "0.6.5"
"""quack-kernels version the kernels were validated against; any other version keeps torch."""


def _quack_version() -> str | None:
    try:
        return importlib.metadata.version("quack-kernels")
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=None)
def _supported(device_index: int) -> bool:
    """Whether this process can run the CuTe experts, warning once when it cannot."""
    if torch.cuda.get_device_capability(device_index)[0] != 9:
        reason = f"needs SM90, device {device_index} is SM{torch.cuda.get_device_capability(device_index)[0]}0"  # fmt: skip
    elif (version := _quack_version()) != QUACK_VERSION:
        reason = f"needs quack-kernels=={QUACK_VERSION}, found {version}"
    else:
        return True
    warnings.warn(f"BF16 CuTe experts disabled ({reason}); using F.grouped_mm", stacklevel=3)
    return False


def enabled(device: torch.device | None = None) -> bool:
    """Whether BF16 experts take the CuTe path in this process."""
    if training.fp8:
        return False
    device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    return _supported(device.index if device.index is not None else torch.cuda.current_device())
