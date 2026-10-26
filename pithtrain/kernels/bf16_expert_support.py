"""
Capability query for the fused BF16 routed experts (the SM90 CuTe-DSL bf16_expert_* kernels, built
on QuACK). Never imports QuACK or cutlass, so it is safe to call on every architecture.
"""

import importlib.metadata
import warnings
from functools import lru_cache

import torch

from pithtrain.kernels.arch import is_sm90_device

__all__ = ["MAX_GROUPS", "QUACK_VERSION", "bf16_experts_supported", "quack_version"]

MAX_GROUPS = 64
"""Most local experts the fused path takes (one int64 bitmask per token in its backward)."""

QUACK_VERSION = "0.6.5"
"""
The quack-kernels version the kernels were validated against (its GEMM classes are not a stable
API); any other version is unsupported.
"""


def quack_version() -> str | None:
    try:
        return importlib.metadata.version("quack-kernels")
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=None)
def supported_on(device_index: int) -> bool:
    if not is_sm90_device(device_index):
        reason = f"needs SM90, device {device_index} is not SM90"
    elif (version := quack_version()) != QUACK_VERSION:
        reason = f"needs quack-kernels=={QUACK_VERSION}, found {version}"
    else:
        return True
    warnings.warn(f"BF16 CuTe experts disabled ({reason}); using F.grouped_mm", stacklevel=3)
    return False


def bf16_experts_supported(num_groups: int | None = None, device: torch.device | None = None) -> bool:
    """
    Whether BF16 routed experts with num_groups local experts (any count when None) can use the
    fused bf16_expert_* kernels on device (the current device by default): needs SM90, quack-kernels
    exactly QUACK_VERSION, and at most MAX_GROUPS (64) local experts. Warns once per device when the
    device or the QuACK version rules it out. Returns False off SM90.
    """
    device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    index = device.index if device.index is not None else torch.cuda.current_device()
    return supported_on(index) and (num_groups is None or num_groups <= MAX_GROUPS)
