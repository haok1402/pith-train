"""
Device capability of this process, the one place in pithtrain that probes it (SM90 and SM100).
"""

import functools

import torch

ARCH_MAJOR, _ = torch.cuda.get_device_capability()
"""Compute-capability major of the current device at import: 9 on SM90, 10 on SM100."""

NATIVE_E8M0 = ARCH_MAJOR >= 10
"""Whether the E8M0 FP8 scale has a native PTX conversion (SM100+); SM90 emulates it."""


def is_sm90_device(device_index: int) -> bool:
    """Whether device device_index is SM90. Uncached, so a test can patch the probe."""
    return torch.cuda.get_device_capability(device_index)[0] == 9


@functools.cache
def sm_count(device: torch.device | int | None = None) -> int:
    """Streaming-multiprocessor count of device (the current device by default)."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count
