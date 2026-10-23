"""
bind_cpu_affinity on a fake two-socket node (sysfs and the CUDA device properties stubbed): which
CPUs each GPU's process is pinned to, and when it leaves the affinity alone.
"""

import builtins
import io
import os
from types import SimpleNamespace

import pytest
import torch

from pithtrain.modules import distributed as D

# 8 GPUs, 4 per socket; 64 CPUs per socket.
GPU_BUS = [0x18, 0x2A, 0x3A, 0x5D, 0x9A, 0xAB, 0xBA, 0xDB]


def sysfs(path: str) -> str:
    for i, bus in enumerate(GPU_BUS):
        if path == "/sys/bus/pci/devices/0000:%02x:00.0/local_cpulist" % bus:
            return "0-63" if i < 4 else "64-127"
    raise FileNotFoundError(path)


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch):
    real_open = builtins.open
    state = SimpleNamespace(affinity=set(range(128)), set_calls=[])

    def fake_open(path, *a, **k):
        if isinstance(path, str) and path.startswith("/sys/"):
            return io.StringIO(sysfs(path) + "\n")
        return real_open(path, *a, **k)

    props = lambda i: SimpleNamespace(pci_domain_id=0, pci_bus_id=GPU_BUS[i], pci_device_id=0)  # noqa: E731
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(state.affinity))
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: state.set_calls.append(sorted(cpus))
    )
    return state


def test_cpulist():
    assert D._cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert D._cpulist("") == []


def test_numa(node):
    D.bind_cpu_affinity(5)
    assert node.set_calls == [list(range(64, 128))]


def test_restricted(node):
    node.affinity = set(range(16))  # already bound by the launcher: kept
    D.bind_cpu_affinity(0)
    assert node.set_calls == []
