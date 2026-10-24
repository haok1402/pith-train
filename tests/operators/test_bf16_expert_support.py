"""
Selection of the BF16 CuTe experts (pithtrain.kernels.bf16_expert_support.supported, composed with
training.fp8 by the models): the CuTe path only with the exact quack-kernels version it was
validated against, on SM90, in BF16 training; otherwise the unfused path, with at most one warning
per process. The check never imports quack.
"""

import subprocess
import sys
import warnings

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.kernels import bf16_expert_support


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(training, "fp8", False, raising=False)
    bf16_expert_support._supported_on.cache_clear()
    yield
    bf16_expert_support._supported_on.cache_clear()


def sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(not sm90(), reason="needs SM90")
def test_enabled_with_validated_version():
    if bf16_expert_support.quack_version() != bf16_expert_support.QUACK_VERSION:
        pytest.skip("venv quack-kernels is not the validated version")
    assert bf16_expert_support.supported()


@pytest.mark.parametrize("version", ["0.6.4", "0.7.0", None])
def test_other_quack_version_falls_back_once(monkeypatch, version):
    monkeypatch.setattr(bf16_expert_support, "quack_version", lambda: version)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not bf16_expert_support.supported()
        assert not bf16_expert_support.supported()
    assert len(w) == 1 and "quack-kernels" in str(w[0].message)


def test_non_sm90_falls_back(monkeypatch):
    monkeypatch.setattr(bf16_expert_support, "quack_version", lambda: bf16_expert_support.QUACK_VERSION)  # fmt: skip
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not bf16_expert_support.supported()
    assert len(w) == 1 and "SM90" in str(w[0].message)


def test_too_many_groups(monkeypatch):
    monkeypatch.setattr(bf16_expert_support, "quack_version", lambda: bf16_expert_support.QUACK_VERSION)  # fmt: skip
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert bf16_expert_support.supported(bf16_expert_support.MAX_GROUPS)
    assert not bf16_expert_support.supported(bf16_expert_support.MAX_GROUPS + 1)


def test_fp8_disables(monkeypatch):
    """FP8 training never takes the BF16 CuTe experts: the models check training.fp8 first."""
    from pithtrain.models.qwen3_moe import Qwen3MoeExperts

    monkeypatch.setattr(training, "fp8", True, raising=False)
    experts = Qwen3MoeExperts.__new__(Qwen3MoeExperts)
    assert not experts.uses_bf16_cute()


def test_guard_does_not_import_quack():
    code = (
        "import sys\n"
        "from pithtrain.contexts import training\n"
        "training.fp8 = True\n"
        "from pithtrain.kernels import bf16_expert_support\n"
        "assert not (not training.fp8 and bf16_expert_support.supported())\n"
        "bf16_expert_support.supported()\n"
        "assert 'quack' not in sys.modules, 'the guard imported quack'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
