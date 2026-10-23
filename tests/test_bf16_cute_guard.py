"""
Selection of the BF16 CuTe experts (pithtrain.operators.bf16_cute.enabled): the CuTe path only with
the exact quack-kernels version it was validated against, on SM90, in BF16 training; otherwise
the unfused path's path, with at most one warning per process. The check never imports quack.
"""

import subprocess
import sys
import warnings

import pytest
import torch

from pithtrain.contexts import training
from pithtrain.operators import bf16_cute


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(training, "fp8", False, raising=False)
    bf16_cute._supported.cache_clear()
    yield
    bf16_cute._supported.cache_clear()


def sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(not sm90(), reason="needs SM90")
def test_enabled_with_validated_version():
    if bf16_cute._quack_version() != bf16_cute.QUACK_VERSION:
        pytest.skip("venv quack-kernels is not the validated version")
    assert bf16_cute.enabled()


@pytest.mark.parametrize("version", ["0.6.4", "0.7.0", None])
def test_other_quack_version_falls_back_once(monkeypatch, version):
    monkeypatch.setattr(bf16_cute, "_quack_version", lambda: version)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not bf16_cute.enabled()
        assert not bf16_cute.enabled()
    assert len(w) == 1 and "quack-kernels" in str(w[0].message)


def test_non_sm90_falls_back(monkeypatch):
    monkeypatch.setattr(bf16_cute, "_quack_version", lambda: bf16_cute.QUACK_VERSION)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert not bf16_cute.enabled()
    assert len(w) == 1 and "SM90" in str(w[0].message)


def test_fp8_disables(monkeypatch):
    monkeypatch.setattr(training, "fp8", True, raising=False)
    assert not bf16_cute.enabled()


def test_guard_does_not_import_quack():
    code = (
        "import sys\n"
        "from pithtrain.contexts import training\n"
        "training.fp8 = True\n"
        "from pithtrain.operators import bf16_cute\n"
        "assert not bf16_cute.enabled()\n"
        "assert 'quack' not in sys.modules, 'the guard imported quack'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
