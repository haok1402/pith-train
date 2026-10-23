"""The all-to-all functions' grad-mode decorator behaves as torch.no_grad() does."""

import pytest
import torch

from pithtrain.operators.all_to_all import _no_grad


@_no_grad
def probe(fail=False):
    if fail:
        raise ValueError
    return torch.is_grad_enabled(), torch.ones(2, requires_grad=True) * 2


@pytest.mark.parametrize("enabled", [True, False])
def test_no_grad_inside_and_restored(enabled):
    with torch.set_grad_enabled(enabled):
        inside, out = probe()
        assert not inside and not out.requires_grad
        assert torch.is_grad_enabled() == enabled
        with pytest.raises(ValueError):
            probe(fail=True)
        assert torch.is_grad_enabled() == enabled


def test_keeps_metadata():
    assert probe.__name__ == "probe"
