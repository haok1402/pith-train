"""
gpt-oss edge-light pipeline partition (pithtrain/models/gpt_oss.py gpt_oss_layer_partition).
Pure function; no GPU needed.

    pytest tests/test_gpt_oss_partition.py -v
"""

import pytest

from pithtrain.models.gpt_oss import gpt_oss_layer_partition
from pithtrain.pipeline.dualpipev import layer_partition


def counts(num_layers: int, stage_count: int, edge_light: bool) -> list[int]:
    ranges = [gpt_oss_layer_partition(num_layers, stage_count, s, edge_light) for s in range(stage_count)]  # fmt: skip
    # Contiguous, in order, covering every layer once: global ids are unchanged.
    assert [i for r in ranges for i in r] == list(range(num_layers))
    return [len(r) for r in ranges]


def test_20b_pp2_edge_light():
    # Rank 0 holds stages 0 (embed) and 3 (lm_head): fewer layers there, not more.
    assert counts(24, 4, edge_light=True) == [5, 7, 7, 5]


def test_off_keeps_even():
    # BF16 runs (edge_light=False) keep r5-base's partition.
    assert counts(24, 4, edge_light=False) == [6, 6, 6, 6]
    assert counts(36, 4, edge_light=False) == [9, 9, 9, 9]


def test_pp1_and_reference_unchanged():
    for n in (24, 36):
        assert counts(n, 2, edge_light=True) == [len(layer_partition(n, 2, s)) for s in range(2)]
        assert counts(n, 1, edge_light=True) == [n]


@pytest.mark.parametrize("num_layers", [8, 24, 36, 61])
@pytest.mark.parametrize("pp", [2, 3, 4, 8])
def test_edge_rank_is_lightest(num_layers, pp):
    if num_layers < 2 * pp:
        pytest.skip("fewer layers than stages")
    c = counts(num_layers, 2 * pp, edge_light=True)
    per_rank = [c[r] + c[2 * pp - 1 - r] for r in range(pp)]
    assert per_rank[0] == min(per_rank)
    assert all(n >= 1 for n in c)
    # DualPipeV pairs a rank's two chunks layer by layer: within one layer of each other.
    assert all(abs(c[r] - c[2 * pp - 1 - r]) <= 1 for r in range(pp))


def test_other_models_partition_unchanged():
    # Generic layer_partition is untouched (other models still call it directly).
    assert [len(layer_partition(24, 4, s)) for s in range(4)] == [6, 6, 6, 6]
    assert [len(layer_partition(26, 4, s)) for s in range(4)] == [6, 7, 7, 6]
    assert [len(layer_partition(27, 4, s)) for s in range(4)] == [7, 7, 7, 6]
