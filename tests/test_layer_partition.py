"""
Pipeline layer partition (pure functions, no GPU): the default split is unchanged, Qwen3.5's
override stays legal, and gpt-oss's edge-light split (gpt_oss_layer_partition) keeps the edge rank
lightest.
"""

import pytest

from pithtrain.pipeline.dualpipev import layer_partition


def reference_partition(num_layers: int, stage_count: int, stage_index: int) -> range:
    """The split layer_partition computed before explicit sizes existed."""
    base, remainder = divmod(num_layers, stage_count)
    layers = [base] * stage_count
    for _ in range(remainder):
        min_val = min(layers)
        best = next((i for i in range(1, stage_count - 1) if layers[i] == min_val), None)
        if best is None:
            best = layers.index(min_val)
        layers[best] += 1
    begin = sum(layers[:stage_index])
    return range(begin, begin + layers[stage_index])


# qwen3-30b-a3b, deepseek-v2-lite, gpt-oss-20b, gpt-oss-120b, qwen3.5-35b-a3b
@pytest.mark.parametrize("num_layers", [48, 27, 24, 36, 40])
@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
def test_default_partition_unchanged(num_layers: int, pp_size: int) -> None:
    for stage_count in (1, 2 * pp_size):
        for i in range(stage_count):
            assert layer_partition(num_layers, stage_count, i) == reference_partition(num_layers, stage_count, i)  # fmt: skip


def test_qwen35_partition() -> None:
    from pithtrain.models.qwen35_moe import pipeline_layer_sizes

    assert pipeline_layer_sizes(40, 1) == [40]
    assert pipeline_layer_sizes(40, 2) == [20, 20]
    # pp=2: rank 0 holds stages 0 and 3 (embed, and norm + lm_head + loss), rank 1 stages 1 and 2.
    assert pipeline_layer_sizes(40, 4) == [10, 10, 11, 9]
    for pp_size in (2, 4, 5, 8):
        stage_count = 2 * pp_size
        sizes = pipeline_layer_sizes(40, stage_count)
        owned = [i for s in range(stage_count) for i in layer_partition(40, stage_count, s, sizes)]
        assert owned == list(range(40))
        assert all(abs(sizes[i] - sizes[stage_count - 1 - i]) <= 1 for i in range(stage_count))


def test_unbalanced_sizes_rejected() -> None:
    with pytest.raises(AssertionError):
        layer_partition(40, 4, 0, [9, 10, 10, 11])  # rank 0 would hold 9 and 11


def gpt_oss_counts(num_layers: int, stage_count: int, edge_light: bool) -> list[int]:
    from pithtrain.models.gpt_oss import gpt_oss_layer_partition

    ranges = [gpt_oss_layer_partition(num_layers, stage_count, s, edge_light) for s in range(stage_count)]  # fmt: skip
    # Contiguous, in order, covering every layer once: global ids are unchanged.
    assert [i for r in ranges for i in r] == list(range(num_layers))
    return [len(r) for r in ranges]


def test_gpt_oss_edge_light() -> None:
    # gpt-oss-20b at pp=2: rank 0 holds stages 0 (embed) and 3 (lm_head), so fewer layers there.
    assert gpt_oss_counts(24, 4, edge_light=True) == [5, 7, 7, 5]
    # BF16 runs (edge_light=False) keep the default partition, and pp=1 is unaffected.
    for n in (24, 36):
        assert gpt_oss_counts(n, 4, edge_light=False) == [len(layer_partition(n, 4, s)) for s in range(4)]  # fmt: skip
        assert gpt_oss_counts(n, 2, edge_light=True) == [len(layer_partition(n, 2, s)) for s in range(2)]  # fmt: skip
        assert gpt_oss_counts(n, 1, edge_light=True) == [n]


@pytest.mark.parametrize("num_layers", [8, 24, 36, 61])
@pytest.mark.parametrize("pp", [2, 3, 4, 8])
def test_gpt_oss_edge_rank_is_lightest(num_layers: int, pp: int) -> None:
    if num_layers < 2 * pp:
        pytest.skip("fewer layers than stages")
    c = gpt_oss_counts(num_layers, 2 * pp, edge_light=True)
    per_rank = [c[r] + c[2 * pp - 1 - r] for r in range(pp)]
    assert per_rank[0] == min(per_rank)
    assert all(n >= 1 for n in c)
    # DualPipeV pairs a rank's two chunks layer by layer: within one layer of each other.
    assert all(abs(c[r] - c[2 * pp - 1 - r]) <= 1 for r in range(pp))
