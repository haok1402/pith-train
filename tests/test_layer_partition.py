"""Pipeline layer partition: the default split is unchanged, and Qwen3.5's override stays legal."""

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
