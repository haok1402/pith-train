import pytest

from pithtrain.modules import training
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.modules.training import TrainingCfg


def recipe(fp8, micro_batch, sequence):
    cfg = TrainingCfg()
    cfg.fp8 = fp8
    cfg.micro_batch_size = micro_batch
    cfg.sequence_length = sequence
    return cfg


@pytest.mark.parametrize("micro_batch,sequence,cp", [(1, 18, 1), (2, 36, 2), (1, 0, 1)])
def test_dense_sm90_fp8_recipe_rejects_compact_k_tail_before_loading_model(
    monkeypatch, micro_batch, sequence, cp
):
    monkeypatch.setattr(training, "fp8_wgrad_supported", lambda: True)
    monkeypatch.setattr(training.distributed, "cp_size", cp, raising=False)
    monkeypatch.setattr(
        training.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: pytest.fail("recipe validation must precede loading the model"),
    )
    cfg = recipe(True, micro_batch, sequence)
    with pytest.raises(ValueError, match="positive multiple of 16"):
        training.setup_model(cfg, DistributedCfg(), shard_dense_model=lambda *_: None)


@pytest.mark.parametrize(
    "fp8,sm90,dense,micro_batch,sequence,cp",
    [
        (True, True, True, 2, 24, 1),
        (True, True, True, 2, 16, 2),
        (True, True, False, 1, 18, 1),
        (False, True, True, 1, 18, 1),
        (True, False, True, 1, 18, 1),
    ],
)
def test_recipe_check_leaves_other_paths_available(
    monkeypatch, fp8, sm90, dense, micro_batch, sequence, cp
):
    class LoadReached(Exception):
        pass

    def load(*args):
        raise LoadReached

    monkeypatch.setattr(training, "fp8_wgrad_supported", lambda: sm90)
    for name in ("fp8", "fp8_attention", "Linear", "GroupedLinear"):
        monkeypatch.setattr(training.training, name, None, raising=False)
    monkeypatch.setattr(training.distributed, "cp_size", cp, raising=False)
    monkeypatch.setattr(training.distributed, "cp_group", None, raising=False)
    monkeypatch.setattr(training.AutoConfig, "from_pretrained", load)
    cfg = recipe(fp8, micro_batch, sequence)
    cfg.model = "test"
    with pytest.raises(LoadReached):
        training.setup_model(
            cfg, DistributedCfg(), shard_dense_model=(lambda *_: None) if dense else None
        )
