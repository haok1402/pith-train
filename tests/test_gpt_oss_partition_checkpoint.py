"""
Checkpoint interchange between gpt-oss pipeline partitions (FP8, PP2/EP4, fp8_param_gather, on 8
GPUs): a reduced gpt-oss-20b (8 layers, 8 experts) trains one step under one layout and saves; the checkpoint
loads into the other layout, which saves it again. Every canonical model, optimizer and scheduler
entry of the two checkpoints must be bitwise equal (same keys, dtypes, shapes and bytes), in both
directions: even [2,2,2,2] -> edge-light [1,3,3,1] and back. Driver:

    bash tests/test_gpt_oss_partition_checkpoint.sh
"""

import json
import os
import sys
from pathlib import Path

import torch

import pithtrain.models.gpt_oss as gpt_oss

DATASET = Path("/mnt/weka/home/hao.kang/haok-pith-train/workspace/datasets/dclm-baseline/toktxt/gpt-oss")  # fmt: skip


def reduced_model(out: Path) -> Path:
    cfg = json.loads(Path("examples/pretrain_lm/gpt-oss-20b/config.json").read_text())
    cfg["num_hidden_layers"] = 8
    cfg["layer_types"] = cfg["layer_types"][:8]
    cfg["num_local_experts"] = 8
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg))
    return out


def run(mode: str, layout: str, work: Path) -> None:
    from pithtrain.contexts import training
    from pithtrain.modules.checkpoint import load_checkpoint, save_checkpoint
    from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
    from pithtrain.operators.fp8_param import precompute_fp8_param_scales
    from pithtrain.tasks import pretrain_lm

    if layout == "even":
        partition = gpt_oss.gpt_oss_layer_partition
        gpt_oss.gpt_oss_layer_partition = lambda n, sc, si, edge_light: partition(n, sc, si, False)
    cfg = pretrain_lm.PretrainLMCfg()
    cfg.dataset = DATASET
    cfg.distributed.pipeline_parallel_size = 2
    cfg.distributed.expert_parallel_size = 4
    t = cfg.training
    t.model = work / "model"
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.lr, t.micro_batch_size, t.sequence_length, t.global_batch_size = 1e-4, 1, 4096, 16
    t.moe_load_balance_type, t.moe_load_balance_coef = "global-batch", 1e-3
    t.fp8 = t.benchmark = True
    t.max_steps = 1
    pretrain_lm.setup_logging(cfg)
    pretrain_lm.setup_distributed(cfg)
    dataset = pretrain_lm.setup_dataset(cfg)
    pretrain_lm.setup_training(cfg)
    layers = [sorted(int(i) for i in m.layers) for m in training.model.module]
    print(f"rank {torch.distributed.get_rank()} {layout} layers {layers}", flush=True)
    if mode == "train":
        pretrain_lm.train_step(cfg, dataset, 0)
    else:
        load_checkpoint(work / f"ckpt-{mode}", 1)
        precompute_fp8_param_scales()
    save_checkpoint(work / f"ckpt-{layout}-{mode}", 1)
    torch.distributed.barrier()


def compare(work: Path, a: str, b: str) -> None:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    sds = []
    for name in (a, b):
        out = work / f"{name}.pt"
        dcp_to_torch_save(work / name / "torch-dcp" / "00000001", out)
        sds.append(torch.load(out, weights_only=False))

    def flat(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    fa, fb = dict(flat(sds[0])), dict(flat(sds[1]))
    assert fa.keys() == fb.keys(), sorted(set(fa) ^ set(fb))[:10]
    tensors = 0
    for k in fa:
        x, y = fa[k], fb[k]
        if isinstance(x, torch.Tensor):
            assert x.dtype == y.dtype and x.shape == y.shape, k
            assert torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)), k  # fmt: skip
            tensors += 1
        elif k.endswith("param_groups"):
            # DCP keeps one pipeline rank's membership list (whose layers depend on the layout),
            # and loading rebuilds membership from the live model (checkpoint.to_localized_optim),
            # so only the group count and the hyperparameters must match.
            def hyper(groups):
                return [{n: v for n, v in g.items() if n != "params"} for g in groups]

            assert hyper(x) == hyper(y), k
        else:
            assert x == y, k
    print(f"{a} == {b}: {len(fa)} entries, {tensors} tensors bitwise equal")


if __name__ == "__main__":
    mode, work = sys.argv[1], Path(sys.argv[-1])
    if mode == "model":
        reduced_model(work / "model")
    elif mode == "compare":
        compare(work, sys.argv[2], sys.argv[3])
    else:
        run(mode, os.environ["LAYOUT"], work)
