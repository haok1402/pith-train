"""Compare pipeline schedules with synthetic tokens and optional Qwen3.5 weights."""

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

from pithtrain.modules.training import (
    make_adamw_optimizer,
    make_muon_optimizer,
    make_wsd_scheduler,
)
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--schedule", choices=("dualpipev", "1f1b"), default="dualpipev")
    parser.add_argument("--layer-counts", type=int, nargs="+")
    parser.add_argument("--compile-mode", choices=("layer", "attention", "eager"), default="layer")
    parser.add_argument("--recompute-mlp", action="store_true")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hf-import", type=Path)
    args = parser.parse_args()

    cfg = PretrainLMCfg()
    cfg.dataset = Path(".")
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.expert_parallel_size = 1
    training = cfg.training
    repo = Path(__file__).resolve().parents[3]
    training.model = args.model or repo / "examples/pretrain_lm/qwen3.5-27b"
    training.hf_import_path = args.hf_import
    training.pipeline_schedule = args.schedule
    training.pipeline_layer_counts = tuple(args.layer_counts) if args.layer_counts else None
    training.pipeline_compile = args.compile_mode
    training.pipeline_recompute_mlp = args.recompute_mlp
    training.optimizer = make_muon_optimizer if args.optimizer == "muon" else make_adamw_optimizer
    training.scheduler = partial(
        make_wsd_scheduler, start_lr=1e-5, warmup_ratio=0.03, final_lr=1e-5
    )
    training.lr = 3e-4
    training.sequence_length = args.seq
    training.micro_batch_size = args.micro_batch
    training.global_batch_size = args.batch
    training.max_steps = args.steps
    training.benchmark = True
    training.fp8 = False
    launch(cfg)


if __name__ == "__main__":
    main()
