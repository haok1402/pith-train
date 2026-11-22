"""Bounded, single-node benchmark using the fork's two-node pretraining recipe."""

import argparse
from pathlib import Path
from runpy import run_path

from pithtrain.modules.training import make_adamw_optimizer, make_constant_scheduler
from pithtrain.tasks.pretrain_lm import launch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--dataset", type=Path, help="Tokenized corpus directory")
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    cfg = run_path(str(bench_dir / "upstream_qwen35_27b.py"))["cfg"]
    if args.dataset is not None:
        cfg.dataset = args.dataset
    cfg.training.benchmark = args.dataset is None
    cfg.training.hf_import_path = bench_dir.parent / "inputs" / "Qwen3.5-27B"
    cfg.training.sequence_length = args.seq
    cfg.training.global_batch_size = args.batch
    cfg.training.micro_batch_size = args.micro_batch
    cfg.training.max_steps = args.steps
    cfg.training.optimizer = make_adamw_optimizer
    cfg.training.scheduler = make_constant_scheduler
    cfg.training.lr = 1e-5
    cfg.training.save_interval = None
    cfg.training.save_location = None
    launch(cfg)
