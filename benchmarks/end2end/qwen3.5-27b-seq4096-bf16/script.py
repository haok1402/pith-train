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
    parser.add_argument(
        "--schedule",
        choices=("dualpipev", "1f1b", "interleaved", "zb-h1", "zbv"),
        default="dualpipev",
    )
    parser.add_argument("--virtual-stages", type=int)
    parser.add_argument("--layer-counts", type=int, nargs="+")
    parser.add_argument("--compile-mode", choices=("attention", "eager"), default="attention")
    recompute = parser.add_mutually_exclusive_group()
    recompute.add_argument("--recompute-mlp", action="store_true")
    recompute.add_argument("--recompute-mlp-layers", type=int, nargs="+")
    parser.add_argument("--recompute-mlp-up-only", action="store_true")
    parser.add_argument("--grad-sync-layers", type=int, nargs="+", default=())
    parser.add_argument("--compute-stream-grad-buffers", action="store_true")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hf-import", type=Path)
    parser.add_argument("--dataset", type=Path, help="Directory of PithTrain token shards.")
    parser.add_argument("--log-first-microbatch-loss", action="store_true")
    parser.add_argument("--allocator-trace-step", type=int)
    parser.add_argument("--allocator-trace-dir", type=Path)
    parser.add_argument("--profile-pipeline-step", type=int)
    args = parser.parse_args()
    if args.allocator_trace_step is not None and (
        args.allocator_trace_dir is None or not 1 <= args.allocator_trace_step <= args.steps
    ):
        parser.error("--allocator-trace-step requires a valid step and --allocator-trace-dir")
    if args.profile_pipeline_step is not None and (
        args.schedule not in ("interleaved", "zb-h1", "zbv")
        or not 1 <= args.profile_pipeline_step <= args.steps
    ):
        parser.error("--profile-pipeline-step requires a valid step and a scheduled dense pipeline")

    cfg = PretrainLMCfg()
    cfg.dataset = args.dataset or Path(".")
    cfg.log_first_microbatch_loss = args.log_first_microbatch_loss
    cfg.allocator_retry_warmup_steps = 3
    cfg.profile_pipeline_step = args.profile_pipeline_step
    cfg.distributed.pipeline_parallel_size = args.pp
    cfg.distributed.context_parallel_size = 1
    cfg.distributed.expert_parallel_size = 1
    training = cfg.training
    repo = Path(__file__).resolve().parents[3]
    training.model = args.model or repo / "examples/pretrain_lm/qwen3.5-27b"
    training.hf_import_path = args.hf_import
    training.pipeline_schedule = args.schedule
    training.pipeline_virtual_stages = args.virtual_stages
    if training.pipeline_virtual_stages is None:
        training.pipeline_virtual_stages = 2 if args.schedule in ("interleaved", "zbv") else 1
    training.pipeline_layer_counts = tuple(args.layer_counts) if args.layer_counts else None
    training.pipeline_compile = args.compile_mode
    training.pipeline_recompute_mlp = (
        tuple(args.recompute_mlp_layers) if args.recompute_mlp_layers else args.recompute_mlp
    )
    training.pipeline_grad_sync_layers = tuple(args.grad_sync_layers)
    training.pipeline_compute_stream_grad_buffers = args.compute_stream_grad_buffers
    training.pipeline_recompute_mlp_up_only = args.recompute_mlp_up_only
    training.optimizer = make_muon_optimizer if args.optimizer == "muon" else make_adamw_optimizer
    training.scheduler = partial(
        make_wsd_scheduler, start_lr=1e-5, warmup_ratio=0.03, final_lr=1e-5
    )
    training.lr = args.lr
    training.sequence_length = args.seq
    if args.allocator_trace_step is not None:
        training.memory_profile_start = args.allocator_trace_step - 1
        training.memory_profile_stop = args.allocator_trace_step
        training.memory_profile_output = args.allocator_trace_dir
    training.micro_batch_size = args.micro_batch
    training.global_batch_size = args.batch
    training.max_steps = args.steps
    training.benchmark = args.dataset is None
    training.fp8 = False
    launch(cfg)


if __name__ == "__main__":
    main()
