"""Run one rank using the verified CPU artifacts and record startup timings."""

import argparse
import os
import runpy
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

from tools.fp8_preflight.artifacts import CuteArtifacts, read_json
from tools.fp8_preflight.imports import ImportArchive
from tools.fp8_preflight.inductor import require_cache
from tools.fp8_preflight.misses import record_miss
from tools.fp8_preflight.recipe import (
    ARCH,
    MICROBATCH,
    SEQUENCE,
    SMS,
    WORLD_SIZE,
    shared_memory_attribute,
)
from tools.fp8_preflight.weights import load_rank, sealed_model_config
from tools.fp8_preflight.scales import load_initial_scales

BENCHMARK = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/end2end/qwen3.5-27b-seq4096-bf16/h200-1n8g/fsdp2/script.py"
)


def mark(phase, state, detail=""):
    now = datetime.now(UTC).isoformat(timespec="milliseconds")
    os.write(
        1,
        f"STARTUP rank={os.environ['RANK']} time={now} phase={phase} state={state} {detail}\n".encode(),
    )


@contextmanager
def measure(phase):
    mark(phase, "begin")
    start = time.monotonic()
    try:
        yield
    finally:
        mark(phase, "end", f"seconds={time.monotonic() - start:.3f}")


def timed(name, function, *, first_only=False):
    count = 0

    @wraps(function)
    def call(*args, **kwargs):
        nonlocal count
        count += 1
        if first_only and count > 1:
            return function(*args, **kwargs)
        with measure(name):
            return function(*args, **kwargs)

    return call


def run(args):
    bundle = args.bundle
    seal = read_json(bundle / "seal.json")
    model_config = sealed_model_config(bundle, seal)
    target = seal["identity"]
    if (target["arch"], target["sm_count"], target["world_size"]) != (ARCH, SMS, WORLD_SIZE):
        raise ValueError("Sealed bundle targets another cluster configuration")
    if (
        target["weights"] != str(args.weights.resolve())
        or int(os.environ["WORLD_SIZE"]) != WORLD_SIZE
    ):
        raise ValueError("Sealed weights or world size do not match the training launch")
    if not os.environ.get("PITHTRAIN_STAGED_RUNTIME"):
        sys.meta_path.insert(0, ImportArchive(bundle / "imports.zip"))
    with measure("torch_import"):
        import torch

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    with measure("cuda_device"):
        torch.cuda.set_device(local_rank)
        properties = torch.cuda.get_device_properties(local_rank)
        if (properties.major, properties.minor, properties.multi_processor_count) != (9, 0, SMS):
            raise RuntimeError(f"Preflight requires an H200 with {SMS} SMs, got {properties}")
    mark("gpu", "ready", f"uuid={properties.uuid}")

    with measure("pithtrain_import"):
        from cutlass import cute
        from cutlass.base_dsl.runtime import cuda as cuda_helpers

        shared_memory = cuda_helpers.get_device_attribute(
            shared_memory_attribute(cuda_helpers), local_rank
        )
        if shared_memory != target["shared_memory_per_sm"]:
            raise RuntimeError(
                f"Sealed CuTe shared-memory target differs from the device: {shared_memory}"
            )
        kernel_root = Path(os.environ.get("PITHTRAIN_KERNEL_ROOT", bundle))
        cute.compile = CuteArtifacts(kernel_root, seal.get("cute", {}), build=False, fallback=True)
        from torch._inductor import compile_fx

        from pithtrain.contexts import training as training_context
        from pithtrain.modules import training
        from pithtrain.tasks import pretrain_lm

    with measure("inductor_cache"):
        require_cache(bundle)

    pretrain_lm.setup_distributed = timed("process_group", pretrain_lm.setup_distributed)
    pretrain_lm.setup_dataset = timed("dataset", pretrain_lm.setup_dataset)
    pretrain_lm.setup_training = timed("training_setup", pretrain_lm.setup_training)
    recompute_scales = pretrain_lm.precompute_fp8_param_scales
    initial_scale_call = True

    def prepare_scales():
        nonlocal initial_scale_call
        if not initial_scale_call:
            return recompute_scales()
        initial_scale_call = False
        with measure("fp8_scale_setup"):
            try:
                if "initial_scales" not in seal:
                    raise ValueError("No sealed initial scale artifact")
                with torch.no_grad():
                    load_initial_scales(bundle / "initial-scales.safetensors", training_context.model)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                record_miss("fp8_scales", "initial-scales", str(error).splitlines()[0])
                return recompute_scales()
            mark("initial_fp8_scales", "loaded")

    pretrain_lm.precompute_fp8_param_scales = prepare_scales
    prepare_loss = timed("first_loss", pretrain_lm.prepare_cross_entropy_sum, first_only=True)
    first_loss_events = None

    @wraps(prepare_loss)
    def measure_loss(*args, **kwargs):
        nonlocal first_loss_events
        if first_loss_events is not None:
            return prepare_loss(*args, **kwargs)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = prepare_loss(*args, **kwargs)
        end.record()
        first_loss_events = begin, end
        return result

    pretrain_lm.prepare_cross_entropy_sum = measure_loss
    torch.nn.Module.to_empty = timed("materialize_fp32_shards", torch.nn.Module.to_empty)
    compile_fx.compile_fx = timed("inductor", compile_fx.compile_fx)
    train_step = timed("train_step", pretrain_lm.train_step)
    reported_loss_gpu_time = False

    @wraps(train_step)
    def report_step(*args, **kwargs):
        nonlocal reported_loss_gpu_time
        train_step(*args, **kwargs)
        if not reported_loss_gpu_time and first_loss_events is not None:
            begin, end = first_loss_events
            mark("first_loss_gpu", "end", f"seconds={begin.elapsed_time(end) / 1000:.3f}")
            reported_loss_gpu_time = True

    pretrain_lm.train_step = report_step

    setup_model = training.setup_model

    @wraps(setup_model)
    def timed_setup_model(cfg, distributed_cfg, *, shard_dense_model=None):
        if shard_dense_model is not None:
            shard_dense_model = timed("fsdp_shard", shard_dense_model)
        return setup_model(
            cfg, distributed_cfg, shard_dense_model=shard_dense_model, model_config=model_config
        )

    training.setup_model = timed("model_and_fsdp", timed_setup_model)

    def load_weights(source, model):
        if Path(source).resolve() != args.weights.resolve():
            raise ValueError("Training requested a different checkpoint from the preflight")
        load_rank(bundle, Path(source), model, rank)

    pretrain_lm.load_hf_into_model = timed("rank_weights", load_weights)
    if args.audit_losses:
        audit_losses(pretrain_lm, torch)

    sys.argv = [
        str(BENCHMARK),
        "--steps",
        str(args.steps),
        "--model",
        str(args.model),
        "--weights",
        str(args.weights),
        "--dataset",
        str(args.dataset),
        "--seq",
        str(SEQUENCE),
        "--batch",
        str(WORLD_SIZE * MICROBATCH),
        "--micro-batch",
        str(MICROBATCH),
        "--fp8",
        "--fp8-attention",
        "--fp8-mlp-recompute",
        "--checkpoint-mlp-tail",
        "4",
        "--checkpoint-replay-mixers",
        "both",
        "--fp8-shard-cache-mib",
        "1024",
    ]
    runpy.run_path(str(BENCHMARK), run_name="__main__")


def audit_losses(task, torch):
    prepare = task.prepare_cross_entropy_sum
    train_step = task.train_step
    losses = []

    @wraps(prepare)
    def audited_prepare(*args, **kwargs):
        backward, loss, count = prepare(*args, **kwargs)
        losses.append(loss)
        return backward, loss, count

    @wraps(train_step)
    def audited_step(cfg, dataset, step):
        before = torch.cuda.memory_stats()["num_alloc_retries"]
        train_step(cfg, dataset, step)
        retries = torch.cuda.memory_stats()["num_alloc_retries"] - before
        bits = ",".join(
            f"{value & 0xFFFFFFFF:08x}" for value in torch.stack(losses).view(torch.int32).tolist()
        )
        losses.clear()
        os.write(
            1,
            f"audit step={step + 1} rank={os.environ['RANK']} retries={retries} outcome=ok loss_bits={bits}\n".encode(),
        )
        if step >= 2 and retries:
            raise RuntimeError(f"Allocator retried in step {step + 1}")

    task.prepare_cross_entropy_sum = audited_prepare
    task.train_step = audited_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--audit-losses", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
