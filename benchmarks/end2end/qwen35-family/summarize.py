"""Summarize a family runner directory with the Megatron dense GDN estimator."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median

H200_BF16_FLOPS = 989.4e12
GLOBAL_BATCH, SEQUENCE_LENGTH, GPUS = 64, 4096, 8
STEP = re.compile(
    r"\bstep (\d+)/(\d+) \| step-time (\S+) sec \| cross-entropy-loss (\S+)"
    r".*?gradient-norm (\S+) \| tokens-per-second ([\d,]+)"
    r" \| peak-gpu-memory (\S+) GB \| peak-gpu-reserved (\S+) GB"
    r" \| cuda-alloc-retries (\d+)"
)


def model_size_and_flops(config):
    """Count text-tower weights and FLOPs per token, excluding the vision and MTP towers."""
    cfg = config.get("text_config", config)
    hidden = cfg["hidden_size"]
    layer_types = cfg["layer_types"]
    linear = layer_types.count("linear_attention")
    full = layer_types.count("full_attention")
    layers = cfg["num_hidden_layers"]
    if linear + full != layers or len(layer_types) != layers:
        raise ValueError("Expected a dense Qwen3.5 layer pattern")
    tied = cfg.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
    q = cfg["head_dim"] * cfg["num_attention_heads"]
    kv = cfg["head_dim"] * cfg["num_key_value_heads"]
    full_projections = hidden * (2 * q + 2 * kv + q * cfg["attn_output_gate"])

    key = cfg["linear_key_head_dim"] * cfg["linear_num_key_heads"]
    value_dim = cfg["linear_value_head_dim"]
    value_heads = cfg["linear_num_value_heads"]
    value = value_dim * value_heads
    gdn_projections = hidden * (2 * key + 3 * value + 2 * value_heads)
    convolution = cfg["linear_conv_kernel_dim"] * (2 * key + value)
    mlp = 3 * hidden * cfg["intermediate_size"]
    head = hidden * cfg["vocab_size"]

    parameters = (
        head * (1 if tied else 2)
        + hidden * (2 * layers + 1)
        + layers * mlp
        + full * (full_projections + 2 * cfg["head_dim"])
        + linear * (gdn_projections + convolution + 2 * value_heads + value_dim)
    )
    # Megatron num_floating_point_operations (dense GDN, causal BSHD, no MTP).
    # Multiply forward MACs by 2 for FMA and by 3 for forward plus backward.
    flops = {
        "mlp": 6 * layers * mlp,
        "lm_head": 6 * head,
        "full_projections": 6 * full * full_projections,
        "full_attention_core": 6 * full * SEQUENCE_LENGTH * q,
        "gdn_projections": 6 * linear * gdn_projections,
        "gdn_convolution": 6 * linear * convolution,
        "gdn_core": 6 * linear * 4 * value_heads * value_dim**2,
    }
    return parameters, flops


def read_arm(path, steps, first, last):
    rows = STEP.findall((path / "train.log").read_text(errors="replace"))
    if [(int(s), int(n)) for s, n, *_ in rows] != [(s, steps) for s in range(1, steps + 1)]:
        raise ValueError(f"{path.name} has {len(rows)} of {steps} expected updates")
    loss = [float(row[3]) for row in rows]
    norm = [float(row[4]) for row in rows]
    if not all(math.isfinite(number) for number in loss + norm):
        raise ValueError(f"{path.name} has a nonfinite loss or gradient norm")
    timed = rows[first - 1 : last]
    rate = median(int(row[5].replace(",", "")) for row in timed) / GPUS
    return {
        "recipe": (path / "recipe.txt").read_text().strip(),
        "tok_s_gpu": rate,
        "peak_allocated_gib": max(float(row[6]) for row in timed),
        "peak_reserved_gib": max(float(row[7]) for row in timed),
        "warmup_max_rank_retries": sum(int(row[8]) for row in rows[: first - 1]),
        "timed_max_rank_retries": sum(int(row[8]) for row in timed),
        "loss_by_step": loss,
        "gradient_norm_by_step": norm,
    }


def summarize(path, first, last):
    metadata = dict(line.split("=", 1) for line in (path / "run.txt").read_text().splitlines())
    steps = int(metadata["steps"])
    if not 1 <= first <= last <= steps:
        raise ValueError(f"Invalid timing window {first}–{last} for {steps} updates")
    params, flops = model_size_and_flops(json.loads((path / "config.json").read_text()))
    arm_names = (
        (f"01-{metadata['mode']}",)
        if metadata["mode"] in ("bf16", "fp8")
        else ("01-bf16", "02-fp8", "03-fp8", "04-bf16")
    )
    arms = {name: read_arm(path / name, steps, first, last) for name in arm_names}
    corpus_digests = {(path / name / "corpus.sha256").read_text() for name in arms}
    if len(corpus_digests) != 1:
        raise ValueError("Arms used different sample order")
    metrics = {
        **metadata,
        "parameters": params,
        "training_flops_per_token": sum(flops.values()),
        "training_flops_breakdown": flops,
        "timing_window": [first, last],
        "corpus_digests": corpus_digests.pop().splitlines(),
        "arms": arms,
    }
    for precision in ("bf16", "fp8"):
        repeats = [arm for name, arm in arms.items() if name.endswith(precision)]
        if not repeats:
            continue
        rate = median(arm["tok_s_gpu"] for arm in repeats)
        metrics[precision] = {
            "tok_s_gpu": rate,
            "mfu_percent": 100 * rate * sum(flops.values()) / H200_BF16_FLOPS
            if precision == "bf16"
            else None,
            "peak_allocated_gib": max(arm["peak_allocated_gib"] for arm in repeats),
            "peak_reserved_gib": max(arm["peak_reserved_gib"] for arm in repeats),
        }
    if len(arms) == 4:
        first_bf16, first_fp8, second_fp8, second_bf16 = arms.values()
        metrics["speedups_in_arm_order"] = [
            first_fp8["tok_s_gpu"] / first_bf16["tok_s_gpu"],
            second_fp8["tok_s_gpu"] / second_bf16["tok_s_gpu"],
        ]
        metrics["speedup"] = median(metrics["speedups_in_arm_order"])
        metrics["max_abs_paired_loss_delta"] = max(
            abs(a - b)
            for bf16, fp8 in ((first_bf16, first_fp8), (second_bf16, second_fp8))
            for a, b in zip(bf16["loss_by_step"], fp8["loss_by_step"], strict=True)
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--first", type=int, default=7)
    parser.add_argument("--last", type=int, default=12)
    args = parser.parse_args()
    try:
        result = summarize(args.run, args.first, args.last)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
