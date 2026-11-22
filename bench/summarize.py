"""Summarize steady-state PithTrain logs and estimate model FLOPs utilization."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median

PATTERN = re.compile(
    r"step (\d+)/\d+ \| step-time ([\d.]+) sec .*?tokens-per-second ([\d,]+)"
    r" \| peak-gpu-memory ([\d.]+) GB"
)


def training_flops_per_token(config, seq_len):
    """Count useful training FLOPs, using the recurrent estimate for Gated DeltaNet."""
    hidden = config["hidden_size"]
    counts = Counter(config["layer_types"])
    if sum(counts.values()) != config["num_hidden_layers"] or counts.keys() - {
        "linear_attention", "full_attention"
    }:
        raise ValueError(f"Unexpected layer types: {counts}")

    q_width = config["num_attention_heads"] * config["head_dim"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    gate_width = q_width if config["attn_output_gate"] else 0
    full_proj = 2 * hidden * (2 * q_width + gate_width + 2 * kv_width)
    full_core = 2 * (seq_len + 1) * q_width

    key_head_dim = config["linear_key_head_dim"]
    value_head_dim = config["linear_value_head_dim"]
    value_heads = config["linear_num_value_heads"]
    key_width = key_head_dim * config["linear_num_key_heads"]
    value_width = value_head_dim * value_heads
    conv_width = 2 * key_width + value_width
    gdn_proj = 2 * hidden * (2 * key_width + 3 * value_width + 2 * value_heads)
    gdn_conv = 2 * config["linear_conv_kernel_dim"] * conv_width
    forward = {
        "SwiGLU projections": counts.total() * 6 * hidden * config["intermediate_size"],
        "full-attention projections": counts["full_attention"] * full_proj,
        "causal full-attention core": counts["full_attention"] * full_core,
        "GDN projections": counts["linear_attention"] * gdn_proj,
        "GDN depthwise convolution": counts["linear_attention"] * gdn_conv,
        "LM head": 2 * hidden * config["vocab_size"],
    }
    training = {name: 3 * flops for name, flops in forward.items()}
    # State decay, reads and rank-one update cost about 7 forward and 15 backward
    # FLOPs per state element. The chunked implementation performs extra work.
    training["GDN recurrent core"] = (
        counts["linear_attention"] * 22 * value_heads * key_head_dim * value_head_dim
    )
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="?")
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--first", type=int, default=5)
    args = parser.parse_args()
    path = (
        Path(__file__).resolve().parent.parent
        / "pith-train/examples/pretrain_lm/qwen3.5-27b/config.json"
    )
    breakdown = training_flops_per_token(json.loads(path.read_text()), args.seq)
    for name, flops in breakdown.items():
        print(f"{name:26} {flops / 1e9:8.4f} GFLOP/token")
    flops = sum(breakdown.values())
    peak = args.gpus * 989e12
    print(f"Total {flops / 1e9:.4f} GFLOP/token")
    print(f"50% target {0.5 * peak / flops:,.0f} aggregate tokens/s")
    if args.log is None:
        return
    rows = [
        (int(step), float(seconds), int(tps.replace(",", "")), float(memory))
        for step, seconds, tps, memory in PATTERN.findall(args.log.read_text())
        if int(step) >= args.first
    ]
    if not rows:
        parser.error(f"No completed step at or after {args.first} in {args.log}")
    tps = median(row[2] for row in rows)
    print(f"Steps {rows[0][0]}–{rows[-1][0]}")
    print(f"Median step time {median(row[1] for row in rows):.3f} s")
    print(f"Median throughput {tps:,.0f} tokens/s aggregate, {tps / args.gpus:,.0f} per GPU")
    print(f"Max peak allocated memory {max(row[3] for row in rows):.2f} GiB per GPU")
    print(f"MFU {100 * tps * flops / peak:.2f}%")


if __name__ == "__main__":
    main()
