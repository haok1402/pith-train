"""Compare full and sequence-sharded BF16 Qwen3.5 loss and all parameter gradients."""

import argparse
import copy
import json
import logging as std_logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--async-tp", action="store_true")
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument("--hf-roundtrip", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    from pithtrain.contexts import distributed, logging, training
    from pithtrain.models.qwen35 import Qwen35TextModel
    from pithtrain.modules.dense_fsdp import DenseFSDPCfg, shard_dense_model
    from pithtrain.modules.dense_tp import DenseTP
    from pithtrain.modules.hf_loader import load_hf_into_model
    from pithtrain.modules.training import init_weights
    from pithtrain.operators.linear import BF16Linear

    training.fp8, training.Linear = False, BF16Linear
    distributed.cp_size = 1
    logging.stdout = std_logging.getLogger("tp-parity")
    std_logging.basicConfig(level=std_logging.INFO)
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        hidden_size=768,
        intermediate_size=1024,
        vocab_size=2048,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=512,
        head_dim=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    reference = Qwen35TextModel(config).to(device=device, dtype=torch.bfloat16)
    init_weights(reference, config.num_hidden_layers, seed=17, pipeline_chunks=False)
    parallel = copy.deepcopy(reference)
    mesh = init_device_mesh("cuda", (1, dist.get_world_size()), mesh_dim_names=("dp", "tp"))
    tp = DenseTP(mesh, async_tp=args.async_tp)
    tp.shard_model(parallel)
    fsdp = DenseFSDPCfg()
    fsdp.compile_blocks = args.compile_blocks
    shard_dense_model(parallel, mesh["dp"], fsdp)
    if args.hf_roundtrip:
        fixture = [None]
        if dist.get_rank() == 0:
            scratch = TemporaryDirectory(dir=os.environ.get("TMPDIR", "."), prefix="tp-parity-")
            fixture[0] = scratch.name
            folder = Path(scratch.name)
            filename = "model.safetensors"
            weights = {
                "model.language_model." + name
                if name != "lm_head.weight"
                else name: param.detach().cpu()
                for name, param in reference.named_parameters()
            }
            save_file(weights, folder / filename)
            (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(weights, filename)})
            )
        dist.broadcast_object_list(fixture)
        with torch.no_grad():
            for param in parallel.parameters():
                param.to_local().fill_(float("nan"))
            load_hf_into_model(Path(fixture[0]), parallel)
        if dist.get_rank() == 0:
            scratch.cleanup()

    tokens = torch.randint(config.vocab_size, (2, 512), device=device)
    labels = torch.randint(config.vocab_size, tokens.shape, device=device)
    labels[:, 3::11] = -100

    reference_logits = reference(tokens).flatten(0, 1)
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.float(), labels.flatten(), reduction="sum"
    )
    reference_loss.backward()
    parallel_logits = parallel(tokens)
    parallel_loss = tp.loss(parallel_logits, labels)
    parallel_loss.backward()
    tp.sync_replicated_grads(parallel)

    def rel_l2(actual, expected):
        error = (actual.float() - expected.float()).norm()
        return (error / expected.float().norm().clamp_min(1e-6)).item()

    expected_logits = reference_logits.chunk(tp.size, dim=-1)[tp.rank]
    logit_error = rel_l2(parallel_logits, expected_logits)
    loss_error = ((parallel_loss - reference_loss).abs() / reference_loss).item()
    worst = 0.0
    for (name, actual), (expected_name, expected) in zip(
        parallel.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == expected_name
        spec = tp.shards.get(name)
        expected_grad = spec.select(expected.grad) if spec else expected.grad
        error = rel_l2(actual.grad.to_local(), expected_grad)
        worst = max(worst, error)
        print(f"rank={tp.rank} grad {name} relative-L2={error:.6f}", flush=True)

    print(
        f"rank={tp.rank} TP={tp.size} async={args.async_tp} compiled={args.compile_blocks} "
        f"hf_roundtrip={args.hf_roundtrip} "
        f"reference_loss={reference_loss.item():.6f} tp_loss={parallel_loss.item():.6f} "
        f"loss_rel={loss_error:.6f} logits_rel_l2={logit_error:.6f} max_grad_rel_l2={worst:.6f}",
        flush=True,
    )
    assert loss_error < 0.001
    assert logit_error < 0.025
    assert worst < 0.06
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
