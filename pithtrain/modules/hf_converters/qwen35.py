"""Dense Qwen3.5 text-tower conversion; vision and MTP weights are intentionally excluded."""

import json
from logging import Logger
from pathlib import Path

import torch

from ._streaming import CopyOp, generic_plan_param


class Qwen35Converter:
    name = "qwen35"

    def detect_hf(self, load_path: Path) -> bool:
        config_path = Path(load_path, "config.json")
        if not config_path.exists():
            return False
        config = json.loads(config_path.read_text())
        text = config.get("text_config", config)
        return isinstance(text, dict) and text.get("model_type") == "qwen3_5_text"

    def detect_dcp(self, metadata) -> bool:
        keys = metadata.state_dict_metadata
        return any(".linear_attn." in k for k in keys) and any(".mlp.gate_proj." in k for k in keys)

    @staticmethod
    def _renest(canon: str) -> str:
        return canon if canon == "lm_head.weight" else "language_model." + canon

    def postprocess_canonical(self, canonical: dict[str, torch.Tensor], stdout: Logger):
        # Match the public multimodal checkpoint's text-tower namespace. As with the MoE
        # converter, conversion produces weight shards; the caller supplies the HF config.
        return {self._renest(key): value for key, value in canonical.items()}

    def plan_param(self, local_fqn, param, weight_map, named_modules):
        # hf_loader strips the leading model. prefix. Accept both the public multimodal
        # checkpoint and a standalone Qwen3_5ForCausalLM text checkpoint.
        def remap(canon):
            nested = self._renest(canon)
            return nested if nested in weight_map else canon

        tp = getattr(named_modules[""], "tensor_parallel", None)
        if tp is None or local_fqn not in tp.shards:
            return generic_plan_param(local_fqn, param, weight_map, named_modules, remap=remap)

        spec = tp.shards[local_fqn]
        key = remap(local_fqn)
        local = param.detach().to_local()
        chunk = -(-param.size(0) // tp.dp_group.size())
        start = min(tp.dp_group.rank() * chunk, param.size(0))
        if spec.dim == 1:
            (col,) = spec.spans
            return [
                CopyOp(
                    local_fqn,
                    local,
                    (key,),
                    (weight_map[key],),
                    (slice(start, start + local.size(0)), slice(col.start, col.stop)),
                    None,
                )
            ]

        # A DP shard may straddle the Q/K or K/V boundary of a packed TP shard.
        copies, segment_start = [], 0
        end = start + local.size(0)
        for source in spec.spans:
            lo = max(start, segment_start)
            hi = min(end, segment_start + len(source))
            if lo < hi:
                shift = source.start - segment_start
                copies.append(
                    CopyOp(
                        local_fqn,
                        local[lo - start : hi - start],
                        (key,),
                        (weight_map[key],),
                        slice(lo + shift, hi + shift),
                        None,
                    )
                )
            segment_start += len(source)
        return copies

    def expected_unmapped(self, hf_keys: set) -> set:
        return {key for key in hf_keys if key.startswith(("visual.", "mtp."))}
