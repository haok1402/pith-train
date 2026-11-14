"""
Static memory profiling: model parameters, FSDP sharding, and optimizer states.

Replicates the layer distribution logic from Qwen3MoeModel and the FSDP sharding
topology from apply_fsdp, without importing any PithTrain code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tensor_spec import MemoryBucket, TensorSpec


@dataclass(slots=True)
class ModelConfig:
    """Parsed model configuration (from HuggingFace config JSON)."""

    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    decoder_sparse_step: int
    mlp_only_layers: list[int]

    @staticmethod
    def from_json(path: str | Path) -> "ModelConfig":
        with open(path) as f:
            cfg = json.load(f)
        hidden_size = cfg["hidden_size"]
        num_attention_heads = cfg["num_attention_heads"]
        model_type = cfg.get("model_type", "qwen3_moe")
        intermediate_size = cfg.get("intermediate_size", 4 * hidden_size)
        # gpt-oss names its expert count num_local_experts and sizes its experts by
        # intermediate_size.
        num_experts = cfg.get("num_experts", cfg.get("n_routed_experts"))
        if num_experts is None:
            num_experts = cfg.get("num_local_experts", 0)
        moe_intermediate_size = intermediate_size if model_type == "gpt_oss" else 0
        return ModelConfig(
            model_type=model_type,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=cfg.get("num_key_value_heads", num_attention_heads),
            head_dim=cfg.get("head_dim", hidden_size // num_attention_heads),
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0),
            moe_intermediate_size=cfg.get("moe_intermediate_size", moe_intermediate_size),
            num_hidden_layers=cfg["num_hidden_layers"],
            vocab_size=cfg["vocab_size"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            decoder_sparse_step=cfg.get("decoder_sparse_step", 1),
            mlp_only_layers=cfg.get("mlp_only_layers", []),
        )


@dataclass(slots=True)
class ParallelismConfig:
    """
    Parallelism and training configuration.

    dp_size is the attention data-parallel degree, how many ways the global batch is split. It
    and cp_size factor the ranks of one pipeline stage; ep_size re-factors the same ranks against
    expt_dp_size.

    attn_sharding_stage and expert_sharding_stage are the FSDP sharding stages (ZeRO-1/2/3) of
    DistributedCfg: the first covers every parameter outside the routed experts, sharded over the
    dp x cp group, the second the routed expert weights, sharded over the expert-dp group.
    """

    pp_size: int
    ep_size: int
    dp_size: int
    cp_size: int
    micro_batch_size: int
    global_batch_size: int
    sequence_length: int
    fp8_training: str = "disabled"
    attn_sharding_stage: int = 1
    expert_sharding_stage: int = 1

    def __post_init__(self) -> None:
        for name in ("attn_sharding_stage", "expert_sharding_stage"):
            if getattr(self, name) not in (1, 2, 3):
                raise ValueError(f"{name} must be 1, 2 or 3, got {getattr(self, name)}")
        if self.stage_size % self.ep_size != 0:
            raise ValueError(
                f"dp_size * cp_size ({self.stage_size}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )

    @property
    def stage_size(self) -> int:
        """
        Ranks holding one pipeline stage: dp * cp == ep * expt_dp.
        """
        return self.dp_size * self.cp_size

    @property
    def expt_dp_size(self) -> int:
        """
        Expert data parallel: how many replicas of the sharded expert set exist.
        """
        return self.stage_size // self.ep_size


def compute_layer_distribution(num_hidden_layers: int, num_stages: int) -> list[int]:
    """
    Distribute decoder layers across pipeline stages.

    Matches the algorithm in pithtrain/pipeline/dualpipev.py:
    even base allocation, remainder layers go to inner stages first
    (edges stay light to compensate for embed/head overhead).
    """
    assert num_hidden_layers >= num_stages >= 1
    base, remainder = divmod(num_hidden_layers, num_stages)
    layers = [base] * num_stages
    for _ in range(remainder):
        min_val = min(layers)
        best = None
        for i in range(1, num_stages - 1):
            if layers[i] == min_val:
                best = i
                break
        if best is None:
            for i in range(num_stages):
                if layers[i] == min_val:
                    best = i
                    break
        layers[best] += 1
    return layers


def layer_is_moe(
    layer_idx: int,
    num_experts: int,
    decoder_sparse_step: int,
    mlp_only_layers: list[int],
) -> bool:
    """Whether a given layer uses MoE (vs dense MLP)."""
    if num_experts <= 0:
        return False
    if (layer_idx + 1) % decoder_sparse_step != 0:
        return False
    if layer_idx in mlp_only_layers:
        return False
    return True


@dataclass(slots=True)
class ShardUnit:
    """
    One FSDP unit of a module: the parameters that gather, reduce and free together.

    apply_fsdp shards each decoder layer's routed experts as one unit (expert=True), the rest of
    the layer as another, and embed_tokens, the final norm and lm_head as one unit each
    (decoder_layer=False).
    """

    name: str
    expert: bool
    decoder_layer: bool
    specs: list[TensorSpec]

    @property
    def numel(self) -> int:
        return sum(spec.numel for spec in self.specs)


class ModelMemoryProfile:
    """Computes static parameter and optimizer memory for a given rank."""

    def __init__(self, model_cfg: ModelConfig, parallel_cfg: ParallelismConfig, pp_rank: int):
        self.model_cfg = model_cfg
        self.parallel_cfg = parallel_cfg
        self.pp_rank = pp_rank

        pp_size = parallel_cfg.pp_size
        num_stages = pp_size * 2
        self.num_stages = num_stages

        # Layer distribution
        self.layer_dist = compute_layer_distribution(model_cfg.num_hidden_layers, num_stages)

        # Module assignments (V-shape)
        self.module0_stage = pp_rank
        self.module1_stage = num_stages - 1 - pp_rank

        # Layer ranges for each module
        self.module0_layers = self._layer_range(self.module0_stage)
        self.module1_layers = self._layer_range(self.module1_stage)

        # Derived
        self.experts_per_rank = (
            model_cfg.num_experts // parallel_cfg.ep_size if model_cfg.num_experts > 0 else 0
        )
        self.is_first_pp_rank = pp_rank == 0
        self.is_last_pp_rank = pp_rank == pp_size - 1

        # module[0] has embed_tokens if stage_id == 0
        # module[1] has embed_tokens if its stage_id == 0
        # module[0] has norm/lm_head if stage_id == num_stages - 1
        # module[1] has norm/lm_head if its stage_id == num_stages - 1
        self.module0_has_embed = self.module0_stage == 0
        self.module1_has_embed = self.module1_stage == 0
        self.module0_has_head = self.module0_stage == num_stages - 1
        self.module1_has_head = self.module1_stage == num_stages - 1

    def _layer_range(self, stage_id: int) -> list[int]:
        """Return global layer indices for a given stage."""
        start = sum(self.layer_dist[:stage_id])
        count = self.layer_dist[stage_id]
        return list(range(start, start + count))

    def _layer_params(self, layer_idx: int, prefix: str) -> list[TensorSpec]:
        """Enumerate all parameters for one decoder layer."""
        cfg = self.model_cfg
        H = cfg.hidden_size
        nheads = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        hd = cfg.head_dim

        specs: list[TensorSpec] = []
        gpt_oss = cfg.model_type == "gpt_oss"

        # Attention parameters
        specs.append(TensorSpec(f"{prefix}.self_attn.q_proj.weight", (nheads * hd, H), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.k_proj.weight", (nkv * hd, H), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.v_proj.weight", (nkv * hd, H), "bf16"))
        specs.append(TensorSpec(f"{prefix}.self_attn.o_proj.weight", (H, nheads * hd), "bf16"))
        if gpt_oss:
            # gpt-oss: biased projections and per-head attention sinks, no q/k norms
            specs.append(TensorSpec(f"{prefix}.self_attn.q_proj.bias", (nheads * hd,), "bf16"))
            specs.append(TensorSpec(f"{prefix}.self_attn.k_proj.bias", (nkv * hd,), "bf16"))
            specs.append(TensorSpec(f"{prefix}.self_attn.v_proj.bias", (nkv * hd,), "bf16"))
            specs.append(TensorSpec(f"{prefix}.self_attn.o_proj.bias", (H,), "bf16"))
            specs.append(TensorSpec(f"{prefix}.self_attn.sinks", (nheads,), "bf16"))
        else:
            specs.append(TensorSpec(f"{prefix}.self_attn.q_norm.weight", (hd,), "bf16"))
            specs.append(TensorSpec(f"{prefix}.self_attn.k_norm.weight", (hd,), "bf16"))

        # Layer norms
        specs.append(TensorSpec(f"{prefix}.input_layernorm.weight", (H,), "bf16"))
        specs.append(TensorSpec(f"{prefix}.post_attention_layernorm.weight", (H,), "bf16"))

        # MLP / MoE
        is_moe = layer_is_moe(
            layer_idx,
            cfg.num_experts,
            cfg.decoder_sparse_step,
            cfg.mlp_only_layers,
        )
        if is_moe and gpt_oss:
            E_local = self.experts_per_rank
            inter = cfg.moe_intermediate_size
            E = cfg.num_experts
            # GptOssTopKRouter and GptOssExperts: fused gate_up and down weights with biases
            specs.append(TensorSpec(f"{prefix}.mlp.router.weight", (E, H), "bf16"))
            specs.append(TensorSpec(f"{prefix}.mlp.router.bias", (E,), "bf16"))
            experts = f"{prefix}.mlp.experts"
            specs.append(TensorSpec(f"{experts}.gate_up_proj", (E_local, 2 * inter, H), "bf16"))
            specs.append(TensorSpec(f"{experts}.gate_up_proj_bias", (E_local, 2 * inter), "bf16"))
            specs.append(TensorSpec(f"{experts}.down_proj", (E_local, H, inter), "bf16"))
            specs.append(TensorSpec(f"{experts}.down_proj_bias", (E_local, H), "bf16"))
        elif is_moe:
            E_local = self.experts_per_rank
            moe_inter = cfg.moe_intermediate_size
            # Gate
            specs.append(TensorSpec(f"{prefix}.mlp.gate.weight", (cfg.num_experts, H), "bf16"))
            # Experts (stacked per rank)
            specs.append(
                TensorSpec(
                    f"{prefix}.mlp.experts.gate_proj.weight",
                    (E_local, moe_inter, H),
                    "bf16",
                )
            )
            specs.append(
                TensorSpec(
                    f"{prefix}.mlp.experts.up_proj.weight",
                    (E_local, moe_inter, H),
                    "bf16",
                )
            )
            specs.append(
                TensorSpec(
                    f"{prefix}.mlp.experts.down_proj.weight",
                    (E_local, H, moe_inter),
                    "bf16",
                )
            )
        else:
            inter = cfg.intermediate_size
            specs.append(TensorSpec(f"{prefix}.mlp.gate_proj.weight", (inter, H), "bf16"))
            specs.append(TensorSpec(f"{prefix}.mlp.up_proj.weight", (inter, H), "bf16"))
            specs.append(TensorSpec(f"{prefix}.mlp.down_proj.weight", (H, inter), "bf16"))

        return specs

    def _is_expert_param(self, spec: TensorSpec) -> bool:
        return ".mlp.experts." in spec.name

    def sharding_stage(self, expert: bool) -> int:
        """
        FSDP sharding stage of a parameter class: the routed experts or everything else.
        """
        pcfg = self.parallel_cfg
        return pcfg.expert_sharding_stage if expert else pcfg.attn_sharding_stage

    def gathers_per_layer(self, expert: bool) -> bool:
        """
        Whether a parameter class keeps its bf16 unsharded weights only around their use.

        Stage 3 frees a unit's unsharded weights after each forward and backward. A shard world
        of 1 holds the full parameter as its shard, so its weights stay resident as at stage 1.
        """
        world = self.parallel_cfg.expt_dp_size if expert else self.parallel_cfg.stage_size
        return self.sharding_stage(expert) == 3 and world > 1

    def _fsdp_shard_world(self, spec: TensorSpec) -> int:
        """
        FSDP shard world size for a parameter: the size of the replica group it lives in.

        Expert weights replicate over expt_dp = stage // ep, everything else over the whole
        dp x cp stage.
        """
        pcfg = self.parallel_cfg
        return pcfg.expt_dp_size if self._is_expert_param(spec) else pcfg.stage_size

    def _fsdp_shard_size(self, spec: TensorSpec) -> int:
        """Number of FSDP-sharded elements on this rank."""
        shard_world = self._fsdp_shard_world(spec)
        # FSDP shards the flattened parameter; each rank holds ceil(numel / world)
        return (spec.numel + shard_world - 1) // shard_world

    def compute_shard_units(self, module_idx: int) -> list[ShardUnit]:
        """
        Enumerate the FSDP units of one module, in execution order, with their bf16 parameters.
        """
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
        has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head
        cfg = self.model_cfg
        H = cfg.hidden_size
        m = f"m[{module_idx}]"

        units: list[ShardUnit] = []
        if has_embed:
            spec = TensorSpec(f"{m}.embed_tokens.weight", (cfg.vocab_size, H), "bf16")
            units.append(ShardUnit(f"{m}.embed_tokens", False, False, [spec]))
        for layer_idx in layers:
            prefix = f"{m}.layer.{layer_idx}"
            specs = self._layer_params(layer_idx, prefix)
            attn = [spec for spec in specs if not self._is_expert_param(spec)]
            experts = [spec for spec in specs if self._is_expert_param(spec)]
            units.append(ShardUnit(prefix, False, True, attn))
            if experts:
                units.append(ShardUnit(f"{prefix}.mlp.experts", True, True, experts))
        if has_head:
            norm = TensorSpec(f"{m}.norm.weight", (H,), "bf16")
            units.append(ShardUnit(f"{m}.norm", False, False, [norm]))
            lm_head = TensorSpec(f"{m}.lm_head.weight", (cfg.vocab_size, H), "bf16")
            units.append(ShardUnit(f"{m}.lm_head", False, False, [lm_head]))
        return units

    def compute_module_params(self, module_idx: int) -> MemoryBucket:
        """
        Compute unsharded parameter memory for one module.

        During the DualPipeV pipeline step, FSDP layers have reshard_after_forward=False,
        so the full unsharded parameters are in memory. A parameter class at sharding stage 3
        (see gathers_per_layer) is left out: the simulator counts its gathered layers per event.
        """
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        stage_id = self.module0_stage if module_idx == 0 else self.module1_stage
        has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
        has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

        bucket = MemoryBucket(f"module[{module_idx}] params (stage {stage_id})")

        cfg = self.model_cfg
        H = cfg.hidden_size

        attn_resident = not self.gathers_per_layer(expert=False)
        expert_resident = not self.gathers_per_layer(expert=True)

        if has_embed and attn_resident:
            bucket.add(f"module[{module_idx}].embed_tokens.weight", (cfg.vocab_size, H), "bf16")

        for layer_idx in layers:
            prefix = f"module[{module_idx}].layer.{layer_idx}"
            for spec in self._layer_params(layer_idx, prefix):
                resident = expert_resident if self._is_expert_param(spec) else attn_resident
                if resident:
                    bucket.specs.append(spec)

        if has_head and attn_resident:
            bucket.add(f"module[{module_idx}].norm.weight", (H,), "bf16")
            bucket.add(f"module[{module_idx}].lm_head.weight", (cfg.vocab_size, H), "bf16")

        # Rotary embedding buffers (not parameters but in memory)
        # cos_cached, sin_cached: [max_pos, head_dim] - relatively small
        max_pos = (
            getattr(cfg, "max_position_embeddings", 40960)
            if hasattr(cfg, "max_position_embeddings")
            else 40960
        )
        bucket.add(
            f"module[{module_idx}].rotary_emb.cos_cached",
            (max_pos, cfg.head_dim),
            "bf16",
        )
        bucket.add(
            f"module[{module_idx}].rotary_emb.sin_cached",
            (max_pos, cfg.head_dim),
            "bf16",
        )

        return bucket

    def compute_fsdp_sharded_params(self) -> MemoryBucket:
        """
        Compute the persistent FSDP-sharded parameter memory.

        FSDP keeps sharded copies alongside unsharded. During the pipeline step,
        both exist in memory for layers with reshard_after_forward=False.

        When FSDP mesh_size=1 (e.g., expert params with dp=1), the sharded copy
        IS the unsharded copy - same tensor, no duplication. We skip these to
        avoid double-counting with params_unsharded.
        """
        bucket = MemoryBucket("FSDP sharded params (persistent)")
        for module_idx in range(2):
            layers = self.module0_layers if module_idx == 0 else self.module1_layers
            has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
            has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

            cfg = self.model_cfg
            H = cfg.hidden_size

            all_specs: list[TensorSpec] = []
            if has_embed:
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].embed_tokens.weight", (cfg.vocab_size, H), "bf16")
                )
            for layer_idx in layers:
                prefix = f"m[{module_idx}].layer.{layer_idx}"
                all_specs.extend(self._layer_params(layer_idx, prefix))
            if has_head:
                all_specs.append(TensorSpec(f"m[{module_idx}].norm.weight", (H,), "bf16"))
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].lm_head.weight", (cfg.vocab_size, H), "bf16")
                )

            for spec in all_specs:
                shard_world = self._fsdp_shard_world(spec)
                if shard_world <= 1:
                    # mesh_size=1: sharded = full param, already counted in params_unsharded
                    continue
                shard_numel = self._fsdp_shard_size(spec)
                bucket.add(
                    f"shard({spec.name})",
                    (shard_numel,),
                    "bf16",
                )
        return bucket

    def compute_optimizer_states(self) -> MemoryBucket:
        """
        Compute Adam optimizer state memory (exp_avg + exp_avg_sq in fp32).

        Optimizer operates on FSDP-sharded parameters.
        """
        bucket = MemoryBucket("Optimizer states (Adam m+v, fp32)")
        for module_idx in range(2):
            layers = self.module0_layers if module_idx == 0 else self.module1_layers
            has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
            has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

            cfg = self.model_cfg
            H = cfg.hidden_size

            all_specs: list[TensorSpec] = []
            if has_embed:
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].embed_tokens.weight", (cfg.vocab_size, H), "bf16")
                )
            for layer_idx in layers:
                prefix = f"m[{module_idx}].layer.{layer_idx}"
                all_specs.extend(self._layer_params(layer_idx, prefix))
            if has_head:
                all_specs.append(TensorSpec(f"m[{module_idx}].norm.weight", (H,), "bf16"))
                all_specs.append(
                    TensorSpec(f"m[{module_idx}].lm_head.weight", (cfg.vocab_size, H), "bf16")
                )

            for spec in all_specs:
                shard_numel = self._fsdp_shard_size(spec)
                # exp_avg (m) in fp32
                bucket.add(f"adam_m({spec.name})", (shard_numel,), "fp32")
                # exp_avg_sq (v) in fp32
                bucket.add(f"adam_v({spec.name})", (shard_numel,), "fp32")
        return bucket

    def compute_gradient_bucket(self, module_idx: int) -> MemoryBucket:
        """
        Compute the gradient memory for one module (all layers).

        During the pipeline step with FSDP's set_requires_gradient_sync(False),
        gradients accumulate on the full unsharded parameter in fp32: the fp32 main_grad of the
        linear and raw expert weights, and a .grad whose grad_dtype FSDP2 sets to the fp32
        reduce dtype for the rest.
        """
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        has_embed = self.module0_has_embed if module_idx == 0 else self.module1_has_embed
        has_head = self.module0_has_head if module_idx == 0 else self.module1_has_head

        bucket = MemoryBucket(f"module[{module_idx}] gradients")

        cfg = self.model_cfg
        H = cfg.hidden_size

        if has_embed:
            bucket.add(f"grad(m[{module_idx}].embed_tokens.weight)", (cfg.vocab_size, H), "fp32")

        for layer_idx in layers:
            prefix = f"grad(m[{module_idx}].layer.{layer_idx})"
            for spec in self._layer_params(layer_idx, prefix):
                bucket.add(spec.name, spec.shape, "fp32")

        if has_head:
            bucket.add(f"grad(m[{module_idx}].norm.weight)", (H,), "fp32")
            bucket.add(f"grad(m[{module_idx}].lm_head.weight)", (cfg.vocab_size, H), "fp32")

        return bucket

    def compute_sharded_gradient_bucket(self, module_idx: int, expert: bool) -> MemoryBucket:
        """
        Compute the fp32 sharded gradients of one parameter class of one module.

        At sharding stages 2 and 3 each reduce-scatter accumulates into a gradient of
        ceil(numel / shard world) fp32 elements per parameter, which lives until the optimizer
        step. At stage 1 the single reduce-scatter runs after the pipeline step, so the simulator
        does not count it there.
        """
        bucket = MemoryBucket(f"module[{module_idx}] sharded gradients")
        for unit in self.compute_shard_units(module_idx):
            if unit.expert != expert:
                continue
            for spec in unit.specs:
                bucket.add(f"grad_shard({spec.name})", (self._fsdp_shard_size(spec),), "fp32")
        return bucket

    def get_module_layer_count(self, module_idx: int) -> int:
        layers = self.module0_layers if module_idx == 0 else self.module1_layers
        return len(layers)

    def get_module_layer_indices(self, module_idx: int) -> list[int]:
        return self.module0_layers if module_idx == 0 else self.module1_layers

    def module_has_embed(self, module_idx: int) -> bool:
        return self.module0_has_embed if module_idx == 0 else self.module1_has_embed

    def module_has_head(self, module_idx: int) -> bool:
        return self.module0_has_head if module_idx == 0 else self.module1_has_head

    def get_total_param_count(self) -> int:
        """Total parameter count across both modules (unsharded, for display)."""
        total = 0
        for module_idx in range(2):
            bucket = self.compute_module_params(module_idx)
            for spec in bucket.specs:
                if "rotary_emb" not in spec.name:
                    total += spec.numel
        return total
