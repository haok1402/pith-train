"""
PithTrain distributed module.
"""

import atexit
import os
import sys
from dataclasses import dataclass
from datetime import timedelta

import torch

from pithtrain.config import SlottedDefault
from pithtrain.contexts import distributed


@dataclass(init=False, slots=True)
class DistributedCfg(SlottedDefault):
    """
    Configuration for distributed runtime.
    """

    pipeline_parallel_size: int = 1
    """
    Degree of pipeline parallelism (PP).

    Partition the model layers across ranks; each rank holds a consecutive slice. Forward and
    backward execution is scheduled by DualPipeV.
    """

    context_parallel_size: int = 1
    """
    Degree of context parallelism (CP).

    Shard the sequence dimension across CP ranks. K/V exchange uses ring attention with a zigzag
    token layout.
    """

    expert_parallel_size: int = 1
    """
    Degree of expert parallelism (EP).

    Distribute the MoE experts across ranks; non-expert layers are unaffected. Token routing uses
    EP dispatch and combine kernels with token deduplication.
    """

    timeout: timedelta = timedelta(minutes=15)
    """
    Timeout for distributed operations.

    Passed to init_process_group, so it bounds every collective. Scale up for multi-node runs;
    keep small to fail fast.
    """

    hsdp_replica: int = 1
    """
    Number of replicas each FSDP shard group is split into.

    At 1, FSDP shards every parameter across the whole replica group for its class: the dp x cp
    stage for the attention parameters, the dp axis of the expert view for the expert parameters.
    Above 1, both of those groups split into this many replicas, so both must divide by it, and
    FSDP shards within one replica and all-reduces across them. Raise it when one replica already
    holds the model, trading memory for a cheaper gradient reduction.
    """


def setup_torch_runtime() -> None:
    """
    Apply the process-wide torch tuning that every launch path shares.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 64


def setup_default_process_group(cfg: DistributedCfg, device_id: int) -> None:
    """
    Create and own the default process group, and register its teardown at exit.

    The teardown runs only on a clean exit. On a crash the excepthook hard-exits first, because
    destroy_process_group shuts NCCL down collectively and would hang draining work that peers
    who already died will never satisfy.
    """
    kwargs = dict(backend="nccl", device_id=device_id, timeout=cfg.timeout)
    torch.distributed.init_process_group(**kwargs)
    atexit.register(torch.distributed.destroy_process_group)

    original = sys.excepthook

    def excepthook(exc_type, exc_value, exc_tb):
        try:
            original(exc_type, exc_value, exc_tb)
        except Exception:
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(1)

    sys.excepthook = excepthook


def setup_device_mesh(cfg: DistributedCfg, device_id: int) -> None:
    """
    Publish the rank and device for this process, then the attention and expert views of the ranks.

    This follows MoE parallel folding (https://arxiv.org/abs/2504.14960): attention and the experts
    each get their own mesh over the same ranks, (pp, dp, cp) for attention and (pp, dp, ep) for
    the experts. Both put pp first, so a rank holds the same pipeline stage either way, and both
    put their busiest axis last, so cp and ep groups are contiguous rank blocks. The attention
    dp_rank alone decides which data a rank loads.
    """
    distributed.rank = torch.distributed.get_rank()
    distributed.world_size = torch.distributed.get_world_size()
    distributed.device = torch.device("cuda", device_id)
    torch.cuda.set_device(distributed.device)

    pp_size = cfg.pipeline_parallel_size
    cp_size = cfg.context_parallel_size
    ep_size = cfg.expert_parallel_size

    world_size = distributed.world_size
    if world_size % pp_size != 0:
        raise RuntimeError(f"{world_size=} not divisible by {pp_size=}")
    stage_size = world_size // pp_size
    if stage_size % cp_size != 0:
        raise RuntimeError(f"{stage_size=} (world_size // pp_size) not divisible by {cp_size=}")
    if stage_size % ep_size != 0:
        raise RuntimeError(f"{stage_size=} (world_size // pp_size) not divisible by {ep_size=}")
    attn_dp_size = stage_size // cp_size
    expt_dp_size = stage_size // ep_size

    # Both views carry pp, so the pp communicator is built twice, at the cost of one extra
    # ncclCommSplit. Only the pp group on attn_mesh is ever read.
    init = torch.distributed.init_device_mesh
    attn_mesh = init("cuda", (pp_size, attn_dp_size, cp_size), mesh_dim_names=("pp", "dp", "cp"))
    expt_mesh = init("cuda", (pp_size, expt_dp_size, ep_size), mesh_dim_names=("pp", "dp", "ep"))
    distributed.attn_mesh, distributed.expt_mesh = attn_mesh, expt_mesh

    distributed.pp_size, distributed.pp_rank = pp_size, attn_mesh.get_local_rank("pp")
    distributed.pp_group = attn_mesh.get_group("pp")

    distributed.cp_size, distributed.cp_rank = cp_size, attn_mesh.get_local_rank("cp")
    distributed.cp_group = attn_mesh.get_group("cp")

    distributed.ep_size, distributed.ep_rank = ep_size, expt_mesh.get_local_rank("ep")
    distributed.ep_group = expt_mesh.get_group("ep")

    # No process group for either dp axis: FSDP reduces off a DeviceMesh, which both views
    # already provide. Only the attention dp is published, since it decides what a rank loads.
    distributed.dp_size, distributed.dp_rank = attn_dp_size, attn_mesh.get_local_rank("dp")


def setup_distributed(cfg: object) -> None:
    """
    Initialize the distributed runtime under torchrun.
    """
    assert hasattr(cfg, "distributed") and isinstance(cfg.distributed, DistributedCfg)
    setup_torch_runtime()
    device_id = int(os.environ["LOCAL_RANK"])
    setup_default_process_group(cfg.distributed, device_id)
    setup_device_mesh(cfg.distributed, device_id)
