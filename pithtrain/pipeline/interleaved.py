"""Interleaved dense 1F1B using PyTorch's schedule planner and PithTrain FSDP hooks."""

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining.schedules import (
    F,
    FULL_BACKWARD,
    RECV_B,
    RECV_F,
    SEND_B,
    SEND_F,
    WAIT_SEND_B,
    WAIT_SEND_F,
    _Action,
    _add_send_recv,
    _add_wait_send,
    _get_1f1b_rank_ops,
    _get_warmup_ops,
)

from pithtrain.contexts import distributed, training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.pipeline.dualpipev import Microbatch


@lru_cache(maxsize=16)
def interleaved_actions(
    pp_size: int, chunks: int, batches: int, train: bool
) -> dict[int, list[_Action]]:
    stage_count = pp_size * chunks
    if train:
        rounds = max(1, batches // pp_size)
        if batches % rounds:
            raise ValueError(
                f"interleaved requires {batches} microbatches to divide evenly into {rounds} rounds"
            )
        group_size = batches // rounds
        actions = {}
        for rank in range(pp_size):
            warmup = _get_warmup_ops(rank, chunks, group_size, pp_size, batches)
            actions[rank] = _get_1f1b_rank_ops(
                chunks,
                pp_size,
                warmup,
                chunks * batches - warmup,
                warmup,
                rank,
                lambda step: rank + (step // group_size % chunks) * pp_size,
                lambda step: rank + (chunks - 1 - ((step - warmup) // group_size % chunks)) * pp_size,
            )
    else:
        actions = {
            rank: [
                _Action(stage, F, index)
                for index in range(batches)
                for stage in range(rank, stage_count, pp_size)
            ]
            for rank in range(pp_size)
        }
    return _add_wait_send(_add_send_recv(actions, lambda stage: stage % pp_size, stage_count))


class InterleavedPipeline(nn.Module):
    """Runs multiple consecutive model segments per rank with depth-first 1F1B."""

    def __init__(
        self, stages: Sequence[nn.Module], grad_sync_layers: tuple[FSDPModule, ...] = ()
    ) -> None:
        super().__init__()
        self.module = nn.ModuleList(stages)
        self.stages = {stage.stage_index: stage for stage in stages}
        self.grad_sync_layers = grad_sync_layers
        self.hidden_size = stages[0].hidden_size
        self.pp_size = distributed.pp_size
        self.last_stage = self.pp_size * len(stages) - 1
        self.objective_pp_rank = self.pp_size - 1

    def step(
        self, microbatches: list[Microbatch], objective: Callable | None = None
    ) -> list[Any]:
        if training.offloaded and self is training.model:
            raise RuntimeError("training state is offloaded; call reload_training_state")
        if self.last_stage in self.stages and objective is None:
            raise ValueError("the last pipeline rank needs an objective")
        if not microbatches:
            return []

        train = torch.is_grad_enabled()
        actions = interleaved_actions(
            self.pp_size, len(self.module), len(microbatches), train
        )[distributed.pp_rank]
        if train:
            for stage in self.module:
                if isinstance(stage, FSDPModule):
                    stage.set_requires_gradient_sync(False)
                    stage.set_reshard_after_backward(False)
                    stage.set_is_last_backward(False)
            for layer in self.grad_sync_layers:
                layer.set_requires_gradient_sync(True, recurse=False)
        FP8WeightCacheControl.step()

        activations = {}
        outgoing = {}
        receives = {}
        sends = {}
        results = {}
        group = distributed.pp_group

        def receive(operation, stage, index):
            tensor, work = receives.pop((operation, stage, index))
            work.wait()
            return tensor

        try:
            for stage_index, operation, index, *_ in actions:
                key = (stage_index, index)
                microbatch = microbatches[index]
                if operation in (SEND_F, SEND_B):
                    peer = (stage_index + (1 if operation == SEND_F else -1)) % self.pp_size
                    tensor = outgoing.pop((operation, *key)).detach().contiguous()
                    work = dist.isend(tensor, group=group, group_dst=peer)
                    sends[operation, *key] = (tensor, work)
                elif operation in (RECV_F, RECV_B):
                    peer = (stage_index + (-1 if operation == RECV_F else 1)) % self.pp_size
                    (tokens,) = microbatch.model_inputs
                    tensor = torch.empty(
                        (*tokens.shape[:2], self.hidden_size),
                        dtype=training.PARAM_DTYPE,
                        device=distributed.device,
                    )
                    receives[operation, *key] = (
                        tensor,
                        dist.irecv(tensor, group=group, group_src=peer),
                    )
                elif operation in (WAIT_SEND_F, WAIT_SEND_B):
                    send = SEND_F if operation == WAIT_SEND_F else SEND_B
                    tensor, work = sends.pop((send, *key))
                    work.wait()
                    del tensor
                elif operation == F:
                    training.current_microbatch = index
                    if stage_index == 0:
                        (activation,) = microbatch.model_inputs
                    else:
                        activation = receive(RECV_F, *key).requires_grad_(train)
                    cu_seqlens = microbatch.cu_seqlens
                    if cu_seqlens is not None:
                        cu_seqlens = cu_seqlens.to(device=distributed.device, dtype=torch.int32)
                    output = self.stages[stage_index](activation, cu_seqlens=cu_seqlens)
                    if stage_index == self.last_stage:
                        output, results[index] = objective((output,), microbatch.objective_inputs)
                    else:
                        outgoing[SEND_F, *key] = output
                    if train:
                        activations[key] = (activation, output)
                    del activation, output
                elif operation == FULL_BACKWARD:
                    activation, output = activations.pop(key)
                    gradient = None if stage_index == self.last_stage else receive(RECV_B, *key)
                    stage = self.stages[stage_index]
                    if isinstance(stage, FSDPModule) and index == len(microbatches) - 1:
                        stage.set_requires_gradient_sync(True)
                        stage.set_reshard_after_backward(True)
                        stage.set_is_last_backward(True)
                    with torch.autograd.set_multithreading_enabled(False):
                        torch.autograd.backward(output, gradient)
                    if stage_index > 0:
                        outgoing[SEND_B, *key] = activation.grad
                    del activation, output, gradient
                else:
                    raise RuntimeError(f"Unexpected interleaved schedule operation: {operation}")
            if not train:
                for module in self.module.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
            return [results[index] for index in range(len(results))]
        finally:
            training.current_microbatch = None
            FP8WeightCacheControl.clear(self)
