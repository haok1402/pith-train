"""Interleaved 1F1B and split-backward schedules for dense Qwen3.5."""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache
from statistics import median
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining._p2p import _build_p2p_edge_groups, _preconnect_p2p_edge_groups
from torch.distributed.pipelining.schedules import (
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FULL_BACKWARD,
    RECV_B,
    RECV_F,
    SEND_B,
    SEND_F,
    WAIT_SEND_B,
    WAIT_SEND_F,
    F,
    _Action,
    _add_send_recv,
    _add_wait_send,
    _add_wait_send_budget,
    _get_1f1b_rank_ops,
    _get_warmup_ops,
)

from pithtrain.contexts import distributed, logging, training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.pipeline.dualpipev import Microbatch


def stage_to_rank(stage: int, pp_size: int, schedule: str) -> int:
    return 2 * pp_size - 1 - stage if schedule == "zbv" and stage >= pp_size else stage % pp_size


def _zbv_rank_actions(rank: int, pp_size: int, batches: int) -> list[_Action]:
    """ZB-V's two local chunks progress in opposite physical directions."""
    target = max(2 * pp_size - 1, batches)
    counters = Counter()
    actions = []

    def emit(operation, chunk):
        stage = rank if chunk == 0 else 2 * pp_size - 1 - rank
        index = counters[operation, chunk]
        if index < batches:
            actions.append(_Action(stage, operation, index))
        counters[operation, chunk] += 1

    for _ in range(2 * (pp_size - rank) - 1):
        emit(F, 0)
    for _ in range(rank):
        emit(F, 1)
        emit(F, 0)
    for _ in range(pp_size - rank):
        emit(F, 1)
        emit(BACKWARD_INPUT, 1)
        emit(BACKWARD_WEIGHT, 1)
    while counters[F, 1] < counters[F, 0] or counters[F, 0] < target:
        if counters[F, 0] < target:
            emit(F, 0)
        emit(BACKWARD_INPUT, 0)
        emit(BACKWARD_WEIGHT, 0)
        emit(F, 1)
        emit(BACKWARD_INPUT, 1)
        emit(BACKWARD_WEIGHT, 1)
    for _ in range(rank):
        emit(BACKWARD_INPUT, 0)
        emit(BACKWARD_INPUT, 1)
    for _ in range(pp_size - rank):
        emit(BACKWARD_INPUT, 0)
        emit(BACKWARD_WEIGHT, 0)
    for chunk in (1, 0):
        while counters[BACKWARD_WEIGHT, chunk] < counters[BACKWARD_INPUT, chunk]:
            emit(BACKWARD_WEIGHT, chunk)
    return actions


@lru_cache(maxsize=16)
def dense_pipeline_actions(
    schedule: str, pp_size: int, chunks: int, batches: int, train: bool
) -> dict[int, list[_Action]]:
    stage_count = pp_size * chunks

    def owner(stage):
        return stage_to_rank(stage, pp_size, schedule)

    if train and schedule == "zbv":
        actions = {rank: _zbv_rank_actions(rank, pp_size, batches) for rank in range(pp_size)}
    elif train:
        split_backward = schedule == "zb-h1"
        rounds = max(1, batches // pp_size)
        if schedule == "interleaved" and batches % rounds:
            raise ValueError(
                f"interleaved requires {batches} microbatches to divide evenly into {rounds} rounds"
            )
        group_size = batches // rounds
        actions = {}
        for rank in range(pp_size):
            warmup = _get_warmup_ops(
                rank,
                chunks,
                group_size,
                pp_size,
                batches,
                multiply_factor=1 if split_backward else 2,
            )
            actions[rank] = _get_1f1b_rank_ops(
                chunks,
                pp_size,
                warmup,
                chunks * batches - warmup,
                warmup,
                rank,
                lambda step: rank + (step // group_size % chunks) * pp_size,
                lambda step: (
                    rank + (chunks - 1 - ((step - warmup) // group_size % chunks)) * pp_size
                ),
                num_1f1b_microbatches=rank,
                enable_zero_bubble=split_backward,
            )
    else:
        actions = {
            rank: [
                _Action(stage, F, index)
                for index in range(batches)
                for stage in range(stage_count)
                if owner(stage) == rank
            ]
            for rank in range(pp_size)
        }
    transfers = _add_wait_send(_add_send_recv(actions, owner, stage_count))
    return _add_wait_send_budget(transfers, owner, max_outstanding_sends=2 * stage_count)


class ScheduledDensePipeline(nn.Module):
    """Execute a static schedule while retaining stock gradient and optimizer semantics."""

    def __init__(
        self,
        schedule: str,
        stages: Sequence[nn.Module],
        grad_sync_layers: tuple[FSDPModule, ...] = (),
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.split_backward = schedule in ("zb-h1", "zbv")
        self.module = nn.ModuleList(stages)
        self.stages = {stage.stage_index: stage for stage in stages}
        self.grad_sync_layers = grad_sync_layers
        self.hidden_size = stages[0].hidden_size
        self.pp_size = distributed.pp_size
        self.pp_rank = distributed.pp_rank
        self.pp_group = distributed.pp_group
        self.last_stage = self.pp_size * len(stages) - 1
        self.objective_pp_rank = stage_to_rank(self.last_stage, self.pp_size, schedule)
        self.peers = dist.get_process_group_ranks(self.pp_group)
        self.edge_groups = {}
        device = distributed.device
        if (
            self.pp_size > 1
            and device.type == "cuda"
            and self.pp_group._get_backend(device).supports_splitting
        ):
            # Eager NCCL otherwise puts all raw P2P traffic on the same communicator.
            dist.all_reduce(torch.zeros(1, dtype=torch.int32, device=device), group=self.pp_group)
            assignment = {
                stage: stage_to_rank(stage, self.pp_size, schedule)
                for stage in range(self.last_stage + 1)
            }
            self.edge_groups, rounds = _build_p2p_edge_groups(self.pp_group, assignment, device)
            _preconnect_p2p_edge_groups(self.pp_group, self.edge_groups, rounds, device)

    def step(
        self,
        microbatches: list[Microbatch],
        objective: Callable | None = None,
        *,
        profile_actions: bool = False,
    ) -> list[Any]:
        if training.offloaded and self is training.model:
            raise RuntimeError("training state is offloaded; call reload_training_state")
        if self.last_stage in self.stages and objective is None:
            raise ValueError("the last pipeline rank needs an objective")
        if not microbatches:
            return []

        train = torch.is_grad_enabled()
        actions = dense_pipeline_actions(
            self.schedule, self.pp_size, len(self.module), len(microbatches), train
        )[self.pp_rank]
        if train:
            for stage in self.module:
                if isinstance(stage, FSDPModule):
                    stage.set_requires_gradient_sync(False)
                    stage.set_reshard_after_backward(False)
                    stage.set_is_last_backward(False)
                    if self.split_backward:
                        stage.set_manual_backward_finalization(True)
            for layer in self.grad_sync_layers:
                layer.set_requires_gradient_sync(True, recurse=False)
        FP8WeightCacheControl.step()

        activations = {}
        outgoing = {}
        receives = {}
        sends = {}
        results = {}
        deferred_wgrads = {}
        reductions = []
        timings = []

        def start_timing():
            if profile_actions:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                return start
            return None

        def end_timing(start, stage, operation):
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                timings.append((stage, operation, start, end))

        def receive(operation, stage, index):
            tensor, work = receives.pop((operation, stage, index))
            work.wait()
            return tensor

        def take_input(receive_op, send_op, stage, index, source):
            if source in self.stages:
                return outgoing.pop((send_op, source, index)).detach()
            return receive(receive_op, stage, index)

        try:
            for stage_index, operation, index, *_ in actions:
                key = (stage_index, index)
                microbatch = microbatches[index]
                if operation in (SEND_F, SEND_B):
                    neighbor = stage_index + (1 if operation == SEND_F else -1)
                    peer = stage_to_rank(neighbor, self.pp_size, self.schedule)
                    group = self.edge_groups.get((self.pp_rank, peer), self.pp_group)
                    tensor = outgoing.pop((operation, *key)).detach().contiguous()
                    work = dist.isend(tensor, group=group, dst=self.peers[peer])
                    sends[operation, *key] = (tensor, work)
                elif operation in (RECV_F, RECV_B):
                    neighbor = stage_index + (-1 if operation == RECV_F else 1)
                    peer = stage_to_rank(neighbor, self.pp_size, self.schedule)
                    group = self.edge_groups.get((peer, self.pp_rank), self.pp_group)
                    (tokens,) = microbatch.model_inputs
                    tensor = torch.empty(
                        (*tokens.shape[:2], self.hidden_size),
                        dtype=training.PARAM_DTYPE,
                        device=distributed.device,
                    )
                    receives[operation, *key] = (
                        tensor,
                        dist.irecv(tensor, group=group, src=self.peers[peer]),
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
                        activation = take_input(
                            RECV_F, SEND_F, *key, stage_index - 1
                        ).requires_grad_(train)
                    cu_seqlens = microbatch.cu_seqlens
                    if cu_seqlens is not None:
                        cu_seqlens = cu_seqlens.to(device=distributed.device, dtype=torch.int32)
                    tick = start_timing()
                    output = self.stages[stage_index](activation, cu_seqlens=cu_seqlens)
                    if stage_index == self.last_stage:
                        output, results[index] = objective((output,), microbatch.objective_inputs)
                    else:
                        outgoing[SEND_F, *key] = output
                    if train:
                        activations[key] = (activation, output)
                    end_timing(tick, stage_index, "F")
                    del activation, output
                elif operation in (FULL_BACKWARD, BACKWARD_INPUT):
                    activation, output = activations.pop(key)
                    gradient = None
                    if stage_index != self.last_stage:
                        gradient = take_input(RECV_B, SEND_B, *key, stage_index + 1)
                    stage = self.stages[stage_index]
                    split = operation == BACKWARD_INPUT
                    if (
                        not split
                        and isinstance(stage, FSDPModule)
                        and index == len(microbatches) - 1
                    ):
                        stage.set_requires_gradient_sync(True)
                        stage.set_reshard_after_backward(True)
                        stage.set_is_last_backward(True)
                    if split:
                        WeightGradStore.enabled = True
                    tick = start_timing()
                    try:
                        with torch.autograd.set_multithreading_enabled(False):
                            torch.autograd.backward(output, gradient)
                    finally:
                        if split:
                            WeightGradStore.enabled = False
                    if split:
                        deferred_wgrads[key] = WeightGradStore.take()
                    if stage_index > 0:
                        outgoing[SEND_B, *key] = activation.grad
                    end_timing(tick, stage_index, "B")
                    del activation, output, gradient
                elif operation == BACKWARD_WEIGHT:
                    tick = start_timing()
                    with torch.no_grad():
                        for accumulate in deferred_wgrads.pop(key):
                            accumulate()
                            del accumulate
                    stage = self.stages[stage_index]
                    if isinstance(stage, FSDPModule) and index == len(microbatches) - 1:
                        stage.set_requires_gradient_sync(True)
                        stage.set_reshard_after_backward(True)
                        reductions.append(stage.finalize_backward(async_op=True))
                    end_timing(tick, stage_index, "W")
                else:
                    raise RuntimeError(f"Unexpected dense pipeline operation: {operation}")
            for reduction in reductions:
                reduction.wait()
            if not train:
                for module in self.module.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
            if timings:
                torch.cuda.synchronize(distributed.device)
                by_action = defaultdict(list)
                for stage, operation, start, end in timings:
                    by_action[stage, operation].append(start.elapsed_time(end))
                summary = [
                    (stage, op, median(times), sum(times))
                    for (stage, op), times in sorted(by_action.items())
                ]
                summaries = [None] * dist.get_world_size()
                dist.all_gather_object(summaries, summary)
                if dist.get_rank() == 0:
                    for rank, rows in enumerate(summaries):
                        for stage, op, p50, total in rows:
                            logging.stdout.info(
                                "pipeline-action rank=%d stage=%d action=%s median_ms=%.3f total_ms=%.1f",
                                rank,
                                stage,
                                op,
                                p50,
                                total,
                            )
            return [results[index] for index in range(len(results))]
        except torch.OutOfMemoryError as error:
            if distributed.device.type == "cuda":
                retries = torch.cuda.memory_stats()["num_alloc_retries"]
                error.add_note(
                    f"Pipeline rank {self.pp_rank}, stage {stage_index}, {operation} on "
                    f"microbatch {index}. Live activation sets {len(activations)}, "
                    f"deferred W sets {len(deferred_wgrads)}, num_alloc_retries_total {retries}."
                )
            raise
        finally:
            training.current_microbatch = None
            if self.split_backward:
                WeightGradStore.take()
            FP8WeightCacheControl.clear(self)
