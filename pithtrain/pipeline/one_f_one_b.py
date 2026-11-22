from collections import deque
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule

from pithtrain.contexts import distributed, training
from pithtrain.contexts.fp8_weight_cache import FP8WeightCacheControl
from pithtrain.pipeline.dualpipev import Microbatch


class OneForwardOneBackward(nn.Module):
    """One contiguous stage per rank, with FIFO activation lifetimes."""

    def __init__(self, stage: nn.Module) -> None:
        super().__init__()
        self.module = nn.ModuleList([stage])
        self.hidden_size = stage.hidden_size
        self.rank = distributed.pp_rank
        self.objective_pp_rank = distributed.pp_size - 1
        group = distributed.pp_group
        self.prev = dist.get_global_rank(group, self.rank - 1) if self.rank > 0 else None
        self.next = (
            dist.get_global_rank(group, self.rank + 1)
            if self.rank < self.objective_pp_rank
            else None
        )

    def _exchange(
        self,
        peer: int | None,
        send: torch.Tensor | None = None,
        receive_for: Microbatch | None = None,
    ) -> torch.Tensor | None:
        if peer is None:
            return None
        if receive_for is None:
            if send is not None:
                dist.isend(send.detach().contiguous(), peer).wait()
            return None

        (tokens,) = receive_for.model_inputs
        incoming = torch.empty(
            (*tokens.shape, self.hidden_size),
            dtype=training.PARAM_DTYPE,
            device=distributed.device,
        )
        if send is None:
            dist.irecv(incoming, peer).wait()
        else:
            # Both sides must post the activation and its opposite-direction gradient together.
            ops = [
                dist.P2POp(dist.isend, send.detach().contiguous(), peer),
                dist.P2POp(dist.irecv, incoming, peer),
            ]
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        return incoming

    def step(
        self,
        microbatches: list[Microbatch],
        objective: Callable | None = None,
    ) -> list[Any]:
        """Run forward-only under no_grad, or train with a loss on the last pipeline rank."""
        if training.offloaded and self is training.model:
            raise RuntimeError("training state is offloaded; call reload_training_state")
        if self.next is None and objective is None:
            raise ValueError("the last pipeline rank needs an objective")
        if not microbatches:
            return []

        stage = self.module[0]
        train = torch.is_grad_enabled()
        fsdp = stage if isinstance(stage, FSDPModule) else None
        if fsdp is not None and train:
            fsdp.set_requires_gradient_sync(False)
            fsdp.set_reshard_after_backward(False)
            fsdp.set_is_last_backward(False)
        FP8WeightCacheControl.step()

        outputs = []
        pending = deque()

        def forward(index: int, activation: torch.Tensor | None) -> torch.Tensor:
            microbatch = microbatches[index]
            training.current_microbatch = index
            if activation is None:
                (stage_input,) = microbatch.model_inputs
            else:
                stage_input = activation.requires_grad_(train)
            cu_seqlens = microbatch.cu_seqlens
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.to(device=distributed.device, dtype=torch.int32)
            output = stage(stage_input, cu_seqlens=cu_seqlens)
            if self.next is None:
                output, result = objective((output,), microbatch.objective_inputs)
                outputs.append(result)
            if train:
                pending.append((index, activation, output))
            return output

        def backward(gradient: torch.Tensor | None) -> torch.Tensor | None:
            index, activation, output = pending.popleft()
            if fsdp is not None and index == len(microbatches) - 1:
                fsdp.set_requires_gradient_sync(True)
                fsdp.set_reshard_after_backward(True)
                fsdp.set_is_last_backward(True)
            with torch.autograd.set_multithreading_enabled(False):
                torch.autograd.backward(output, gradient)
            return activation.grad if activation is not None else None

        try:
            if not train:
                for index, microbatch in enumerate(microbatches):
                    activation = self._exchange(self.prev, receive_for=microbatch)
                    self._exchange(self.next, forward(index, activation))
                if fsdp is not None:
                    for module in stage.modules():
                        if isinstance(module, FSDPModule):
                            module.reshard()
                return outputs

            # Fill the pipeline, alternating F/B once full, then finish the outstanding backwards.
            warmup = min(self.objective_pp_rank - self.rank, len(microbatches))
            for index in range(warmup):
                activation = self._exchange(self.prev, receive_for=microbatches[index])
                self._exchange(self.next, forward(index, activation))

            if warmup < len(microbatches):
                activation = self._exchange(self.prev, receive_for=microbatches[warmup])
            for index in range(warmup, len(microbatches)):
                output = forward(index, activation)
                gradient = self._exchange(self.next, output, microbatches[pending[0][0]])
                input_gradient = backward(gradient)
                next_batch = microbatches[index + 1] if index + 1 < len(microbatches) else None
                activation = self._exchange(self.prev, input_gradient, next_batch)

            while pending:
                gradient = self._exchange(self.next, receive_for=microbatches[pending[0][0]])
                self._exchange(self.prev, backward(gradient))
            return outputs
        finally:
            training.current_microbatch = None
            FP8WeightCacheControl.clear(stage)
