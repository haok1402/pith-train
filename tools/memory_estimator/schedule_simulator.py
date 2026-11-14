"""
DualPipeV 8-step schedule simulator.

Replays the exact scheduling algorithm from DualPipeV.step() (dualpipev.py:482-547),
tracking live activations, gradients, communication buffers, and deferred weight
gradient storage at each event boundary. Produces a timeline of MemorySnapshots
with the peak high-water mark identified.

FSDP sharding stages are modeled per parameter class (attention: everything outside the routed
experts; experts: the routed expert weights), each with its own stage:

- Stage 1: the unsharded weights of both modules are resident for the whole step, and a module's
  unsharded fp32 gradients exist from its first backward to the end of the step.
- Stage 2: each FSDP unit's fp32 main_grad is reduce-scattered and freed after the unit's
  backward, so the unsharded gradient that is live is that of the largest unit of the module in
  backward. A routed-expert backward deferred to the zero-bubble W phase keeps the expert
  main_grads of every layer of its module until its weight_chunk. A module's fp32 sharded
  gradients exist from its first backward to the optimizer step, and an event that reduces a class
  holds one reduce-scatter input buffer, the fp32 gradient of the class's largest unit.
- Stage 3: gradients as at stage 2. The bf16 unsharded weights of each module active at an event
  (both modules in an overlapped forward/backward, none during a weight_chunk) are its two largest
  decoder-layer units, the one running and one prefetched, plus its embed_tokens, norm and lm_head.

Approximations: every event stands for a whole pass over a module, so stages 2 and 3 count the
largest unit (or two largest decoder-layer units) of the module rather than the unit running at
the memory peak within the pass, and the embedding and head units for the whole pass rather than
only while in use. A module's sharded gradients appear at its first backward even when the first
reduce of its experts waits for a W phase. A class whose shard world is 1 keeps its unsharded
weights resident at stage 3.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .activation_profile import ActivationEstimator, TokenCounts
from .model_profile import ModelConfig, ModelMemoryProfile, ParallelismConfig
from .tensor_spec import MemoryBucket, MemorySnapshot


@dataclass(slots=True)
class ScheduleEvent:
    """One event in the simulated schedule timeline."""

    name: str
    snapshot: MemorySnapshot
    delta_bytes: int  # change from previous event


@dataclass(slots=True)
class ScheduleResult:
    """Output of the schedule simulation."""

    timeline: list[ScheduleEvent]
    peak_event_idx: int
    peak_bytes: int
    static_bytes: int


class ScheduleSimulator:
    """Simulate the DualPipeV 8-step schedule and track memory at each point."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        parallel_cfg: ParallelismConfig,
        profile: ModelMemoryProfile,
        activation_est: ActivationEstimator,
        token_counts: TokenCounts,
        pp_rank: int,
        fragmentation_factor: float = 0.10,
        non_pytorch_bytes: int = 0,
    ):
        self.model_cfg = model_cfg
        self.parallel_cfg = parallel_cfg
        self.profile = profile
        self.act_est = activation_est
        self.tc = token_counts
        self.pp_rank = pp_rank
        self.pp_size = parallel_cfg.pp_size
        self.frag_factor = fragmentation_factor

        self.is_first_pp_rank = pp_rank == 0
        self.is_last_pp_rank = pp_rank == self.pp_size - 1
        self.non_pytorch_bytes = non_pytorch_bytes

        # Precompute per-module activation costs
        self._chunk_records: dict[int, MemoryBucket] = {}
        self._chunk_autograd: dict[int, MemoryBucket] = {}
        for mod_idx in range(2):
            records, autograd = activation_est.compute_chunk_activations(mod_idx, 0, 0)
            self._chunk_records[mod_idx] = records
            self._chunk_autograd[mod_idx] = autograd

        # Precompute static memory buckets
        self._params = [profile.compute_module_params(i) for i in range(2)]
        self._fsdp_sharded = profile.compute_fsdp_sharded_params()
        self._optimizer = profile.compute_optimizer_states()
        self._grad_buckets = [profile.compute_gradient_bucket(i) for i in range(2)]

        # Sharding stages 2 and 3, per module and class (False: attention, True: experts).
        classes = (False, True)
        self._stage = {c: profile.sharding_stage(c) for c in classes}
        units = [profile.compute_shard_units(i) for i in range(2)]
        self._unit_max = [
            {c: max((u.numel for u in units[i] if u.expert == c), default=0) for c in classes}
            for i in range(2)
        ]
        self._class_numel = [
            {c: sum(u.numel for u in units[i] if u.expert == c) for c in classes} for i in range(2)
        ]
        self._gathered_numel = [{c: 0 for c in classes} for _ in range(2)]
        for i in range(2):
            for c in classes:
                if not profile.gathers_per_layer(c):
                    continue
                layers = sorted(u.numel for u in units[i] if u.expert == c and u.decoder_layer)
                edges = sum(u.numel for u in units[i] if u.expert == c and not u.decoder_layer)
                self._gathered_numel[i][c] = sum(layers[-2:]) + edges
        self._rs_numel = {c: max(self._unit_max[i][c] for i in range(2)) for c in classes}
        self._sharded_grads = [
            {c: profile.compute_sharded_gradient_bucket(i, c) for c in classes} for i in range(2)
        ]

        # Precompute per-chunk costs
        self._wgrad_bytes = [
            activation_est.compute_wgrad_store_bytes_per_chunk(i) for i in range(2)
        ]
        self._comm_bytes = activation_est.compute_comm_buffer_bytes()
        self._pp_transfer_bytes = activation_est.compute_pp_transfer_bytes()

        # Num chunks. dp is the only axis that splits the batch; ep shards experts, not data.
        dp_size = parallel_cfg.dp_size
        micro_bs = parallel_cfg.micro_batch_size
        global_bs = parallel_cfg.global_batch_size
        self.num_chunks = global_bs // (micro_bs * dp_size)

    def simulate(self) -> ScheduleResult:
        pp_rank = self.pp_rank
        pp_size = self.pp_size
        num_chunks = self.num_chunks

        # State tracking
        # Which (phase, chunk_id) have live activations
        live_acts: dict[tuple[int, int], int] = {}  # -> module_idx
        live_autograd: dict[tuple[int, int], int] = {}  # -> module_idx
        # Gradients: once backward completes a chunk on a module, all its layers have .grad
        grads_allocated: set[int] = set()  # module indices that have gradients
        # WeightGradStore: accumulated bytes from deferred wgrads
        wgrad_store_bytes = 0
        # PP transfer buffers: detached tensors at last PP rank
        pp_transfers: set[int] = set()  # chunk_ids with live transfers
        # Modules of the backward chunks whose expert weight gradients wait for a weight_chunk
        pending_wgrad: deque[int] = deque()

        # Forward chunk counters
        fwd_chunk_id = [0, 0]
        bwd_chunk_id = [0, 0]

        timeline: list[ScheduleEvent] = []
        prev_total = 0

        def _module_for_phase(phase: int) -> int:
            """Phase 0 -> module 0, phase 1 -> module 1."""
            return phase

        def _snapshot(
            event_name: str,
            active: tuple[int, ...] = (),
            bwd_module: int | None = None,
            reduced: tuple[bool, ...] = (),
            wgrad_module: int | None = None,
        ) -> None:
            """
            Record the memory of one event.

            active are the modules running forward or backward, bwd_module the one in backward,
            reduced the parameter classes reduce-scattered at this event and wgrad_module the
            module whose deferred weight gradients a weight_chunk is computing.
            """
            nonlocal prev_total
            snap = MemorySnapshot(event_name=event_name)
            snap.fragmentation_factor = self.frag_factor
            snap.non_pytorch_bytes = self.non_pytorch_bytes

            # Static
            snap.params_unsharded = list(self._params)
            snap.fsdp_sharded_params = self._fsdp_sharded
            snap.optimizer_states = self._optimizer

            # Activations
            for key, mod_idx in live_acts.items():
                phase, chunk_id = key
                records, _ = self.act_est.compute_chunk_activations(mod_idx, phase, chunk_id)
                snap.activations[key] = records
            for key, mod_idx in live_autograd.items():
                phase, chunk_id = key
                _, autograd = self.act_est.compute_chunk_activations(mod_idx, phase, chunk_id)
                snap.autograd_overhead[key] = autograd

            # Gradients
            grad_bucket = MemoryBucket("gradients")
            for mod_idx in sorted(grads_allocated):
                for spec in self._grad_buckets[mod_idx].specs:
                    if self._stage[".mlp.experts." in spec.name] == 1:
                        grad_bucket.specs.append(spec)
            for c in (False, True):
                if self._stage[c] == 1:
                    continue
                name = "experts" if c else "attn"
                # Modules whose fp32 main_grads of the class are all live, waiting for a W phase
                whole = set(pending_wgrad) if c else set()
                if c and wgrad_module is not None:
                    whole.add(wgrad_module)
                for mod_idx in sorted(whole):
                    numel = self._class_numel[mod_idx][c]
                    grad_bucket.add(f"main_grad(m[{mod_idx}].{name}, deferred)", (numel,), "fp32")
                if bwd_module is not None and bwd_module not in whole:
                    numel = self._unit_max[bwd_module][c]
                    grad_bucket.add(f"main_grad(m[{bwd_module}].{name}, unit)", (numel,), "fp32")
                for mod_idx in sorted(grads_allocated):
                    grad_bucket.specs.extend(self._sharded_grads[mod_idx][c].specs)
            snap.gradients = grad_bucket

            # Comm buffers
            comm_bucket = MemoryBucket("comm_buffers")
            comm_bucket.add("p2p_recv_buffers", (self._comm_bytes,), "int8")
            for c in reduced:
                if self._stage[c] > 1 and self._rs_numel[c] > 0:
                    name = "experts" if c else "attn"
                    comm_bucket.add(f"reduce_scatter_input({name})", (self._rs_numel[c],), "fp32")
            snap.comm_buffers = comm_bucket

            # Gathered params
            if any(n > 0 for numels in self._gathered_numel for n in numels.values()):
                gathered_bucket = MemoryBucket("params_gathered")
                for mod_idx in sorted(active):
                    for c in (False, True):
                        numel = self._gathered_numel[mod_idx][c]
                        if numel > 0:
                            name = f"m[{mod_idx}].{'experts' if c else 'attn'} gathered units"
                            gathered_bucket.add(name, (numel,), "bf16")
                snap.params_gathered = gathered_bucket

            # WeightGradStore
            wgrad_bucket = MemoryBucket("wgrad_store")
            if wgrad_store_bytes > 0:
                wgrad_bucket.add("deferred_wgrad_tensors", (wgrad_store_bytes,), "int8")
            snap.weight_grad_store = wgrad_bucket

            # PP transfer buffers
            pp_bucket = MemoryBucket("pp_transfer")
            for cid in sorted(pp_transfers):
                pp_bucket.add(
                    f"pp_transfer_chunk_{cid}",
                    (self._pp_transfer_bytes,),
                    "int8",
                )
            snap.pp_transfer_buffers = pp_bucket

            total = snap.total_bytes
            delta = total - prev_total
            prev_total = total
            timeline.append(ScheduleEvent(name=event_name, snapshot=snap, delta_bytes=delta))

        def forward_chunk(phase: int, event_prefix: str) -> None:
            nonlocal wgrad_store_bytes
            mod_idx = _module_for_phase(phase)
            chunk_id = fwd_chunk_id[phase]
            fwd_chunk_id[phase] += 1
            key = (phase, chunk_id)
            live_acts[key] = mod_idx
            live_autograd[key] = mod_idx

            # At last PP rank, phase 0 forward creates a transfer to phase 1
            if self.is_last_pp_rank and phase == 0:
                pp_transfers.add(chunk_id)

            _snapshot(f"{event_prefix} fwd(p{phase}) c{chunk_id}", active=(mod_idx,))

        def backward_chunk(phase: int, enable_zb: bool, event_prefix: str) -> None:
            nonlocal wgrad_store_bytes
            mod_idx = _module_for_phase(phase)
            chunk_id = bwd_chunk_id[phase]
            bwd_chunk_id[phase] += 1
            key = (phase, chunk_id)

            # Free activations
            live_acts.pop(key, None)
            live_autograd.pop(key, None)

            # Allocate gradients (first backward on this module allocates all grads)
            grads_allocated.add(mod_idx)

            # At last PP rank, phase 1 backward produces grads that go to phase 0
            # but the output_grad_chunks[0] is just a reference, not new memory

            if enable_zb:
                wgrad_store_bytes += self._wgrad_bytes[mod_idx]
                pending_wgrad.append(mod_idx)

            # Consume PP transfer if this is phase 1 at last rank
            if self.is_last_pp_rank and phase == 1:
                pp_transfers.discard(chunk_id)

            _snapshot(
                f"{event_prefix} bwd(p{phase},zb={enable_zb}) c{chunk_id}",
                active=(mod_idx,),
                bwd_module=mod_idx,
                reduced=(False,) if enable_zb else (False, True),
            )

        def forward_backward_chunk(phase0: int, phase1: int, event_prefix: str) -> None:
            """
            Overlapped forward (phase0) + backward (phase1).

            Memory peaks when both activation sets coexist.
            We model this by first adding the forward chunk, then taking snapshot
            (which includes both old backward activations and new forward activations),
            then removing the backward chunk.
            """
            nonlocal wgrad_store_bytes
            mod_idx0 = _module_for_phase(phase0)
            mod_idx1 = _module_for_phase(phase1)

            fwd_cid = fwd_chunk_id[phase0]
            fwd_chunk_id[phase0] += 1
            bwd_cid = bwd_chunk_id[phase1]
            bwd_chunk_id[phase1] += 1

            fwd_key = (phase0, fwd_cid)
            bwd_key = (phase1, bwd_cid)

            # Add forward activations (both sets coexist at peak)
            live_acts[fwd_key] = mod_idx0
            live_autograd[fwd_key] = mod_idx0

            if self.is_last_pp_rank and phase0 == 0:
                pp_transfers.add(fwd_cid)

            # Gradients from backward
            grads_allocated.add(mod_idx1)

            # PP transfer consumed
            if self.is_last_pp_rank and phase1 == 1:
                pp_transfers.discard(bwd_cid)

            # Snapshot with both sets alive (peak of overlap)
            _snapshot(
                f"{event_prefix} fwd_bwd(p{phase0},p{phase1}) fc{fwd_cid}/bc{bwd_cid}",
                active=(mod_idx0, mod_idx1),
                bwd_module=mod_idx1,
                reduced=(False, True),
            )

            # Now free backward activations (post-overlap)
            live_acts.pop(bwd_key, None)
            live_autograd.pop(bwd_key, None)

        def weight_chunk(event_prefix: str) -> None:
            nonlocal wgrad_store_bytes
            # Pops one batch of deferred wgrads - free the oldest module's worth
            # In practice, weight_chunk pops from a FIFO queue
            if wgrad_store_bytes > 0:
                # Each pop frees one chunk's worth of wgrad from whichever module
                # We use a simple approximation: free the average of both modules
                wgrad_store_bytes = max(0, wgrad_store_bytes - max(self._wgrad_bytes))
            wgrad_module = pending_wgrad.popleft() if pending_wgrad else None
            reduced = (True,) if wgrad_module is not None else ()
            _snapshot(f"{event_prefix} weight_chunk", reduced=reduced, wgrad_module=wgrad_module)

        # === Static baseline ===
        _snapshot("static_baseline")

        # === Step 1: nF0 = (pp_size - pp_rank - 1) * 2 ===
        step_1 = (pp_size - pp_rank - 1) * 2
        for i in range(step_1):
            forward_chunk(0, f"S1.i{i}")

        # === Step 2: nF0F1 = pp_rank + 1 ===
        step_2 = pp_rank + 1
        for i in range(step_2):
            forward_chunk(0, f"S2.i{i}")
            forward_chunk(1, f"S2.i{i}")

        # === Step 3: nB1W1F1 = pp_size - pp_rank - 1 ===
        step_3 = pp_size - pp_rank - 1
        for i in range(step_3):
            backward_chunk(1, enable_zb=True, event_prefix=f"S3.i{i}")
            weight_chunk(f"S3.i{i}")
            forward_chunk(1, f"S3.i{i}")

        # === Step 4: nF0B1F1B0 = num_chunks - pp_size * 2 + pp_rank + 1 ===
        step_4 = num_chunks - pp_size * 2 + pp_rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_last_pp_rank:
                    # Special case: no overlap for first iteration on last rank
                    forward_chunk(0, "S4.i0.last_rank")
                    backward_chunk(1, enable_zb=False, event_prefix="S4.i0.last_rank")
                else:
                    forward_backward_chunk(0, 1, f"S4.i{i}")
            else:
                forward_backward_chunk(0, 1, f"S4.i{i}")
            forward_backward_chunk(1, 0, f"S4.i{i}")

        # === Step 5: nB1F1B0 = pp_size - pp_rank - 1 ===
        step_5 = pp_size - pp_rank - 1
        for i in range(step_5):
            backward_chunk(1, enable_zb=False, event_prefix=f"S5.i{i}")
            forward_backward_chunk(1, 0, f"S5.i{i}")

        # === Step 6: nB1B0 = pp_rank + 1 (second half uses zero bubble) ===
        step_6 = pp_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and pp_rank % 2 == 1:
                enable_zb = True
            backward_chunk(1, enable_zb=enable_zb, event_prefix=f"S6.i{i}")
            if i == step_6 // 2 and pp_rank % 2 == 0:
                enable_zb = True
            backward_chunk(0, enable_zb=enable_zb, event_prefix=f"S6.i{i}")

        # === Step 7: nWB0 = pp_size - pp_rank - 1 ===
        step_7 = pp_size - pp_rank - 1
        for i in range(step_7):
            weight_chunk(f"S7.i{i}")
            backward_chunk(0, enable_zb=True, event_prefix=f"S7.i{i}")

        # === Step 8: nW = pp_rank + 1 ===
        step_8 = pp_rank + 1
        for i in range(step_8):
            weight_chunk(f"S8.i{i}")

        # Find peak
        peak_idx = 0
        peak_bytes = 0
        for idx, event in enumerate(timeline):
            total = event.snapshot.total_bytes
            if total > peak_bytes:
                peak_bytes = total
                peak_idx = idx

        static_bytes = timeline[0].snapshot.static_bytes if timeline else 0

        return ScheduleResult(
            timeline=timeline,
            peak_event_idx=peak_idx,
            peak_bytes=peak_bytes,
            static_bytes=static_bytes,
        )
