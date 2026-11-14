"""
Loss check of the pp_size 1 schedules on the pretraining benchmark path: a few steps of
pretrain_lm.launch with TrainingCfg.pp1_sequential_schedule off or on.

usage: torchrun ... check_pretrain.py <model> <overlapped|sequential>
"""

import sys

model, schedule = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"benchmarks/pretraining/{model}")

from setup import cfg  # noqa: E402

from pithtrain.tasks import pretrain_lm  # noqa: E402

cfg.distributed.pipeline_parallel_size = 1
cfg.distributed.expert_parallel_size = 8
training = cfg.training
training.micro_batch_size = 1
training.global_batch_size = 64
training.sequence_length = 2048
training.max_steps = 5
training.pp1_sequential_schedule = schedule == "sequential"

if __name__ == "__main__":
    pretrain_lm.launch(cfg)
