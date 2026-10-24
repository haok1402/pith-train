"""
Routing metadata that stage 1 hands to the dispatch / combine all-to-alls and to stage 3 and 5.

A torch-only leaf so that pipeline/, models/ and the dispatch operator can share the types without
importing one another.
"""

from typing import List, NamedTuple, Optional

import torch


class AllToAllSplits(NamedTuple):
    input_splits: List[int]
    output_splits: List[int]


class RoutingInfo(NamedTuple):
    topk_weight: torch.Tensor
    expert_idxs: torch.Tensor
    moe_local_idxs: Optional[torch.Tensor] = None
    expand_idx: Optional[torch.Tensor] = None
    dispatch_splits: Optional[AllToAllSplits] = None
    combine_splits: Optional[AllToAllSplits] = None
    expert_counts: Optional[list[int]] = None
    """Rows routed to each local expert, on the host, when dispatch already knows them; lets stage 3
    size its grouped GEMMs without another device-to-host sync."""
