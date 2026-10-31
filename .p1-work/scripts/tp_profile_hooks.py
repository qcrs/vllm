# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVTX annotations only for the separate TP diagnostic processes."""

from functools import wraps

import torch
from tp_worker_hooks import P1TPWorker

from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker

_active = None
_sequence = 0
_reclaim = KVReclamationWorker.reclaim
_all_reduce = torch.distributed.all_reduce
_broadcast = torch.distributed.broadcast


def label(kind, tensor=None):
    rank, sequence = _active
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    return f"p1-tp|{rank}|{sequence}|{kind}|{size}"


@wraps(_reclaim)
def reclaim(self, *args, **kwargs):
    global _active, _sequence
    _sequence += 1
    _active = (self.tp_group.rank_in_group, _sequence)
    try:
        with torch.cuda.nvtx.range(label("reclaim")):
            return _reclaim(self, *args, **kwargs)
    finally:
        _active = None


@wraps(_all_reduce)
def all_reduce(tensor, *args, **kwargs):
    if _active is None:
        return _all_reduce(tensor, *args, **kwargs)
    kind = "score" if tensor.dtype == torch.float32 else "completion"
    with torch.cuda.nvtx.range(label(kind, tensor)):
        return _all_reduce(tensor, *args, **kwargs)


@wraps(_broadcast)
def broadcast(tensor, *args, **kwargs):
    if _active is None:
        return _broadcast(tensor, *args, **kwargs)
    with torch.cuda.nvtx.range(label("plan", tensor)):
        return _broadcast(tensor, *args, **kwargs)


KVReclamationWorker.reclaim = reclaim
torch.distributed.all_reduce = all_reduce
torch.distributed.broadcast = broadcast


class P1TPProfileWorker(P1TPWorker):
    pass
