# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host CPU placement; the ordinary Worker execution path is inherited."""

import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


class AffinityWorker(Worker):
    def init_device(self):
        super().init_device()
        cpus = json.loads(os.environ["P1_WORKER_CPUS"])
        cpu = cpus[self.local_rank]
        previous = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        record = {
            "pid": os.getpid(),
            "rank": self.rank,
            "local_rank": self.local_rank,
            "dp_rank": self.parallel_config.data_parallel_rank,
            "previous_affinity": previous,
            "affinity": sorted(os.sched_getaffinity(0)),
            "scope": "Worker main thread; existing helper threads unchanged",
        }
        run = Path(os.environ["P1_RUN_DIR"])
        (run / f"worker-affinity-{self.local_rank}.json").write_text(
            json.dumps(record, indent=2)
        )
