# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in request/step timestamps and NVTX for one TP TTFT diagnostic."""

import json
import os
import time
from functools import wraps
from pathlib import Path

import torch
from worker_affinity import AffinityWorker

from vllm.v1.core.sched.scheduler import Scheduler


def save(name, events):
    path = Path(os.environ["P1_RUN_DIR"]) / name
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def clock_mark():
    stamp = time.monotonic_ns()
    torch.cuda.nvtx.mark(f"p1-ttft-clock|{os.getpid()}|{stamp}")


class TimelineScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeline = []
        clock_mark()

    def add_request(self, request):
        stamp = time.monotonic_ns()
        super().add_request(request)
        self.timeline.append(
            dict(
                event="enqueue",
                time_ns=stamp,
                request_id=request.request_id,
                prompt_tokens=request.num_prompt_tokens,
            )
        )

    def schedule(self, *args, **kwargs):
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        output = super().schedule(*args, **kwargs)
        end = time.monotonic_ns()
        # SchedulerOutput is a pickled dataclass; only this diagnostic adds a tag.
        output._p1_ttft_step = self.current_step
        self.timeline.append(
            dict(
                event="schedule",
                step=self.current_step,
                start_ns=start,
                end_ns=end,
                thread_ns=time.thread_time_ns() - cpu,
                requests={
                    rid: dict(
                        logical_start=self.requests[rid].num_computed_tokens - q,
                        effective_start=self.requests[rid].effective_kv_len,
                        prompt_tokens=self.requests[rid].num_prompt_tokens,
                        q=q,
                    )
                    for rid, q in output.num_scheduled_tokens.items()
                },
                waiting=[request.request_id for request in self.waiting],
                reclaim=(
                    dict(
                        request_id=output.reclaim_request.request_id,
                        source_e=output.reclaim_request.source_effective_kv_len,
                        target_e=output.reclaim_request.target_keep_tokens,
                    )
                    if output.reclaim_request is not None
                    else None
                ),
            )
        )
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        self.timeline.append(
            dict(
                event="complete",
                step=scheduler_output._p1_ttft_step,
                start_ns=start,
                end_ns=time.monotonic_ns(),
                thread_ns=time.thread_time_ns() - cpu,
                output_counts={
                    output.request_id: len(output.new_token_ids)
                    for batch in outputs.values()
                    for output in batch.outputs
                    if output.new_token_ids
                },
            )
        )
        return outputs

    def shutdown(self):
        clock_mark()
        save("timeline-scheduler.jsonl", self.timeline)
        return super().shutdown()


class TimelineWorker(AffinityWorker):
    def init_device(self):
        super().init_device()
        self.timeline = []
        self.timeline_step = None
        self.timeline_ready = False
        clock_mark()
        self.install_timeline()

    def install_timeline(self):
        from vllm.v1.worker.gpu.async_utils import AsyncOutput
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        from vllm.v1.worker.gpu.kv_compaction import StableKVCompactor
        from vllm.v1.worker.gpu.kv_reclamation import KVReclamationWorker
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        worker = self

        def instrument(cls, name):
            original = getattr(cls, name)
            phase = cls.__name__ + "." + name

            @wraps(original)
            def measured(*args, **kwargs):
                if worker.timeline_step is None:
                    return original(*args, **kwargs)
                start, cpu = time.monotonic_ns(), time.thread_time_ns()
                label = f"p1-ttft|{worker.rank}|{worker.timeline_step}|{phase}"
                with torch.cuda.nvtx.range(label):
                    result = original(*args, **kwargs)
                worker.timeline.append(
                    dict(
                        event=phase,
                        step=worker.timeline_step,
                        start_ns=start,
                        end_ns=time.monotonic_ns(),
                        thread_ns=time.thread_time_ns() - cpu,
                    )
                )
                return result

            setattr(cls, name, measured)

        for name in (
            "execute_model",
            "prepare_inputs",
            "prepare_attn",
            "sample_tokens",
            "postprocess_sampled",
        ):
            instrument(GPUModelRunner, name)
        instrument(AsyncOutput, "get_output")
        instrument(KVReclamationWorker, "reclaim")
        instrument(StableKVCompactor, "compact")
        instrument(ModelCudaGraphManager, "run_fullgraph")

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.timeline_ready = True
        return result

    def execute_model(self, scheduler_output):
        if self.timeline_ready:
            self.timeline_step = scheduler_output._p1_ttft_step
        return super().execute_model(scheduler_output)

    def shutdown(self):
        clock_mark()
        save(f"timeline-rank-{self.rank}.jsonl", self.timeline)
        return super().shutdown()
