# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic Qwen3-8B native MRV2 baseline, including rolling arrivals."""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    if os.environ.get("P1_TRACE") == "1":
        os.environ["VLLM_KV_TRACE_FILE"] = str(run / "kv_trace.jsonl")
        os.environ["VLLM_KV_TRACE_IDS"] = "1"
    torch.cuda.set_per_process_memory_fraction(0.50)
    model = "/data/models/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    args = EngineArgs(
        model=model,
        dtype="bfloat16",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        block_size=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_cascade_attn=True,
        scheduler_reserve_full_isl=False,
        max_model_len=40960,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=6 * 2**30,
        gpu_memory_utilization=0.40,
        seed=0,
        disable_log_stats=True,
    )
    pressure = os.environ.get("P1_PRESSURE") == "1"
    if pressure:
        args.max_model_len = 512
        args.max_num_batched_tokens = 256
        args.kv_cache_memory_bytes = 256 * 2**20
        args.num_gpu_blocks_override = 33
    engine = LLMEngine.from_engine_args(args)
    config = engine.vllm_config
    core = engine.engine_core.engine_core
    worker = core.model_executor.driver_worker.worker
    runner = worker.model_runner
    identity = {
        "runner": f"{type(runner).__module__}.{type(runner).__name__}",
        "max_concurrent_batches": config.max_concurrent_batches,
        "batch_queue_is_none": core.batch_queue is None,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "cache_config": str(config.cache_config),
        "attention_config": str(config.attention_config),
        "kv_cache_config": str(core.scheduler.kv_cache_config),
        "python_source": __import__("vllm").__file__,
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "args": vars(args),
    }
    views = {
        name: layer.kv_cache
        for name, layer in config.compilation_config.static_forward_context.items()
        if isinstance(getattr(layer, "kv_cache", None), torch.Tensor)
    }
    identity["kv_views"] = {
        name: {
            "shape": list(view.shape),
            "stride": list(view.stride()),
            "dtype": str(view.dtype),
            "data_ptr": view.data_ptr(),
        }
        for name, view in views.items()
    }
    assert len(views) == 36
    (run / "identity.json").write_text(json.dumps(identity, indent=2, default=str))
    assert config.max_concurrent_batches == 1 and core.batch_queue is None
    assert "gpu.model_runner" in identity["runner"]
    params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, seed=0)
    prompts = {
        "A": "The following is a memory test. "
        + "Mary went to the garden. " * 256
        + "Where is Mary? Reply briefly.",
        "B": "Calculate 17 * 19 and explain the answer briefly.",
    }
    starts = {}
    outputs = {}
    ticks = []
    for step in range(256):
        for rid, arrival in [("A", 0), ("B", 5)]:
            if step == arrival:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompts[rid]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if pressure:
                    token_ids = (token_ids * 16)[
                        : int(os.environ.get("P1_PRESSURE_PROMPT_TOKENS", "256"))
                    ]
                starts[rid] = time.perf_counter()
                engine.add_request(rid, {"prompt_token_ids": token_ids}, params)
        t0 = time.perf_counter()
        batch = engine.step()
        ticks.append({"step": step, "latency_s": time.perf_counter() - t0})
        for output in batch:
            outputs[output.request_id] = {
                "token_ids": list(output.outputs[0].token_ids),
                "text": output.outputs[0].text,
                "finished": output.finished,
                "elapsed_s": time.perf_counter() - starts[output.request_id],
            }
        if step >= 5 and not engine.has_unfinished_requests():
            break
    assert set(outputs) == {"A", "B"} and all(o["finished"] for o in outputs.values())
    (run / "outputs.json").write_text(json.dumps(outputs, indent=2))
    (run / "steps.json").write_text(json.dumps(ticks, indent=2))
    print(json.dumps({"status": "PASS", "steps": len(ticks), "outputs": outputs}))


if __name__ == "__main__":
    main()
