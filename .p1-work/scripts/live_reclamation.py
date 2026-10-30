# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-8B live release/reuse and completion-boundary verification."""

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
    pressure = True
    if os.environ.get("P1_RECLAIM") == "1":
        args.additional_config = {
            "kv_reclamation": {
                "min_keep_tokens": 32,
                "policy": os.environ.get("P1_POLICY", "sink_recent"),
                "stats_path": os.environ.get("P1_STATS"),
            }
        }
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
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_reclamation import sink_recent_ranges, whole_block_row

    scheduler = core.scheduler
    path = os.environ.get("P1_PATH", "whole")
    lifecycle = os.environ.get("P1_LIFECYCLE", "decode")
    steps, events = [], []
    original_schedule = scheduler.schedule
    original_reclaim = runner.reclamation.reclaim
    original_update = scheduler.update_from_output
    original_aborts = core._process_aborts_queue
    original_prepare = runner.reclamation.prepare_attn
    pending_abort = False
    released = set()
    reused = set()
    reuse_before = {}
    reference = None
    reference_row = None
    previous_logical = {}
    completed_reclaims = 0
    completion_event = None
    max_tokens = 40 if lifecycle == "eos" else 64
    internal_ids = {}
    active_batch = None
    attention_checks = 0
    preempted = False
    first_layer = next(iter(config.compilation_config.static_forward_context.values()))
    original_attention = first_layer.impl.forward

    def checked_prepare(batch):
        nonlocal active_batch
        physical = original_prepare(batch)
        active_batch = batch
        logical_positions = batch.positions.cpu().tolist()
        physical_positions = physical.positions.cpu().tolist()
        kv_lengths = physical.seq_lens.cpu().tolist()
        offset = 0
        for i, rid in enumerate(batch.req_ids):
            q = int(batch.num_scheduled_tokens[i])
            logical = scheduler.requests[rid].num_computed_tokens - q
            effective = runner.reclamation.effective_kv_len_np[batch.idx_mapping_np[i]]
            assert logical_positions[offset : offset + q] == list(
                range(logical, logical + q)
            )
            assert physical_positions[offset : offset + q] == list(
                range(effective, effective + q)
            )
            assert kv_lengths[i] == effective + q
            assert physical.seq_lens_cpu_upper_bound[i] == effective + q
            offset += q
        return physical

    def checked_attention(layer, query, key, value, cache, metadata, output, **kwargs):
        nonlocal attention_checks
        answer = original_attention(
            layer, query, key, value, cache, metadata, output, **kwargs
        )
        rid = internal_ids.get("A")
        if completed_reclaims and rid in active_batch.req_ids:
            i = active_batch.req_ids.index(rid)
            starts = metadata.query_start_loc.cpu().tolist()
            if starts[i + 1] - starts[i] != 1:
                return answer
            length = int(metadata.seq_lens[i].cpu())
            row = metadata.block_table[i].cpu().tolist()
            kv = read_members(cache.cpu(), row, range(length)).float()
            k, v = kv.chunk(2, dim=-1)
            q = query[starts[i]].cpu().float()
            groups = q.shape[0] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            logits = torch.einsum("hd,thd->ht", q, k) * q.shape[-1] ** -0.5
            wanted = torch.einsum("ht,thd->hd", logits.softmax(-1), v)
            actual = output[starts[i]].cpu().float().reshape_as(wanted)
            torch.testing.assert_close(actual, wanted, atol=1.6e-2, rtol=1.6e-2)
            attention_checks += 1
        return answer

    def read_members(view, row, keep):
        return torch.stack([view[row[i // 16], :, i % 16, :] for i in keep])

    def controlled_schedule(*args, **kwargs):
        nonlocal pending_abort, reference, reference_row, preempted
        step = len(steps)
        request = scheduler.requests.get(internal_ids.get("A"))
        if lifecycle == "preempt" and step == 24:
            assert request.effective_kv_len < request.num_computed_tokens
            scheduler.running.remove(request)
            scheduler._preempt_request(request, time.monotonic())
            assert request.effective_kv_len == request.num_computed_tokens == 0
            reference = reference_row = None
            preempted = True
            events.append({"event": "native_preempt_reset", "L": 0, "E": 0})
        excluded = lifecycle == "q0" and step == 20
        if excluded:
            scheduler.running.remove(request)
        maintenance = lifecycle == "maintenance" and step == 16
        output = (
            SchedulerOutput.make_empty()
            if maintenance
            else original_schedule(*args, **kwargs)
        )
        if excluded:
            scheduler.running.insert(0, request)
            assert output.total_num_scheduled_tokens > 0
        request = scheduler.requests.get(internal_ids.get("A"))
        if request is not None and request.num_computed_tokens >= 256:
            q = output.num_scheduled_tokens.get(request.request_id, 0)
            source = request.effective_kv_len + q
            trigger = step == (16 if path == "whole" else 17)
            if lifecycle == "repeat":
                trigger = step in ((16, 32) if path == "whole" else (17, 34))
            if lifecycle == "maintenance":
                trigger = maintenance
            if lifecycle == "eos":
                trigger = step == 30
            if lifecycle == "q0":
                trigger = excluded
            if trigger:
                keep = (
                    128 if completed_reclaims == 0 else (64 if path == "whole" else 63)
                )
                if lifecycle == "maintenance":
                    keep = 127
                intent = scheduler.make_reclaim_request(request, output, keep)
                output.reclaim_request = intent
                events.append(
                    {
                        "event": "intent",
                        "step": step,
                        "L": request.num_computed_tokens,
                        "E_source": source,
                        "q": q,
                        "old_row": intent.source_block_ids,
                    }
                )
                if lifecycle == "abort":
                    pending_abort = True
        # Observe freshly allocated B pages before its real forward writes them.
        row = (
            scheduler.kv_cache_manager.get_block_ids(internal_ids["B"])[0]
            if "B" in internal_ids
            else []
        )
        new_reused = set(row) & released - reused
        for page in new_reused:
            reuse_before[page] = views[next(iter(views))][page].cpu()
        reused.update(new_reused)
        return output

    def checked_reclaim(intent, tables):
        nonlocal reference, reference_row, completion_event
        # This is an experiment-only small snapshot, never a serving backend.
        assert intent.source_effective_kv_len <= 1024
        snapshot = [t.cpu() for t in views.values()]
        prefix, recent = sink_recent_ranges(
            intent.source_effective_kv_len, intent.target_keep_tokens, 16
        )
        keep = [*range(prefix), *range(recent, intent.source_effective_kv_len)]
        policy = runner.reclamation.config.policy
        if policy != "sink_recent":
            recent = intent.source_effective_kv_len - 16
            idx = runner.req_states.req_id_to_index[intent.request_id]
            assert policy == "triattention"
            logical_gpu = runner.req_states.num_computed_tokens.gpu[idx : idx + 1]
            assert (
                int(logical_gpu.cpu())
                == scheduler.requests[intent.request_id].num_computed_tokens
            )
            keep = (
                runner.reclamation.scorer.plan(
                    list(views.values()),
                    tables.block_tables[0].gpu[idx],
                    logical_gpu,
                    intent.source_effective_kv_len,
                    intent.target_keep_tokens,
                    prefix,
                    recent,
                )
                .cpu()
                .tolist()
            )
            assert len(keep) == len(set(keep)) == intent.target_keep_tokens
            assert keep == sorted(keep)
        whole = policy == "sink_recent" and (
            whole_block_row(
                intent.source_block_ids,
                intent.source_effective_kv_len,
                len(keep),
                prefix,
                recent,
                16,
            )
            is not None
        )
        expected = [read_members(t, intent.source_block_ids, keep) for t in snapshot]
        row = whole_block_row(
            intent.source_block_ids,
            intent.source_effective_kv_len,
            len(keep),
            prefix,
            recent,
            16,
        )
        if row is None or not whole:
            row = intent.source_block_ids[: (len(keep) + 15) // 16]
        result = original_reclaim(intent, tables)
        if lifecycle == "failure":
            # The real destructive kernel was enqueued; continuing is invalid.
            raise RuntimeError("injected failure after destructive compaction")
        completion_event = torch.cuda.Event()
        completion_event.record(runner.main_stream)
        reference = expected
        reference_row = row
        assert result.retained_block_ids == row
        events.append(
            {
                "event": "payload_exact_before_free",
                "whole": whole,
                "K": len(keep),
                "row": row,
            }
        )
        return result

    def process_aborts():
        nonlocal pending_abort
        if pending_abort:
            engine.abort_request(["A"])
            pending_abort = False
            events.append({"event": "abort_before_update"})
        return original_aborts()

    def checked_update(output, result):
        nonlocal completed_reclaims
        intent = output.reclaim_request
        if intent:
            assert completion_event.query(), "Output returned before GPU publication"
            for view, wanted in zip(views.values(), reference):
                actual = read_members(view.cpu(), reference_row, range(wanted.shape[0]))
                assert torch.equal(actual, wanted)
            if lifecycle == "eos":
                request = scheduler.requests[internal_ids["A"]]
                request.sampling_params.ignore_eos = False
                request.sampling_params.update_from_generation_config(
                    {}, tokenizer.eos_token_id
                )
                result.sampled_token_ids[result.req_id_to_index[request.request_id]] = [
                    request.sampling_params.eos_token_id
                ]
                events.append({"event": "controlled_eos_before_update"})
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        answer = original_update(output, result)
        if intent:
            completed_reclaims += 1
            request = scheduler.requests.get(internal_ids["A"])
            active = request is not None and not request.is_finished()
            row = scheduler.kv_cache_manager.get_block_ids(internal_ids["A"])[0]
            removed = set(intent.source_block_ids) - set(
                result.reclaim_result.retained_block_ids
            )
            released.update(removed)
            events.append(
                {
                    "event": "canonical_commit",
                    "active": active,
                    "released": sorted(removed),
                    "new_row": row,
                    "free_before": free_before,
                    "free_after": (
                        scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                    ),
                }
            )
        return answer

    scheduler.schedule = controlled_schedule
    runner.reclamation.reclaim = checked_reclaim
    scheduler.update_from_output = checked_update
    core._process_aborts_queue = process_aborts
    runner.reclamation.prepare_attn = checked_prepare
    first_layer.impl.forward = checked_attention
    params = SamplingParams(
        temperature=0, max_tokens=max_tokens, ignore_eos=True, seed=0
    )
    outputs = {}
    for step in range(180):
        if step == 0 or step == 19:
            rid = "A" if step == 0 else "B"
            text = (
                "Mary went to the garden. "
                if rid == "A"
                else "John visited the office. "
            )
            ids = tokenizer.encode(text * 80, add_special_tokens=False)[:256]
            if lifecycle == "q0" and rid == "B":
                ids = ids[:128]
            internal_ids[rid] = engine.add_request(
                rid, {"prompt_token_ids": ids}, params
            )
        t0 = time.perf_counter()
        try:
            batch_outputs = engine.step()
        except RuntimeError as exc:
            if lifecycle != "failure" or str(exc) != (
                "injected failure after destructive compaction"
            ):
                raise
            assert completed_reclaims == 0
            intent = next(e for e in events if e["event"] == "intent")
            row = scheduler.kv_cache_manager.get_block_ids(internal_ids["A"])[0]
            assert row == intent["old_row"]
            assert not any(e["event"] == "canonical_commit" for e in events)
            (run / "evidence.json").write_text(
                json.dumps({"status": "PASS", "fail_stop": str(exc), "row": row})
            )
            print("PASS: destructive failure propagated; no scheduler commit/free")
            return
        for output in batch_outputs:
            outputs[output.request_id] = {
                "tokens": list(output.outputs[0].token_ids),
                "finished": output.finished,
            }
        for rid, request in scheduler.requests.items():
            logical = request.num_computed_tokens
            if request.num_preemptions == 0:
                assert logical >= previous_logical.get(rid, 0)
            previous_logical[rid] = logical
        if reference is not None and internal_ids["A"] in scheduler.requests:
            row = scheduler.kv_cache_manager.get_block_ids(internal_ids["A"])[0]
            assert row[: len(reference_row)] == reference_row
            # Later decode and B reuse must preserve all retained historical bytes.
            for view, wanted in zip(views.values(), reference):
                actual = read_members(view.cpu(), row, range(wanted.shape[0]))
                assert torch.equal(actual, wanted)
        steps.append(
            {
                "step": step,
                "elapsed_s": time.perf_counter() - t0,
                "requests": {
                    rid: {"L": r.num_computed_tokens, "E": r.effective_kv_len}
                    for rid, r in scheduler.requests.items()
                },
            }
        )
        if preempted and internal_ids["A"] in scheduler.requests:
            request = scheduler.requests[internal_ids["A"]]
            assert request.effective_kv_len == request.num_computed_tokens
        if step >= 19 and not engine.has_unfinished_requests():
            break
    assert completed_reclaims == (2 if lifecycle == "repeat" else 1)
    if lifecycle == "preempt":
        assert preempted
    if lifecycle not in ("abort", "eos"):
        assert attention_checks > 0
    assert reused, "B never obtained a released canonical block"
    overwritten = [
        p
        for p in reused
        if not torch.equal(views[next(iter(views))][p].cpu(), reuse_before[p])
    ]
    assert overwritten, "B did not actually write new KV into reused pages"
    assert outputs["B"]["finished"]
    if lifecycle != "abort":
        assert outputs["A"]["finished"]
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 32
    events.append({"event": "B_wrote_released_ids", "ids": overwritten})
    (run / "evidence.json").write_text(
        json.dumps(
            {
                "events": events,
                "steps": steps,
                "outputs": outputs,
                "attention_fp32_checks": attention_checks,
                "attention_atol": 1.6e-2,
                "attention_rtol": 1.6e-2,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "path": path,
                "lifecycle": lifecycle,
                "reused_ids": overwritten,
            }
        )
    )


if __name__ == "__main__":
    main()
