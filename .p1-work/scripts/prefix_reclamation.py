# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-8B Prefix Gate: canonical APC, protected payload and physical reuse."""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import KVEventsConfig
from vllm.v1.kv_reclamation import sink_recent_ranges


@torch.inference_mode()
def main():
    run = Path(os.environ["P1_RUN_DIR"])
    mode = os.environ.get("P1_PREFIX_MODE", "token")
    lifecycle = os.environ.get("P1_LIFECYCLE", "repeat")
    policy = os.environ.get("P1_POLICY", "sink_recent")
    feature = mode != "native"
    forced = mode in ("whole", "token", "gap")
    os.environ["VLLM_KV_TRACE_FILE"] = str(run / "kv_trace.jsonl")
    os.environ["VLLM_KV_TRACE_IDS"] = "1"
    torch.cuda.set_per_process_memory_fraction(0.50)
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/models/Qwen3-8B", local_files_only=True
    )
    options = EngineArgs(
        model="/data/models/Qwen3-8B",
        dtype="bfloat16",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        block_size=16,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_cascade_attn=True,
        scheduler_reserve_full_isl=False,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=256 * 2**20,
        num_gpu_blocks_override=33 if mode == "pressure" else 49,
        gpu_memory_utilization=0.40,
        seed=0,
        disable_log_stats=True,
        kv_events_config=KVEventsConfig(enable_kv_cache_events=True, publisher="null"),
    )
    if feature:
        options.additional_config = {
            "kv_reclamation": {
                "pressure": mode == "pressure",
                "retention": 1.0 if mode == "identity" else 0.5,
                "min_keep_tokens": 32,
                "sink_tokens": 16,
                "recent_tokens": 16,
                "policy": policy,
                "stats_path": os.environ.get("P1_STATS"),
            }
        }
    engine = LLMEngine.from_engine_args(options)
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    runner = core.model_executor.driver_worker.worker.model_runner
    manager = scheduler.kv_cache_manager
    owner = manager.coordinator.single_type_managers[0]
    pool = manager.block_pool
    context = engine.vllm_config.compilation_config.static_forward_context
    views = [
        layer.kv_cache
        for layer in context.values()
        if isinstance(getattr(layer, "kv_cache", None), torch.Tensor)
    ]
    assert len(views) == 36
    assert engine.vllm_config.max_concurrent_batches == 1 and core.batch_queue is None
    (run / "identity.json").write_text(
        json.dumps(
            {
                "args": vars(options),
                "mode": mode,
                "lifecycle": lifecycle,
                "runner": f"{type(runner).__module__}.{type(runner).__name__}",
                "torch": torch.__version__,
                "triton": __import__("triton").__version__,
                "gpu": torch.cuda.get_device_name(0),
                "kv_views": [
                    {
                        "shape": list(v.shape),
                        "stride": list(v.stride()),
                        "ptr": v.data_ptr(),
                    }
                    for v in views
                ],
            },
            indent=2,
            default=str,
        )
    )
    common = tokenizer.encode("Read the following facts carefully. " * 30)[:64]
    a_prompt = common + tokenizer.encode("Alice visited the garden. " * 60)[:192]
    b_prompt = common + tokenizer.encode("Bob stayed at the office. " * 40)[:96]
    assert len(common) == 64 and len(a_prompt) == 256 and len(b_prompt) == 160
    assert a_prompt[64] != b_prompt[64]
    prompts = {"A": a_prompt, "B": b_prompt, "C": a_prompt}
    if mode == "pressure":
        prompts["C"] = common + tokenizer.encode("Carol crossed the river. " * 60)[:192]
    if mode == "no_gain":
        prompts["B"] = a_prompt
    ids, outputs, events, steps = {}, {}, [], []
    references, released, reuse_before, rewritten = {}, set(), {}, set()
    published_prefix = {}
    gap_hold = None
    pending, completion, active_batch = None, None, None
    checks = 0
    abort_pending = False
    shared_ids = []
    preempted = False
    original_schedule = scheduler.schedule
    original_update = scheduler.update_from_output
    original_evict = pool.evict_blocks
    original_aborts = core._process_aborts_queue

    def members(view, row, indices):
        return torch.stack([view[row[i // 16], :, i % 16, :] for i in indices])

    def checked_evict(block_ids):
        before = {i: pool.blocks[i].ref_cnt for i in block_ids}
        free = pool.get_num_free_blocks()
        hashes = sum(
            int(pool.blocks[i].block_hash is not None)
            + len(pool.cached_block_hashes_by_block.get(i, ()))
            for i in block_ids
        )
        start = time.perf_counter_ns()
        answer = original_evict(block_ids)
        elapsed = time.perf_counter_ns() - start
        assert pool.get_num_free_blocks() == free
        assert before == {i: pool.blocks[i].ref_cnt for i in block_ids}
        assert all(pool.blocks[i].block_hash is None for i in block_ids)
        assert all(i not in pool.cached_block_hashes_by_block for i in block_ids)
        events.append(
            {
                "event": "deindex",
                "ids": sorted(block_ids),
                "hashes": hashes,
                "ns": elapsed,
            }
        )
        return answer

    def checked_schedule(*args, **kwargs):
        nonlocal abort_pending, shared_ids, preempted, gap_hold
        step = len(steps)
        if mode == "gap" and step == 17:
            gap_hold = owner.req_to_blocks[ids["A"]][8]
            pool.touch([gap_hold])
            owner.req_to_blocks["fixture_prefix_hold"] = [gap_hold]
            events.append({"event": "shared_gap_hold", "index": 8, "P": 144})
        if mode == "gap" and step == 21:
            del owner.req_to_blocks["fixture_prefix_hold"]
            pool.free_blocks([gap_hold])
            events.append({"event": "shared_gap_release"})
        if lifecycle == "preempt" and step == 28:
            request = scheduler.requests[ids["A"]]
            assert request.effective_kv_len < request.num_computed_tokens
            scheduler.running.remove(request)
            scheduler._preempt_request(request, time.monotonic())
            assert request.num_computed_tokens == request.effective_kv_len == 0
            references.pop(ids["A"])
            preempted = True
            events.append({"event": "preempt_reset"})
        output = original_schedule(*args, **kwargs)
        targets = {16: ("A", 160), 32: ("A", 128), 33: ("B", 128)}
        if mode in ("token", "gap"):
            targets = {17: ("A", 159), 34: ("B", 127), 35: ("A", 111)}
        if mode == "gap":
            targets[17] = ("A", 191)
        if lifecycle != "repeat":
            targets = {min(targets): targets[min(targets)]}
        if forced and step in targets:
            external, keep = targets[step]
            request = scheduler.requests[ids[external]]
            output.reclaim_request = scheduler.make_reclaim_request(
                request, output, keep
            )
            assert output.reclaim_request.protected_prefix_tokens == (
                144 if mode == "gap" and step == 17 else 64
            )
            abort_pending = lifecycle == "abort"
        for new in output.scheduled_new_reqs:
            events.append(
                {
                    "event": "apc_admission",
                    "id": new.req_id,
                    "hit_tokens": new.num_computed_tokens,
                    "row": new.block_ids[0],
                }
            )
            if new.req_id == ids.get("B"):
                expected = 240 if mode == "no_gain" else 64
                assert new.num_computed_tokens == expected
                assert (
                    new.block_ids[0][: expected // 16]
                    == manager.get_block_ids(ids["A"])[0][: expected // 16]
                )
                shared_ids = new.block_ids[0][: expected // 16]
            if new.req_id == ids.get("C") and forced:
                assert new.num_computed_tokens == (144 if mode == "gap" else 64)
                assert new.block_ids[0][:4] == manager.get_block_ids(ids["B"])[0][:4]
        if output.reclaim_request is not None:
            intent = output.reclaim_request
            assert all(
                pool.blocks[i].ref_cnt == 1
                for i in intent.source_block_ids[intent.protected_prefix_tokens // 16 :]
            )
            events.append({"event": "intent", **vars(intent), "step": step})
        for rid in output.num_scheduled_tokens:
            if rid != ids.get("C"):
                continue
            for page in set(manager.get_block_ids(rid)[0]) & released:
                if page not in reuse_before:
                    reuse_before[page] = views[0][page].cpu()
        return output

    def process_aborts():
        nonlocal abort_pending
        if abort_pending:
            engine.abort_request(["A"])
            abort_pending = False
            events.append({"event": "abort_before_update"})
        return original_aborts()

    if feature:
        original_reclaim = runner.reclamation.reclaim
        original_prepare = runner.reclamation.prepare_attn
        original_compact = runner.reclamation.compactor.compact
        first_layer = next(
            iter(engine.vllm_config.compilation_config.static_forward_context.values())
        )
        original_attention = first_layer.impl.forward

        def checked_prepare(batch):
            nonlocal active_batch
            physical = original_prepare(batch)
            active_batch = batch
            for i, rid in enumerate(batch.req_ids):
                q = int(batch.num_scheduled_tokens[i])
                index = batch.idx_mapping_np[i]
                effective = int(runner.reclamation.effective_kv_len_np[index])
                assert int(physical.seq_lens[i].cpu()) == effective + q
                assert scheduler.requests[rid].effective_kv_len == effective
            return physical

        def checked_attention(
            layer, query, key, value, cache, metadata, output, **kwargs
        ):
            nonlocal checks
            answer = original_attention(
                layer, query, key, value, cache, metadata, output, **kwargs
            )
            starts = metadata.query_start_loc.cpu().tolist()
            for i, rid in enumerate(active_batch.req_ids):
                if rid not in references or starts[i + 1] - starts[i] != 1:
                    continue
                length = int(metadata.seq_lens[i].cpu())
                row = metadata.block_table[i].cpu().tolist()
                k, v = members(cache, row, range(length)).cpu().float().chunk(2, dim=-1)
                q = query[starts[i]].cpu().float()
                groups = q.shape[0] // k.shape[1]
                k, v = (
                    k.repeat_interleave(groups, dim=1),
                    v.repeat_interleave(groups, dim=1),
                )
                logits = torch.einsum("hd,thd->ht", q, k) * q.shape[-1] ** -0.5
                wanted = torch.einsum("ht,thd->hd", logits.softmax(-1), v)
                torch.testing.assert_close(
                    output[starts[i]].cpu().float().reshape_as(wanted),
                    wanted,
                    atol=1.6e-2,
                    rtol=1.6e-2,
                )
                checks += 1
            return answer

        def checked_compact(row, keep, **kwargs):
            pending["keep"] = keep.cpu().tolist()
            p = pending["p"]
            assert pending["keep"][:p] == list(range(p))
            assert pending["keep"] == sorted(set(pending["keep"]))
            return original_compact(row, keep, **kwargs)

        def checked_reclaim(intent, tables):
            nonlocal pending, completion
            p = intent.protected_prefix_tokens
            source = [
                members(
                    v, intent.source_block_ids, range(intent.source_effective_kv_len)
                ).cpu()
                for v in views
            ]
            protected = [
                members(v, intent.source_block_ids, range(p)).cpu() for v in views
            ]
            prefix, recent = sink_recent_ranges(
                intent.source_effective_kv_len, intent.target_keep_tokens, 16, p
            )
            pending = {"source": source, "protected": protected, "p": p}
            pending["other_requests"] = {}
            for rid, index in runner.req_states.req_id_to_index.items():
                if rid == intent.request_id or rid not in scheduler.requests:
                    continue
                row = list(manager.get_block_ids(rid)[0])
                length = int(runner.reclamation.effective_kv_len_np[index])
                pending["other_requests"][rid] = (
                    row,
                    [members(v, row, range(length)).cpu() for v in views],
                )
            pending["keep"] = [
                *range(prefix),
                *range(recent, intent.source_effective_kv_len),
            ]
            begin, completion = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record(runner.main_stream)
            result = original_reclaim(intent, tables)
            completion.record(runner.main_stream)
            pending["begin"] = begin
            pending["row"] = result.retained_block_ids
            pending["expected"] = [s[pending["keep"]] for s in source]
            return result

        runner.reclamation.prepare_attn = checked_prepare
        runner.reclamation.reclaim = checked_reclaim
        runner.reclamation.compactor.compact = checked_compact
        first_layer.impl.forward = checked_attention

    def checked_update(output, result):
        intent = output.reclaim_request
        free_before = pool.get_num_free_blocks()
        if intent is not None:
            assert completion.query(), "GPU publication must precede canonical free"
            for v, expected, protected in zip(
                views, pending["expected"], pending["protected"]
            ):
                assert torch.equal(
                    members(v, pending["row"], range(len(pending["keep"]))).cpu(),
                    expected,
                )
                assert torch.equal(
                    members(
                        v,
                        intent.source_block_ids,
                        range(intent.protected_prefix_tokens),
                    ).cpu(),
                    protected,
                )
            for row, snapshots in pending["other_requests"].values():
                for view, wanted in zip(views, snapshots):
                    assert torch.equal(
                        members(view, row, range(wanted.shape[0])).cpu(), wanted
                    )
            if lifecycle == "eos":
                request = scheduler.requests[intent.request_id]
                request.sampling_params.ignore_eos = False
                request.sampling_params.update_from_generation_config(
                    {}, tokenizer.eos_token_id
                )
                result.sampled_token_ids[result.req_id_to_index[intent.request_id]] = [
                    request.sampling_params.eos_token_id
                ]
        answer = original_update(output, result)
        if intent is not None:
            removed = set(intent.source_block_ids) - set(pending["row"])
            released.update(removed)
            request = scheduler.requests.get(intent.request_id)
            if request is not None and not request.is_finished():
                assert (
                    owner.num_cached_block[intent.request_id]
                    == intent.protected_prefix_tokens // 16
                )
                assert pool.get_num_free_blocks() == free_before + len(removed)
                references[intent.request_id] = (pending["row"], pending["expected"])
                published_prefix[intent.request_id] = (
                    intent.protected_prefix_tokens // 16
                )
            events.append(
                {
                    "event": "completed",
                    "id": intent.request_id,
                    "released": sorted(removed),
                    "keep": pending["keep"],
                    "ms": pending["begin"].elapsed_time(completion),
                    "free_before": free_before,
                    "free_after": pool.get_num_free_blocks(),
                }
            )
        return answer

    pool.evict_blocks = checked_evict
    scheduler.schedule = checked_schedule
    scheduler.update_from_output = checked_update
    core._process_aborts_queue = process_aborts
    for step in range(140):
        arrivals = {0: "A", 1: "B", (6 if mode == "pressure" else 18): "C"}
        external = arrivals.get(step)
        if external is not None:
            params = SamplingParams(
                temperature=0,
                max_tokens=4 if external == "C" else (16 if mode == "no_gain" else 60),
                ignore_eos=True,
                seed=0,
            )
            ids[external] = engine.add_request(
                external, {"prompt_token_ids": prompts[external]}, params
            )
        start = time.perf_counter()
        for output in engine.step():
            outputs[output.request_id] = {
                "tokens": list(output.outputs[0].token_ids),
                "finished": output.finished,
            }
        for rid, (row, expected) in references.items():
            if rid not in scheduler.requests:
                continue
            assert manager.get_block_ids(rid)[0][: len(row)] == row
            for view, wanted in zip(views, expected):
                assert torch.equal(
                    members(view, row, range(wanted.shape[0])).cpu(), wanted
                )
        for page, before in reuse_before.items():
            if not torch.equal(views[0][page].cpu(), before):
                rewritten.add(page)
        free_ids = {b.block_id for b in pool.free_block_queue.get_all_free_blocks()}
        for page in shared_ids:
            owners = sum(
                any(b.block_id == page for b in row)
                for row in owner.req_to_blocks.values()
            )
            assert pool.blocks[page].ref_cnt == owners
            assert (page in free_ids) == (owners == 0)
        if preempted and ids["A"] in scheduler.requests:
            a = scheduler.requests[ids["A"]]
            assert a.effective_kv_len == a.num_computed_tokens
            assert owner.num_cached_block[ids["A"]] >= 16
        for rid in references:
            if rid not in scheduler.requests:
                continue
            r = scheduler.requests[rid]
            assert r.effective_kv_len < r.num_computed_tokens
            p = published_prefix[rid]
            assert owner.num_cached_block[rid] == p
            assert all(b.block_hash is None for b in owner.req_to_blocks[rid][p:])
        if mode == "no_gain" and step == 1:
            assert scheduler._select_reclaim_candidate({}) is None
            events.append(
                {
                    "event": "no_gain",
                    "P": 240,
                    "E": scheduler.requests[ids["A"]].effective_kv_len,
                    "min_private": 32,
                }
            )
        steps.append(
            {
                "step": step,
                "s": time.perf_counter() - start,
                "free": pool.get_num_free_blocks(),
                "requests": {
                    rid: {
                        "L": r.num_computed_tokens,
                        "E": r.effective_kv_len,
                        "cached_blocks": owner.num_cached_block.get(rid),
                        "preemptions": r.num_preemptions,
                    }
                    for rid, r in scheduler.requests.items()
                },
            }
        )
        if step >= 18 and not engine.has_unfinished_requests():
            break
    completed = [e for e in events if e["event"] == "completed"]
    if forced:
        assert len(completed) == (3 if lifecycle == "repeat" else 1)
    if lifecycle == "preempt":
        assert preempted
    if mode == "pressure":
        assert completed
        trace = [
            json.loads(line)
            for line in (run / "kv_trace.jsonl").read_text().splitlines()
        ]
        assert any(e["event"] == "allocation_miss" for e in trace)
        assert any(e["event"] == "reclaim_select" for e in trace)
    if forced or mode == "pressure":
        assert rewritten, "C must write into genuinely released IDs"
        if lifecycle not in ("eos", "abort"):
            assert checks > 0
    assert outputs["B"]["finished"] and outputs["C"]["finished"]
    if lifecycle != "abort":
        assert outputs["A"]["finished"]
    assert pool.get_num_free_blocks() == options.num_gpu_blocks_override - 1
    evidence = {
        "status": "PASS",
        "mode": mode,
        "lifecycle": lifecycle,
        "policy": policy,
        "events": events,
        "steps": steps,
        "outputs": outputs,
        "rewritten_ids": sorted(rewritten),
        "attention_fp32_checks": checks,
        "attention_atol_rtol": 1.6e-2,
    }
    (run / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(
        json.dumps(
            {
                k: evidence[k]
                for k in (
                    "status",
                    "mode",
                    "lifecycle",
                    "policy",
                    "rewritten_ids",
                    "attention_fp32_checks",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
