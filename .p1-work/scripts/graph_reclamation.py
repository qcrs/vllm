# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-8B graph gate with real canonical release, reuse and replay."""

import argparse
import dataclasses
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.v1.kv_reclamation import sink_recent_ranges, whole_block_row


def layout(tensor):
    return {
        "ptr": tensor.data_ptr(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
    }


def members(view, row, count):
    ids = torch.arange(count, device=view.device)
    pages = torch.tensor(row, device=view.device)[ids // 16]
    return view[pages, :, ids % 16, :]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["full", "piecewise", "eager"], default="full"
    )
    parser.add_argument("--path", choices=["whole", "token"], default="token")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--diagnose-numerics", action="store_true")
    parser.add_argument(
        "--policy",
        choices=["sink_recent", "random", "triattention"],
        default="sink_recent",
    )
    args = parser.parse_args()
    run = Path(os.environ["P1_RUN_DIR"])
    if args.mode == "piecewise":
        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
    os.environ["VLLM_KV_TRACE_FILE"] = str(run / "kv_trace.jsonl")
    os.environ["VLLM_KV_TRACE_IDS"] = "1"
    torch.cuda.set_per_process_memory_fraction(0.50)
    options = EngineArgs(
        model="/data/models/Qwen3-8B",
        dtype="bfloat16",
        enforce_eager=args.mode == "eager",
        attention_backend="FLASH_ATTN",
        block_size=16,
        enable_prefix_caching=args.prefix,
        async_scheduling=False,
        disable_cascade_attn=True,
        scheduler_reserve_full_isl=False,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        kv_cache_memory_bytes=256 * 2**20,
        num_gpu_blocks_override=33,
        gpu_memory_utilization=0.40,
        seed=0,
        disable_log_stats=True,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode="PIECEWISE"
            if args.mode == "piecewise"
            else "FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[2, 4],
        ),
        additional_config={
            "kv_reclamation": {
                "min_keep_tokens": 32,
                "policy": args.policy,
                "stats_path": os.environ.get("P1_STATS"),
            }
        },
    )
    started = time.perf_counter()
    engine = LLMEngine.from_engine_args(options)
    startup_s = time.perf_counter() - started
    config = engine.vllm_config
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    manager, pool = scheduler.kv_cache_manager, scheduler.kv_cache_manager.block_pool
    runner = core.model_executor.driver_worker.worker.model_runner
    graph_manager = runner.cudagraph_manager
    context = config.compilation_config.static_forward_context
    layers = [
        layer
        for layer in context.values()
        if isinstance(getattr(layer, "kv_cache", None), torch.Tensor)
    ]
    views = [layer.kv_cache for layer in layers]
    assert (
        len(views) == 36
        and config.max_concurrent_batches == 1
        and core.batch_queue is None
    )
    captures = dataclasses.asdict(compilation_counter)
    graph_ids = {str(desc): id(graph) for desc, graph in graph_manager.graphs.items()}
    if args.mode == "full":
        assert (
            graph_ids and graph_manager.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        )

    def static_layouts():
        return {
            **{
                name: layout(value)
                for name, value in vars(runner.input_buffers).items()
                if isinstance(value, torch.Tensor)
            },
            "physical_seq_lens": layout(runner.reclamation.kv_seq_lens),
            "cache_positions": layout(runner.reclamation.cache_positions),
            "effective_kv_len": layout(runner.reclamation.effective_kv_len.gpu),
            "resident_row": layout(runner.block_tables.block_tables[0].gpu),
            "input_row": layout(runner.block_tables.input_block_tables[0]),
            "slots": layout(runner.block_tables.slot_mappings),
            "kv": [layout(view) for view in views],
        }

    static = static_layouts()
    (run / "identity.json").write_text(
        json.dumps(
            {
                "args": vars(options),
                "mode": args.mode,
                "resolved_graph_mode": str(graph_manager.cudagraph_mode),
                "startup_s": startup_s,
                "torch": torch.__version__,
                "triton": __import__("triton").__version__,
                "gpu": torch.cuda.get_device_name(0),
                "static": static,
                "captures": captures,
                "graphs": graph_ids,
                "runner": type(runner).__module__,
            },
            indent=2,
            default=str,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    common = tokenizer.encode("Read the following facts carefully. " * 20)[:64]
    prompts = {
        "A": common + tokenizer.encode("Alice visited the garden. " * 60)[:192],
        "B": common + tokenizer.encode("Bob stayed at the office. " * 40)[:64],
        "C": common + tokenizer.encode("Carol crossed the river. " * 60)[:128],
    }
    ids, outputs, steps, events = {}, {}, [], []
    references, released, rewritten, reuse_before = {}, set(), set(), {}
    active = physical = last_desc = pending = completion = None
    counts = Counter()
    original_schedule, original_update = (
        scheduler.schedule,
        scheduler.update_from_output,
    )
    original_inputs, original_prepare = (
        runner.prepare_inputs,
        runner.reclamation.prepare_attn,
    )
    original_execute, original_dispatch = runner.execute_model, graph_manager.dispatch
    original_reclaim, original_compact = (
        runner.reclamation.reclaim,
        runner.reclamation.compactor.compact,
    )
    original_attention = layers[0].impl.forward

    def checked_schedule(*a, **kw):
        output = original_schedule(*a, **kw)
        step = len(steps)
        if step in (16, 32):
            request = scheduler.requests[ids["A"]]
            keep = (
                (160 if step == 16 else 128)
                if args.prefix
                else (128 if step == 16 else 64)
            )
            if args.path == "token":
                keep -= 1 if step == 16 else 2
            output.reclaim_request = scheduler.make_reclaim_request(
                request, output, keep
            )
            assert output.reclaim_request.protected_prefix_tokens == (
                64 if args.prefix else 0
            )
        if step % 2:
            output.num_scheduled_tokens = dict(
                reversed(list(output.num_scheduled_tokens.items()))
            )
        for new in output.scheduled_new_reqs:
            events.append(
                {
                    "event": "admission",
                    "step": step,
                    "id": new.req_id,
                    "hit": new.num_computed_tokens,
                    "row": new.block_ids[0],
                }
            )
            assert new.num_computed_tokens == (
                64 if args.prefix and new.req_id != ids["A"] else 0
            )
        return output

    def checked_inputs(*a, **kw):
        nonlocal active
        active = original_inputs(*a, **kw)
        return active

    def checked_prepare(batch):
        nonlocal physical
        physical = original_prepare(batch)
        lengths = physical.seq_lens.cpu().tolist()
        positions = batch.positions.cpu().tolist()
        physical_positions = physical.positions.cpu().tolist()
        starts = batch.query_start_loc_np.tolist()
        reuse_before.clear()
        for i, rid in enumerate(batch.req_ids):
            index, q = batch.idx_mapping_np[i], int(batch.num_scheduled_tokens[i])
            effective = int(runner.reclamation.effective_kv_len_np[index])
            logical = int(runner.req_states.num_computed_tokens_np[index])
            assert lengths[i] == physical.seq_lens_cpu_upper_bound[i] == effective + q
            assert batch.seq_lens[i] == logical + q
            assert positions[starts[i] : starts[i + 1]] == list(
                range(logical, logical + q)
            )
            assert physical_positions[starts[i] : starts[i + 1]] == list(
                range(effective, effective + q)
            )
            if rid == ids.get("C"):
                row = manager.get_block_ids(rid)[0]
                for position in range(effective, effective + q):
                    page, offset = row[position // 16], position % 16
                    if page in released:
                        reuse_before[page, offset] = [
                            view[page, :, offset, :].clone() for view in views
                        ]
        assert lengths[batch.num_reqs :] == [0] * (
            batch.num_reqs_after_padding - batch.num_reqs
        )
        assert static_layouts() == static
        return physical

    def checked_dispatch(*a, **kw):
        nonlocal last_desc
        last_desc = original_dispatch(*a, **kw)
        return last_desc

    def checked_attention(layer, query, key, value, cache, metadata, output, **kw):
        answer = original_attention(
            layer, query, key, value, cache, metadata, output, **kw
        )
        starts = metadata.query_start_loc.cpu().tolist()
        for i, rid in enumerate(active.req_ids):
            if rid not in references or starts[i + 1] - starts[i] != 1:
                continue
            count, row = (
                int(metadata.seq_lens[i].cpu()),
                metadata.block_table[i].cpu().tolist(),
            )
            k, v = members(cache, row, count).cpu().float().chunk(2, dim=-1)
            q = query[starts[i]].cpu().float()
            group = q.shape[0] // k.shape[1]
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
            scores = torch.einsum("hd,thd->ht", q, k) * q.shape[-1] ** -0.5
            expected = torch.einsum("ht,thd->hd", scores.softmax(-1), v)
            torch.testing.assert_close(
                output[starts[i]].cpu().float().reshape_as(expected),
                expected,
                atol=1.6e-2,
                rtol=1.6e-2,
            )
            counts["fp32_attention"] += 1
        return answer

    def checked_execute(*a, **kw):
        answer = original_execute(*a, **kw)
        state = runner.execute_model_state
        if state is None:
            return answer
        mode = last_desc.cg_mode
        counts[mode.name] += 1
        batch = state.input_batch
        metadata = next(iter(state.attn_metadata.values()))
        assert metadata.seq_lens.data_ptr() == runner.reclamation.kv_seq_lens.data_ptr()
        assert (
            metadata.block_table.data_ptr()
            == runner.block_tables.input_block_tables[0].data_ptr()
        )
        assert (
            metadata.query_start_loc.data_ptr()
            == runner.input_buffers.query_start_loc.data_ptr()
        )
        expected_slots = []
        for i, rid in enumerate(batch.req_ids):
            idx = batch.idx_mapping_np[i]
            effective = int(runner.reclamation.effective_kv_len_np[idx])
            row = manager.get_block_ids(rid)[0]
            q = int(batch.num_scheduled_tokens[i])
            expected_slots.extend(
                row[j // 16] * 16 + j % 16 for j in range(effective, effective + q)
            )
            assert metadata.block_table[i, : len(row)].cpu().tolist() == row
        slots = (
            runner.block_tables.slot_mappings[0, : batch.num_tokens_after_padding]
            .cpu()
            .tolist()
        )
        assert slots == expected_slots + [-1] * (
            batch.num_tokens_after_padding - batch.num_tokens
        )
        if batch.num_reqs_after_padding > batch.num_reqs:
            assert metadata.block_table[batch.num_reqs :].count_nonzero() == 0
            counts["padded_batches"] += 1
        for (page, offset), snapshots in reuse_before.items():
            for view, before in zip(views, snapshots):
                after = view[page, :, offset, :]
                assert torch.isfinite(after).all(), (page, offset)
                assert not torch.equal(
                    after.view(torch.int16), before.view(torch.int16)
                )
            rewritten.add(page)
        if mode == CUDAGraphMode.FULL and references:
            # Re-run the same append slots with eager attention on the same retained KV.
            # Restore graph-written slots afterwards to preserve the tested trajectory.
            graph_hidden = state.hidden_states.clone()
            gpu_slots = torch.tensor(expected_slots, device=runner.device)
            pages, offsets = gpu_slots // 16, gpu_slots % 16
            appended = [view[pages, :, offsets, :].clone() for view in views]
            actual = graph_hidden[: batch.num_tokens]
            graph_logits = runner.model.compute_logits(actual[batch.logits_indices])
            graph_tokens = graph_logits.argmax(-1)
            comparison_references = (
                ("runtime", "capture_bound")
                if args.diagnose_numerics
                else ("capture_bound",)
            )
            for reference in comparison_references:
                reference_metadata = {
                    name: dataclasses.replace(value, max_seq_len=options.max_model_len)
                    if reference == "capture_bound"
                    else value
                    for name, value in state.attn_metadata.items()
                }
                with set_forward_context(
                    reference_metadata,
                    config,
                    num_tokens=batch.num_tokens_after_padding,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=state.slot_mappings_by_layer,
                    is_padding=batch.is_padding,
                ):
                    eager = runner.model(
                        input_ids=batch.input_ids,
                        positions=batch.positions,
                        **runner.model_state.prepare_inputs(batch, runner.req_states),
                    )
                expected = eager[: batch.num_tokens]
                eager_logits = runner.model.compute_logits(
                    expected[batch.logits_indices]
                )
                close = torch.isclose(actual, expected, atol=6.25e-2, rtol=3e-2)
                comparison = {
                    "step": len(steps),
                    "reference": reference,
                    "descriptor": dataclasses.asdict(last_desc),
                    "req_ids": batch.req_ids,
                    "seq_lens": metadata.seq_lens.cpu().tolist(),
                    "runtime_max_seq_len": metadata.max_seq_len,
                    "hidden_max_abs": (actual.float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                    "hidden_mismatches": (~close).sum().item(),
                    "logit_max_abs": (graph_logits.float() - eager_logits.float())
                    .abs()
                    .max()
                    .item(),
                    "graph_argmax": graph_tokens.cpu().tolist(),
                    "eager_argmax": eager_logits.argmax(-1).cpu().tolist(),
                    "kv_max_abs": max(
                        (view[pages, :, offsets, :].float() - values.float())
                        .abs()
                        .max()
                        .item()
                        for view, values in zip(views, appended)
                    ),
                }
                with (run / "comparisons.jsonl").open("a") as log:
                    log.write(json.dumps(comparison, default=str) + "\n")
                if not close.all():
                    torch.save(
                        {
                            "graph": actual.cpu(),
                            "eager": expected.cpu(),
                            "graph_logits": graph_logits.cpu(),
                            "eager_logits": eager_logits.cpu(),
                            "context": comparison,
                        },
                        run / f"difference-{len(steps)}-{reference}.pt",
                    )
                for view, values in zip(views, appended):
                    view[pages, :, offsets, :] = values
                if reference == "capture_bound":
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    assert comparison["kv_max_abs"] == 0
                    assert torch.equal(graph_tokens, eager_logits.argmax(-1))
            counts["full_vs_eager"] += 1
        steps.append(
            {
                "step": len(steps),
                "mode": mode.name,
                "descriptor": dataclasses.asdict(last_desc),
                "req_ids": batch.req_ids,
                "mapping": batch.idx_mapping_np.tolist(),
                "starts": batch.query_start_loc_np.tolist(),
                "seq_lens": metadata.seq_lens.cpu().tolist(),
                "logical_seq_lens": batch.seq_lens.cpu().tolist(),
                "slots": slots,
            }
        )
        with (run / "steps.jsonl").open("a") as log:
            log.write(json.dumps(steps[-1], default=str) + "\n")
        return answer

    def checked_compact(row, keep, **kw):
        pending["keep"] = keep.cpu().tolist()
        return original_compact(row, keep, **kw)

    def checked_reclaim(intent, tables):
        nonlocal pending, completion
        assert not torch.cuda.is_current_stream_capturing()
        prefix, recent = sink_recent_ranges(
            intent.source_effective_kv_len,
            intent.target_keep_tokens,
            16,
            intent.protected_prefix_tokens,
        )
        keep = [*range(prefix), *range(recent, intent.source_effective_kv_len)]
        pending = {
            "keep": keep,
            "source": [
                members(
                    view, intent.source_block_ids, intent.source_effective_kv_len
                ).cpu()
                for view in views
            ],
        }
        pending["others"] = [
            (
                row,
                [
                    members(
                        view, row, int(runner.reclamation.effective_kv_len_np[index])
                    ).cpu()
                    for view in views
                ],
            )
            for rid, index in runner.req_states.req_id_to_index.items()
            if rid != intent.request_id and rid in scheduler.requests
            for row in [manager.get_block_ids(rid)[0]]
        ]
        begin, completion = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record(runner.main_stream)
        result = original_reclaim(intent, tables)
        completion.record(runner.main_stream)
        pending.update(result=result, begin=begin)
        return result

    def checked_update(output, result):
        intent = output.reclaim_request
        free = pool.get_num_free_blocks()
        if intent:
            assert completion.query(), "Canonical release preceded local GPU completion"
            keep, row = pending["keep"], pending["result"].retained_block_ids
            assert keep == sorted(set(keep)) and len(keep) == intent.target_keep_tokens
            expected = [source[keep] for source in pending["source"]]
            for view, values in zip(views, expected):
                assert torch.equal(members(view, row, len(keep)).cpu(), values)
            for other_row, snapshots in pending["others"]:
                for view, values in zip(views, snapshots):
                    assert torch.equal(
                        members(view, other_row, values.shape[0]).cpu(), values
                    )
        answer = original_update(output, result)
        if intent:
            removed = sorted(set(intent.source_block_ids) - set(row))
            assert pool.get_num_free_blocks() == free + len(removed)
            references[intent.request_id] = (row, expected)
            released.update(removed)
            # Poison only now-free pages. A stale graph read must fail visibly.
            page_indices = torch.tensor(removed, device=runner.device)
            for view in views:
                view[page_indices] = float("nan")
            prefix, recent = sink_recent_ranges(
                intent.source_effective_kv_len,
                len(keep),
                16,
                intent.protected_prefix_tokens,
            )
            whole = (
                args.policy == "sink_recent"
                and whole_block_row(
                    intent.source_block_ids,
                    intent.source_effective_kv_len,
                    len(keep),
                    prefix,
                    recent,
                    16,
                )
                is not None
            )
            assert whole == (args.path == "whole" and args.policy == "sink_recent")
            events.append(
                {
                    "event": "reclaim",
                    "step": len(steps) - 1,
                    "intent": vars(intent),
                    "keep": keep,
                    "row": row,
                    "released": removed,
                    "free_before": free,
                    "free_after": pool.get_num_free_blocks(),
                    "gpu_ms": pending["begin"].elapsed_time(completion),
                    "whole": whole,
                }
            )
        return answer

    scheduler.schedule, scheduler.update_from_output = checked_schedule, checked_update
    runner.prepare_inputs, runner.reclamation.prepare_attn = (
        checked_inputs,
        checked_prepare,
    )
    runner.execute_model, graph_manager.dispatch = checked_execute, checked_dispatch
    runner.reclamation.reclaim, runner.reclamation.compactor.compact = (
        checked_reclaim,
        checked_compact,
    )
    layers[0].impl.forward = checked_attention
    for step in range(120):
        external = {0: "A", 2: "B", 17: "C"}.get(step)
        if external:
            params = SamplingParams(
                temperature=0,
                seed=0,
                ignore_eos=True,
                max_tokens={"A": 64, "B": 40, "C": 20}[external],
            )
            ids[external] = engine.add_request(
                external, {"prompt_token_ids": prompts[external]}, params
            )
        for output in engine.step():
            outputs[output.request_id] = {
                "tokens": list(output.outputs[0].token_ids),
                "finished": output.finished,
            }
        for rid, (row, expected) in references.items():
            if rid in scheduler.requests:
                assert manager.get_block_ids(rid)[0][: len(row)] == row
                for view, values in zip(views, expected):
                    assert torch.equal(
                        members(view, row, values.shape[0]).cpu(), values
                    )
        assert static_layouts() == static
        assert {
            str(desc): id(graph) for desc, graph in graph_manager.graphs.items()
        } == graph_ids
        assert (
            compilation_counter.num_cudagraph_captured
            == captures["num_cudagraph_captured"]
        )
        if step >= 17 and not engine.has_unfinished_requests():
            break
    assert len([event for event in events if event["event"] == "reclaim"]) == 2
    assert rewritten and all(result["finished"] for result in outputs.values())
    assert pool.get_num_free_blocks() == 32
    assert counts["fp32_attention"] > 0
    if args.mode == "full":
        assert (
            counts["FULL"] > 0
            and counts["full_vs_eager"] > 0
            and counts["padded_batches"] > 0
        )
    if args.mode == "piecewise":
        assert counts["PIECEWISE"] > 0
    evidence = {
        "status": "DIAGNOSTIC" if args.diagnose_numerics else "PASS",
        "args": vars(args),
        "events": events,
        "steps": steps,
        "outputs": outputs,
        "counts": dict(counts),
        "rewritten": sorted(rewritten),
        "capture_delta": compilation_counter.num_cudagraph_captured
        - captures["num_cudagraph_captured"],
    }
    (run / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
    print(
        json.dumps(
            {
                key: evidence[key]
                for key in ("status", "counts", "rewritten", "capture_delta")
            }
        )
    )


if __name__ == "__main__":
    main()
