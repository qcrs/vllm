# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic-only TP worker extension; never used by Serving measurements."""

import dataclasses
import os
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from vllm.compilation.counter import compilation_counter
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.v1.kv_reclamation import sink_recent_ranges


def members(view, row, count):
    ids = torch.arange(count, device=view.device)
    pages = torch.tensor(row, device=view.device)[ids // 16]
    return view[pages, :, ids % 16, :]


def layout(value):
    return (value.data_ptr(), tuple(value.shape), value.stride(), str(value.dtype))


class EventProbe:
    def __init__(self, event, directory, number):
        self.event, self.directory, self.number = event, directory, number

    def record(self, stream=None):
        self.event.record(stream)

        def observe():
            torch.accelerator.set_device_index(self.event.device.index)
            while not self.event.query():
                time.sleep(0.001)
            (self.directory / f"local-done-{self.number}").write_text(
                str(time.monotonic_ns())
            )

        threading.Thread(target=observe, daemon=True).start()


class P1TPWorker:
    def p1_install(self, options):
        runner = self.model_runner
        worker = runner.reclamation
        rank = get_tp_group().rank_in_group
        directory = Path(os.environ["P1_RUN_DIR"]) / f"rank-{rank}"
        directory.mkdir()
        state = self._p1 = {
            "options": options,
            "rank": rank,
            "directory": directory,
            "pending": None,
            "references": {},
            "released": set(),
            "rewritten": set(),
            "counts": Counter(),
            "number": 0,
        }
        views = worker.layers
        assert len(views) == 36 and all(v.shape[1] == 4 for v in views)
        context = self.vllm_config.compilation_config.static_forward_context
        attention = next(
            layer
            for layer in context.values()
            if isinstance(getattr(layer, "kv_cache", None), torch.Tensor)
        )
        original_attention = attention.impl.forward

        def static():
            return {
                **{
                    name: layout(value)
                    for name, value in vars(runner.input_buffers).items()
                    if isinstance(value, torch.Tensor)
                },
                "E": layout(worker.effective_kv_len.gpu),
                "physical_lengths": layout(worker.kv_seq_lens),
                "positions": layout(worker.cache_positions),
                "resident_row": layout(runner.block_tables.block_tables[0].gpu),
                "input_row": layout(runner.block_tables.input_block_tables[0]),
                "slots": layout(runner.block_tables.slot_mappings),
                "kv": [layout(v) for v in views],
            }

        state["static"] = static()
        manager = runner.cudagraph_manager
        state["graphs"] = {str(k): id(v) for k, v in manager.graphs.items()}
        state["captures"] = compilation_counter.num_cudagraph_captured
        original_reclaim, original_compact = worker.reclaim, worker.compactor.compact
        original_execute, original_prepare = runner.execute_model, worker.prepare_attn
        original_dispatch = manager.dispatch

        def compact(row, keep, **kwargs):
            state["pending"]["compacted"] = True
            state["pending"]["keep"] = keep.cpu().tolist()
            if rank == options["delay_rank"] and state["number"] == 1:
                (directory / "delay-start-1").write_text(str(time.monotonic_ns()))
                time.sleep(options["delay_s"])
                (directory / "delay-end-1").write_text(str(time.monotonic_ns()))
            if rank == options["fail_rank"]:
                (directory / "injected-failure").write_text(str(time.monotonic_ns()))
                raise RuntimeError("P1 injected non-output-rank compaction failure")
            return original_compact(row, keep, **kwargs)

        def reclaim(intent, tables):
            assert not torch.cuda.is_current_stream_capturing()
            state["number"] += 1
            prefix, recent = sink_recent_ranges(
                intent.source_effective_kv_len,
                intent.target_keep_tokens,
                worker.config.sink_tokens,
                intent.protected_prefix_tokens,
            )
            state["pending"] = {
                "intent": intent,
                "source": [
                    members(
                        v, intent.source_block_ids, intent.source_effective_kv_len
                    ).cpu()
                    for v in views
                ],
                "keep": [
                    *range(prefix),
                    *range(recent, intent.source_effective_kv_len),
                ],
                "others": [],
                "compacted": False,
            }
            for rid, idx in runner.req_states.req_id_to_index.items():
                if rid == intent.request_id:
                    continue
                length = int(worker.effective_kv_len_np[idx])
                row = (
                    tables.block_tables[0]
                    .gpu[idx, : (length + 15) // 16]
                    .cpu()
                    .tolist()
                )
                state["pending"]["others"].append(
                    (row, [members(v, row, length).cpu() for v in views])
                )
            event = worker.local_reclaim_done
            if isinstance(event, EventProbe):
                event = event.event
            worker.local_reclaim_done = EventProbe(event, directory, state["number"])
            result = original_reclaim(intent, tables)
            state["pending"]["result"] = result
            return result

        def prepare(batch):
            state["active"] = batch
            state["reuse_before"] = {}
            physical = original_prepare(batch)
            starts = batch.query_start_loc_np.tolist()
            for i, rid in enumerate(batch.req_ids):
                idx, q = batch.idx_mapping_np[i], int(batch.num_scheduled_tokens[i])
                logical = int(runner.req_states.num_computed_tokens_np[idx])
                effective = int(worker.effective_kv_len_np[idx])
                assert batch.positions[
                    starts[i] : starts[i + 1]
                ].cpu().tolist() == list(range(logical, logical + q))
                assert physical.positions[
                    starts[i] : starts[i + 1]
                ].cpu().tolist() == list(range(effective, effective + q))
                assert int(physical.seq_lens[i]) == effective + q
                row = runner.block_tables.block_tables[0].gpu[idx].cpu().tolist()
                for position in range(effective, effective + q):
                    page, offset = row[position // 16], position % 16
                    if page in state["released"]:
                        state["reuse_before"][page, offset] = [
                            v[page, :, offset, :].clone() for v in views
                        ]
            assert physical.seq_lens[batch.num_reqs :].count_nonzero() == 0
            return physical

        def checked_attention(layer, query, key, value, cache, metadata, output, **kw):
            answer = original_attention(
                layer, query, key, value, cache, metadata, output, **kw
            )
            starts = metadata.query_start_loc.cpu().tolist()
            for i, rid in enumerate(state["active"].req_ids):
                if rid not in state["references"] or starts[i + 1] - starts[i] != 1:
                    continue
                count = int(metadata.seq_lens[i].cpu())
                row = metadata.block_table[i].cpu().tolist()
                k, v = members(cache, row, count).cpu().float().chunk(2, dim=-1)
                q = query[starts[i]].cpu().float()
                group = q.shape[0] // k.shape[1]
                k = k.repeat_interleave(group, dim=1)
                v = v.repeat_interleave(group, dim=1)
                scores = torch.einsum("hd,thd->ht", q, k) * q.shape[-1] ** -0.5
                expected = torch.einsum("ht,thd->hd", scores.softmax(-1), v)
                torch.testing.assert_close(
                    output[starts[i]].cpu().float().reshape_as(expected),
                    expected,
                    atol=1.6e-2,
                    rtol=1.6e-2,
                )
                state["counts"]["fp32_attention"] += 1
            return answer

        def dispatch(*args, **kwargs):
            descriptor = original_dispatch(*args, **kwargs)
            state["descriptor"] = descriptor
            return descriptor

        def execute(*args, **kwargs):
            answer = original_execute(*args, **kwargs)
            current = runner.execute_model_state
            if current is None:
                return answer
            batch = current.input_batch
            metadata = next(iter(current.attn_metadata.values()))
            mode = state["descriptor"].cg_mode
            state["counts"][mode.name] += 1
            slots = []
            for i, rid in enumerate(batch.req_ids):
                idx = batch.idx_mapping_np[i]
                effective = int(worker.effective_kv_len_np[idx])
                row = runner.block_tables.block_tables[0].gpu[idx].cpu().tolist()
                q = int(batch.num_scheduled_tokens[i])
                slots.extend(
                    row[p // 16] * 16 + p % 16 for p in range(effective, effective + q)
                )
                assert (
                    metadata.block_table[i, : (effective + q + 15) // 16].cpu().tolist()
                    == row[: (effective + q + 15) // 16]
                )
                for position in range(effective, effective + q):
                    page = row[position // 16]
                    if page in state["released"]:
                        assert all(
                            torch.isfinite(v[page, :, position % 16, :]).all()
                            for v in views
                        )
                        state["rewritten"].add(page)
            assert runner.block_tables.slot_mappings[
                0, : batch.num_tokens_after_padding
            ].cpu().tolist() == slots + [-1] * (
                batch.num_tokens_after_padding - batch.num_tokens
            )
            if batch.num_reqs_after_padding > batch.num_reqs:
                assert metadata.block_table[batch.num_reqs :].count_nonzero() == 0
                state["counts"]["padded"] += 1
            for (page, offset), before in state["reuse_before"].items():
                for view, previous in zip(views, before):
                    value = view[page, :, offset, :]
                    assert torch.isfinite(value).all()
                    assert not torch.equal(
                        value.view(torch.int16), previous.view(torch.int16)
                    )
            if mode == CUDAGraphMode.FULL and state["references"]:
                graph = current.hidden_states[: batch.num_tokens].clone()
                slot_tensor = torch.tensor(slots, device=runner.device)
                pages, offsets = slot_tensor // 16, slot_tensor % 16
                appended = [v[pages, :, offsets, :].clone() for v in views]
                graph_argmax = runner.model.compute_logits(
                    graph[batch.logits_indices]
                ).argmax(-1)
                reference = {
                    name: dataclasses.replace(
                        value, max_seq_len=runner.model_config.max_model_len
                    )
                    for name, value in current.attn_metadata.items()
                }
                with set_forward_context(
                    reference,
                    self.vllm_config,
                    num_tokens=batch.num_tokens_after_padding,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=current.slot_mappings_by_layer,
                    is_padding=batch.is_padding,
                ):
                    eager = runner.model(
                        input_ids=batch.input_ids,
                        positions=batch.positions,
                        **runner.model_state.prepare_inputs(batch, runner.req_states),
                    )
                torch.testing.assert_close(
                    graph, eager[: batch.num_tokens], atol=0, rtol=0
                )
                assert torch.equal(
                    graph_argmax,
                    runner.model.compute_logits(eager[batch.logits_indices]).argmax(-1),
                )
                for v, expected in zip(views, appended):
                    torch.testing.assert_close(
                        v[pages, :, offsets, :], expected, atol=0, rtol=0
                    )
                    v[pages, :, offsets, :] = expected
                state["counts"]["graph_eager"] += 1
            assert static() == state["static"]
            assert {str(k): id(v) for k, v in manager.graphs.items()} == state["graphs"]
            assert compilation_counter.num_cudagraph_captured == state["captures"]
            return answer

        worker.reclaim, worker.compactor.compact = reclaim, compact
        attention.impl.forward = checked_attention
        worker.prepare_attn, runner.execute_model, manager.dispatch = (
            prepare,
            execute,
            dispatch,
        )
        return {
            "rank": rank,
            "static": state["static"],
            "captures": state["captures"],
            "graphs": state["graphs"],
        }

    def p1_check(self, active_ids):
        state, runner = self._p1, self.model_runner
        worker, views = runner.reclamation, runner.reclamation.layers
        pending = state["pending"]
        result = None
        if pending:
            assert worker.local_reclaim_done.event.query()
            keep, row = pending["keep"], pending["result"].retained_block_ids
            assert (
                keep == sorted(set(keep))
                and len(keep) == pending["intent"].target_keep_tokens
            )
            expected = [source[keep] for source in pending["source"]]
            idx = runner.req_states.req_id_to_index[pending["intent"].request_id]
            assert int(worker.effective_kv_len.gpu[idx]) == len(keep)
            resident = runner.block_tables.block_tables[0].gpu[idx].cpu().tolist()
            assert resident[: len(row)] == row
            assert int(runner.block_tables.num_blocks.np[0, idx]) == len(row)
            for view, value in zip(views, expected):
                assert torch.equal(members(view, row, len(keep)).cpu(), value)
            for other_row, values in pending["others"]:
                for view, value in zip(views, values):
                    assert torch.equal(
                        members(view, other_row, len(value)).cpu(), value
                    )
            state["references"][pending["intent"].request_id] = (row, expected)
            result = {
                "rank": state["rank"],
                "number": state["number"],
                "keep": keep,
                "row": row,
                "E": len(keep),
                "path": "token" if pending["compacted"] else "whole",
            }
            state["pending"] = None
        for rid, (row, values) in state["references"].items():
            if rid not in active_ids:
                continue
            for view, value in zip(views, values):
                assert torch.equal(members(view, row, len(value)).cpu(), value)
        return {
            "reclaim": result,
            "rank": state["rank"],
            "counts": dict(state["counts"]),
            "rewritten": sorted(state["rewritten"]),
        }

    def p1_poison(self, released):
        state = self._p1
        state["released"].update(released)
        index = torch.tensor(released, device=self.model_runner.device)
        for view in self.model_runner.reclamation.layers:
            view[index] = float("nan")
        torch.accelerator.synchronize()
