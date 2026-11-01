# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import queue
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    ECTransferConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.core import (
    DPEngineCoreProc,
    EngineCore,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_reclamation import PolicyBinding, ReclamationConfig
from vllm.v1.metrics.stats import DPLoadSnapshot
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from ...utils import create_new_process_for_each_test, multi_gpu_test

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.", allow_module_level=True)

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
# test_engine_core_concurrent_batches assumes exactly 12 tokens per prompt.
# Adjust prompt if changing model to maintain 12-token length.
PROMPT = "I am Gyoubu Masataka Oniwa"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids

_REQUEST_COUNTER = 0


def reclamation_core():
    core = object.__new__(EngineCoreProc)
    core.has_coordinator = False
    core.reclamation_config = ReclamationConfig()
    core.reclamation_policies = {0: PolicyBinding(0, core.reclamation_config)}
    core.default_reclamation_version = 0
    core.next_reclamation_version = 1
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, tensor_parallel_size=2)
    )
    core.scheduler = SimpleNamespace(requests={})
    core.scheduler.add_request = lambda r: core.scheduler.requests.update(
        {r.request_id: r}
    )
    return core


def test_reclamation_utility_publishes_after_all_acks_and_retires_after_last_request():
    core = reclamation_core()
    retired = []

    def rpc(method, args):
        if method == "prepare_reclamation_policy":
            version, config = args
            assert core.default_reclamation_version == 0
            return [
                dict(rank=r, binding=PolicyBinding(version, config)) for r in (0, 1)
            ]
        assert method == "retire_reclamation_policies"
        retired.extend(args[0])
        return [dict(rank=r, versions=args[0]) for r in (0, 1)]

    core.model_executor = SimpleNamespace(collective_rpc=rpc)
    old = Request("old", [1, 2], SamplingParams(), None)
    core.add_request(old)
    core.shutdown_state = EngineShutdownState.RUNNING
    core.output_queue = queue.Queue()
    payload = MsgpackDecoder().decode(
        MsgpackEncoder().encode(
            (0, 7, "update_reclamation_policy", ({"policy": "full"},))
        )
    )
    core._handle_client_request(EngineCoreRequestType.UTILITY, payload)
    _, output = core.output_queue.get_nowait()
    assert output.utility_output.failure_message is None
    assert core.default_reclamation_version == 1
    new = Request("new", [1, 2], SamplingParams(), None)
    core.add_request(new)
    assert old.policy_binding.version == 0 and new.policy_binding.version == 1
    assert new.policy_binding.config.policy == "full"
    assert not core._retire_unused_reclamation_policies() and not retired
    del core.scheduler.requests["old"]
    assert core._retire_unused_reclamation_policies() == [0]
    assert retired == [0] and set(core.reclamation_policies) == {1}
    with pytest.raises(ValueError, match="Unknown or unprepared"):
        core.add_request(
            Request(
                "stale",
                [1],
                SamplingParams(extra_args={"kv_reclamation_version": 0}),
                None,
            )
        )


@pytest.mark.parametrize("failure", ["raise", "missing", "duplicate", "config", "hash"])
def test_reclamation_failed_prepare_never_publishes_or_reuses_version(failure):
    from dataclasses import replace

    core = reclamation_core()
    before = core.reclamation_policies.copy()

    def rpc(method, args):
        version, config = args
        if failure == "raise":
            raise RuntimeError("injected prepare failure")
        binding = PolicyBinding(version, config, "same")
        acks = [dict(rank=r, binding=binding) for r in (0, 1)]
        if failure == "missing":
            acks.pop()
        elif failure == "duplicate":
            acks[1]["rank"] = 0
        elif failure == "config":
            acks[1]["binding"] = replace(
                binding, config=replace(config, retention=0.75)
            )
        else:
            acks[1]["binding"] = replace(binding, calibration_sha256="different")
        return acks

    core.model_executor = SimpleNamespace(collective_rpc=rpc)
    with pytest.raises((ValueError, RuntimeError)):
        core.update_reclamation_policy(
            {"policy": "triattention", "stats_path": "stats"}
        )
    assert core.default_reclamation_version == 0
    assert core.reclamation_policies == before and core.next_reclamation_version == 2
    with pytest.raises(ValueError, match="unsupported"):
        core.update_reclamation_policy({"budget_mode": "demand"})
    core.vllm_config.parallel_config.data_parallel_size = 2
    with pytest.raises(ValueError, match="single engine"):
        core.update_reclamation_policy({"policy": "full"})
    core.vllm_config.parallel_config.data_parallel_size = 1
    core.has_coordinator = True
    with pytest.raises(ValueError, match="single engine"):
        core.update_reclamation_policy({"policy": "full"})


def test_dense_dp_busy_loop_publishes_queue_transitions_and_idle_zero():
    core = object.__new__(EngineCoreProc)
    core.publish_dp_lb_stats = True
    core.last_counts = (0, 0)
    core.last_dp_load = (0, 0.0, {})
    core.dp_add_watermarks = {}
    core.dp_load_sequence = core.dp_load_sampled_ns = 0
    core.output_queue = queue.Queue()
    state = SimpleNamespace(counts=(0, 0))
    core.scheduler = SimpleNamespace(
        get_request_counts=lambda: state.counts,
        get_dp_load_snapshot=lambda seq, now: DPLoadSnapshot(
            unfinished_tokens=sum(state.counts), sequence=seq, generated_ns=now
        ),
    )
    running = iter([True, True, False])
    core._handle_shutdown = lambda: next(running)
    inputs = iter([(0, 2), (1, 0)])
    outputs = iter([(1, 0), (0, 0)])
    core._process_input_queue = lambda: setattr(state, "counts", next(inputs))
    core._process_engine_step = lambda: setattr(state, "counts", next(outputs))

    with pytest.raises(SystemExit):
        core.run_busy_loop()

    decoder = MsgpackDecoder(EngineCoreOutputs)
    observed = []
    while not core.output_queue.empty():
        client_index, output = core.output_queue.get_nowait()
        assert client_index == -1
        stats = decoder.decode(MsgpackEncoder().encode(output)).scheduler_stats
        observed.append((stats.num_running_reqs, stats.num_waiting_reqs))
    assert observed == [(0, 2), (1, 0), (0, 0)]


def test_dense_dp_known_work_keeps_inflight_and_preempted_history():
    scheduler = object.__new__(Scheduler)
    requests = [
        Request(name, [1] * length, SamplingParams(max_tokens=10000), None)
        for name, length in (
            ("waiting", 1024),
            ("chunk", 8192),
            ("preempt", 512),
            ("decode", 20),
            ("done", 1024),
        )
    ]
    waiting, chunk, preempted, decode, finished = requests
    chunk.status = decode.status = RequestStatus.RUNNING
    chunk.num_computed_tokens = chunk.num_in_flight_tokens = 512
    preempted.status = RequestStatus.PREEMPTED
    decode.append_output_token_ids([2, 3])
    decode.num_computed_tokens = 21
    finished.status = RequestStatus.FINISHED_ABORTED
    scheduler.requests = {r.request_id: r for r in requests}
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.5)
    scheduler.current_step = 7
    snapshot = scheduler.get_dp_load_snapshot(2, 100)
    assert snapshot == DPLoadSnapshot(9729, 0.5, 2, 7, 100)
    chunk.num_in_flight_tokens = 0
    assert scheduler.get_dp_load_snapshot(3, 101).unfinished_tokens == 9729 - 512


def test_dense_dp_load_only_changes_coalesce_but_idle_zero_is_immediate():
    core = object.__new__(EngineCoreProc)
    core.publish_dp_lb_stats = True
    core.last_counts, core.last_dp_load = (0, 0), (0, 0.0, {})
    core.dp_add_watermarks = {}
    core.dp_load_sequence = core.dp_load_sampled_ns = 0
    core.output_queue = queue.Queue()
    state = SimpleNamespace(counts=(1, 0), work=8192, usage=0.5)
    core.scheduler = SimpleNamespace(
        get_request_counts=lambda: state.counts,
        get_dp_load_snapshot=lambda seq, now: DPLoadSnapshot(
            state.work, state.usage, seq, seq, now
        ),
    )
    with patch("vllm.v1.engine.core.time.monotonic_ns") as clock:
        clock.return_value = 1_000_000_000
        core._maybe_publish_request_counts()
        state.work = 7680
        clock.return_value += 50_000_000
        core._maybe_publish_request_counts()
        assert core.output_queue.qsize() == 1
        clock.return_value += 50_000_000
        core._maybe_publish_request_counts()
        clock.return_value += 100_000_000
        core._maybe_publish_request_counts()
        assert core.output_queue.qsize() == 2
        state.usage = 0.6
        core._maybe_publish_request_counts()
        assert core.output_queue.qsize() == 2
        clock.return_value += 100_000_000
        core._maybe_publish_request_counts()
        state.counts, state.work, state.usage = (0, 0), 0, 0.0
        clock.return_value += 1
        core._maybe_publish_request_counts()
    snapshots = []
    while not core.output_queue.empty():
        _, output = core.output_queue.get_nowait()
        output = MsgpackDecoder(EngineCoreOutputs).decode(
            MsgpackEncoder().encode(output)
        )
        snapshots.append(output.scheduler_stats.dp_load)
    assert [s.unfinished_tokens for s in snapshots] == [8192, 7680, 7680, 0]
    assert [s.kv_cache_usage for s in snapshots] == [0.5, 0.5, 0.6, 0.0]
    assert [s.sequence for s in snapshots] == [1, 2, 3, 4]


def test_moe_dp_request_counts_keep_step_and_wave():
    core = object.__new__(DPEngineCoreProc)
    core.publish_dp_lb_stats = True
    core.last_counts = (0, 0)
    core.output_queue = queue.Queue()
    core.scheduler = SimpleNamespace(get_request_counts=lambda: (2, 3))
    core.step_counter, core.current_wave = 7, 4
    core._maybe_publish_request_counts()
    _, output = core.output_queue.get_nowait()
    stats = output.scheduler_stats
    assert (stats.num_running_reqs, stats.num_waiting_reqs) == (2, 3)
    assert (stats.step_counter, stats.current_wave) == (7, 4)


def test_dense_dp_add_ack_follows_main_thread_admission_and_is_published():
    """Same-count snapshots must still acknowledge processed ADDs."""
    core = object.__new__(EngineCoreProc)
    core.publish_dp_lb_stats = True
    core.shutdown_state = EngineShutdownState.RUNNING
    core.last_counts, core.last_dp_load = (0, 0), (0, 0.0, {})
    core.dp_add_watermarks = {}
    core.dp_load_sequence = core.dp_load_sampled_ns = 0
    core.output_queue = queue.Queue()
    core.scheduler = SimpleNamespace(
        get_request_counts=lambda: (0, 0),
        get_dp_load_snapshot=lambda seq, now: DPLoadSnapshot(sequence=seq),
    )
    wire = make_request()
    wire.client_index, wire.dp_add_sequence = 1, 7
    decoded = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(wire))
    request = Request.from_engine_core_request(decoded, None)
    assert request.dp_add_sequence == 7

    def admit(req, wave):
        assert req is request and wave == 0
        assert core.dp_add_watermarks == {}

    core.add_request = admit
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))
    core._maybe_publish_request_counts()
    _, output = core.output_queue.get_nowait()
    snapshot = (
        MsgpackDecoder(EngineCoreOutputs)
        .decode(MsgpackEncoder().encode(output))
        .scheduler_stats.dp_load
    )
    assert snapshot.add_watermarks == {1: 7}
    core.dp_add_watermarks[1] = 8
    assert output.scheduler_stats.dp_load.add_watermarks == {1: 7}
    assert core.last_dp_load == (0, 0.0, {1: 7})


def make_request() -> EngineCoreRequest:
    global _REQUEST_COUNTER
    _REQUEST_COUNTER += 1
    request_id = f"request-{_REQUEST_COUNTER}"
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-{uuid.uuid4()}",
        prompt_token_ids=PROMPT_TOKENS,
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@create_new_process_for_each_test()
def test_engine_core():
    """Setup the EngineCore."""
    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )
    """Test basic request lifecycle."""

    # First request.
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # Second request.
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 1

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    # Add two requests in a row.
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    engine_core.add_request(*engine_core.preprocess_add_request(make_request()))
    assert len(engine_core.scheduler.waiting) == 2
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 4

    # Loop through until they are all done.
    while (outs := engine_core.step_fn()[0].get(0)) and outs.outputs:
        pass

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0
    """Test abort cycle."""

    # Basic abort.
    req = make_request()
    request_id = req.request_id

    engine_core.add_request(*engine_core.preprocess_add_request(req))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0
    assert engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1
    assert engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    engine_core.abort_requests([request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0
    assert not engine_core.scheduler.has_unfinished_requests()
    assert engine_core.scheduler.has_finished_requests()

    _ = engine_core.step_fn()
    assert not engine_core.scheduler.has_unfinished_requests()
    assert not engine_core.scheduler.has_finished_requests()

    # Add, step, abort 1 of the 3.
    req0 = make_request()
    req1 = make_request()
    req2 = make_request()

    engine_core.add_request(*engine_core.preprocess_add_request(req0))
    engine_core.add_request(*engine_core.preprocess_add_request(req1))
    assert len(engine_core.scheduler.waiting) == 2
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    engine_core.add_request(*engine_core.preprocess_add_request(req2))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 3

    # Abort just one.
    engine_core.abort_requests([req1.request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    _ = engine_core.step_fn()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    # Abort the other requests at the same time.
    engine_core.abort_requests([req2.request_id, req0.request_id])
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0

    # Sending duplicate requests with same request_id
    req0 = make_request()
    req1 = make_request()
    req0.request_id = req1.request_id = "test"
    engine_core.add_request(*engine_core.preprocess_add_request(req0))

    while engine_core.scheduler.has_requests():
        engine_core.step_fn()

    engine_core.add_request(*engine_core.preprocess_add_request(req1))
    while engine_core.scheduler.has_requests():
        engine_core.step_fn()

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0


@create_new_process_for_each_test()
def test_engine_core_advanced_sampling():
    """
    A basic end-to-end test to verify that the engine functions correctly
    when additional sampling parameters, such as top_p, min_tokens, and
    presence_penalty, are set.
    """
    """Setup the EngineCore."""
    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )
    """Test basic request lifecycle."""
    # First request.
    request: EngineCoreRequest = make_request()
    request.sampling_params = SamplingParams(
        min_tokens=4,
        presence_penalty=1.0,
        frequency_penalty=1.0,
        repetition_penalty=0.1,
        stop_token_ids=[1001, 1002],
    )
    engine_core.add_request(*engine_core.preprocess_add_request(request))

    def _check_engine_state():
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0
        # Loop through until they are all done.
        while engine_core.scheduler.has_requests():
            engine_core.step_fn()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 0

    _check_engine_state()

    # Second request.
    request2 = make_request()
    request2.sampling_params = SamplingParams(
        top_p=0.99,
        top_k=50,
    )
    engine_core.add_request(*engine_core.preprocess_add_request(request2))
    _check_engine_state()


@create_new_process_for_each_test()
def test_engine_core_concurrent_batches():
    """
    Test that the engine can handle multiple concurrent batches.
    """

    def make_request_with_max_tokens(req_id: str, max_tokens: int) -> EngineCoreRequest:
        request = make_request()
        request.request_id = req_id
        request.sampling_params.max_tokens = max_tokens
        return request

    class DummyExecutor(UniProcExecutor):
        def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
            super().initialize_from_config(kv_cache_configs)

            # Create a thread pool with a single worker
            self.thread_pool = ThreadPoolExecutor(max_workers=1)

        def execute_model(
            self,
            scheduler_output,
            non_block=False,
        ) -> Future[ModelRunnerOutput | None]:
            """Make execute_model non-blocking."""

            # DummyExecutor used only for testing async case.
            assert non_block

            def _execute():
                output = self.collective_rpc("execute_model", args=(scheduler_output,))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        def sample_tokens(
            self, grammar_output, non_block=False
        ) -> Future[ModelRunnerOutput]:
            """Make sample_tokens non-blocking."""

            # DummyExecutor used only for testing async case.
            assert non_block

            def _execute():
                output = self.collective_rpc("sample_tokens", args=(grammar_output,))
                # Make a copy because output[0] may be reused
                # by the next batch.
                return copy.deepcopy(output[0])

            # Use the thread pool instead of creating a new thread
            return self.thread_pool.submit(_execute)

        def shutdown(self):
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)

    engine_args = EngineArgs(
        model=MODEL_NAME,
        # To test concurrent batches.
        max_num_seqs=2,
        # Avoid all requests being scheduled once.
        enable_prefix_caching=False,
        max_num_batched_tokens=10,
        # Reduce startup time.
        enforce_eager=True,
        # Test concurrent batch behaviour independently of async scheduling.
        async_scheduling=False,
    )
    vllm_config = engine_args.create_engine_config()
    # Force two concurrent batches to exercise the batch queue independently
    # of async scheduling (which is disabled above).
    with (
        set_default_torch_num_threads(1),
        patch.object(
            VllmConfig,
            "max_concurrent_batches",
            new_callable=PropertyMock,
            return_value=2,
        ),
    ):
        engine_core = EngineCore(
            vllm_config=vllm_config, log_stats=False, executor_class=DummyExecutor
        )
    assert engine_core.batch_queue is not None

    # Add two requests in a row. Each request have 12 prompt tokens.
    req0 = make_request_with_max_tokens("0", 5)
    engine_core.add_request(*engine_core.preprocess_add_request(req0))
    req1 = make_request_with_max_tokens("1", 5)
    engine_core.add_request(*engine_core.preprocess_add_request(req1))

    # Schedule Batch 1: (10, req0)
    assert engine_core.step_with_batch_queue()[0] is None
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 10
    # num_computed_tokens should have been updated immediately.
    assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 10

    # Schedule Batch 2: (2, req0), (8, req1)
    assert engine_core.step_with_batch_queue()[0] == {}
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 2
    assert scheduler_output.num_scheduled_tokens["1"] == 8
    # num_computed_tokens should have been updated immediately.
    assert engine_core.scheduler.requests["0"].num_computed_tokens == 12
    assert engine_core.scheduler.requests["1"].num_computed_tokens == 8

    assert engine_core.scheduler.get_num_unfinished_requests() == 2

    # Finish Batch 1 and schedule Batch 3: (4, req1).
    # Note that req0 cannot be scheduled
    # because it is in the decoding stage now.
    engine_core.step_with_batch_queue()
    assert len(engine_core.batch_queue) == 1
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["1"] == 4

    # Finish Batch 2. Get first token of req0.
    # Schedule Batch 4: (1, req0).
    output = engine_core.step_with_batch_queue()[0].get(0)
    assert output is not None
    assert len(output.outputs) == 1
    assert engine_core.scheduler.requests[req0.request_id].num_tokens == 13
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["0"] == 1

    # Finish Batch 3. Get first token of req1. Schedule Batch 5: (1, req1).
    output = engine_core.step_with_batch_queue()[0].get(0)
    assert output is not None
    assert len(output.outputs) == 1
    assert engine_core.scheduler.requests[req1.request_id].num_tokens == 13
    scheduler_output = engine_core.batch_queue[-1][1]
    assert scheduler_output.num_scheduled_tokens["1"] == 1

    # Loop until req0 is finished.
    req_id = 0
    expected_num_tokens = [
        engine_core.scheduler.requests["0"].num_tokens + 1,
        engine_core.scheduler.requests["1"].num_tokens + 1,
    ]
    while engine_core.scheduler.get_num_unfinished_requests() == 2:
        output = engine_core.step_with_batch_queue()[0]
        # Every step consumes an output.
        assert output is not None
        assert len(output[0].outputs) == 1
        if req_id in engine_core.scheduler.requests:
            assert (
                engine_core.scheduler.requests[req_id].num_tokens
                == expected_num_tokens[req_id]
            )
        expected_num_tokens[req_id] += 1
        req_id = (req_id + 1) % 2


@multi_gpu_test(num_gpus=2)
def test_engine_core_tp():
    """
    Test engine can initialize worker in tp properly
    """

    """Setup the EngineCore."""
    engine_args = EngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        # Reduce startup time.
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )

    def get_worker_cache_config_field(worker, key: str):
        return getattr(worker.cache_config, key)

    num_gpu_blocks = engine_core.collective_rpc(
        get_worker_cache_config_field, args=("num_gpu_blocks",)
    )
    num_cpu_blocks = engine_core.collective_rpc(
        get_worker_cache_config_field, args=("num_cpu_blocks",)
    )
    assert all(x is not None for x in num_gpu_blocks)
    assert all(x is not None for x in num_cpu_blocks)


@create_new_process_for_each_test()
def test_engine_core_invalid_request_id_type():
    """Test that engine raises TypeError for non-string request_id."""
    engine_args = EngineArgs(model=MODEL_NAME)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )

    # Test with UUID object (common mistake)
    uuid_request = make_request()
    uuid_request.request_id = uuid.uuid4()  # UUID object instead of string

    with pytest.raises(TypeError, match="request_id must be a string, got.*UUID"):
        engine_core.add_request(*engine_core.preprocess_add_request(uuid_request))

    # Test with integer
    int_request = make_request()
    int_request.request_id = 12345

    with pytest.raises(TypeError, match="request_id must be a string, got.*int"):
        engine_core.add_request(*engine_core.preprocess_add_request(int_request))

    # Test with None
    none_request = make_request()
    none_request.request_id = None

    with pytest.raises(TypeError, match="request_id must be a string, got.*NoneType"):
        engine_core.add_request(*engine_core.preprocess_add_request(none_request))

    # Verify engine is still functional after errors
    valid_request = make_request()
    engine_core.add_request(*engine_core.preprocess_add_request(valid_request))
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0


@create_new_process_for_each_test()
@pytest.mark.parametrize(
    ("ec_role", "gpu_memory_utilization", "enable_prefix_caching"),
    [
        ("ec_producer", 0.01, False),
        # NOTE: ec_producer never allows prefix caching
        ("ec_consumer", 0.7, True),
        ("ec_consumer", 0.7, False),
    ],
)
@pytest.mark.parametrize("use_kv_connector", [False, True])
def test_encoder_instance_zero_kv_cache(
    ec_role: str,
    gpu_memory_utilization: float,
    enable_prefix_caching: bool,
    use_kv_connector: bool,
):
    """EPD (Encoder-Prefill-Decode) Encoder-cache-specific tests

    This test verifies encoder-only instance initializes with 0 KV cache blocks.
    Under EPD disagg mode, Encoder instances (EC producer role) only execute
    vision encoder, so they don't need KV cache for text generation.
    """
    # Form vllm config
    model_config = ModelConfig(
        model="llava-hf/llava-1.5-7b-hf",  # Multimodal model
        enforce_eager=True,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        disable_hybrid_kv_cache_manager=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=gpu_memory_utilization,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    kv_transfer_config = (
        KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        )
        if use_kv_connector
        else None
    )
    ec_transfer_config = ECTransferConfig(
        ec_connector="ECExampleConnector",
        ec_role=ec_role,
        ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test_encoder"},
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        kv_transfer_config=kv_transfer_config,
        ec_transfer_config=ec_transfer_config,
    )

    executor_class = Executor.get_class(vllm_config)
    print(f"executor_class: {executor_class}")

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config, executor_class=executor_class, log_stats=True
        )

    # Check encoder cache manager exists
    assert engine_core.scheduler.encoder_cache_manager is not None, (
        "encoder_cache_manager should exist"
    )

    if ec_role == "ec_producer":
        # Check 1: num_blocks should be 0
        # NOTE: num_blocks=1 as BlockPool always needs a null_block.
        kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config
        print(f"kv_cache_config: {kv_cache_config}")
        assert kv_cache_config.num_blocks == 1, (
            f"ec_producer should only have 1 KV blocks, "
            f"got {kv_cache_config.num_blocks}"
        )

        # Check 2: kv_cache_groups should be empty
        assert len(kv_cache_config.kv_cache_groups) == 0, (
            f"ec_producer should have 0 KV cache groups, "
            f"got {len(kv_cache_config.kv_cache_groups)}"
        )

        # Check 3: kv_cache_tensors should be empty
        assert len(kv_cache_config.kv_cache_tensors) == 0, (
            f"Encoder instance should have 0 KV cache tensors, "
            f"got {len(kv_cache_config.kv_cache_tensors)}"
        )

        # Check 4: Verify EC connector is initialized and is producer
        assert engine_core.scheduler.ec_connector is not None, (
            "Encoder instance should have EC connector"
        )
        assert engine_core.scheduler.ec_connector.is_producer, (
            "Encoder instance EC connector should be producer"
        )

        # Check 5: Verify chunked prefill is disabled
        assert not vllm_config.scheduler_config.enable_chunked_prefill, (
            "Encoder instance should disable chunked prefill (no KV cache)"
        )

    elif ec_role == "ec_consumer":
        # Check 1: num_blocks should be > 1
        kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config
        print(f"kv_cache_config: {kv_cache_config}")
        assert kv_cache_config.num_blocks > 1, (
            f"ec_consumer should have >1 KV blocks, got {kv_cache_config.num_blocks}"
        )

        # Check 2: kv_cache_groups should NOT be empty
        assert len(kv_cache_config.kv_cache_groups) > 0, (
            f"ec_consumer should have KV cache groups, "
            f"got {len(kv_cache_config.kv_cache_groups)}"
        )

        # Check 3: Verify EC connector is consumer
        assert engine_core.scheduler.ec_connector is not None, (
            "Consumer instance should have EC connector"
        )
        assert not engine_core.scheduler.ec_connector.is_producer, (
            "Consumer instance EC connector should be consumer"
        )
