# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
    _select_kv_seq_lens,
)
from vllm.v1.worker.gpu.input_batch import post_update, prepare_pos_seq_lens
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.states import RequestState


def make_request_state() -> RequestState:
    return RequestState(
        max_num_reqs=1,
        max_model_len=256,
        max_num_batched_tokens=8,
        num_speculative_steps=0,
        vocab_size=128,
        device=torch.device("cuda"),
    )


def add_request(state: RequestState, req_id: str, initial_tokens: int) -> int:
    state.add_request(
        req_id=req_id,
        prompt_len=initial_tokens,
        all_token_ids=list(range(initial_tokens)),
        num_computed_tokens=initial_tokens,
        max_tokens=8,
    )
    state.apply_staged_writes()
    torch.accelerator.synchronize()
    return state.req_id_to_index[req_id]


def advance_one_token(state: RequestState, req_idx: int) -> None:
    post_update(
        idx_mapping=torch.tensor([req_idx], dtype=torch.int32, device="cuda"),
        num_computed_tokens=state.num_computed_tokens.gpu,
        effective_kv_len=state.effective_kv_len.gpu,
        last_sampled_tokens=state.last_sampled_tokens,
        output_bin_counts=None,
        sampled_tokens=torch.tensor([[7]], dtype=torch.int64, device="cuda"),
        num_sampled=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        all_token_ids=state.all_token_ids.gpu,
        total_len=state.total_len.gpu,
    )
    torch.accelerator.synchronize()


@unittest.skipUnless(current_platform.is_cuda(), "requires CUDA")
class TestEffectiveKVState(unittest.TestCase):
    def test_initialization(self) -> None:
        state = make_request_state()
        req_idx = add_request(state, "request-a", 37)

        self.assertEqual(state.num_computed_tokens.gpu[req_idx].item(), 37)
        self.assertEqual(state.effective_kv_len.gpu[req_idx].item(), 37)

    def test_baseline_advancement(self) -> None:
        state = make_request_state()
        req_idx = add_request(state, "request-a", 128)

        advance_one_token(state, req_idx)

        self.assertEqual(state.num_computed_tokens.gpu[req_idx].item(), 129)
        self.assertEqual(state.effective_kv_len.gpu[req_idx].item(), 129)

    def test_independent_divergence_advancement(self) -> None:
        state = make_request_state()
        req_idx = add_request(state, "request-a", 128)
        state.effective_kv_len.gpu[req_idx] = 64

        advance_one_token(state, req_idx)

        self.assertEqual(state.num_computed_tokens.gpu[req_idx].item(), 129)
        self.assertEqual(state.effective_kv_len.gpu[req_idx].item(), 65)

    def test_v2_override_is_absolute_when_computed_delta_is_zero(self) -> None:
        state = make_request_state()
        req_idx = add_request(state, "request-a", 128)
        override_valid = torch.tensor([True], dtype=torch.bool, device="cuda")
        override_value = torch.tensor([5], dtype=torch.int32, device="cuda")

        post_update(
            idx_mapping=torch.tensor([req_idx], dtype=torch.int32, device="cuda"),
            num_computed_tokens=state.num_computed_tokens.gpu,
            effective_kv_len=state.effective_kv_len.gpu,
            last_sampled_tokens=state.last_sampled_tokens,
            output_bin_counts=None,
            sampled_tokens=torch.tensor([[7]], dtype=torch.int64, device="cuda"),
            num_sampled=torch.tensor([1], dtype=torch.int32, device="cuda"),
            num_rejected=torch.tensor([1], dtype=torch.int32, device="cuda"),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            all_token_ids=state.all_token_ids.gpu,
            total_len=state.total_len.gpu,
            effective_kv_len_override_valid=override_valid,
            effective_kv_len_override=override_value,
        )
        torch.accelerator.synchronize()

        self.assertEqual(state.num_computed_tokens.gpu[req_idx].item(), 128)
        self.assertEqual(state.effective_kv_len.gpu[req_idx].item(), 5)

    def test_v2_override_does_not_add_computed_delta(self) -> None:
        state = make_request_state()
        req_idx = add_request(state, "request-a", 128)
        override_valid = torch.tensor([True], dtype=torch.bool, device="cuda")
        override_value = torch.tensor([5], dtype=torch.int32, device="cuda")

        post_update(
            idx_mapping=torch.tensor([req_idx], dtype=torch.int32, device="cuda"),
            num_computed_tokens=state.num_computed_tokens.gpu,
            effective_kv_len=state.effective_kv_len.gpu,
            last_sampled_tokens=state.last_sampled_tokens,
            output_bin_counts=None,
            sampled_tokens=torch.tensor([[7]], dtype=torch.int64, device="cuda"),
            num_sampled=torch.tensor([1], dtype=torch.int32, device="cuda"),
            num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            all_token_ids=state.all_token_ids.gpu,
            total_len=state.total_len.gpu,
            effective_kv_len_override_valid=override_valid,
            effective_kv_len_override=override_value,
        )
        torch.accelerator.synchronize()

        self.assertEqual(state.num_computed_tokens.gpu[req_idx].item(), 129)
        self.assertEqual(state.effective_kv_len.gpu[req_idx].item(), 5)

    def test_request_slot_reuse_overwrites_state(self) -> None:
        state = make_request_state()
        req_idx_a = add_request(state, "request-a", 128)
        state.effective_kv_len.gpu[req_idx_a] = 64

        self.assertEqual(state.remove_request("request-a"), req_idx_a)
        req_idx_b = add_request(state, "request-b", 37)

        self.assertEqual(req_idx_b, req_idx_a)
        self.assertEqual(state.num_computed_tokens.gpu[req_idx_b].item(), 37)
        self.assertEqual(state.effective_kv_len.gpu[req_idx_b].item(), 37)


@unittest.skipUnless(current_platform.is_cuda(), "requires CUDA")
class TestLogicalPhysicalInputContract(unittest.TestCase):
    def assert_position_contract(
        self,
        logical_base: int,
        effective_base: int,
        query_len: int,
        expected_positions: list[int],
        expected_cache_positions: list[int],
        expected_logical_len: int,
        expected_effective_len: int,
    ) -> None:
        device = torch.device("cuda")
        idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
        query_start_loc = torch.tensor(
            [0, query_len], dtype=torch.int32, device=device
        )
        logical = torch.tensor([logical_base], dtype=torch.int32, device=device)
        effective = torch.tensor([effective_base], dtype=torch.int32, device=device)
        positions = torch.zeros(query_len, dtype=torch.int64, device=device)
        cache_positions = torch.zeros_like(positions)
        seq_lens = torch.zeros(1, dtype=torch.int32, device=device)
        effective_seq_lens = torch.zeros_like(seq_lens)

        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            logical,
            positions,
            seq_lens,
            cache_pos=cache_positions,
            effective_kv_seq_lens=effective_seq_lens,
            effective_kv_len=effective,
        )
        torch.accelerator.synchronize()

        self.assertEqual(positions.cpu().tolist(), expected_positions)
        self.assertEqual(cache_positions.cpu().tolist(), expected_cache_positions)
        self.assertEqual(seq_lens.item(), expected_logical_len)
        self.assertEqual(effective_seq_lens.item(), expected_effective_len)

    def test_baseline_equality(self) -> None:
        self.assert_position_contract(128, 128, 1, [128], [128], 129, 129)
        logical_block_index = 128 // 16
        cache_block_index = 128 // 16
        self.assertEqual(logical_block_index, 8)
        self.assertEqual(cache_block_index, 8)

    def test_single_token_divergence(self) -> None:
        self.assert_position_contract(128, 64, 1, [128], [64], 129, 65)
        logical_block_index = 128 // 16
        cache_block_index = 64 // 16
        self.assertEqual(logical_block_index, 8)
        self.assertEqual(cache_block_index, 4)

    def test_divergent_positions_and_lengths(self) -> None:
        self.assert_position_contract(
            128,
            64,
            4,
            [128, 129, 130, 131],
            [64, 65, 66, 67],
            132,
            68,
        )

    def test_effective_lengths_follow_logical_padding_contract(self) -> None:
        device = torch.device("cuda")
        logical_seq_lens = torch.full((4,), -1, dtype=torch.int32, device=device)
        effective_seq_lens = torch.full_like(logical_seq_lens, -1)

        prepare_pos_seq_lens(
            idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            num_computed_tokens=torch.tensor([128], dtype=torch.int32, device=device),
            pos=torch.zeros(1, dtype=torch.int64, device=device),
            seq_lens=logical_seq_lens,
            cache_pos=torch.zeros(1, dtype=torch.int64, device=device),
            effective_kv_seq_lens=effective_seq_lens,
            effective_kv_len=torch.tensor([64], dtype=torch.int32, device=device),
        )
        torch.accelerator.synchronize()

        self.assertEqual(logical_seq_lens.cpu().tolist(), [129, 0, 0, 0])
        self.assertEqual(effective_seq_lens.cpu().tolist(), [65, 0, 0, 0])

    def test_common_metadata_preserves_logical_seq_lens(self) -> None:
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
        logical = torch.tensor([129, 98], dtype=torch.int32, device=device)
        effective = torch.tensor([65, 82], dtype=torch.int32, device=device)
        metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=logical,
            effective_kv_seq_lens=effective,
            num_reqs=2,
            num_actual_tokens=3,
            max_query_len=2,
            max_seq_len=129,
            block_table_tensor=torch.zeros((2, 1), dtype=torch.int32, device=device),
            slot_mapping=torch.zeros(3, dtype=torch.int64, device=device),
        )
        unpadded = metadata.unpadded(2, 1)

        self.assertEqual(metadata.seq_lens.cpu().tolist(), [129, 98])
        self.assertEqual(metadata.effective_kv_seq_lens.cpu().tolist(), [65, 82])
        self.assertEqual(unpadded.seq_lens.cpu().tolist(), [129])
        self.assertEqual(unpadded.effective_kv_seq_lens.cpu().tolist(), [65])

    def test_flash_attention_length_selection_and_fallback(self) -> None:
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
        logical = torch.tensor([129, 98], dtype=torch.int32, device=device)
        effective = torch.tensor([65, 82], dtype=torch.int32, device=device)
        kwargs = dict(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=logical,
            num_reqs=2,
            num_actual_tokens=3,
            max_query_len=2,
            max_seq_len=129,
            block_table_tensor=torch.zeros((2, 1), dtype=torch.int32, device=device),
            slot_mapping=torch.zeros(3, dtype=torch.int64, device=device),
        )

        physical_metadata = CommonAttentionMetadata(
            effective_kv_seq_lens=effective, **kwargs
        )
        fallback_metadata = CommonAttentionMetadata(**kwargs)

        self.assertIs(_select_kv_seq_lens(physical_metadata), effective)
        self.assertIs(_select_kv_seq_lens(fallback_metadata), logical)

    def test_prepare_attn_wires_cache_positions(self) -> None:
        cache_positions = torch.tensor([64], dtype=torch.int64, device="cuda")
        logical_positions = torch.tensor([128], dtype=torch.int64, device="cuda")
        captured: dict[str, torch.Tensor] = {}

        class FakeBlockTables:
            def gather_block_tables(self, idx_mapping, num_reqs_padded):
                return (torch.zeros((1, 1), dtype=torch.int32, device="cuda"),)

            def compute_slot_mappings(
                self, idx_mapping, query_start_loc, positions, num_tokens_padded
            ):
                captured["positions"] = positions
                return torch.zeros((1, 1), dtype=torch.int64, device="cuda")

        runner = SimpleNamespace(
            pcp_manager=None,
            block_tables=FakeBlockTables(),
        )
        input_batch = SimpleNamespace(
            idx_mapping=torch.tensor([0], dtype=torch.int32, device="cuda"),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            cache_positions=cache_positions,
            positions=logical_positions,
            num_reqs_after_padding=1,
            num_tokens_after_padding=1,
        )

        GPUModelRunner.prepare_attn(runner, input_batch)

        self.assertIs(captured["positions"], cache_positions)
        self.assertIsNot(captured["positions"], logical_positions)

    def _make_minimal_flash_builder(self) -> FlashAttentionMetadataBuilder:
        builder = FlashAttentionMetadataBuilder.__new__(FlashAttentionMetadataBuilder)
        builder.aot_schedule = False
        builder.aot_sliding_window = None
        builder.use_full_cuda_graph = False
        builder.max_cudagraph_size = None
        builder.max_num_splits = 0
        builder.dcp_world_size = 1
        builder.dcp_rank = 0
        builder.cp_kv_cache_interleave_size = 1
        builder.device = torch.device("cuda")
        builder.cache_config = SimpleNamespace(cache_dtype="auto")
        builder.kv_cache_dtype = torch.bfloat16
        builder.num_heads_q = 1
        builder.num_heads_kv = 1
        builder.headdim = 16
        builder.block_size = 16
        builder.kv_cache_spec = SimpleNamespace(sliding_window=None)
        builder.rswa_window = None
        builder.persistent_rswa_prefix_lens = None
        builder.persistent_rswa_window_tensor = None
        return builder

    def _make_builder_metadata(
        self, effective_kv_seq_lens: torch.Tensor | None
    ) -> CommonAttentionMetadata:
        device = torch.device("cuda")
        return CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
            query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
            seq_lens=torch.tensor([129, 98], dtype=torch.int32, device=device),
            effective_kv_seq_lens=effective_kv_seq_lens,
            num_reqs=2,
            num_actual_tokens=3,
            max_query_len=2,
            max_seq_len=129,
            block_table_tensor=torch.zeros((2, 1), dtype=torch.int32, device=device),
            slot_mapping=torch.zeros(3, dtype=torch.int64, device=device),
        )

    def test_flash_builder_outputs_effective_lengths_and_fallback(self) -> None:
        builder = self._make_minimal_flash_builder()
        effective = torch.tensor([65, 82], dtype=torch.int32, device="cuda")

        physical = builder.build(0, self._make_builder_metadata(effective))
        fallback = builder.build(0, self._make_builder_metadata(None))

        self.assertEqual(physical.seq_lens.cpu().tolist(), [65, 82])
        self.assertEqual(fallback.seq_lens.cpu().tolist(), [129, 98])

    def test_effective_length_scope_guard(self) -> None:
        metadata = self._make_builder_metadata(
            torch.tensor([65, 82], dtype=torch.int32, device="cuda")
        )
        self.assertIs(
            _select_kv_seq_lens(metadata, use_effective=True),
            metadata.effective_kv_seq_lens,
        )
        self.assertIs(
            _select_kv_seq_lens(metadata, use_effective=False), metadata.seq_lens
        )


if __name__ == "__main__":
    unittest.main()
