# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NOTE: Coding style guide for this file:
This model runner is shared by all models: text and multimodal, generative
and embedding, public and private. As a result, this file must only contain
code that is common to every model. Model-specific behavior belongs in the
appropriate model-specific files.

In other words:
* Be paranoid about changing this file. It should remain stable.
* Be even more paranoid about adding new lines. It should remain minimal.

Even for shared features (for example, different parallelism modes), keep the
complexity out of this path. The less common the feature, the more it should be
hidden. Prefer utility functions defined elsewhere and call them from here,
instead of embedding feature-specific logic directly.
"""

import functools
import gc
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import PIN_MEMORY, STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import (
    CompactionPlanData,
    CompactionResultData,
    GrammarOutput,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.async_utils import AsyncOutput, AsyncPoolingOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import (
    async_copy_to_gpu,
    set_default_max_concurrency,
)
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ModelCudaGraphManager,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.eplb_utils import EPLBController, step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    post_update,
    post_update_num_computed_tokens,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_compaction import compact_paged_kv_triton_2d
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import (
    LoraState,
    create_lora_capture_hook,
    get_lora_capture_cases,
    get_num_active_loras_for_dispatch,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.lora import set_active_mm_loras
from vllm.v1.worker.gpu.model_states import init_model_state
from vllm.v1.worker.gpu.pool.pooling_runner import PoolingRunner
from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.shutdown import free_before_shutdown
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import KVBlockZeroer, copy_kv_cache_blocks_inplace

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PreparedCompaction:
    request_id: str
    req_state_idx: int
    batch_idx: int
    source_effective_kv_len: int
    source_num_blocks: int
    block_ids: tuple[int, ...]
    block_size: int
    keep_member_indices: tuple[int, ...]
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None


class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]

        # Lazily initialized in _init_kv_zero_meta() when the KV cache needs
        # zeroing (e.g. hybrid models with fp8 KV cache).
        self.kv_block_zeroer: KVBlockZeroer | None = None

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.is_encoder_decoder = self.model_config.is_encoder_decoder

        self.output_copy_stream = torch.cuda.Stream(self.device)

        # Pipeline parallelism.
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.is_first_pp_rank = get_pp_group().is_first_rank
        self.is_last_pp_rank = get_pp_group().is_last_rank

        # Size the UVA buffer pools to the max number of concurrent in-flight
        # steps. Must run before any pooled buffer is constructed
        set_default_max_concurrency(vllm_config.max_concurrent_batches)

        # PP broadcast/recv helper. Runs the collective on a side stream.
        self.pp_handler: PPHandler | None = None

        # Persistent buffer for intermediate tensors (non-first PP ranks).
        self.intermediate_tensors: IntermediateTensors | None = None

        # Data parallelism.
        self.dp_size = self.parallel_config.data_parallel_size
        self.dp_rank = self.parallel_config.data_parallel_rank

        # Decode context parallelism.
        self.dcp_size = self.parallel_config.decode_context_parallel_size
        self.use_dcp = self.dcp_size > 1
        self.dcp_rank = get_dcp_group().rank_in_group if self.use_dcp else 0
        self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        self.encoder_cache = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            self.encoder_cache = EncoderCache()

        # Speculative decoding.
        self.speculator = None
        self.use_aux_hidden_state_outputs = False
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method in ("eagle3", "dflash", "dspark"):
                # Drafting may require auxiliary hidden states from target model outputs
                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        self.pcp_manager: pcp.PCPManager | None = None

        # Pooling models.
        self.is_pooling_model = self.model_config.runner_type == "pooling"
        self.pooling_runner: PoolingRunner | None = None

        # General request states.
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        if self.use_pp:
            self.pp_handler = PPHandler(
                max_num_reqs=self.max_num_reqs,
                num_speculative_steps=self.num_speculative_steps,
                device=self.device,
            )

        # Samplers and decode_query_len created in load_model() after
        # model_state exists (num_new_sampled_tokens_per_step from ModelState).
        self.sampler: Sampler | None = None
        self.rejection_sampler: RejectionSampler | None = None
        self.prompt_logprobs_worker: PromptLogprobsWorker | None = None
        self.structured_outputs_worker: StructuredOutputsWorker | None = None
        self.cudagraph_manager: ModelCudaGraphManager | None = None

        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)
        self.lora_capture_cases = [0]
        if self.lora_config:
            self.lora_capture_cases = get_lora_capture_cases(
                self.lora_config, self.compilation_config
            )

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

        # For transferring state from execute_model to subsequent sample_tokens call.
        self.execute_model_state: ExecuteModelState | None = None

        # Expert parallelism load balancer.
        self.eplb = EPLBController(self.parallel_config, self.device)

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks: list[SupportedTask] = []
        if self.model_config.runner_type == "generate":
            tasks.extend(self.model_state.get_supported_generation_tasks())
        if self.is_pooling_model:
            # Do not rely on pooling_runner here, since this information is needed
            # on the first PP rank, while pooling_runner is only initialized
            # on the last PP rank.
            tasks.extend(PoolingRunner.get_supported_tasks(self.model))
        return tuple(tasks)

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        self.eplb.prepare_load()
        eplb_models_added = False
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.vllm_config.model_config
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )

            if self.use_aux_hidden_state_outputs:
                assert self.speculative_config is not None
                set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
            if isinstance(self.speculator, DraftModelSpeculator):
                self.speculator.load_model(self.model)
                eplb_models_added = self.eplb.maybe_register_speculator(
                    self.speculator, self.speculative_config, load_dummy_weights
                )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        if not load_dummy_weights:
            prepare_communication_buffer_for_model(self.model)
            if self.speculator is not None:
                prepare_communication_buffer_for_model(self.speculator.model)

        # Initialize the components that require the model.
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )

        self.decode_query_len = (
            self.num_speculative_steps
            + self.model_state.num_new_sampled_tokens_per_step
        )

        # Initialize samplers. Model states may override via custom_sampler().
        if self.is_last_pp_rank and not self.is_pooling_model:
            self.sampler = Sampler(
                max_num_reqs=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                req_states=self.req_states,
                logprobs_mode=self.model_config.logprobs_mode,
                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )
            custom = self.model_state.custom_sampler(self.sampler)

            if custom:
                self.sampler, self.rejection_sampler = custom
            elif self.speculative_config is not None:
                self.rejection_sampler = RejectionSampler(
                    self.sampler,
                    self.speculative_config,
                    self.device,
                )
            self.prompt_logprobs_worker = PromptLogprobsWorker(
                self.max_num_reqs,
                logprobs_mode=self.model_config.logprobs_mode,
            )
            self.structured_outputs_worker = StructuredOutputsWorker(
                max_num_logits=self.max_num_reqs * self.decode_query_len,
                vocab_size=self.vocab_size,
                device=self.device,
            )

        if self.is_pooling_model and self.is_last_pp_rank:
            self.pooling_runner = PoolingRunner(self.model)
        eplb_models_added |= self.eplb.maybe_register_model(
            self.model,
            self.model_config,
            load_dummy_weights,
        )
        self.eplb.maybe_start_async_loop(eplb_models_added)

        if not self.is_first_pp_rank:
            # For non-first PP ranks, create intermediate tensors sized
            # for the max capture size so they can be sliced per batch.
            # Save as persistent member so runtime can copy received data
            # into the same addresses that the CUDA graphs captured.
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )

    def get_model(self) -> nn.Module:
        return self.model

    def get_draft_model(self) -> nn.Module | None:
        speculator = self.speculator
        if not isinstance(speculator, DraftModelSpeculator):
            return None
        return speculator.model

    def reload_weights(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_weights(self, *args, **kwargs)  # type: ignore[arg-type]

    def update_config(self, *args, **kwargs) -> None:
        # TODO(Wentao): Use full version instead of import when fully migrated to v2
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.update_config(self, *args, **kwargs)  # type: ignore[arg-type]

        # v2 reads config via self.vllm_config (e.g. in load_model), so keep it
        # in sync with the attributes the v1 helper just replaced.
        self.vllm_config.model_config = self.model_config
        self.vllm_config.load_config = self.load_config

    @functools.cached_property
    def main_stream(self) -> torch.cuda.Stream:
        # Cache the default CUDA stream to avoid lookup overhead.
        return torch.cuda.current_stream(self.device)

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        # 深拉票呢哦 建立 本地得runtime
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            # Cross-attention block tables need to index encoder tokens, which
            # can exceed the decoder's max_model_len.
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            # When using DCP, each request's KV cache is sharded among different ranks.
            # As a result, one block on the current rank covers `block_size * cp_size`
            # tokens in the full, global (unsharded) sequence.
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
            # Align to a multiple of (128 / block_size) as required by some attention
            # backends such as TRTLLM (#39324)
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            # For Mamba/Hybrid Model, KVCaches need extra blocks for speculative tokens
            if isinstance(spec, MambaSpec):
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
            max_num_blocks_per_group.append(max_num_blocks)

        self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        self.pcp_manager = pcp.maybe_build_pcp_manager(
            self.vllm_config,
            self.device,
            self.supports_mm_inputs,
            self.req_states,
            self.block_tables,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            attn_cg_support.min_cg_support,
            attn_cg_support.min_cg_attn_backend,
            self.decode_query_len,
            use_v2_model_runner=True,
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            kv_cache_config=self.kv_cache_config,
            max_num_reqs=self.max_num_reqs,
        )
        self.cudagraph_manager = ModelCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.decode_query_len,
            lora_capture_cases=self.lora_capture_cases,
        )
        check_attention_cp_compatibility(self.vllm_config)
        if isinstance(self.speculator, DraftModelSpeculator):
            # HACK(woosuk)
            self.speculator.set_attn(
                self.model_state,
                self.kv_cache_config,
                self.block_tables,
                self.input_buffers,
                self.attn_groups,
            )
        if self.speculator is not None:
            # After set_attn, so the speculator can size its cudagraph mode
            # to its own attention support.
            self.speculator.init_cudagraph_manager(cudagraph_mode)

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            #  runner 持有最终 tensor references
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_groups,
            self.device,
            self.cache_config.cache_dtype,
            self.kernel_block_sizes,
            self.vllm_config,
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            pin_memory=PIN_MEMORY,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
            max_concurrency=self.vllm_config.max_concurrent_batches,
        )

    @torch.inference_mode()
    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if skip_attn and not is_profile:
            raise ValueError(
                "skip_attn must only be True for initial memory profiling."
            )

        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        if uniform_decode:
            # HACK(lucas): for now since the worker is shared between MRV1 and MRV2,
            # and for spec-decode with MTP we want to make sure the dummy runs use
            # 1+num_speculative_tokens we use max here, this will likely be eventually
            # changed in the worker: https://github.com/vllm-project/vllm/pull/35243
            num_tokens = max(num_tokens, self.decode_query_len)
            num_reqs = num_tokens // self.decode_query_len
            assert num_tokens % self.decode_query_len == 0
        num_tokens_per_request = [num_tokens // num_reqs] * num_reqs
        num_tokens_per_request[-1] += num_tokens % num_reqs

        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Get the intermediate tensors for the dummy run.
        intermediate_tensors = None
        if not self.is_first_pp_rank:
            assert self.intermediate_tensors is not None
            intermediate_tensors = self.intermediate_tensors[:num_tokens]

        max_loras = self.lora_config.max_loras if self.lora_config is not None else 0
        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens=np.array(num_tokens_per_request, dtype=np.int32),
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            # Execute the model.
            self.execute_model(
                dummy_scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                is_profile=is_profile,
            )
        self.kv_connector.set_disabled(False)

        # Non-last PP ranks don't produce output for sampling.
        if not self.is_last_pp_rank:
            return None, None

        assert self.execute_model_state is not None
        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        self.execute_model_state = None

        # dummy run the eagle speculator's propose to ensure DP/EP sync.
        if self.speculator is not None:
            assert self.sampler is not None
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
            if self.speculator.supports_mm_inputs:
                mm_inputs = (
                    [],
                    torch.zeros(
                        input_batch.num_tokens,
                        dtype=torch.bool,
                        device=self.device,
                    ),
                )

            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            self.speculator.propose(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings=slot_mappings_by_layer,
                last_hidden_states=spec_hidden_states,
                aux_hidden_states=aux_hidden_states,
                num_sampled=torch.ones(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                num_rejected=torch.zeros(
                    input_batch.num_reqs, dtype=torch.int32, device=self.device
                ),
                last_sampled=self.req_states.last_sampled_tokens,
                next_prefill_tokens=self.req_states.next_prefill_tokens,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seeds=self.sampler.sampling_states.seeds.gpu,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn,
                mm_inputs=mm_inputs,
                is_profile=is_profile,
            )

        assert hidden_states is not None  # Last PP rank always has hidden_states
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        dummy_input_batch = InputBatch.make_dummy(
            num_reqs, num_reqs, self.input_buffers
        )

        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        assert self.sampler is not None
        self.sampler(logits, dummy_input_batch)

    @torch.inference_mode()
    def _dummy_pooler_run(self, hidden_states: torch.Tensor) -> None:
        assert self.pooling_runner is not None
        self.pooling_runner.dummy_pooler_run(hidden_states)

    @torch.inference_mode()
    def profile_run(self) -> None:
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
        if self.is_last_pp_rank:
            assert sample_hidden_states is not None
            if self.pooling_runner is None:
                self._dummy_sampler_run(sample_hidden_states)
            else:
                self._dummy_pooler_run(hidden_states)

        torch.accelerator.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def post_kv_cache_wake_up(self) -> None:
        self.block_tables.init_block_table_layout_tensors()

    def reset_mm_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        # SP is not supported yet.
        return num_scheduled_tokens

    def profile_cudagraph_memory(self) -> int:
        # NOTE(woosuk): It is TBD whether we keep this API or not.
        return 0

    @torch.inference_mode()
    def capture_model(self) -> int:
        assert self.cudagraph_manager is not None
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        gc.collect()
        torch.accelerator.empty_cache()
        start_free_gpu_memory = torch.accelerator.get_memory_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            self.cudagraph_manager.capture(
                self.model,
                self.model_state,
                self.input_buffers,
                self.intermediate_tensors,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                has_lora=self.lora_config is not None,
                use_aux_hidden_state_outputs=self.use_aux_hidden_state_outputs,
                lora_capture_hook=create_lora_capture_hook(self.lora_config, self),
            )
            if self.speculator is not None:
                self.speculator.capture()

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def _remove_request(self, req_id: str) -> bool:
        # Call model_state.remove_request *before* req_states.remove_request
        # so the model_state can still look up the slot index.
        self.model_state.remove_request(req_id)
        req_idx = self.req_states.remove_request(req_id)
        if req_idx is None:
            return False
        if self.pp_handler is not None:
            self.pp_handler.on_req_idx_freed(req_idx)
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)
        if self.prompt_logprobs_worker is not None:
            self.prompt_logprobs_worker.remove_request(req_id)
        self.lora_state.remove_request(req_id)
        return True

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            self._remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.encoder_cache is not None:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.free_encoder_cache(mm_hash)

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = len(new_req_data.prompt_token_ids)
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=sampling_params.max_tokens if sampling_params else 1,  # type: ignore[arg-type]
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

            if self.is_last_pp_rank and new_req_data.sampling_params is not None:
                assert self.sampler is not None
                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )
                assert self.prompt_logprobs_worker is not None
                self.prompt_logprobs_worker.add_request(
                    req_id, req_index, new_req_data.sampling_params
                )

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.model_state.apply_staged_writes()
        if self.sampler is not None:
            self.sampler.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        reclaim_transitions = reqs.reclaim_transitions or [None] * len(reqs.req_ids)
        for req_id, num_computed_tokens, req_new_block_ids, reclaim_transition in zip(
            reqs.req_ids,
            reqs.num_computed_tokens,
            reqs.new_block_ids,
            reclaim_transitions,
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if reclaim_transition is not None:
                # Reclaim and normal append are mutually exclusive for one row.
                self._commit_reclaim_transition(
                    request_id=req_id,
                    retained_block_ids=reclaim_transition.retained_block_ids,
                    new_effective_kv_len=reclaim_transition.new_effective_kv_len,
                    expected_old_num_blocks=(
                        reclaim_transition.expected_old_num_blocks
                    ),
                    same_step_new_block_ids=req_new_block_ids,
                )
            elif req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        # Update CPU num_computed_prefill_tokens.
        np.minimum(
            self.req_states.num_computed_tokens_np,
            self.req_states.prefill_len.np,
            out=self.req_states.num_computed_prefill_tokens,
        )

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Apply copy-on-write block copies for partial prefix-cache hits, after
        # zeroing new blocks and before the forward pass reads them.
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

    def _commit_reclaim_transition(
        self,
        request_id: str,

        # reclaim 之后，从“旧 Worker block row”中保留下来的 block IDs。
        #
        # 例如旧 row：
        #   [B0, B1, B2, B3, B4, B5]
        #
        # 删除 B2/B3 后：
        #   retained_block_ids = [B0, B1, B4, B5]
        #
        # 注意：
        # 这里“不应该”包含当前 scheduler step 新申请的 B6。
        retained_block_ids: list[int],

        # reclaim 后的有效 physical KV token 数。
        #
        # 例如：
        #   old effective = 94
        #   删除 2 个完整 block
        #   block_size = 16
        #
        #   new effective = 94 - 2*16 = 62
        #
        # 注意：
        # 它描述的是“已经真正有效的 KV token 数”，
        # 不是 block capacity。
        new_effective_kv_len: int,

        # 一个 stale-decision validation fence。
        #
        # producer 在产生 reclaim decision 时认为：
        #   “这个 request 当前应该有 6 个 active blocks”
        #
        # 所以传：
        #   expected_old_num_blocks = 6
        #
        # Worker 真正执行时也必须看到 6。
        #
        # 如果 Worker 已经变成 7，
        # 说明这个 reclaim decision 已经过期，不能继续执行。
        expected_old_num_blocks: int,

        # 当前 scheduler step “额外新分配”的 block IDs。
        #
        # 例如：
        #   retained = [B0, B1, B4, B5]
        #
        # 当前 query 又跨 block boundary，
        # scheduler allocate_slots() 新分配：
        #   B6
        #
        # 那么：
        #   same_step_new_block_ids = ([B6],)
        #
        # 为什么是 tuple[list[int], ...]？
        # 因为 vLLM block table API 原本支持多个 KV cache groups：
        #
        #   (
        #       group0_block_ids,
        #       group1_block_ids,
        #       ...
        #   )
        #
        # 但我们 V1 只允许 single KV group。
        same_step_new_block_ids: tuple[list[int], ...] | None = None,
    ) -> None:

        """Stage a validated single-group worker physical-view transition."""

        # ============================================================
        # Phase 1:
        # 确认这个 request 在 Worker 里还存在。
        # ============================================================

        # 如果 request 已经 finish / remove / preempt 掉，
        # 一个旧的 reclaim decision 就不能继续作用在它身上。
        if request_id not in self.req_states.req_id_to_index:
            raise ValueError(f"Unknown worker request: {request_id}")


        # ============================================================
        # Phase 2:
        # V1 scope guard：只支持一个 KV cache group。
        # ============================================================

        # 当前 reclaim contract 只审计过：
        #   num_kv_cache_groups = 1
        #
        # multi-group / hybrid KV 的 block row layout 更复杂，
        # 所以不能让这个 helper 悄悄泛化过去。
        if self.block_tables.num_kv_cache_groups != 1:
            raise ValueError(
                "Worker reclaim commit requires one KV cache group"
            )


        # ============================================================
        # Phase 3:
        # 读取当前 Worker 的“旧状态”。
        #
        # 后面的 validation 全部针对这份 old state。
        # ============================================================

        # request_id 是逻辑 ID，
        # Worker 内部真正的 GPU/CPU state 都按 row index 保存。
        #
        # 例如：
        #   "req-A" -> req_index = 0
        req_index = self.req_states.req_id_to_index[request_id]


        # 当前 Worker active block-table prefix 长度。
        #
        # 例如：
        #   [B0 B1 B2 B3 B4 B5]
        #
        # active count:
        #   old_num_blocks = 6
        #
        # 这里读的是 CPU-side num_blocks.np，
        # 不需要 GPU readback。
        old_num_blocks = int(
            self.block_tables.num_blocks.np[0, req_index]
        )


        # 当前 reclaim 发生之前的 physical KV valid length。
        #
        # canonical:
        #   old_effective_kv_len = 94
        #
        # 它现在是 GPU persistent state，
        # 所以这里 .item() 会有 GPU -> CPU scalar read。
        #
        # correctness 没问题；
        # 只是未来可以优化掉这个同步。
        old_effective_kv_len = int(
            self.req_states.effective_kv_len.gpu[req_index].item()
        )


        # 当前 logical progress。
        #
        # canonical:
        #   logical_num_computed_tokens = 94
        #
        # 和 physical 不同：
        #
        # reclaim 后：
        #   logical 仍然 94
        #   physical 变成 62
        #
        # 这里已经有 CPU mirror，所以直接读 np。
        logical_num_computed_tokens = int(
            self.req_states.num_computed_tokens_np[req_index]
        )


        # reclaim 后保留下来的“旧 blocks”数量。
        #
        # canonical:
        #   retained = [B0 B1 B4 B5]
        #   retained_num_blocks = 4
        retained_num_blocks = len(retained_block_ids)


        # 当前 KV block token capacity。
        #
        # canonical:
        #   block_size = 16
        block_size = self.block_tables.block_sizes[0]


        # ============================================================
        # Phase 4:
        # 把 current-step new block IDs 规范化成普通 list。
        # ============================================================

        if same_step_new_block_ids is None:

            # 当前 step 没有跨新 block boundary，
            # 没有新增 capacity。
            #
            # 例如：
            # E=62，只计算 physical position 62，
            # 仍然落在已有 B5 中。
            new_block_ids: list[int] = []

        elif len(same_step_new_block_ids) != 1:

            # API shape 本来允许：
            #
            #   (
            #       group0_ids,
            #       group1_ids,
            #   )
            #
            # 但 P1 V1 只支持 single group，
            # 所以如果 tuple 里不是恰好一个 group，
            # 直接拒绝。
            raise ValueError(
                "same-step new block IDs require one KV cache group"
            )

        else:

            # single group：
            #
            #   ([B6],)
            #
            # 展开成：
            #
            #   [B6]
            new_block_ids = same_step_new_block_ids[0]


        # ============================================================
        # Phase 5:
        # Validation 1 —— decision 有没有过期？
        # ============================================================

        # producer 产生 decision 时认为 old row 有 6 blocks。
        #
        # Worker 真正执行时也必须还是 6。
        #
        # 如果变成：
        #   expected = 6
        #   actual   = 7
        #
        # 说明两边 state version 不一致。
        #
        # 此时不能继续用旧 retained row。
        if expected_old_num_blocks != old_num_blocks:
            raise ValueError(
                f"expected {expected_old_num_blocks} blocks, "
                f"found {old_num_blocks}"
            )


        # ============================================================
        # Phase 6:
        # Validation 2 —— retained row 本身有没有明显问题？
        # ============================================================

        # V1 不允许把整个 physical KV row reclaim 成空。
        if retained_num_blocks == 0:
            raise ValueError(
                "retained block list must be non-empty"
            )


        # 不可能：
        #
        # old row 只有 6 个 blocks，
        # reclaim 后反而说 retained 有 7 个“旧 blocks”。
        if retained_num_blocks > old_num_blocks:
            raise ValueError(
                "retained block count exceeds current active count"
            )


        # retained 里面不能同一个 physical block 出现两次。
        #
        # 错误例子：
        #   [B0 B1 B4 B4]
        #
        # 否则一个 physical block 被多个 logical physical columns 引用，
        # 当前 V1 contract 不允许。
        if len(set(retained_block_ids)) != retained_num_blocks:
            raise ValueError(
                "retained block IDs must be unique"
            )


        # ============================================================
        # Phase 7:
        # 构造真正 forward 前 Worker 应该看到的 final row。
        # ============================================================

        # canonical:
        #
        # retained:
        #   [B0 B1 B4 B5]
        #
        # same-step new:
        #   [B6]
        #
        # final:
        #   [B0 B1 B4 B5 B6]
        #
        # 注意：
        # 这里仅仅拼“block ID metadata list”。
        #
        # 没有 copy K/V payload。
        final_physical_row = [
            *retained_block_ids,
            *new_block_ids,
        ]


        # final row 同样不允许 duplicate。
        #
        # 这一条尤其能抓出：
        #
        # retained 本身已经错误包含 B6，
        # 然后 same_step_new 又带一次 B6。
        #
        # 例如：
        #
        # retained:
        #   [B0 B1 B4 B5 B6]
        #
        # new:
        #   [B6]
        #
        # final:
        #   [B0 B1 B4 B5 B6 B6]
        #
        # 必须拒绝。
        if len(set(final_physical_row)) != len(final_physical_row):
            raise ValueError(
                "final physical block IDs must be unique"
            )


        # ============================================================
        # Phase 8:
        # Validation 3 —— new effective KV length 合不合法？
        # ============================================================

        # 当前 scope 不支持 effective=0。
        #
        # 即不能把 request 的全部 physical KV 都删空。
        if new_effective_kv_len <= 0:
            raise ValueError(
                "new effective KV length must be positive"
            )


        # reclaim 的语义是 shrink。
        #
        # 不能：
        #   old_E = 94
        #   new_E = 110
        #
        # 那不是 reclaim。
        if old_effective_kv_len < new_effective_kv_len:
            raise ValueError(
                "reclaim cannot increase effective KV length"
            )


        # ============================================================
        # Phase 9:
        # 算“physical token 实际减少多少”。
        # ============================================================

        # canonical:
        #
        # old_E = 94
        # new_E = 62
        #
        # physical_shrink = 32
        physical_shrink = (
            old_effective_kv_len - new_effective_kv_len
        )


        # 再从“删掉多少整个 blocks”计算理论上应该 shrink 多少 token。
        #
        # old_num_blocks = 6
        # retained_num_blocks = 4
        #
        # 说明从 old row 中去掉了：
        #   6 - 4 = 2 blocks
        #
        # block_size = 16
        #
        # expected_shrink = 2 * 16 = 32
        #
        # 注意：
        # current-step 的 B6 不算在这里。
        #
        # 因为 B6 是新 capacity，
        # 不是“旧 row 中被保留的 block”。
        expected_shrink = (
            old_num_blocks - retained_num_blocks
        ) * block_size


        # ============================================================
        # Phase 10:
        # 核心 whole-block reclaim correctness check。
        # ============================================================

        # canonical：
        #
        # physical_shrink = 94 - 62 = 32
        #
        # expected_shrink = (6 - 4) * 16 = 32
        #
        # PASS。
        #
        # 错误例子：
        #
        # new_E = 63
        #
        # physical_shrink = 31
        # expected_shrink = 32
        #
        # 说明你不是删了完整两个 blocks，
        # 而是出现 token-level / inconsistent transition。
        if physical_shrink != expected_shrink:
            raise ValueError(
                "effective KV shrink does not match "
                "reclaimed whole blocks"
            )


        # 这一条从 token 维度再次 enforce：
        #
        # physical shrink 必须是 block_size 的整数倍。
        #
        # 例如：
        #   32 % 16 == 0
        #
        # 虽然在前一个 equality check 下某种程度上是冗余防御，
        # 但它直接表达了 V1 contract：
        #
        #   whole-block reclaim only
        if physical_shrink % block_size != 0:
            raise ValueError(
                "effective KV shrink must be block aligned"
            )


        # ============================================================
        # Phase 11:
        # physical progress 不能超过 logical progress。
        # ============================================================

        # reclaim 后正确关系：
        #
        #   E <= L
        #
        # canonical：
        #   62 <= 94
        #
        # 如果：
        #   E = 100
        #   L = 94
        #
        # 意味着 physical KV 居然比模型 logical progress 更多，
        # 不符合当前 contract。
        if new_effective_kv_len > logical_num_computed_tokens:
            raise ValueError(
                "effective KV length cannot exceed logical progress"
            )


        # ============================================================
        # Phase 12:
        # 保证 logical / physical divergence 是 whole-block 对齐。
        # ============================================================

        # whole-block reclaim 要求：
        #
        #   L - E = R * block_size
        #
        # canonical：
        #
        #   94 - 62 = 32
        #   32 % 16 = 0
        #
        # 所以：
        #
        #   L % S == E % S
        #
        # 这条 invariant 非常重要，
        # 因为它保证未来 logical 和 physical
        # 跨 block boundary 的 cadence 保持一致。
        if (
            logical_num_computed_tokens - new_effective_kv_len
        ) % block_size != 0:
            raise ValueError(
                "logical/physical divergence must be block aligned"
            )


        # ============================================================
        # Phase 13:
        # strict no-op。
        # ============================================================

        # 如果：
        #
        # retained block 数没减少
        # E 也没减少
        # 当前 step 也没有新增 block
        #
        # 那么实际上什么都没发生。
        #
        # 不应该为了 no-op 制造 staged descriptor。
        if (
            retained_num_blocks == old_num_blocks
            and new_effective_kv_len == old_effective_kv_len
            and not new_block_ids
        ):
            return


        # ============================================================
        # Phase 14:
        # 真正开始 mutation。
        #
        # 重点：
        # 到这里之前没有改任何 persistent/staged state。
        # 所以上面任何 validation 失败，都属于 fail-before-mutation。
        # ============================================================


        # ------------------------------------------------------------
        # Mutation A：
        # 把 reclaim + current-step allocation
        # 合成“一个最终 block row descriptor”。
        # ------------------------------------------------------------

        # canonical：
        #
        # old:
        #   [B0 B1 B2 B3 B4 B5]
        #
        # final:
        #   [B0 B1 B4 B5 B6]
        #
        # overwrite=True：
        # 从 column 0 重建整个 active prefix。
        #
        # 最重要的是：
        #
        # 这里只有“一次 append_block_ids()”
        #
        # 因此 BlockTables 只产生：
        #
        #   ONE staged descriptor
        #
        # 这就是 R1 用来解决原 01A
        # two-descriptor capacity conflict 的核心。
        self.block_tables.append_block_ids(
            req_index,
            (final_physical_row,),
            overwrite=True,
        )


        # ------------------------------------------------------------
        # Mutation B：
        # 如果 physical valid length 真的 shrink，
        # 再 stage effective_kv_len 的 absolute replacement。
        # ------------------------------------------------------------

        # canonical：
        #
        # old_E = 94
        # new_E = 62
        #
        # 注意：
        #
        # final block row 有 5 blocks：
        #   [B0 B1 B4 B5 B6]
        #
        # 但是 E 仍然是 62，
        # 不能写成：
        #
        #   5 * 16 = 80
        #
        # 因为 B6 只是当前 step 新分配出来的 capacity，
        # 当前 forward 尚未真正完成写入这些 future tokens。
        if new_effective_kv_len != old_effective_kv_len:
            self.req_states.effective_kv_len.stage_write_elem(
                req_index,
                new_effective_kv_len,
            )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, batch_desc: BatchExecutionDescriptor
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        num_tokens_after_padding = batch_desc.num_tokens
        assert num_tokens > 0
        if envs.VLLM_MOE_SKIP_PADDING:
            # Mark trailing cudagraph-padding rows so kernels can skip work for
            # them when supported.
            self.input_buffers.is_padding[:num_tokens].fill_(False)
            self.input_buffers.is_padding[num_tokens:num_tokens_after_padding].fill_(
                True
            )
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # batch_idx -> req_id
        req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.decode_query_len
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Get query_start_loc.
        # num_reqs_padded is None for PIECEWISE graphs (no request padding needed)
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
        prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
        computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens
        num_computed_prefill_tokens_np = computed_prefill_tokens_np[idx_mapping_np]
        is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np

        # Get prefill tokens if any.
        if np.any(is_prefilling_np):
            # 去准备把对应的片段拿出来
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens. 计算position的位置
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
            cache_pos=self.input_buffers.cache_positions,
            effective_kv_seq_lens=self.input_buffers.effective_kv_seq_lens,
            effective_kv_len=self.req_states.effective_kv_len.gpu,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]
        effective_kv_seq_lens = self.input_buffers.effective_kv_seq_lens[
            :num_reqs_padded
        ]

        dcp_local_seq_lens = None
        if self.use_dcp:
            # Prepare dcp local seq_lens.
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.dcp_size,
                self.dcp_rank,
                self.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        # decode 在这里处理
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )

        # CPU upper bound on seq_lens; padded entries left at zero.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        max_seq_len_np = None
        if self.use_pp:
            # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
            max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=prefill_len_np,
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
            is_prefilling_np=is_prefilling_np,
            max_seq_len_np=max_seq_len_np,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            cache_positions=self.input_buffers.cache_positions[
                :num_tokens_after_padding
            ],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            effective_kv_seq_lens=effective_kv_seq_lens,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            prompt_lens=prompt_lens,
        )
        return pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

    def prepare_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self.pcp_manager is not None:
            return self.pcp_manager.prepare_attn(input_batch)

        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        # Slot mappings: [num_kv_cache_groups, num_tokens_padded].
        # Kernel pads beyond num_tokens with PAD_SLOT_ID.
        # Logical positions remain model/RoPE coordinates; cache_positions track
        # effective KV append coordinates.
        slot_mappings = self.block_tables.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            # 函数内 这是计算地图 所以 不许哟啊原本的 positon

            # block table 只告诉你“第 N 个逻辑 block 对应哪个物理 block”。
            # 传入 postion block_index = 18 // 16 = 1 block_offset = 18 % 16 = 2
            input_batch.cache_positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        return block_tables, slot_mappings

    def prepare_dummy_attn(
        self, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = pcp.maybe_get_pcp_dummy_slot_mappings(
            self.pcp_manager, self.block_tables, input_batch.num_tokens
        )
        return block_tables, slot_mappings

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    def postprocess_sampled(
        self,
        idx_mapping: torch.Tensor,  # May include -1 for masked entries
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
        effective_kv_len_override_valid: torch.Tensor | None = None,
        effective_kv_len_override: torch.Tensor | None = None,
    ) -> None:
        # Update the number of computed tokens.
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.effective_kv_len.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            effective_kv_len_override_valid,
            effective_kv_len_override,
        )

        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )

    @staticmethod
    def extract_compaction_plans(
        scheduler_output: SchedulerOutput,
    ) -> dict[str, CompactionPlanData]:
        """Read the Scheduler→Worker V2 plan carrier without executing it."""
        plans = scheduler_output.compaction_plans
        if not isinstance(plans, dict):
            raise TypeError("compaction_plans must be a request-indexed dict")
        for request_id, plan in plans.items():
            if not isinstance(plan, CompactionPlanData):
                raise TypeError("compaction_plans values must be CompactionPlanData")
            if request_id != plan.request_id:
                raise ValueError("Compaction plan request identity mismatch")
        return plans

    def _prepare_v2_compactions(
        self,
        compaction_plans: dict[str, CompactionPlanData],
        input_batch: InputBatch,
    ) -> dict[str, _PreparedCompaction]:
        '''
        它就是逐项核验 Plan 里面声称的东西，转换成 Worker 真正能执行的 descriptor。
        Scheduler
        │
        └─ compaction_plans
            │
            ├─ request_id
            ├─ expected_source_E
            ├─ expected_source_num_blocks
            └─ keep_member_indices


        Current Batch
        │
        └─ input_batch
            │
            ├─ req_ids
            ├─ idx_mapping_np（req_state--> batch）
            └─ effective_kv_seq_lens


        Persistent Worker State
        │
        ├─ self.req_states
        │     └─ req_id_to_index
        │
        ├─ self.block_tables
        │     ├─ num_blocks.np
        │     └─ block_tables[0].gpu
        │
        └─ self.kv_caches
            └─ 每一层真实 KV cache tensor
        '''
        """Validate post-forward V2 sources without mutating runtime state."""
        if not compaction_plans:
            return {}
        if self.block_tables.num_kv_cache_groups != 1:
            raise ValueError("V2 compaction requires exactly one KV cache group")

        block_size = self.block_tables.block_sizes[0]
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("V2 compaction block_size must be positive")
        if not self.kv_caches:
            raise ValueError("V2 compaction requires initialized layer KV caches")
        '''
        enumerate 迭代器 返回 一个 元组 带索引
        req_ids 存放本轮需要forward的 id
        得到一个字典 是因为我们可能需要ids 去找batch_idx
        items 返回 (键，值) 元组
        '''
        batch_indices = {
            request_id: batch_idx
            for batch_idx, request_id in enumerate(input_batch.req_ids)
        }
        prepared: dict[str, _PreparedCompaction] = {}
        for request_id, plan in compaction_plans.items():
            if request_id != plan.request_id:
                raise ValueError("Compaction plan request identity mismatch")
            # 基于 id 得到 index
            req_state_idx = self.req_states.req_id_to_index.get(request_id)
            if req_state_idx is None:
                raise ValueError(f"V2 compaction request is missing: {request_id}")
            batch_idx = batch_indices.get(request_id)
            if batch_idx is None:
                raise ValueError(
                    f"V2 compaction request is not in the current batch: {request_id}"
                )
            if int(input_batch.idx_mapping_np[batch_idx]) != req_state_idx:
                raise ValueError(
                    f"V2 compaction request index mapping is stale: {request_id}"
                )
            # item 把单元素tensor转化为Python scalar 可能存在隐含的GPU->CPU sync
            source_effective_kv_len = int(
                input_batch.effective_kv_seq_lens[batch_idx].item()
            )
            if source_effective_kv_len <= 0:
                raise ValueError(
                    "V2 compaction source effective KV length must be positive"
                )
            if source_effective_kv_len != plan.expected_source_effective_kv_len:
                raise ValueError(
                    f"V2 compaction source effective KV length mismatch: {request_id}"
                )
            # block_tables req_state 的 变 需要 idx 来检索
            source_num_blocks = int(
                self.block_tables.num_blocks.np[0, req_state_idx]
            )
            if source_num_blocks <= 0:
                raise ValueError("V2 compaction source block count must be positive")
            if source_num_blocks != plan.expected_source_num_blocks:
                raise ValueError(
                    f"V2 compaction source block count mismatch: {request_id}"
                )
            expected_source_num_blocks = cdiv(source_effective_kv_len, block_size)
            if source_num_blocks != expected_source_num_blocks:
                raise ValueError(
                    "V2 compaction source block count is not the exact page count"
                )
            keep_member_indices = plan.keep_member_indices
            if not isinstance(keep_member_indices, list) or not keep_member_indices:
                raise ValueError("keep_member_indices must be a non-empty list")
            if any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in keep_member_indices
            ):
                raise TypeError("keep_member_indices values must be integers")
            if keep_member_indices[0] < 0:
                raise ValueError("keep_member_indices must be non-negative")
            if any(
                current <= previous
                for previous, current in zip(
                    keep_member_indices, keep_member_indices[1:]
                )
            ):
                raise ValueError("keep_member_indices must be strictly increasing")
            if keep_member_indices[-1] >= source_effective_kv_len:
                raise ValueError("keep_member_indices exceed the source extent")

            new_effective_kv_len = len(keep_member_indices)
            new_num_blocks = cdiv(new_effective_kv_len, block_size)
            if not 0 < new_num_blocks <= source_num_blocks:
                raise ValueError("V2 compaction destination block count is invalid")
            '''
            二维矩阵
            [max_num_reqs, max_num_blocks_per_req] 所以存放的 physical_block_id
            变成不可变、可哈希的元组
            '''
            block_ids_tensor = self.block_tables.block_tables[0].gpu[
                req_state_idx, :source_num_blocks
            ]

            block_ids = tuple(int(block_id) for block_id in block_ids_tensor.tolist())
            if len(set(block_ids)) != source_num_blocks or any(
                block_id < 0 for block_id in block_ids
            ):
                raise ValueError("V2 compaction active block IDs are invalid")
            max_block_id = max(block_ids)
            for layer_idx, kv_cache in enumerate(self.kv_caches):
                if kv_cache.ndim != 4:
                    raise ValueError(
                        f"V2 compaction layer {layer_idx} KV cache must be 4D"
                    )
                if kv_cache.shape[2] != block_size:
                    raise ValueError(
                        f"V2 compaction layer {layer_idx} block size mismatch"
                    )
                if max_block_id >= kv_cache.shape[0]:
                    raise ValueError(
                        f"V2 compaction layer {layer_idx} lacks a source block"
                    )
                if kv_cache.device != block_ids_tensor.device:
                    raise ValueError(
                        f"V2 compaction layer {layer_idx} device mismatch"
                    )

            prepared[request_id] = _PreparedCompaction(
                request_id=request_id,
                req_state_idx=req_state_idx,
                batch_idx=batch_idx,
                source_effective_kv_len=source_effective_kv_len,
                source_num_blocks=source_num_blocks,
                block_ids=block_ids,
                block_size=block_size,
                keep_member_indices=tuple(keep_member_indices),
                new_effective_kv_len=new_effective_kv_len,
                new_num_blocks=new_num_blocks,
                step_seq=plan.step_seq,
            )
        return prepared
    # 推理 不训练 不构建 autograd graph
    @torch.inference_mode()
    def _execute_v2_compactions(
        self,
        prepared_compactions: dict[str, _PreparedCompaction],
        input_batch: InputBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, list[CompactionResultData]]:
        """Execute validated V2 compactions and stage the worker next state."""
        '''
        override_valid
        = 当前 batch 哪些 request 的 effective_kv_len
        需要被 V2 强制覆盖（区分是0 还是 没有修改）

        override_value
        = 要覆盖成多少，也就是 K

        results
        = Worker → Scheduler 的 CompactionResultData 回执
        '''
        override_valid = torch.zeros(
            input_batch.num_reqs, dtype=torch.bool, device=self.device
        )
        override_value = torch.zeros(
            input_batch.num_reqs, dtype=torch.int32, device=self.device
        )
        results: list[CompactionResultData] = []
        prepared_tensors: list[
            tuple[_PreparedCompaction, torch.Tensor, torch.Tensor]
        ] = []

        # Phase 1: execute every request × every layer before touching any
        # Worker BlockTable metadata or constructing a success result.
        for request_id, prepared in prepared_compactions.items():
            keep = torch.tensor(
                prepared.keep_member_indices, dtype=torch.int64, device=self.device
            )
            block_ids = torch.tensor(
                prepared.block_ids, dtype=torch.int32, device=self.device
            )
            for kv_cache in self.kv_caches:
                new_effective_kv_len, new_num_blocks = compact_paged_kv_triton_2d(
                    kv_cache,
                    block_ids,
                    prepared.source_effective_kv_len,
                    keep,
                )
                if (
                    new_effective_kv_len != prepared.new_effective_kv_len
                    or new_num_blocks != prepared.new_num_blocks
                ):
                    raise RuntimeError(
                        "V2 compaction primitive returned an unexpected shape"
                    )
            prepared_tensors.append((prepared, keep, block_ids))

        # Phase 2: all payload calls succeeded; now stage metadata and publish
        # the per-batch absolute override/result state.
        for prepared, _, _ in prepared_tensors:
            retained_block_ids = list(prepared.block_ids[: prepared.new_num_blocks])
            self.block_tables.append_block_ids(
                prepared.req_state_idx,
                (retained_block_ids,),
                overwrite=True,
            )
            override_valid[prepared.batch_idx] = True
            override_value[prepared.batch_idx] = prepared.new_effective_kv_len
            results.append(
                CompactionResultData(
                    request_id=prepared.request_id,
                    expected_source_effective_kv_len=prepared.source_effective_kv_len,
                    expected_source_num_blocks=prepared.source_num_blocks,
                    new_effective_kv_len=prepared.new_effective_kv_len,
                    new_num_blocks=prepared.new_num_blocks,
                    step_seq=prepared.step_seq,
                )
            )
        return override_valid, override_value, results

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        compaction_plans = self.extract_compaction_plans(scheduler_output)
        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.req_states.effective_kv_len.apply_write()
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)

        num_active_loras = 0
        if self.lora_config:
            req_ids = list(scheduler_output.num_scheduled_tokens.keys())
            num_active_loras = get_num_active_loras_for_dispatch(
                self.lora_config, self.lora_state, req_ids, dummy_run
            )

        skip_compiled = False
        if self.is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Encoder-decoder models such as Whisper should run eager/non-compiled
            # when encoder inputs are scheduled, because this step updates
            # cross-attention cache with dynamic encoder outputs.
            skip_compiled = True

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=is_profile or skip_compiled,
            num_active_loras=num_active_loras,
        )

        if batch_desc.num_tokens == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            input_batch = self.prepare_inputs(scheduler_output, batch_desc)
            block_tables, slot_mappings = self.prepare_attn(input_batch)
            # Mamba "align" pre-copy: migrate recurrent state across block
            # boundaries before the forward. Runs only on real batches, and
            # before model_state.prepare_attn gathers num_accepted_tokens so the
            # boundary reset is visible to the attention metadata.
            self.model_state.preprocess_state(
                input_batch,
                block_tables,
                self.kv_cache_config,
                self.req_states.num_computed_tokens.gpu,
            )

            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            input_batch = InputBatch.make_dummy(
                batch_desc.num_reqs or num_reqs,
                batch_desc.num_tokens,
                self.input_buffers,
            )
            if not skip_attn_for_dummy_run:
                block_tables, slot_mappings = self.prepare_dummy_attn(input_batch)
            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        attn_metadata = None
        slot_mappings_by_layer = None
        if not (dummy_run and skip_attn_for_dummy_run):
            assert slot_mappings is not None
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            assert block_tables is not None
            attn_metadata = self.model_state.prepare_attn(
                input_batch,
                batch_desc.cg_mode,
                block_tables,
                slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )

        input_ids = input_batch.input_ids
        inputs_embeds = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            # Run MM encoder (if needed) and get multimodal embeddings.
            # Only first PP rank prepares multimodal embeddings.
            if dummy_run:
                # Obtain mm embeddings of correct shape for compiled model.
                inputs_embeds = self.model_state.dummy_inputs_embeds(
                    input_batch.num_tokens_after_padding
                )
            else:
                scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
                if self.lora_config is not None:
                    set_active_mm_loras(
                        model=self.model,
                        lora_manager=self.lora_manager,
                        encoder_cache=self.encoder_cache,
                        req_id_to_index=self.req_states.req_id_to_index,
                        lora_state=self.lora_state,
                        scheduled_encoder_inputs=scheduled_encoder_inputs,
                    )
                inputs_embeds = self.model_state.get_mm_embeddings(
                    scheduled_encoder_inputs, input_batch, self.req_states
                )
            if inputs_embeds is not None and not self.model.requires_raw_input_tokens:
                input_ids = None

        model_inputs = {
            "input_ids": input_ids,
            "positions": input_batch.positions,
            "inputs_embeds": inputs_embeds,
            "intermediate_tensors": None,
            # NOTE: Values returned by `prepare_inputs` will override the default
            # values above.
            **self.model_state.prepare_inputs(input_batch, self.req_states),
        }
        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None

            # Prepare the intermediate tensors.
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            n = input_batch.num_tokens_after_padding
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }
            model_inputs["intermediate_tensors"] = IntermediateTensors(new_tensors)
            del intermediate_tensors

        # Update the EPLB meta.
        self.eplb.prepare_forward(self.model_config, input_batch.num_tokens)

        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Use explicit cudagraph replay for FULL mode.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            assert self.cudagraph_manager is not None
            self.kv_connector.pre_forward(scheduler_output)
            model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            # For piecewise and eager mode, just call model().
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )

            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=batch_desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings_by_layer,
                skip_compiled=skip_compiled,
                is_padding=input_batch.is_padding,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # Run the PIECEWISE graph (compiled PW cudagraph or breakable
                    # cudagraph, chosen inside run_pw_graph). cg_mode is only
                    # PIECEWISE after the cudagraph manager exists.
                    assert self.cudagraph_manager is not None
                    model_output = self.cudagraph_manager.run_pw_graph(
                        self.model, model_inputs
                    )
                else:
                    # Eager (NONE): call the raw model directly.
                    model_output = self.model(**model_inputs)

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
            else:
                assert isinstance(model_output, torch.Tensor)
                hidden_states = model_output
                aux_hidden_states = None
            output_intermediate_tensors = None
        else:
            assert isinstance(model_output, IntermediateTensors)
            hidden_states = None
            aux_hidden_states = None
            output_intermediate_tensors = model_output

        prepared_compactions = self._prepare_v2_compactions(
            compaction_plans, input_batch
        )
        (
            effective_kv_len_override_valid,
            effective_kv_len_override,
            compaction_results,
        ) = self._execute_v2_compactions(prepared_compactions, input_batch)
        finished_req_ids = scheduler_output.finished_req_ids
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            finished_req_ids=finished_req_ids,
            effective_kv_len_override_valid=effective_kv_len_override_valid,
            effective_kv_len_override=effective_kv_len_override,
            compaction_results=compaction_results,
        )

        if not self.is_last_pp_rank:
            # Non-last PP rank: return IntermediateTensors for sending.
            return output_intermediate_tensors
        return None

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        effective_kv_len_override_valid = (
            self.execute_model_state.effective_kv_len_override_valid
        )
        effective_kv_len_override = self.execute_model_state.effective_kv_len_override
        compaction_results = self.execute_model_state.compaction_results
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
            # Will be adjusted for rejections if necessary in update_requests.
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # Might contain non-final prefill chunks, which will be scheduled
                # in the immediate next step (rather than in pp_size steps).
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # Last rank: sample tokens
        hidden_states, input_batch = pcp.maybe_restore_pcp_for_sampling(
            self.pcp_manager, hidden_states, input_batch
        )

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            compaction_results=compaction_results,
        )
        # Start async output copy here so that it can overlap with speculator proposal.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            # The EAGLE/MTP drafter reads one position ahead of the target.
            mm_inputs = self.model_state.gather_mm_embeddings(
                input_batch, draft_lookahead=1
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
            effective_kv_len_override_valid,
            effective_kv_len_override,
        )

        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                spec_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                mm_inputs=mm_inputs,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()

    @torch.inference_mode()
    @step_eplb_after()
    def pool(self) -> AsyncPoolingOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)

        if not self.is_last_pp_rank:
            self.postprocess_num_computed_tokens(input_batch)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        assert self.pooling_runner is not None
        pooler_output, is_valid = self.pooling_runner.pool(
            hidden_states, input_batch, self.req_states
        )

        # Build the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncPoolingOutput(
            model_runner_output=model_runner_output,
            pooler_output=pooler_output,
            is_valid=is_valid,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
        )

        self.postprocess_num_computed_tokens(input_batch)
        return async_output

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        # Update the number of computed tokens.
        post_update_num_computed_tokens(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            input_batch.query_start_loc,
        )

    def shutdown(self) -> None:
        """Release GPU tensors (model weights, KV caches, workspace) so that
        memory is reclaimable when running in the same process."""
        torch.accelerator.synchronize()
        if hasattr(self, "kv_caches"):
            self.kv_caches.clear()
        if hasattr(self, "attn_groups"):
            self.attn_groups.clear()
        if hasattr(self, "kv_cache_config"):
            del self.kv_cache_config
        free_before_shutdown(self.vllm_config)
        if hasattr(self, "model_state"):
            del self.model_state
        if getattr(self, "speculator", None) is not None:
            self.speculator = None
        if hasattr(self, "model"):
            del self.model

        gc.collect()
        torch.accelerator.empty_cache()
        logger.debug("Cleaned up model weights, KV caches, and workspace")

    ########### EPLB methods start ###########
    @property
    def eplb_state(self):
        return self.eplb.state

    @eplb_state.setter
    def eplb_state(self, state) -> None:
        self.eplb.state = state

    @property
    def eep_eplb_suppressed(self) -> bool:
        return self.eplb.suppressed

    @eep_eplb_suppressed.setter
    def eep_eplb_suppressed(self, suppressed: bool) -> None:
        self.eplb.suppressed = suppressed

    def setup_eplb_from_mapping(
        self,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        self.eplb.setup_from_mapping(
            self.model,
            self.model_config,
            expanded_physical_to_logical,
            old_num_physical_experts,
        )

    ########### EPLB methods end ###########


class ExecuteModelState(NamedTuple):
    input_batch: InputBatch
    attn_metadata: dict[str, Any] | None
    slot_mappings_by_layer: dict[str, torch.Tensor] | None
    hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    finished_req_ids: set[str]
    effective_kv_len_override_valid: torch.Tensor
    effective_kv_len_override: torch.Tensor
    compaction_results: list[CompactionResultData]


def sort_batch_req_ids(
    num_tokens_per_req: dict[str, int], decode_query_len: int
) -> list[str]:
    # Order decode -> short_extend -> prefill; split_decodes_and_prefills
    # relies on uniform decodes (query_len == decode_query_len) leading.
    key = lambda r: ((num := num_tokens_per_req[r]) != decode_query_len, num)
    return sorted(num_tokens_per_req, key=key)
