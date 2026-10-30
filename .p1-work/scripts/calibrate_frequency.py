# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrate normalized, pre-RoPE Q from the actual eager Qwen3 MRV2 model."""

import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams


def main():
    run = Path(os.environ["P1_RUN_DIR"])
    model = Path("/data/models/Qwen3-8B")
    data = Path("/data_ssd/qcrs/datasets/raw/wikitext-2-raw-v1/wiki.valid.raw")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = tokenizer.encode(
        data.read_text(), add_special_tokens=False, truncation=True, max_length=16384
    )
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=str(model),
            dtype="bfloat16",
            enforce_eager=True,
            attention_backend="FLASH_ATTN",
            block_size=16,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_cascade_attn=True,
            scheduler_reserve_full_isl=False,
            max_model_len=20480,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            kv_cache_memory_bytes=3 * 2**30,
            gpu_memory_utilization=0.4,
            seed=0,
            disable_log_stats=True,
        )
    )
    runner = (
        engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    )
    modules = [
        m for name, m in runner.model.named_modules() if name.endswith(".q_norm")
    ]
    assert len(modules) == 36
    sums = torch.zeros((36, 32, 3, 64), device="cuda", dtype=torch.float32)
    counts = [0] * 36

    def collect(layer):
        def hook(module, args, output):
            q = output.float()
            assert q.ndim == 3 and q.shape[1:] == (32, 128)
            real, imag = q[..., :64], q[..., 64:]
            sums[layer, :, 0].add_(real.sum(dim=0))
            sums[layer, :, 1].add_(imag.sum(dim=0))
            sums[layer, :, 2].add_((real.square() + imag.square()).sqrt().sum(dim=0))
            counts[layer] += q.shape[0]

        return hook

    handles = [
        module.register_forward_hook(collect(i)) for i, module in enumerate(modules)
    ]
    engine.add_request(
        "calibration",
        {"prompt_token_ids": ids},
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    while engine.has_unfinished_requests():
        engine.step()
    for handle in handles:
        handle.remove()
    assert counts == [len(ids)] * 36
    stats = (sums / len(ids)).cpu()
    assert torch.isfinite(stats).all()
    hf = engine.vllm_config.model_config.hf_text_config
    metadata = {
        "model_type": hf.model_type,
        "num_layers": hf.num_hidden_layers,
        "num_attention_heads": hf.num_attention_heads,
        "num_kv_heads": hf.num_key_value_heads,
        "head_dim": hf.head_dim,
        "rope_parameters": hf.rope_parameters,
        "rope_style": "half",
        "q_normalized": True,
        "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()
        ).hexdigest(),
        "calibration": {
            "path": str(data),
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "tokens": len(ids),
            "selection": "first 16384 raw text tokens; no chat template",
            "q_source": "q_norm output before rotary_emb in Qwen3Attention.forward",
        },
    }
    torch.save({"metadata": metadata, "stats": stats}, run / "frequency.pt")
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"status": "PASS", "counts": counts, "metadata": metadata}))


if __name__ == "__main__":
    main()
