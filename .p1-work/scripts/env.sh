#!/usr/bin/env bash
# Source this file from the P1 v8 worktree before running experiments.
export P1_CUDA_VISIBLE_DEVICES="${P1_CUDA_VISIBLE_DEVICES:-1}"
p1() { source /home/qcrs/learning/llm-kv-lab/activate p1; }
p1
export PYTHONPATH="$P1_WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
