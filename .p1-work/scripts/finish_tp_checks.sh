#!/usr/bin/env bash
set -euo pipefail
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
export P1_STATS=/data_ssd/qcrs/experiments/p1-v8/M2/20261031-000856-normalized-q-calibration/frequency.pt
.venv/bin/python .p1-work/scripts/summarize_tp_serving.py
.venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
    --name complete-gqa-stable-ties-nccl -- \
    @python .p1-work/scripts/tp_collectives.py
for lifecycle in q0 maintenance; do
    .venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
        --name "full-token-$lifecycle-delay" -- \
        @python .p1-work/scripts/tp_reclamation.py \
        --mode full --path token --lifecycle "$lifecycle" --delay
done
.venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
    --name nsys-http-tri-diagnostic -- \
    @python .p1-work/scripts/profile_tp.py --tp 2 --graph-mode full \
    --policy triattention --pool-gib 0.75 --stats "$P1_STATS" --repeats 1 --trace
.venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
    --name nsys-small-tri-diagnostic -- \
    @python .p1-work/scripts/profile_tp.py --trajectory --mode full \
    --path token --policy triattention
.venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
    --name nsys-slow-rank-diagnostic -- \
    @python .p1-work/scripts/profile_tp.py --trajectory --mode full --path token --delay
for policy in full noop; do
    .venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
        --name "nsys-short-eager-$policy-profile" -- \
        @python .p1-work/scripts/profile_tp.py --tp 2 --graph-mode eager \
        --workload short --policy "$policy" --pool-gib 8 --repeats 1 --full-warmup
done
for run in /data_ssd/qcrs/experiments/p1-v8/TP/*-nsys-*-diagnostic; do
    /opt/nvidia/nsight-systems/2024.7.1/bin/nsys export --type sqlite \
        --output "$run/trace.sqlite" "$run/tp.nsys-rep"
    .venv/bin/python .p1-work/scripts/analyze_tp_profile.py "$run"
done
for run in /data_ssd/qcrs/experiments/p1-v8/TP/*-nsys-short-eager-*-profile; do
    /opt/nvidia/nsight-systems/2024.7.1/bin/nsys export --type sqlite \
        --output "$run/trace.sqlite" "$run/tp.nsys-rep"
    .venv/bin/python .p1-work/scripts/analyze_tp_eager_profile.py "$run"
done
.venv/bin/python .p1-work/scripts/summarize_tp_serving.py
export P1_CUDA_VISIBLE_DEVICES=1
source .p1-work/scripts/env.sh
.venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
    --name final-reclamation-regression -- \
    @python -m pytest tests/v1/core/test_scheduler.py \
    tests/v1/core/test_kv_cache_utils.py \
    tests/v1/core/test_single_type_kv_cache_manager.py \
    tests/v1/core/test_prefix_caching.py tests/v1/worker/test_kv_compaction.py \
    tests/v1/executor/test_executor.py \
    -k 'reclamation or reclaim or physical or frequency or graph_attention' -q
