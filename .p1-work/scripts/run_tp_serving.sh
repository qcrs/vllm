#!/usr/bin/env bash
set -euo pipefail
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
stats=/data_ssd/qcrs/experiments/p1-v8/M2/20261031-000856-normalized-q-calibration/frequency.pt
for pool in 8 0.75; do
    policies=(full noop sink_recent)
    if [[ "$pool" == 0.75 ]]; then
        policies+=(random triattention)
    fi
    for policy in "${policies[@]}"; do
        .venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
            --name "http-long-full-$pool-$policy" -- \
            @python .p1-work/scripts/serving_reclamation.py --tp 2 --graph-mode full \
            --policy "$policy" --pool-gib "$pool" --stats "$stats"
    done
done
for mode in full eager; do
    for policy in full noop; do
        .venv/bin/python .p1-work/scripts/run_guard.py --stage TP \
            --name "http-short-$mode-8-$policy" -- \
            @python .p1-work/scripts/serving_reclamation.py --tp 2 --graph-mode "$mode" \
            --workload short --policy "$policy" --pool-gib 8 --full-warmup
    done
done
