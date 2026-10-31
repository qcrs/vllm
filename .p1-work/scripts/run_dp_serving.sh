#!/usr/bin/env bash
set -euo pipefail
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
stats=/data_ssd/qcrs/experiments/p1-v8/M2/20261031-000856-normalized-q-calibration/frequency.pt
for pool in 4 1.5; do
    policies=(full noop sink_recent)
    if [[ "$pool" == 1.5 ]]; then
        policies+=(random triattention)
    fi
    for policy in "${policies[@]}"; do
        .venv/bin/python .p1-work/scripts/run_guard.py --stage DP \
            --name "http-native-$pool-$policy" -- \
            @python .p1-work/scripts/dp_serving.py --scenario native \
            --policy "$policy" --pool-gib "$pool" --stats "$stats"
    done
done
for policy in full sink_recent; do
    .venv/bin/python .p1-work/scripts/run_guard.py --stage DP \
        --name "http-skewed-1.5-$policy" -- \
        @python .p1-work/scripts/dp_serving.py --scenario skewed \
        --policy "$policy" --pool-gib 1.5 --stats "$stats"
done
for scenario in native skewed; do
    for policy in full sink_recent; do
        .venv/bin/python .p1-work/scripts/run_guard.py --stage DP \
            --name "http-$scenario-1.5-$policy-diagnostic" -- \
            @python .p1-work/scripts/dp_serving.py --scenario "$scenario" \
            --policy "$policy" --pool-gib 1.5 --stats "$stats" --repeats 1 --trace
    done
done
.venv/bin/python .p1-work/scripts/run_guard.py --stage DP \
    --name http-native-4-sink_recent-diagnostic -- \
    @python .p1-work/scripts/dp_serving.py --scenario native \
    --policy sink_recent --pool-gib 4 --stats "$stats" --repeats 1 --trace
.venv/bin/python .p1-work/scripts/summarize_dp_serving.py
