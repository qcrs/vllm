#!/usr/bin/env bash
set -euo pipefail
export P1_CUDA_VISIBLE_DEVICES=0,1
source .p1-work/scripts/env.sh
export P1_STATS=/data_ssd/qcrs/experiments/p1-v8/M2/20261031-000856-normalized-q-calibration/frequency.pt
for variant in eager-whole eager-token-delay eager-token-fail eager-random eager-tri prefix-eager-token full-whole full-token-delay prefix-full-token; do
    options=()
    case "$variant" in
        eager-whole) options=(--mode eager --path whole) ;;
        eager-token-delay) options=(--mode eager --path token --delay) ;;
        eager-token-fail) options=(--mode eager --path token --fail) ;;
        eager-random) options=(--mode eager --path token --policy random) ;;
        eager-tri) options=(--mode eager --path token --policy triattention) ;;
        prefix-eager-token) options=(--mode eager --path token --prefix) ;;
        full-whole) options=(--mode full --path whole) ;;
        full-token-delay) options=(--mode full --path token --delay) ;;
        prefix-full-token) options=(--mode full --path token --prefix) ;;
    esac
    .venv/bin/python .p1-work/scripts/run_guard.py --stage TP --name "$variant" -- \
        @python .p1-work/scripts/tp_reclamation.py "${options[@]}"
done
