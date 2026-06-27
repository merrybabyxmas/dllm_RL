#!/bin/bash
# ===========================================================================
# Ablation launcher: cw_grpo + delta_v_only × 3 datasets
# Output dir: experiments/outputs/official_9exp_ablation/
#
# Ablation table (for paper):
#   baseline     : no training (already in official_9exp)
#   diffu_grpo   : GRPO uniform reward (already in official_9exp)
#   cw_grpo      : confidence weighting only, no value head       ← NEW
#   delta_v_only : delta-V credit only, no confidence weighting   ← NEW
#   stage2       : delta-V + confidence weighting (already in v4) ← reference
# ===========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp_ablation"
PYTHON="python"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

mkdir -p "${OUT_BASE}"

echo "============================================================"
echo "Ablation Launcher (cw_grpo + delta_v_only)"
echo "Started: $(date)"
echo "============================================================"

run_exp() {
    local DATASET="$1"
    local METHOD="$2"
    local RESULT_FILE="${OUT_BASE}/${DATASET}/${METHOD}/result.json"
    local LOG_FILE="${OUT_BASE}/${DATASET}_${METHOD}.log"
    mkdir -p "${OUT_BASE}/${DATASET}/${METHOD}"

    if [ -f "${RESULT_FILE}" ]; then
        SCORE=$(python3 -c "import json; print(json.load(open('${RESULT_FILE}'))['mean_score'])" 2>/dev/null || echo "?")
        echo "  [SKIP] ${DATASET}/${METHOD} already done  score=${SCORE}"
        return 0
    fi

    echo ""
    echo "=== [$(date '+%H:%M:%S')] Starting ${DATASET}/${METHOD} ==="
    ${PYTHON} -u experiments/run_experiment.py \
        --dataset "${DATASET}" \
        --method  "${METHOD}"  \
        --output_dir "${OUT_BASE}" \
        --max_value_states 4   \
        > "${LOG_FILE}" 2>&1
    echo "  [$(date '+%H:%M:%S')] ${DATASET}/${METHOD} DONE"
}

echo ""
echo "=== Phase 1: CW-GRPO (confidence weighting only) ==="
run_exp mbpp      cw_grpo
run_exp humaneval cw_grpo
run_exp svamp     cw_grpo

echo ""
echo "=== Phase 2: Delta-V only (no confidence weighting) ==="
run_exp mbpp      delta_v_only
run_exp humaneval delta_v_only
run_exp svamp     delta_v_only

echo ""
echo "============================================================"
echo "ALL ABLATION DONE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Full ablation table:"
V2="${PROJECT_ROOT}/experiments/outputs/official_9exp"
V4="${PROJECT_ROOT}/experiments/outputs/official_9exp_v4"
AB="${OUT_BASE}"

for DS in mbpp humaneval svamp; do
    echo "  --- ${DS} ---"
    for METHOD_DIR in \
        "baseline:${V2}" \
        "diffu_grpo:${V2}" \
        "cw_grpo:${AB}" \
        "delta_v_only:${AB}" \
        "stage2:${V4}"; do
        M="${METHOD_DIR%%:*}"
        BASE="${METHOD_DIR##*:}"
        RFILE="${BASE}/${DS}/${M}/result.json"
        if [ -f "${RFILE}" ]; then
            S=$(python3 -c "import json; d=json.load(open('${RFILE}')); print(f'{d[\"mean_score\"]:.4f}')" 2>/dev/null || echo "?")
            echo "    ${M}: ${S}"
        else
            echo "    ${M}: NOT FOUND"
        fi
    done
done
echo "============================================================"
