#!/bin/bash
# ===========================================================================
# Stage2 rerun with max_value_states=4 (vs 2 in launch_fast.sh)
# Output dir: experiments/outputs/official_9exp_v4/
# Baseline + diffu_grpo already done; only stage2 × 3 datasets run here.
# ===========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp_v4"
PYTHON="python"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

mkdir -p "${OUT_BASE}"

echo "============================================================"
echo "Stage2 v4 Launcher (max_value_states=4)"
echo "Started: $(date)"
echo "============================================================"

run_stage2() {
    local DATASET="$1"
    local RESULT_FILE="${OUT_BASE}/${DATASET}/stage2/result.json"
    local LOG_FILE="${OUT_BASE}/${DATASET}_stage2.log"
    mkdir -p "${OUT_BASE}/${DATASET}/stage2"

    if [ -f "${RESULT_FILE}" ]; then
        SCORE=$(python3 -c "import json; print(json.load(open('${RESULT_FILE}'))['mean_score'])" 2>/dev/null || echo "?")
        echo "  [SKIP] ${DATASET}/stage2 already done  score=${SCORE}"
        return 0
    fi

    echo ""
    echo "=== [$(date '+%H:%M:%S')] Starting ${DATASET}/stage2 (max_value_states=4) ==="
    ${PYTHON} -u experiments/run_experiment.py \
        --dataset "${DATASET}" \
        --method  stage2       \
        --output_dir "${OUT_BASE}" \
        --max_value_states 4   \
        > "${LOG_FILE}" 2>&1
    echo "  [$(date '+%H:%M:%S')] ${DATASET}/stage2 DONE"
}

run_stage2 mbpp
run_stage2 humaneval
run_stage2 svamp

echo ""
echo "============================================================"
echo "ALL STAGE2 v4 DONE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results (stage2 v4 vs v2):"
V2_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp"
for DS in mbpp humaneval svamp; do
    V4="${OUT_BASE}/${DS}/stage2/result.json"
    V2="${V2_BASE}/${DS}/stage2/result.json"
    S4=$(python3 -c "import json; print(f\"{json.load(open('${V4}'))['mean_score']:.4f}\")" 2>/dev/null || echo "?")
    S2=$(python3 -c "import json; print(f\"{json.load(open('${V2}'))['mean_score']:.4f}\")" 2>/dev/null || echo "?")
    echo "  ${DS}: v4=${S4}  v2=${S2}"
done
echo "============================================================"
