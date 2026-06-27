#!/bin/bash
# ablation 완료 후 v4 humaneval/svamp 재실행
# OOM으로 죽은 v4 launcher의 나머지 실험을 ablation 완료 뒤에 순차 실행

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp_v4"
AB_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp_ablation"
PYTHON="python"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

echo "[v4-remaining] Waiting for ablation to finish..."
# svamp/delta_v_only가 마지막 ablation 실험
while [ ! -f "${AB_BASE}/svamp/delta_v_only/result.json" ]; do
    sleep 300
done
echo "[v4-remaining] Ablation done. Starting v4 remaining at $(date)"

run_v4() {
    local DATASET="$1"
    local RESULT_FILE="${OUT_BASE}/${DATASET}/stage2/result.json"
    local LOG_FILE="${OUT_BASE}/${DATASET}_stage2.log"
    mkdir -p "${OUT_BASE}/${DATASET}/stage2"

    if [ -f "${RESULT_FILE}" ]; then
        SCORE=$(python3 -c "import json; print(json.load(open('${RESULT_FILE}'))['mean_score'])" 2>/dev/null || echo "?")
        echo "  [SKIP] ${DATASET}/stage2-v4 already done  score=${SCORE}"
        return 0
    fi

    echo "=== [$(date '+%H:%M:%S')] Starting ${DATASET}/stage2 v4 ==="
    ${PYTHON} -u experiments/run_experiment.py \
        --dataset "${DATASET}" \
        --method  stage2       \
        --output_dir "${OUT_BASE}" \
        --max_value_states 4   \
        > "${LOG_FILE}" 2>&1
    echo "  [$(date '+%H:%M:%S')] ${DATASET}/stage2 v4 DONE"
}

run_v4 humaneval
run_v4 svamp

echo "=== v4 remaining ALL DONE at $(date) ==="
