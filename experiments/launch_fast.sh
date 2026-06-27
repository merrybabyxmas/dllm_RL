#!/bin/bash
# ===========================================================================
# Fast experiment launcher: mbpp + humaneval + svamp  × 3 methods
# All datasets complete in 1-4h each → full sweep in ~1 day sequential.
#
# Dataset sizes (1 epoch):
#   MBPP     :   500 examples → ~3-4h per training run
#   HumanEval:   164 examples → ~1-2h per training run
#   SVAMP    :   800 examples → ~3-4h per training run
#
# Stage2 OOM fixes applied:
#   - value_hidden_size=256 (was 1024 → 4x smaller AdamW states)
#   - max_value_states=2 (subsample V evaluations: 2 passes vs 8)
#   - PYTORCH_ALLOC_CONF=expandable_segments:True (set in run_experiment.py)
#
# Usage:
#   bash experiments/launch_fast.sh
#   bash experiments/launch_fast.sh 2>&1 | tee experiments/outputs/fast_launch.log
# ===========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp"
PYTHON="python"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

echo "============================================================"
echo "Fast Experiment Launcher (mbpp + humaneval + svamp)"
echo "Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: run one experiment, write stdout to log, block until done
# ---------------------------------------------------------------------------
run_exp() {
    local DATASET="$1"
    local METHOD="$2"
    local EXTRA="${3:-}"

    local RESULT_FILE="${OUT_BASE}/${DATASET}/${METHOD}/result.json"
    local LOG_FILE="${OUT_BASE}/${DATASET}_${METHOD}.log"
    mkdir -p "${OUT_BASE}/${DATASET}/${METHOD}"

    # Skip if already completed
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
        --max_value_states 2 \
        ${EXTRA} \
        > "${LOG_FILE}" 2>&1
    echo "  [$(date '+%H:%M:%S')] ${DATASET}/${METHOD} DONE"
}

# ---------------------------------------------------------------------------
# 1. Baselines (fast, no training)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: Baselines ==="
run_exp mbpp      baseline
run_exp humaneval baseline
run_exp svamp     baseline

# ---------------------------------------------------------------------------
# 2. Diffu-GRPO training
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: Diffu-GRPO training ==="
run_exp mbpp      diffu_grpo
run_exp humaneval diffu_grpo
run_exp svamp     diffu_grpo

# ---------------------------------------------------------------------------
# 3. Stage 2 training (delta-V credit assignment)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 3: Stage 2 training ==="
run_exp mbpp      stage2
run_exp humaneval stage2
run_exp svamp     stage2

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "ALL 9 EXPERIMENTS DONE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results:"
for DS in mbpp humaneval svamp; do
    for M in baseline diffu_grpo stage2; do
        RFILE="${OUT_BASE}/${DS}/${M}/result.json"
        if [ -f "${RFILE}" ]; then
            SCORE=$(python3 -c "import json; d=json.load(open('${RFILE}')); print(f'{d[\"mean_score\"]:.4f}')" 2>/dev/null || echo "?")
            echo "  ${DS}/${M}: ${SCORE}"
        else
            echo "  ${DS}/${M}: NOT FOUND"
        fi
    done
done
echo "============================================================"
