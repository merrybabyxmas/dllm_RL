#!/bin/bash
# ===========================================================================
# Launch 9 official experiments: 3 datasets × 3 methods
# Run EXACTLY 2 at a time using proper foreground-wait pattern.
#
# Dataset sizes (training, 1 epoch):
#   GSM8K : 7,473 examples  → ~56h per training run
#   MBPP  :   500 examples  → ~3.8h per training run
#   Spider: 10,000 examples (capped from 78k) → ~75h per training run
#
# Execution order (fast experiments first):
#   Pair 1: gsm8k/baseline   + mbpp/baseline      (~2h   + ~0.5h)
#   Pair 2: spider/baseline  + mbpp/diffu_grpo     (~1h   + ~3.8h)
#   Pair 3: mbpp/stage2      + spider/diffu_grpo   (~3.8h + ~75h)
#   Pair 4: gsm8k/diffu_grpo + spider/stage2       (~56h  + ~75h)
#   Pair 5: gsm8k/stage2     (single)              (~56h)
#
# NOTE: Uses foreground `wait $PID` on directly-backgrounded processes
#       (not subshell wrapping), so wait actually blocks.
# ===========================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="${PROJECT_ROOT}/experiments/outputs/official_9exp"
PYTHON="python"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

mkdir -p "${OUT_BASE}"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "9-Experiment Launch  (2 at a time)"
echo "Project: ${PROJECT_ROOT}"
echo "Output:  ${OUT_BASE}"
echo "Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: launch one experiment directly in parent shell (no subshell)
# Usage: launch_bg DATASET METHOD MAX_TRAIN
# Sets global variable LAST_PID
# ---------------------------------------------------------------------------
launch_bg() {
    local DS=$1 M=$2 MT=$3
    local LOG="${OUT_BASE}/${DS}_${M}.log"
    echo "[$(date +%H:%M:%S)] Launching ${DS}/${M}  log→${LOG}"

    nohup "${PYTHON}" -u \
        "${PROJECT_ROOT}/experiments/run_experiment.py" \
        --dataset "${DS}" --method "${M}" \
        --max_train_examples "${MT}" \
        --output_dir "${OUT_BASE}" \
        >> "${LOG}" 2>&1 &

    LAST_PID=$!
    echo "  PID: ${LAST_PID}"
}

wait_pid() {
    local PID=$1
    echo "[$(date +%H:%M:%S)] Waiting for PID ${PID} ..."
    wait "${PID}" 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] PID ${PID} finished"
}

# --------------------------------------------------------------------------
# Pair 1: gsm8k/baseline + mbpp/baseline  (each ~18GB after no_grad fix)
# --------------------------------------------------------------------------
echo; echo "=== Pair 1: gsm8k/baseline + mbpp/baseline ==="
launch_bg gsm8k baseline 100000; P1=${LAST_PID}
launch_bg mbpp  baseline 100000; P2=${LAST_PID}
wait_pid ${P1}; wait_pid ${P2}

# --------------------------------------------------------------------------
# Pair 2: spider/baseline + mbpp/diffu_grpo
# --------------------------------------------------------------------------
echo; echo "=== Pair 2: spider/baseline + mbpp/diffu_grpo ==="
launch_bg spider baseline   100000; P3=${LAST_PID}
launch_bg mbpp   diffu_grpo 100000; P4=${LAST_PID}
wait_pid ${P3}; wait_pid ${P4}

# --------------------------------------------------------------------------
# Pair 3: mbpp/stage2 + spider/diffu_grpo
# --------------------------------------------------------------------------
echo; echo "=== Pair 3: mbpp/stage2 + spider/diffu_grpo ==="
launch_bg mbpp   stage2      100000; P5=${LAST_PID}
launch_bg spider diffu_grpo   10000; P6=${LAST_PID}
wait_pid ${P5}; wait_pid ${P6}

# --------------------------------------------------------------------------
# Pair 4: spider/stage2 + gsm8k/diffu_grpo
# --------------------------------------------------------------------------
echo; echo "=== Pair 4: spider/stage2 + gsm8k/diffu_grpo ==="
launch_bg spider stage2      10000; P7=${LAST_PID}
launch_bg gsm8k  diffu_grpo 100000; P8=${LAST_PID}
wait_pid ${P7}; wait_pid ${P8}

# --------------------------------------------------------------------------
# Pair 5: gsm8k/stage2 (single — longest run)
# --------------------------------------------------------------------------
echo; echo "=== Pair 5: gsm8k/stage2 ==="
launch_bg gsm8k stage2 100000; P9=${LAST_PID}
wait_pid ${P9}

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
echo
echo "============================================================"
echo "ALL 9 EXPERIMENTS DONE"
echo "Finished: $(date)"
echo "============================================================"
echo
echo "Results:"
for DS in gsm8k mbpp spider; do
    for M in baseline diffu_grpo stage2; do
        RFILE="${OUT_BASE}/${DS}/${M}/result.json"
        if [ -f "${RFILE}" ]; then
            SCORE=$(python -c "import json; d=json.load(open('${RFILE}')); print(f'{d[\"mean_score\"]:.4f}')" 2>/dev/null || echo "parse_err")
            echo "  ${DS}/${M}: ${SCORE}"
        else
            echo "  ${DS}/${M}: MISSING"
        fi
    done
done
