#!/usr/bin/env bash
# =============================================================================
# launch_main_experiments.sh
# 4x RTX 4090 parallel experiment launcher for cc_rl main experiments.
#
# Design: GPU pool (0-3). Each experiment runs on 1 GPU.
# Completed experiments (result.json exists) are skipped.
# After each batch of 4, waits for all to finish before starting next batch.
#
# Usage:
#   bash experiments/launch_main_experiments.sh
#   # Override GPU list:
#   GPU_IDS="0,1,2,3" bash experiments/launch_main_experiments.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
NUM_GPUS=4
GPU_IDS="0,1,2,3"            # comma-separated GPU indices
METHODS=("baseline" "delta_v_only")
DATASETS=("mbpp" "humaneval" "svamp" "gsm8k" "countdown" "spider")
GEN_LENGTHS=("128" "256" "512")
MAX_TRAIN=10000               # cap training examples per dataset

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_BASE="${PROJECT_ROOT}/experiments/outputs/main_experiments"
LOG_DIR="${OUT_BASE}/launcher_logs"
PYTHON="${PYTHON:-python3}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

mkdir -p "${LOG_DIR}"

# Parse GPU_IDS into array
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

# ---------------------------------------------------------------------------
# Build experiment list
# ---------------------------------------------------------------------------
declare -a EXPERIMENTS=()
for DS in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for GL in "${GEN_LENGTHS[@]}"; do
            EXPERIMENTS+=("${DS}|${METHOD}|${GL}")
        done
    done
done

TOTAL=${#EXPERIMENTS[@]}
echo "============================================================"
echo "Total experiments: ${TOTAL}"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Gen lengths: ${GEN_LENGTHS[*]}"
echo "Output base: ${OUT_BASE}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# GPU pool management
# ---------------------------------------------------------------------------
declare -a GPU_PIDS=()       # PID running on each GPU slot (0 = free)
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_PIDS+=("0")
done

get_free_gpu() {
    # Returns index into GPU_ARRAY of a free GPU, or -1 if none free
    for ((i=0; i<NUM_GPUS; i++)); do
        pid="${GPU_PIDS[$i]}"
        if [[ "$pid" == "0" ]]; then
            echo "$i"
            return
        fi
        # Check if PID is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            GPU_PIDS[$i]="0"
            echo "$i"
            return
        fi
    done
    echo "-1"
}

wait_for_free_gpu() {
    while true; do
        idx=$(get_free_gpu)
        if [[ "$idx" != "-1" ]]; then
            echo "$idx"
            return
        fi
        sleep 10
    done
}

# ---------------------------------------------------------------------------
# Run one experiment on a specific GPU slot
# ---------------------------------------------------------------------------
run_on_gpu_slot() {
    local SLOT=$1
    local DS=$2
    local METHOD=$3
    local GL=$4

    local GPU_ID="${GPU_ARRAY[$SLOT]}"
    local RESULT_DIR="${OUT_BASE}/${DS}/gl${GL}/${METHOD}"
    local RESULT_FILE="${RESULT_DIR}/result.json"
    local LOG_FILE="${LOG_DIR}/${DS}_gl${GL}_${METHOD}.log"

    # Skip if result already exists
    if [[ -f "${RESULT_FILE}" ]]; then
        echo "[SKIP]  ${DS}/gl${GL}/${METHOD}  (result.json exists)"
        echo "0"   # return fake PID 0 so caller doesn't register it
        return
    fi

    echo "[START] GPU${GPU_ID}  ${DS}/gl${GL}/${METHOD}  -> ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    ${PYTHON} -u "${PROJECT_ROOT}/experiments/run_experiment.py" \
        --dataset "${DS}" \
        --method  "${METHOD}" \
        --gen_length "${GL}" \
        --max_train_examples "${MAX_TRAIN}" \
        --output_dir "${OUT_BASE}" \
        > "${LOG_FILE}" 2>&1 &

    echo "$!"
}

# ---------------------------------------------------------------------------
# Main scheduling loop
# ---------------------------------------------------------------------------
DONE_COUNT=0
SKIP_COUNT=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r DS METHOD GL <<< "${EXP}"

    RESULT_DIR="${OUT_BASE}/${DS}/gl${GL}/${METHOD}"
    RESULT_FILE="${RESULT_DIR}/result.json"

    if [[ -f "${RESULT_FILE}" ]]; then
        echo "[SKIP]  ${DS}/gl${GL}/${METHOD}  (result.json exists)"
        ((SKIP_COUNT++)) || true
        continue
    fi

    # Wait until a GPU is free
    SLOT=$(wait_for_free_gpu)
    PID=$(run_on_gpu_slot "$SLOT" "$DS" "$METHOD" "$GL")

    if [[ "$PID" != "0" ]]; then
        GPU_PIDS[$SLOT]="$PID"
        ((DONE_COUNT++)) || true
        echo "  [${DONE_COUNT}/${TOTAL}]  slot=${SLOT}  pid=${PID}"
    fi
done

# Wait for all remaining jobs
echo ""
echo "All experiments dispatched. Waiting for remaining jobs..."
for pid_val in "${GPU_PIDS[@]}"; do
    if [[ "$pid_val" != "0" ]]; then
        wait "$pid_val" 2>/dev/null || true
    fi
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SUMMARY TABLE"
echo "============================================================"
printf "%-12s %-12s %-8s %-12s %-10s\n" "DATASET" "METHOD" "GL" "SCORE" "STATUS"
printf "%-12s %-12s %-8s %-12s %-10s\n" "-------" "------" "--" "-----" "------"

for DS in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for GL in "${GEN_LENGTHS[@]}"; do
            RESULT_FILE="${OUT_BASE}/${DS}/gl${GL}/${METHOD}/result.json"
            if [[ -f "${RESULT_FILE}" ]]; then
                SCORE=$(python3 -c "import json; d=json.load(open('${RESULT_FILE}')); print(f\"{d['mean_score']:.4f}\")" 2>/dev/null || echo "err")
                printf "%-12s %-12s %-8s %-12s %-10s\n" "${DS}" "${METHOD}" "${GL}" "${SCORE}" "DONE"
            else
                printf "%-12s %-12s %-8s %-12s %-10s\n" "${DS}" "${METHOD}" "${GL}" "---" "MISSING"
            fi
        done
    done
done

echo "============================================================"
echo "Launcher finished."
