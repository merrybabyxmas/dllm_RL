#!/bin/bash
# Run 6 training experiments SEQUENTIALLY (1 at a time) to avoid OOM.
# Memory usage: diffu_grpo ~44GB, stage2 ~61GB — cannot safely run 2 training together.
# Estimated total: ~170 hours (7 days)
set -euo pipefail

cd /home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl
export PYTHONPATH="src:../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

OUT="experiments/outputs/official_9exp"
PY="python"

run1() {
    local DS=$1 M=$2 MT=$3
    local LOG="${OUT}/${DS}_${M}.log"
    echo ""
    echo "=== [$(date +%H:%M:%S)] Starting ${DS}/${M} ==="
    rm -f "${LOG}"

    nohup ${PY} -u experiments/run_experiment.py \
        --dataset "${DS}" --method "${M}" --max_train_examples "${MT}" \
        --output_dir "${OUT}" >> "${LOG}" 2>&1 &
    local PID=$!
    echo "  PID: ${PID}"
    wait "${PID}" 2>/dev/null || true
    echo "  [$(date +%H:%M:%S)] ${DS}/${M} DONE"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    sleep 5  # allow GPU memory to clear
}

echo "============================================================"
echo "Sequential Training (1 at a time — avoids OOM)"
echo "Started: $(date)"
echo "Order: mbpp/diffu_grpo → mbpp/stage2 → spider/diffu_grpo"
echo "       → spider/stage2 → gsm8k/diffu_grpo → gsm8k/stage2"
echo "============================================================"

run1 mbpp   diffu_grpo 100000
run1 mbpp   stage2     100000
run1 spider diffu_grpo  10000
run1 spider stage2      10000
run1 gsm8k  diffu_grpo 100000
run1 gsm8k  stage2     100000

echo ""
echo "============================================================"
echo "ALL 6 TRAINING EXPERIMENTS DONE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Final results:"
for DS in gsm8k mbpp spider; do
    for M in baseline diffu_grpo stage2; do
        R="${OUT}/${DS}/${M}/result.json"
        if [ -f "${R}" ]; then
            S=$(${PY} -c "import json; d=json.load(open('${R}')); print(f'{d[\"mean_score\"]:.4f}')" 2>/dev/null || echo "err")
            echo "  ${DS}/${M}: ${S}"
        fi
    done
done
