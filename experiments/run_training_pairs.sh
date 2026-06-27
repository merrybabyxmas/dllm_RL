#!/bin/bash
# Train all 6 training experiments (3 datasets × diffu_grpo + stage2), 2 at a time.
# Uses direct nohup & so wait $PID correctly blocks on child processes.
set -euo pipefail

cd /home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl
export PYTHONPATH="src:../d1/diffu-grpo"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true

OUT="experiments/outputs/official_9exp"
PY="python"

run2() {
    local DS1=$1 M1=$2 MT1=$3
    local DS2=$4 M2=$5 MT2=$6
    echo ""
    echo "=== [$(date +%H:%M:%S)] Starting ${DS1}/${M1} + ${DS2}/${M2} ==="

    nohup ${PY} -u experiments/run_experiment.py \
        --dataset "${DS1}" --method "${M1}" --max_train_examples "${MT1}" \
        --output_dir "${OUT}" >> "${OUT}/${DS1}_${M1}.log" 2>&1 &
    P1=$!

    nohup ${PY} -u experiments/run_experiment.py \
        --dataset "${DS2}" --method "${M2}" --max_train_examples "${MT2}" \
        --output_dir "${OUT}" >> "${OUT}/${DS2}_${M2}.log" 2>&1 &
    P2=$!

    echo "  PIDs: ${P1}, ${P2}"
    wait "${P1}" 2>/dev/null || true
    echo "  [$(date +%H:%M:%S)] ${DS1}/${M1} done (PID ${P1})"
    wait "${P2}" 2>/dev/null || true
    echo "  [$(date +%H:%M:%S)] ${DS2}/${M2} done (PID ${P2})"
}

echo "============================================================"
echo "Training Pairs Launch"
echo "Started: $(date)"
echo "Memory per experiment: ~35GB → 2 at a time fits in 94GB"
echo "============================================================"

# Pair 1: mbpp (fastest) — ~24h each
rm -f "${OUT}/mbpp_diffu_grpo.log" "${OUT}/mbpp_stage2.log"
run2 mbpp diffu_grpo 100000  mbpp stage2 100000

# Pair 2: spider — ~12h each (10k examples)
rm -f "${OUT}/spider_diffu_grpo.log" "${OUT}/spider_stage2.log"
run2 spider diffu_grpo 10000  spider stage2 10000

# Pair 3: gsm8k (longest) — ~56h each
rm -f "${OUT}/gsm8k_diffu_grpo.log" "${OUT}/gsm8k_stage2.log"
run2 gsm8k diffu_grpo 100000  gsm8k stage2 100000

echo ""
echo "============================================================"
echo "ALL 6 TRAINING EXPERIMENTS DONE"
echo "Finished: $(date)"
echo "============================================================"

# Print results
echo ""
echo "Results:"
for DS in gsm8k mbpp spider; do
    for M in baseline diffu_grpo stage2; do
        R="${OUT}/${DS}/${M}/result.json"
        if [ -f "${R}" ]; then
            S=$(${PY} -c "import json; d=json.load(open('${R}')); print(f'{d[\"mean_score\"]:.4f}')" 2>/dev/null || echo "err")
            echo "  ${DS}/${M}: ${S}"
        fi
    done
done
