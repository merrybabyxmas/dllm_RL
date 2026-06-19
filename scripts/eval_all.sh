#!/bin/bash
# Evaluate all trained checkpoints on GSM8K test set
set -e

PACKAGE_DIR="/home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl"
RESULTS_DIR="${PACKAGE_DIR}/outputs/eval_results"
mkdir -p "${RESULTS_DIR}"

MODELS=(
  "gsm8k_baseline"
  "gsm8k_stage1"
  "gsm8k_stage2"
)

for MODEL_NAME in "${MODELS[@]}"; do
  CKPT_DIR="${PACKAGE_DIR}/outputs/${MODEL_NAME}"
  if [ -d "${CKPT_DIR}" ]; then
    echo "Evaluating ${MODEL_NAME}..."
    CUDA_VISIBLE_DEVICES=0 python -m cc_rl.evaluate \
      --model_path "${CKPT_DIR}" \
      --dataset gsm8k \
      --split test \
      --output_dir "${RESULTS_DIR}/${MODEL_NAME}" \
      --batch_size 16 \
      --gen_length 256 \
      --steps 64
    echo "Done: ${MODEL_NAME}"
  else
    echo "Skipping ${MODEL_NAME}: checkpoint not found at ${CKPT_DIR}"
  fi
done

echo "All evaluations complete. Results in ${RESULTS_DIR}"
