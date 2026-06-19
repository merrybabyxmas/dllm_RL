#!/bin/bash
# Experiment: Official diffu-GRPO baseline (3000 steps)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/diffu_grpo_3000step"
RUN_NAME="diffu_grpo_gsm8k_3000step"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/../src:${SCRIPT_DIR}/../../d1/diffu-grpo:${PYTHONPATH}"

cd "${SCRIPT_DIR}"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

CUDA_VISIBLE_DEVICES=0 python train_experiment.py \
  --method diffu_grpo \
  --config "${SCRIPT_DIR}/configs/common.yaml" \
  --model_path "${MODEL_PATH}" \
  --dataset gsm8k \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --max_steps 3000 \
  --logging_steps 10 \
  --save_steps 500 \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "diffu-GRPO training complete. Results in ${OUTPUT_DIR}"
