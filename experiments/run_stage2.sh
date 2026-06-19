#!/bin/bash
# Experiment: Stage 2 State-Value Confidence Credit (3000 steps)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/stage2_value_credit_3000step"
RUN_NAME="stage2_value_credit_gsm8k_3000step"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/../src:${SCRIPT_DIR}/../../d1/diffu-grpo:${PYTHONPATH}"

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=0 python train_experiment.py \
  --method stage2 \
  --config "${SCRIPT_DIR}/configs/common.yaml" \
  --model_path "${MODEL_PATH}" \
  --dataset gsm8k \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --max_steps 3000 \
  --logging_steps 10 \
  --save_steps 500 \
  --credit_alpha 1.0 \
  --credit_eps 1e-6 \
  --credit_clip_min 0.25 \
  --credit_clip_max 4.0 \
  --critic_lr 5e-6 \
  --critic_loss_coef 0.5 \
  --value_hidden_size 1024 \
  --value_mlp_layers 2 \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Stage 2 training complete. Results in ${OUTPUT_DIR}"
