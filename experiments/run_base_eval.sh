#!/bin/bash
# Experiment: Evaluate base LLaDA-8B-Instruct (no training)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/base_eval"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/../src:${SCRIPT_DIR}/../../d1/diffu-grpo:${PYTHONPATH}"

echo "Evaluating base model: ${MODEL_PATH}"
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/eval_base.py" \
  --model_path "${MODEL_PATH}" \
  --n_examples 256 \
  --gen_length 256 \
  --diffusion_steps 64 \
  --output "${OUTPUT_DIR}/gsm8k_results.json" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "Base evaluation complete. Results in ${OUTPUT_DIR}/gsm8k_results.json"
