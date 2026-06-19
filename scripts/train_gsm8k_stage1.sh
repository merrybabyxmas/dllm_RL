#!/bin/bash
# Stage 1: Confidence-Weighted GRPO on GSM8K
set -e

PACKAGE_DIR="/home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl"
D1_DIR="/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"

cd "${D1_DIR}"

CUDA_VISIBLE_DEVICES=0 python -m cc_rl.train \
  --config "${PACKAGE_DIR}/configs/stage1_cw_grpo.yaml" \
  --model_path MDLM-hf/LLaDA-8B-Instruct \
  --dataset gsm8k \
  --output_dir "${PACKAGE_DIR}/outputs/gsm8k_stage1" \
  --max_steps 3000
