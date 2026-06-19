#!/usr/bin/env bash
# Run standalone DiffuGRPO (baseline, no confidence weighting)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}/src"
unset HF_TOKEN || true
unset HUGGING_FACE_HUB_TOKEN || true

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/train_standalone.py" \
    --method diffu_grpo \
    --model_path /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir "${SCRIPT_DIR}/outputs/diffu_grpo_3000step" \
    --max_steps 3000 \
    --num_generations 4 \
    --batch_size 1 \
    --learning_rate 1e-6 \
    --diffusion_steps 64 \
    --block_length 32 \
    --max_completion_length 256 \
    --max_prompt_length 256 \
    --beta 0.04 \
    --epsilon 0.2 \
    --lora_r 64 \
    --lora_alpha 64 \
    --logging_steps 10 \
    --save_steps 500 \
    --seed 42 \
    2>&1 | tee "${SCRIPT_DIR}/outputs/diffu_grpo_3000step/stdout.log"
