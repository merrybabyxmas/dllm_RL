#!/usr/bin/env bash
# Run standalone Stage 2: Stage 1 + Value Head (delta-V advantages)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PROJECT_ROOT}/src"
unset HF_TOKEN || true
unset HUGGING_FACE_HUB_TOKEN || true

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/train_standalone.py" \
    --method stage2 \
    --model_path /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir "${SCRIPT_DIR}/outputs/stage2_3000step" \
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
    --credit_alpha 1.0 \
    --credit_eps 1e-6 \
    --credit_clip_min 0.25 \
    --credit_clip_max 4.0 \
    --value_hidden_size 1024 \
    --critic_lr 5e-6 \
    --lora_r 64 \
    --lora_alpha 64 \
    --logging_steps 10 \
    --save_steps 500 \
    --seed 42 \
    2>&1 | tee "${SCRIPT_DIR}/outputs/stage2_3000step/stdout.log"
