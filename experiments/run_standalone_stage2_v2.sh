#!/bin/bash
# Stage 2 v2: replay buffer + K-step value update + critic quality gate
cd /home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl
export PYTHONPATH="src:../d1/diffu-grpo"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

CUDA_VISIBLE_DEVICES=0 python experiments/train_standalone_v2.py \
  --method stage2 \
  --model_path /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct \
  --dataset gsm8k \
  --output_dir experiments/outputs/stage2_v2_3000step \
  --max_steps 3000 \
  --num_generations 4 \
  --batch_size 1 \
  --learning_rate 1e-6 \
  --temperature 0.9 \
  --diffusion_steps 64 \
  --block_length 32 \
  --max_completion_length 256 \
  --max_prompt_length 256 \
  --beta 0.04 \
  --epsilon 0.2 \
  --lora_r 64 \
  --lora_alpha 64 \
  --credit_alpha 1.0 \
  --critic_lr 5e-6 \
  --value_k_steps 5 \
  --value_replay_size 64 \
  --value_replay_batch 16 \
  --critic_expvar_gate 0.05 \
  --critic_warmup_steps 200 \
  --value_loss_fn huber \
  --logging_steps 10 \
  --save_steps 500 \
  --seed 42 \
  > experiments/outputs/stage2_v2_3000step/train.log 2>&1
echo "done $?"
