#!/bin/bash
# v2 완료 후 diffu_grpo 자동 재시작
LOG_V2="experiments/outputs/stage2_v2_3000step/train.log"
LOG_DIFFU="experiments/outputs/diffu_grpo_3000step/train.log"
cd /home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl

echo "[watcher] Waiting for stage2_v2 to reach step 3000..."
while true; do
    last_step=$(grep -o '"step": [0-9]*' "$LOG_V2" 2>/dev/null | tail -1 | grep -o '[0-9]*')
    if [ -n "$last_step" ] && [ "$last_step" -ge 3000 ]; then
        echo "[watcher] stage2_v2 completed at step $last_step. Starting diffu_grpo..."
        break
    fi
    # also check if process died
    if ! pgrep -f "train_standalone_v2.py" > /dev/null; then
        echo "[watcher] v2 process not found. Checking last step..."
        last_step=$(grep -o '"step": [0-9]*' "$LOG_V2" 2>/dev/null | tail -1 | grep -o '[0-9]*')
        if [ -n "$last_step" ] && [ "$last_step" -ge 2900 ]; then
            echo "[watcher] v2 likely done (step $last_step). Starting diffu_grpo..."
            break
        fi
    fi
    sleep 60
done

export PYTHONPATH="src:../d1/diffu-grpo"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

echo "[watcher] Launching diffu_grpo from scratch (new 3000-step run)..."
mkdir -p experiments/outputs/diffu_grpo_v2_3000step

CUDA_VISIBLE_DEVICES=0 python experiments/train_standalone.py \
  --method diffu_grpo \
  --model_path /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct \
  --dataset gsm8k \
  --output_dir experiments/outputs/diffu_grpo_v2_3000step \
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
  --logging_steps 10 \
  --save_steps 500 \
  --seed 42 \
  > experiments/outputs/diffu_grpo_v2_3000step/train.log 2>&1

echo "[watcher] diffu_grpo done, exit=$?"
