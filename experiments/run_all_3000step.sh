#!/bin/bash
# Run all 3 experiments sequentially (base eval → diffu-grpo → stage2)
# Usage: bash run_all_3000step.sh [model_path]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${1:-/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct}"

echo "=================================================="
echo "Confidence-Credit DLLM RL - 3000 Step Experiments"
echo "Model: ${MODEL_PATH}"
echo "=================================================="

# Verify model is available
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "ERROR: Model not found at ${MODEL_PATH}"
  echo "Please download: GSAI-ML/LLaDA-8B-Instruct"
  exit 1
fi

# Verify model weights
TOTAL_SIZE=$(du -sm "${MODEL_PATH}"/*.safetensors 2>/dev/null | awk '{sum+=$1} END {print sum}')
if [ "${TOTAL_SIZE:-0}" -lt 15000 ]; then
  echo "WARNING: Model weights incomplete (${TOTAL_SIZE}MB < 15000MB expected)"
  echo "The model may not be fully downloaded yet."
  read -p "Continue anyway? [y/N]: " cont
  [[ "$cont" == "y" || "$cont" == "Y" ]] || exit 1
fi

echo ""
echo "=== EXPERIMENT 1: Base Model Evaluation ==="
bash "${SCRIPT_DIR}/run_base_eval.sh"
echo ""

echo "=== EXPERIMENT 2: Diffu-GRPO (3000 steps) ==="
bash "${SCRIPT_DIR}/run_diffu_grpo.sh"
echo ""

echo "=== EXPERIMENT 3: Stage 2 Value Credit (3000 steps) ==="
bash "${SCRIPT_DIR}/run_stage2.sh"
echo ""

echo "=================================================="
echo "All experiments complete!"
echo "Results in: ${SCRIPT_DIR}/outputs/"
echo "=================================================="
ls -lh "${SCRIPT_DIR}/outputs/" 2>/dev/null
