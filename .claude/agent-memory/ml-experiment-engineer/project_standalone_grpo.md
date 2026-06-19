---
name: project-standalone-grpo
description: Standalone TRL-free GRPO training script for LLaDA-8B-Instruct, implementing diffu_grpo/stage1/stage2 methods
metadata:
  type: project
---

Standalone PyTorch training loop for LLaDA-8B-Instruct GRPO without TRL.

**Why:** TRL 0.15.1 has hard imports (vllm, llm_blender, TRANSFORMERS_CACHE) that break
the existing DiffuGRPOTrainer-based code. This script bypasses TRL entirely.

**Script:** `/home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl/experiments/train_standalone.py`

**Run scripts:**
- `experiments/run_standalone_diffu_grpo.sh`
- `experiments/run_standalone_stage1.sh`
- `experiments/run_standalone_stage2.sh`

**How to apply:** Use this script for all GRPO training runs. No TRL, no accelerate required.

**Key design notes:**
- All diffusion primitives (generate, forward_process, _get_per_token_logps) are copied
  inline from d1/diffu-grpo/diffu_grpo_trainer.py — no import from d1.
- Data loading uses local GSM8K loader (no d1 import) with same SYSTEM_PROMPT.
- LoRA via peft directly; reference logprobs via `model.disable_adapter()`.
- Stage 2 value head uses detached hidden states; policy loss and value loss are
  on SEPARATE computation graphs — `loss.backward()` only updates policy,
  `value_loss.backward()` only updates value head.
- Stage 2 advantage fallback: if residual std < 1e-6, falls back to group_adv
  (prevents NaN/inf from division by near-zero).

**Smoke test (verified passing, all 3 methods):**
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/train_standalone.py \
  --method diffu_grpo \  # or stage1 or stage2
  --model_path /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct \
  --output_dir experiments/outputs/smoke_test \
  --max_steps 2 --num_generations 2 \
  --diffusion_steps 8 --block_length 8 --max_completion_length 32
```
Expected: zero loss (both completions reward=0, std=0, no gradient) — correct behavior.

**Numerical notes:**
- `compute_responsibility_weights_batch`: vectorized, uses `.clamp(min=1.0)` for masked mean
- forward_process seed: always converts via `int(seed.item())` for Python 3.13 compat
- All metrics detached before `.float()` conversion to avoid autograd warnings
