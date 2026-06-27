#!/usr/bin/env python3
"""
Stage 2 Tier 3 Short Policy Training Diagnostic (300 steps).

Runs 300 training steps sequentially for three methods:
  Method A: diffu_grpo   (standard GRPO)
  Method B: stage1       (confidence-weighted advantages)
  Method C: stage2       (true block delta-V + value head replay)

Each method trains on the SAME 64 GSM8K examples (same seed order),
is evaluated every 20 steps on 32 held-out examples, and the
comparison summary is printed at the end.

Tier 3 pass criteria (Stage 2 method):
  eval_reward not worse than Diffu-GRPO: PASS/FAIL
  ema_expvar >= 0.05 at end: PASS/FAIL
  stage2_adv active in last 50 steps: PASS/FAIL

Model: LLaDA-8B-Instruct + LoRA (r=8, alpha=16)
Model path: /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct

Usage:
  python experiments/run_tier3.py
  python experiments/run_tier3.py --seed 42 --output_dir experiments/outputs/tier3
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STANDALONE_V2 = os.path.join(_PROJECT_ROOT, "experiments", "train_standalone_v2.py")

# Add src to path for cc_rl imports (not needed here but kept for consistency)
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# ---------------------------------------------------------------------------
# Import from train_standalone_v2 (safe because of if __name__ == '__main__' guard)
# ---------------------------------------------------------------------------
import importlib.util

_spec = importlib.util.spec_from_file_location("train_standalone_v2", _STANDALONE_V2)
_tsv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tsv2)

# Pull in the functions and classes we need
seed_everything                      = _tsv2.seed_everything
get_gsm8k_questions                  = _tsv2.get_gsm8k_questions
_extract_xml_answer                  = _tsv2._extract_xml_answer
correctness_reward_func              = _tsv2.correctness_reward_func
int_reward_func                      = _tsv2.int_reward_func
soft_format_reward_func              = _tsv2.soft_format_reward_func
generate                             = _tsv2.generate
generate_with_confidence             = _tsv2.generate_with_confidence
tokenise_prompt                      = _tsv2.tokenise_prompt
rollout_and_compute_advantages       = _tsv2.rollout_and_compute_advantages
compute_policy_loss                  = _tsv2.compute_policy_loss
ValueHead                            = _tsv2.ValueHead
ValueReplayBuffer                    = _tsv2.ValueReplayBuffer
SYSTEM_PROMPT                        = _tsv2.SYSTEM_PROMPT

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Hyper-parameters (fixed per spec)
# ---------------------------------------------------------------------------

N_TRAIN          = 64
N_EVAL           = 32
MAX_STEPS        = 300
EVAL_EVERY       = 30
NUM_GENERATIONS  = 4
LORA_R           = 8
LORA_ALPHA       = 16
TEMPERATURE      = 0.9
DIFFUSION_STEPS  = 64
BLOCK_LENGTH     = 32
MAX_COMPLETION_LENGTH = 256
MAX_PROMPT_LENGTH     = 256
LEARNING_RATE    = 1e-6
CRITIC_LR        = 5e-6
VALUE_K_STEPS    = 5
VALUE_REPLAY_SIZE     = 32
VALUE_REPLAY_BATCH    = 16
CRITIC_WARMUP_STEPS   = 30
CRITIC_EXPVAR_GATE    = 0.05

MODEL_PATH = "/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct"
MASK_ID    = 126336


# ---------------------------------------------------------------------------
# Config object (mirrors the argparse namespace that train_standalone_v2 uses)
# ---------------------------------------------------------------------------

class MethodCfg:
    """Flat config object compatible with train_standalone_v2 function signatures."""

    def __init__(self, method: str, seed: int) -> None:
        self.method                 = method
        self.seed                   = seed

        # Data
        self.dataset                = "gsm8k"

        # Diffusion generation
        self.diffusion_steps        = DIFFUSION_STEPS
        self.block_length           = BLOCK_LENGTH
        self.max_completion_length  = MAX_COMPLETION_LENGTH
        self.max_prompt_length      = MAX_PROMPT_LENGTH
        self.temperature            = TEMPERATURE
        self.cfg_scale              = 0.0
        self.remasking              = "low_confidence"
        self.mask_id                = MASK_ID
        self.p_mask_prompt          = 0.3
        self.num_generations        = NUM_GENERATIONS

        # PPO / GRPO
        self.beta                   = 0.04
        self.epsilon                = 0.2

        # Stage 1 credit
        self.credit_alpha           = 1.0
        self.credit_eps             = 1e-6
        self.credit_clip_min        = 0.25
        self.credit_clip_max        = 4.0

        # Stage 2 value head
        self.value_hidden_size      = 1024
        self.critic_lr              = CRITIC_LR
        self.value_k_steps          = VALUE_K_STEPS
        self.value_replay_size      = VALUE_REPLAY_SIZE
        self.value_replay_batch     = VALUE_REPLAY_BATCH
        self.critic_expvar_gate     = CRITIC_EXPVAR_GATE
        self.critic_warmup_steps    = CRITIC_WARMUP_STEPS
        self.value_loss_fn          = "huber"

        # LoRA
        self.lora_r                 = LORA_R
        self.lora_alpha             = LORA_ALPHA

        # Optimizer
        self.learning_rate          = LEARNING_RATE
        self.max_grad_norm          = 1.0


# ---------------------------------------------------------------------------
# Evaluation: generate completions on eval set, compute mean reward
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer,
    eval_examples: List[dict],
    cfg: MethodCfg,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run greedy generation on eval_examples, compute mean + std reward.

    Uses temperature=0.0 (greedy) for deterministic eval.
    Returns (mean_reward, std_reward).
    """
    model.eval()
    all_rewards: List[float] = []

    for example in eval_examples:
        prompt_ids_1d = tokenise_prompt(tokenizer, example["prompt"], cfg.max_prompt_length)
        prompt_ids_1d = prompt_ids_1d.to(device)
        prompt_ids    = prompt_ids_1d.unsqueeze(0)  # [1, prompt_len]

        full_ids = generate(
            model=model,
            prompt=prompt_ids,
            steps=cfg.diffusion_steps,
            gen_length=cfg.max_completion_length,
            block_length=cfg.block_length,
            temperature=0.0,   # greedy eval
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=cfg.mask_id,
        )  # [1, prompt_len + completion_len]

        prompt_length  = prompt_ids.shape[1]
        completion_ids = full_ids[:, prompt_length:]
        completion_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True)

        # Correctness reward
        extracted = _extract_xml_answer(completion_text)
        correct = (extracted == example.get("answer", ""))
        # Also add int_reward + soft_format_reward for consistency with training
        reward = 2.0 if correct else 0.0
        reward += 0.5 if extracted.isdigit() else 0.0

        import re
        fmt_pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        if re.search(fmt_pat, completion_text, re.DOTALL):
            reward += 0.5

        all_rewards.append(reward)

    rewards_arr = np.array(all_rewards, dtype=np.float32)
    return float(rewards_arr.mean()), float(rewards_arr.std())


# ---------------------------------------------------------------------------
# LoRA re-initialisation between methods
# ---------------------------------------------------------------------------

def reset_lora_weights(model: nn.Module) -> None:
    """
    Re-initialize all LoRA A and B weight matrices to their default init.

    LoRA A is init with kaiming_uniform, B is zero-init (so the adapter
    starts as identity). This resets the policy to the pre-trained base
    without reloading the full model from disk.
    """
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for key in module.lora_A:
                nn.init.kaiming_uniform_(module.lora_A[key].weight, a=math.sqrt(5))
            for key in module.lora_B:
                nn.init.zeros_(module.lora_B[key].weight)


# ---------------------------------------------------------------------------
# Single-method training runner
# ---------------------------------------------------------------------------

def run_method(
    method: str,
    model: nn.Module,
    tokenizer,
    train_examples: List[dict],
    eval_examples: List[dict],
    cfg: MethodCfg,
    device: torch.device,
    seed: int,
) -> Dict:
    """
    Run MAX_STEPS training steps for one method.

    Resets LoRA weights at the start so all methods start from the same base.
    Evaluates every EVAL_EVERY steps on the held-out eval set.

    Returns a result dict with:
      final_eval_reward    : float
      eval_history         : List[(step, eval_reward)]
      grad_steps_nonzero   : int
      ema_expvar_final     : float  (0.0 for non-stage2)
      stage2_adv_count     : int    (steps using stage2 adv in last 50)
      training_rewards     : List[float]  (mean reward per step)
    """
    # Reset LoRA weights → fresh start for this method
    reset_lora_weights(model)

    # Policy optimizer — rebuild to reset momentum
    policy_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(policy_params, lr=cfg.learning_rate, weight_decay=0.0)

    # Value head for stage2
    value_head:      Optional[ValueHead]         = None
    value_optimizer: Optional[torch.optim.AdamW] = None
    replay_buffer:   Optional[ValueReplayBuffer] = None
    ema_expvar: float = 0.0

    if method == "stage2":
        hidden_size = model.config.hidden_size
        value_head  = ValueHead(
            hidden_size=hidden_size,
            mlp_hidden_size=cfg.value_hidden_size,
            n_layers=2,
        ).to(device)
        value_optimizer = torch.optim.AdamW(
            value_head.parameters(), lr=cfg.critic_lr, weight_decay=0.0
        )
        replay_buffer = ValueReplayBuffer(capacity=cfg.value_replay_size)

    # Deterministic example order (same seed for all methods)
    rng = random.Random(seed)
    shuffled_train = list(range(N_TRAIN))
    rng.shuffle(shuffled_train)

    eval_history:       List[Tuple[int, float]] = []
    grad_steps_nonzero: int  = 0
    stage2_adv_count:   int  = 0       # how many steps in LAST 20 used stage2 adv
    last20_start        = MAX_STEPS - 50 + 1
    training_rewards:   List[float] = []
    replay_val_loss_avg = 0.0

    method_label = {"diffu_grpo": "A: Diffu-GRPO",
                    "stage1":     "B: Stage 1 (confidence-weighted)",
                    "stage2":     "C: Stage 2 (true block delta-V)"}[method]

    print(f"\n[Method {method_label}]")

    for step in range(1, MAX_STEPS + 1):
        # Sample example (cycle through shuffled_train)
        idx      = shuffled_train[(step - 1) % N_TRAIN]
        example  = train_examples[idx]
        batch    = [example]

        # ----- Rollout --------------------------------------------------------
        try:
            rollout = rollout_and_compute_advantages(
                model=model,
                tokenizer=tokenizer,
                batch_examples=batch,
                cfg=cfg,
                device=device,
                value_head=value_head,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step:3d}] OOM during rollout, skipping.")
            torch.cuda.empty_cache()
            continue

        mean_reward = float(rollout["rewards"].mean())
        training_rewards.append(mean_reward)

        # ----- Policy loss + backward ----------------------------------------
        model.train()
        optimizer.zero_grad()

        try:
            loss, metrics = compute_policy_loss(
                model=model,
                rollout=rollout,
                cfg=cfg,
                value_head=value_head,
                ema_expvar=ema_expvar,
                step=step,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  [step {step:3d}] OOM during loss computation, skipping.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

        loss.backward()

        # Check gradient norm (proxy for non-zero gradient steps)
        total_gnorm = 0.0
        for p in policy_params:
            if p.grad is not None:
                total_gnorm += p.grad.data.norm(2).item() ** 2
        total_gnorm = math.sqrt(total_gnorm)
        if total_gnorm > 1e-8:
            grad_steps_nonzero += 1

        torch.nn.utils.clip_grad_norm_(policy_params, cfg.max_grad_norm)
        optimizer.step()

        # ----- Value head replay (stage2 only) --------------------------------
        replay_val_loss = 0.0
        if value_head is not None and replay_buffer is not None and value_optimizer is not None:
            pooled_h = rollout.get("pooled_hidden")
            if pooled_h is not None:
                rwd_cpu = rollout["rewards"].detach().cpu()
                for g in range(pooled_h.shape[0]):
                    replay_buffer.push(pooled_h[g], float(rwd_cpu[g]))

            if len(replay_buffer) >= cfg.value_replay_batch:
                vlosses = []
                for _ in range(cfg.value_k_steps):
                    h_b, r_b = replay_buffer.sample(cfg.value_replay_batch, device)
                    v_pred_r = value_head.forward_pooled(h_b)
                    if cfg.value_loss_fn == "huber":
                        v_loss_r = F.huber_loss(v_pred_r, r_b.float(), delta=1.0)
                    else:
                        v_loss_r = F.mse_loss(v_pred_r, r_b.float())
                    value_optimizer.zero_grad()
                    v_loss_r.backward()
                    torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
                    value_optimizer.step()
                    vlosses.append(v_loss_r.item())
                replay_val_loss = sum(vlosses) / len(vlosses)
                replay_val_loss_avg = 0.9 * replay_val_loss_avg + 0.1 * replay_val_loss

            # Update EMA explained variance
            ev = metrics.get("explained_var", float("nan"))
            if not math.isnan(ev):
                ema_expvar = 0.95 * ema_expvar + 0.05 * ev

        # Track stage2 adv usage in last 20 steps
        if step >= last20_start and metrics.get("using_stage2_adv", False):
            stage2_adv_count += 1

        # ----- Eval every EVAL_EVERY steps ------------------------------------
        if step % EVAL_EVERY == 0:
            eval_reward, eval_std = evaluate(model, tokenizer, eval_examples, cfg, device)
            eval_history.append((step, eval_reward))

            # Print line
            if method == "stage2":
                print(
                    f"  step {step:3d}: reward={mean_reward:.3f}  "
                    f"eval_reward={eval_reward:.3f}  "
                    f"ema_expvar={ema_expvar:.3f}  "
                    f"adv_ON={int(metrics.get('using_stage2_adv', False) * 100)}%"
                )
            else:
                print(
                    f"  step {step:3d}: reward={mean_reward:.3f}  "
                    f"eval_reward={eval_reward:.3f}"
                )

        torch.cuda.empty_cache()

    # Final eval if not already at step MAX_STEPS
    if not eval_history or eval_history[-1][0] != MAX_STEPS:
        eval_reward, _ = evaluate(model, tokenizer, eval_examples, cfg, device)
        eval_history.append((MAX_STEPS, eval_reward))

    final_eval_reward = eval_history[-1][1]

    return {
        "method":              method,
        "final_eval_reward":   final_eval_reward,
        "eval_history":        eval_history,
        "grad_steps_nonzero":  grad_steps_nonzero,
        "ema_expvar_final":    ema_expvar,
        "stage2_adv_count":    stage2_adv_count,
        "training_rewards":    training_rewards,
        "replay_val_loss_avg": replay_val_loss_avg,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 Tier 3 Short Policy Training (100 steps, 3 methods)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROJECT_ROOT, "experiments", "outputs", "tier3"))
    p.add_argument("--model_path", type=str, default=MODEL_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print("Stage 2 Tier 3 Short Policy Training (100 steps)")
    print("=" * 50)
    print(f"Config: {N_TRAIN} train / {N_EVAL} eval / {MAX_STEPS} steps / "
          f"{NUM_GENERATIONS} rollouts")
    print(f"Model: LLaDA-8B-Instruct + LoRA(r={LORA_R})")
    print(f"Device: {device}")
    print()

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Load model (once, shared across methods via LoRA weight reset)
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    base_model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.config.use_cache = False
    print(f"Base model loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Apply LoRA (once — re-init weights between methods)
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    print()

    # ------------------------------------------------------------------
    # Load GSM8K and split into train / eval
    # ------------------------------------------------------------------
    print("Loading GSM8K dataset...")
    full_dataset = get_gsm8k_questions("train")
    # Deterministic split using seed
    rng_split = random.Random(args.seed)
    all_indices = list(range(len(full_dataset)))
    rng_split.shuffle(all_indices)

    train_indices = all_indices[:N_TRAIN]
    eval_indices  = all_indices[N_TRAIN:N_TRAIN + N_EVAL]

    train_examples = [full_dataset[i] for i in train_indices]
    eval_examples  = [full_dataset[i] for i in eval_indices]

    print(f"Split: {len(train_examples)} train, {len(eval_examples)} eval")
    print()

    # ------------------------------------------------------------------
    # Run all 3 methods sequentially
    # ------------------------------------------------------------------
    methods  = ["diffu_grpo", "stage1", "stage2"]
    results: Dict[str, dict] = {}

    for method in methods:
        cfg = MethodCfg(method=method, seed=args.seed)
        result = run_method(
            method=method,
            model=model,
            tokenizer=tokenizer,
            train_examples=train_examples,
            eval_examples=eval_examples,
            cfg=cfg,
            device=device,
            seed=args.seed,
        )
        results[method] = result

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print("COMPARISON SUMMARY (at step 100)")
    print("=" * 50)
    print(f"{'Method':<30} {'eval_reward':>11}  {'grad_steps':>10}  {'notes'}")
    print("-" * 70)

    grpo_reward = results["diffu_grpo"]["final_eval_reward"]
    s1_reward   = results["stage1"]["final_eval_reward"]
    s2_reward   = results["stage2"]["final_eval_reward"]
    s2_expvar   = results["stage2"]["ema_expvar_final"]

    grpo_grad = results["diffu_grpo"]["grad_steps_nonzero"]
    s1_grad   = results["stage1"]["grad_steps_nonzero"]
    s2_grad   = results["stage2"]["grad_steps_nonzero"]

    print(f"{'Diffu-GRPO':<30} {grpo_reward:>11.3f}  {grpo_grad:>6}/{MAX_STEPS}")
    print(f"{'Stage 1':<30} {s1_reward:>11.3f}  {s1_grad:>6}/{MAX_STEPS}")
    s2_notes = f"expvar={s2_expvar:.2f}"
    print(f"{'Stage 2 delta-V':<30} {s2_reward:>11.3f}  {s2_grad:>6}/{MAX_STEPS}  "
          f"{s2_notes}")

    # ------------------------------------------------------------------
    # Tier 3 pass criteria (Stage 2)
    # ------------------------------------------------------------------
    print()
    print("Tier 3 pass (Stage 2):")

    # Criterion 1: eval_reward not worse than Diffu-GRPO
    # "not worse" = Stage2 reward >= GRPO reward - 0.05 tolerance
    reward_ok   = (s2_reward >= grpo_reward - 0.05)

    # Criterion 2: ema_expvar >= 0.05 at end
    expvar_ok   = (s2_expvar >= CRITIC_EXPVAR_GATE)

    # Criterion 3: stage2_adv active in at least 1 of the last 50 steps
    s2_adv_ok   = (results["stage2"]["stage2_adv_count"] >= 1)

    def _pf(b: bool) -> str:
        return "PASS" if b else "FAIL"

    print(f"  eval_reward not worse than Diffu-GRPO: {_pf(reward_ok)}"
          f"  ({s2_reward:.3f} vs {grpo_reward:.3f})")
    print(f"  ema_expvar >= {CRITIC_EXPVAR_GATE:.2f} at end:          {_pf(expvar_ok)}"
          f"  ({s2_expvar:.4f})")
    adv_count = results["stage2"]["stage2_adv_count"]
    print(f"  stage2_adv active in last 50 steps:   {_pf(s2_adv_ok)}"
          f"  ({adv_count}/20 steps)")

    all_pass = reward_ok and expvar_ok and s2_adv_ok
    n_pass   = sum([reward_ok, expvar_ok, s2_adv_ok])
    print()
    print(f"Overall Tier 3: {_pf(all_pass)} ({n_pass}/3 criteria met)")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    import json
    summary = {
        "seed":     args.seed,
        "n_train":  N_TRAIN,
        "n_eval":   N_EVAL,
        "max_steps": MAX_STEPS,
        "methods":  {
            m: {
                "final_eval_reward":  r["final_eval_reward"],
                "grad_steps_nonzero": r["grad_steps_nonzero"],
                "ema_expvar_final":   r["ema_expvar_final"],
                "stage2_adv_count":   r["stage2_adv_count"],
                "eval_history":       r["eval_history"],
            }
            for m, r in results.items()
        },
        "tier3_pass": all_pass,
        "criteria":   {
            "reward_ok":  reward_ok,
            "expvar_ok":  expvar_ok,
            "s2_adv_ok":  s2_adv_ok,
        },
    }

    summary_path = os.path.join(args.output_dir, "tier3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved -> {summary_path}")


if __name__ == "__main__":
    main()
