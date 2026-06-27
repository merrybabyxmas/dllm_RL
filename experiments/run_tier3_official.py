#!/usr/bin/env python3
"""
Tier 3 Short Policy Training — Official DiffuGRPOTrainer hierarchy.

Compares three methods on GSM8K over 300 training steps each:
  Method A: DiffuGRPOTrainer        — standard GRPO baseline
  Method B: CWGRPOTrainer           — Stage 1 confidence-weighted GRPO
  Method C: ValueCreditTrainer      — Stage 2 block-level delta-V credit

All three share the same 64 train / 32 eval examples (same seed-derived split),
identical hyperparameters, and identical LoRA configuration.  Between methods,
LoRA A/B matrices are re-initialized to zero-B / kaiming-A so training restarts
from the same pre-trained base without reloading from disk.

Evaluation is performed via a TrainerCallback every 30 steps, which runs
greedy diffusion generation on the 32 held-out examples and scores them with
the same three reward functions used during training.

Usage:
    python experiments/run_tier3_official.py
    python experiments/run_tier3_official.py --seed 42 --output_base experiments/outputs/tier3_official
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: d1 package and cc_rl package
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_D1_PATH      = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
_SRC_PATH     = os.path.join(_PROJECT_ROOT, "src")

for _p in [_D1_PATH, _SRC_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# TRL 1.6.0 compat: patch missing is_rich_available in trl.import_utils
import trl.import_utils as _trl_iu
if not hasattr(_trl_iu, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_iu.is_rich_available = _ira
    except ImportError:
        _trl_iu.is_rich_available = lambda: False

# ---------------------------------------------------------------------------
# Imports from official d1 package
# ---------------------------------------------------------------------------
from diffu_grpo_trainer import DiffuGRPOTrainer       # noqa: E402
from diffu_grpo_config import DiffuGRPOConfig          # noqa: E402
from data_utils import get_gsm8k_questions             # noqa: E402
from reward_func import (                               # noqa: E402
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
)

# ---------------------------------------------------------------------------
# Imports from cc_rl package
# ---------------------------------------------------------------------------
from cc_rl.algorithms.stage1_cw_grpo import CWGRPOTrainer          # noqa: E402
from cc_rl.algorithms.stage2_value_credit import ValueCreditTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# HF / PEFT
# ---------------------------------------------------------------------------
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Lock all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH  = "/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct"
MASK_ID     = 126336

N_TRAIN     = 64
N_EVAL      = 32
MAX_STEPS   = 300
EVAL_EVERY  = 30


# ---------------------------------------------------------------------------
# LoRA weight re-initialization
# ---------------------------------------------------------------------------

def reset_lora_weights(model: nn.Module) -> None:
    """
    Re-initialize all LoRA adapter matrices to their default state:
      - lora_A: kaiming_uniform (matches default PEFT initialization)
      - lora_B: zeros           (ensures adapter starts as identity)

    This lets us reuse the same model object across methods without
    reloading the full 8B base model from disk.
    """
    for _name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for key in module.lora_A:
                nn.init.kaiming_uniform_(module.lora_A[key].weight, a=math.sqrt(5))
            for key in module.lora_B:
                nn.init.zeros_(module.lora_B[key].weight)


# ---------------------------------------------------------------------------
# Eval callback: greedy generation + reward scoring every eval_every steps
# ---------------------------------------------------------------------------

class EvalCallback(TrainerCallback):
    """
    Runs greedy diffusion generation on a fixed eval set every `eval_every`
    training steps and records mean reward.

    The HF Trainer's native eval loop does not suit the diffusion generation
    format (it expects the model's forward() to return logits, not a denoised
    sequence).  This callback bypasses that limitation by using the trainer's
    own generate() method directly.

    Results are stored in self.eval_history for retrieval after training.
    """

    def __init__(
        self,
        trainer_ref: "DiffuGRPOTrainer",
        eval_examples: List[dict],
        tokenizer,
        eval_every: int = 30,
    ) -> None:
        super().__init__()
        self.trainer_ref   = trainer_ref
        self.eval_examples = eval_examples
        self.tokenizer     = tokenizer
        self.eval_every    = eval_every
        self.eval_history: List[Tuple[int, float]] = []  # (step, mean_reward)

    @torch.no_grad()
    def _run_eval(self, global_step: int) -> float:
        """
        Generate completions on eval_examples with temperature=0.0 (greedy),
        then score with the three reward functions.

        Returns mean reward over all eval examples.
        """
        trainer  = self.trainer_ref
        args     = trainer.args
        device   = trainer.accelerator.device

        all_rewards: List[float] = []
        model = trainer.model
        model.eval()

        for example in self.eval_examples:
            # Tokenize prompt (left-padded to match training behavior)
            prompt_text = self.tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids = enc["input_ids"].to(device)[:, -args.max_prompt_length:]
            # prompt_ids: [1, prompt_len]

            # Diffusion generation (greedy: temperature=0.0)
            from trl.models import unwrap_model_for_generation
            with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as unwrapped:
                full_ids = trainer.generate(
                    model=unwrapped,
                    prompt=prompt_ids,
                    steps=args.diffusion_steps,
                    gen_length=args.max_completion_length,
                    block_length=args.block_length,
                    temperature=0.0,   # greedy eval
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                )  # [1, prompt_len + comp_len]

            prompt_length   = prompt_ids.size(1)
            completion_ids_ = full_ids[:, prompt_length:]
            completion_text = self.tokenizer.decode(
                completion_ids_[0], skip_special_tokens=True
            )

            # Score with reward functions (conversational format expected by d1)
            completions_ = [[{"role": "assistant", "content": completion_text}]]
            prompts_     = [example["prompt"]]

            r_correct = correctness_reward_func(
                prompts=prompts_,
                completions=completions_,
                answer=[example.get("answer", "")],
            )[0]
            r_int    = int_reward_func(completions=completions_)[0]
            r_fmt    = soft_format_reward_func(completions=completions_)[0]

            all_rewards.append(r_correct + r_int + r_fmt)

        model.train()
        mean_r = float(np.mean(all_rewards))
        self.eval_history.append((global_step, mean_r))
        return mean_r

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if state.global_step % self.eval_every == 0 and state.global_step > 0:
            mean_r = self._run_eval(state.global_step)
            print(
                f"  [EvalCallback step={state.global_step:4d}]"
                f"  eval_reward={mean_r:.4f}"
            )
        return control


# ---------------------------------------------------------------------------
# Build DiffuGRPOConfig for a given method
# ---------------------------------------------------------------------------

def build_config(method: str, output_base: str) -> DiffuGRPOConfig:
    """
    Build a DiffuGRPOConfig for the given method.

    All three methods share identical hyperparameters; only output_dir differs.
    """
    return DiffuGRPOConfig(
        output_dir=os.path.join(output_base, method),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        num_generations=4,
        generation_batch_size=1,
        learning_rate=1e-6,
        beta=0.04,
        epsilon=0.2,
        max_completion_length=256,
        max_prompt_length=256,
        diffusion_steps=64,
        block_length=32,
        temperature=0.9,
        mask_id=MASK_ID,
        remasking="low_confidence",
        # Disable native trainer eval (we use EvalCallback instead)
        eval_strategy="no",
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        seed=42,
        # Misc
        dataloader_drop_last=False,
        report_to=[],            # no wandb
        cfg_scale=0.0,
    )


# ---------------------------------------------------------------------------
# Run a single method
# ---------------------------------------------------------------------------

def run_method(
    method: str,
    model: nn.Module,
    tokenizer,
    train_dataset: Dataset,
    eval_examples: List[dict],
    peft_config: LoraConfig,
    output_base: str,
    seed: int,
) -> Dict:
    """
    Train one method for MAX_STEPS steps and return results dict.

    Steps:
    1. Reset LoRA weights to identity (fresh start, no model reload).
    2. Build DiffuGRPOConfig and appropriate Trainer subclass.
    3. Attach EvalCallback.
    4. Call trainer.train().
    5. Return eval history + final metrics.

    Parameters
    ----------
    method       : "method_a", "method_b", or "method_c"
    model        : LoRA-wrapped model (re-used across methods)
    tokenizer    : tokenizer
    train_dataset: HF Dataset with 64 GSM8K examples
    eval_examples: list of 32 GSM8K dicts
    peft_config  : LoraConfig (used only by Trainer for checkpoint saving)
    output_base  : base output directory
    seed         : global seed

    Returns
    -------
    dict with keys: method, final_eval_reward, eval_history, train_time_s
    """
    print(f"\n{'='*60}")
    print(f"  Method: {method}")
    print(f"{'='*60}")

    # Reset LoRA to fresh state
    reset_lora_weights(model)
    print("  LoRA weights re-initialized to identity.")

    cfg = build_config(method, output_base)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Reward functions used during training
    reward_funcs = [
        correctness_reward_func,
        int_reward_func,
        soft_format_reward_func,
    ]

    # Trainer-specific kwargs (Stage 1 and Stage 2 have extra params)
    extra_kwargs: Dict[str, Any] = {}
    if method in ("method_b", "method_c"):
        extra_kwargs.update(
            credit_alpha=1.0,
            credit_eps=1e-6,
            credit_clip_min=0.25,
            credit_clip_max=4.0,
        )
    if method == "method_c":
        extra_kwargs.update(
            value_hidden_size=1024,
            value_mlp_layers=2,
            critic_lr=5e-6,
            critic_loss_coef=0.5,
            delta_v_gate=0.01,
        )

    # Select trainer class
    trainer_cls = {
        "method_a": DiffuGRPOTrainer,
        "method_b": CWGRPOTrainer,
        "method_c": ValueCreditTrainer,
    }[method]

    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        **extra_kwargs,
    )

    # Attach eval callback
    eval_cb = EvalCallback(
        trainer_ref=trainer,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        eval_every=EVAL_EVERY,
    )
    trainer.add_callback(eval_cb)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Final eval if not already evaluated at step MAX_STEPS
    if not eval_cb.eval_history or eval_cb.eval_history[-1][0] != MAX_STEPS:
        final_r = eval_cb._run_eval(MAX_STEPS)
    else:
        final_r = eval_cb.eval_history[-1][1]

    return {
        "method":           method,
        "final_eval_reward": final_r,
        "eval_history":     eval_cb.eval_history,
        "train_time_s":     train_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tier 3: 3-method comparison using official DiffuGRPOTrainer hierarchy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_base", type=str,
                   default=os.path.join(_PROJECT_ROOT, "experiments", "outputs", "tier3_official"))
    p.add_argument("--model_path",  type=str, default=MODEL_PATH)
    p.add_argument("--methods",     nargs="+",
                   default=["method_a", "method_b", "method_c"],
                   choices=["method_a", "method_b", "method_c"],
                   help="Which methods to run (default: all three)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_base, exist_ok=True)

    print("=" * 60)
    print("Tier 3: Official DiffuGRPOTrainer 3-method comparison")
    print("=" * 60)
    print(f"  Train: {N_TRAIN} examples  |  Eval: {N_EVAL} examples")
    print(f"  Steps: {MAX_STEPS}  |  Eval every: {EVAL_EVERY}")
    print(f"  Methods: {args.methods}")
    print(f"  Model: {args.model_path}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Load model (once — LoRA weights reset between methods)
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model_path} ...")
    t0 = time.time()
    base_model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.config.use_cache = False
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Apply LoRA (once — re-init between methods, not reload)
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    print()

    # ------------------------------------------------------------------
    # Load GSM8K and split deterministically
    # ------------------------------------------------------------------
    print("Loading GSM8K ...")
    full_data = get_gsm8k_questions("train")

    rng_split = random.Random(args.seed)
    all_idx   = list(range(len(full_data)))
    rng_split.shuffle(all_idx)

    train_idx = all_idx[:N_TRAIN]
    eval_idx  = all_idx[N_TRAIN: N_TRAIN + N_EVAL]

    # HF Dataset for trainer (must have "prompt" and "answer" columns)
    train_dataset = full_data.select(train_idx)
    eval_examples = [full_data[i] for i in eval_idx]  # plain list of dicts

    print(f"  Split: {len(train_dataset)} train / {len(eval_examples)} eval")
    print()

    # ------------------------------------------------------------------
    # Run all requested methods sequentially
    # ------------------------------------------------------------------
    results: Dict[str, Dict] = {}

    for method in args.methods:
        result = run_method(
            method=method,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_examples=eval_examples,
            peft_config=peft_config,
            output_base=args.output_base,
            seed=args.seed,
        )
        results[method] = result
        # Free any cached memory between methods
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Comparison summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    labels = {
        "method_a": "A: DiffuGRPO (baseline)",
        "method_b": "B: CW-GRPO (Stage 1)",
        "method_c": "C: delta-V (Stage 2)",
    }
    print(f"{'Method':<30} {'eval_reward':>12}  {'time(s)':>10}")
    print("-" * 58)
    for m, r in results.items():
        lbl = labels.get(m, m)
        print(f"  {lbl:<28} {r['final_eval_reward']:>12.4f}  {r['train_time_s']:>10.1f}")

    # Tier 3 pass criteria (applied to Method C vs Method A)
    print()
    if "method_a" in results and "method_c" in results:
        grpo_r   = results["method_a"]["final_eval_reward"]
        delta_v_r = results["method_c"]["final_eval_reward"]
        reward_ok = delta_v_r >= grpo_r - 0.05
        print("Tier 3 pass criteria (Method C vs Method A):")
        pf = lambda b: "PASS" if b else "FAIL"
        print(f"  eval_reward not worse than baseline: {pf(reward_ok)}"
              f"  ({delta_v_r:.4f} vs {grpo_r:.4f})")
        all_pass = reward_ok
        print(f"\nOverall Tier 3: {pf(all_pass)}")
    print()

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    summary = {
        "seed":       args.seed,
        "n_train":    N_TRAIN,
        "n_eval":     N_EVAL,
        "max_steps":  MAX_STEPS,
        "eval_every": EVAL_EVERY,
        "methods": {
            m: {
                "final_eval_reward": r["final_eval_reward"],
                "eval_history":      [(int(s), float(v)) for s, v in r["eval_history"]],
                "train_time_s":      r["train_time_s"],
            }
            for m, r in results.items()
        },
    }

    if "method_a" in results and "method_c" in results:
        summary["tier3_pass"] = reward_ok

    out_path = os.path.join(args.output_base, "tier3_official_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
