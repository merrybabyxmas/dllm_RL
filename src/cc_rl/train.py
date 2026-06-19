"""
Central training entry point for cc_rl.

Parses a YAML config file (with optional CLI overrides) and launches the
appropriate trainer: baseline DiffuGRPO, Stage 1 CW-GRPO, Stage 2 Value
Credit, or Stage 3 Q Credit.

Usage
-----
python -m cc_rl.train --config configs/stage1_cw_grpo.yaml \
    --model_path MDLM-hf/LLaDA-8B-Instruct \
    --dataset gsm8k \
    --output_dir outputs/gsm8k_stage1 \
    --max_steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# Ensure d1/diffu-grpo is importable
_DIFFU_GRPO_PATH = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
if _DIFFU_GRPO_PATH not in sys.path:
    sys.path.insert(0, _DIFFU_GRPO_PATH)


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
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and return as nested dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge flat CLI key=value overrides into nested config."""
    for key, value in overrides.items():
        # Support dot-notation for nested keys: model.precision -> config["model"]["precision"]
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Type coercion
        d[parts[-1]] = _coerce(value)
    return config


def _coerce(value: str) -> Any:
    """Attempt to parse string as int, float, bool, or leave as string."""
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.lower() == "null" or value.lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(cfg: Dict[str, Any], split: str = "train"):
    """Instantiate the appropriate dataset based on config."""
    name = cfg.get("dataset", {}).get("name", "gsm8k")
    max_examples = cfg.get("dataset", {}).get("max_train_examples") if split == "train" \
        else cfg.get("dataset", {}).get("max_eval_examples")

    if name == "gsm8k":
        from cc_rl.data.gsm8k import get_gsm8k_dataset
        return get_gsm8k_dataset(split=split, max_examples=max_examples)
    elif name in ("math500", "math_500"):
        from cc_rl.data.math500 import get_math500_dataset
        return get_math500_dataset(split=split, max_examples=max_examples)
    elif name == "synthetic_arithmetic":
        from cc_rl.data.synthetic_arithmetic import get_synthetic_arithmetic_dataset
        return get_synthetic_arithmetic_dataset(
            n=max_examples or 1000, split=split
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")


# ---------------------------------------------------------------------------
# Reward factory
# ---------------------------------------------------------------------------

def build_reward_funcs(cfg: Dict[str, Any]):
    """Instantiate reward function(s) based on config."""
    reward_type = cfg.get("reward", {}).get("type", "exact_match")
    dataset_name = cfg.get("dataset", {}).get("name", "gsm8k")

    if reward_type == "exact_match" and dataset_name == "gsm8k":
        from cc_rl.rewards.exact_match import reward_gsm8k_batch
        return [reward_gsm8k_batch]
    elif reward_type == "exact_match" and dataset_name in ("math500", "math_500"):
        from cc_rl.rewards.math_normalize import reward_math500_batch
        return [reward_math500_batch]
    else:
        from cc_rl.rewards.exact_match import reward_gsm8k_batch
        return [reward_gsm8k_batch]


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_trainer(
    algorithm: str,
    model,
    processing_class,
    train_dataset,
    eval_dataset,
    reward_funcs,
    grpo_config,
    cfg: Dict[str, Any],
):
    """Instantiate the appropriate trainer based on algorithm name."""
    credit_cfg = cfg.get("credit", {})
    credit_kwargs = {
        "credit_alpha": credit_cfg.get("alpha", 1.0),
        "credit_eps": credit_cfg.get("eps", 1e-6),
        "credit_clip_min": credit_cfg.get("clip_min", 0.25),
        "credit_clip_max": credit_cfg.get("clip_max", 4.0),
    }

    common_kwargs = dict(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
    )

    if algorithm in ("diffu_grpo", "baseline"):
        from cc_rl.algorithms.diffu_grpo import WrappedDiffuGRPOTrainer
        return WrappedDiffuGRPOTrainer(**common_kwargs)

    elif algorithm == "stage1_cw_grpo":
        from cc_rl.algorithms.stage1_cw_grpo import CWGRPOTrainer
        return CWGRPOTrainer(**common_kwargs, **credit_kwargs)

    elif algorithm == "stage2_value_credit":
        from cc_rl.algorithms.stage2_value_credit import ValueCreditTrainer
        critic_cfg = cfg.get("critic", {})
        return ValueCreditTrainer(
            **common_kwargs,
            **credit_kwargs,
            value_hidden_size=critic_cfg.get("value_hidden_size", 1024),
            value_mlp_layers=critic_cfg.get("value_mlp_layers", 2),
            critic_lr=cfg.get("optimizer", {}).get("critic_lr", 5e-6),
            critic_loss_coef=critic_cfg.get("loss_coef", 0.5),
        )

    elif algorithm == "stage3_q_credit":
        from cc_rl.algorithms.stage3_q_credit import QCreditTrainer
        critic_cfg = cfg.get("critic", {})
        return QCreditTrainer(
            **common_kwargs,
            **credit_kwargs,
            value_hidden_size=critic_cfg.get("value_hidden_size", 1024),
            value_mlp_layers=critic_cfg.get("value_mlp_layers", 2),
            critic_lr=cfg.get("optimizer", {}).get("critic_lr", 5e-6),
            critic_loss_coef=critic_cfg.get("loss_coef", 0.5),
            q_hidden_size=critic_cfg.get("q_hidden_size", 1024),
            q_mlp_layers=critic_cfg.get("q_mlp_layers", 2),
            q_lr=cfg.get("optimizer", {}).get("q_lr", 5e-6),
            q_loss_coef=critic_cfg.get("q_loss_coef", 0.5),
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cc_rl training entry point")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Dot-notation key=value overrides e.g. train.algorithm=stage2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI flat overrides
    cli_overrides = {}
    if args.model_path:
        cli_overrides["model.name_or_path"] = args.model_path
    if args.dataset:
        cli_overrides["dataset.name"] = args.dataset
    if args.output_dir:
        cli_overrides["project.output_dir"] = args.output_dir
    if args.max_steps:
        cli_overrides["train.max_steps"] = args.max_steps
    if args.seed:
        cli_overrides["project.seed"] = args.seed
    for override in args.overrides:
        if "=" in override:
            k, v = override.split("=", 1)
            cli_overrides[k] = v
    cfg = merge_cli_overrides(cfg, cli_overrides)

    # Seed
    seed = cfg.get("project", {}).get("seed", 42)
    seed_everything(seed)

    # Output dir
    output_dir = cfg.get("project", {}).get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # Model + tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = cfg["model"]["name_or_path"]
    precision = cfg.get("model", {}).get("precision", "bf16")

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(precision, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Datasets
    train_dataset = build_dataset(cfg, split="train")
    eval_dataset = build_dataset(cfg, split="test")

    # Reward functions
    reward_funcs = build_reward_funcs(cfg)

    # Build GRPOConfig
    from trl import GRPOConfig
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    opt_cfg = cfg.get("optimizer", {})
    sampling_cfg = cfg.get("sampling", {})

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=train_cfg.get("max_steps", 3000),
        per_device_train_batch_size=train_cfg.get("batch_prompts", 1),
        num_generations=train_cfg.get("num_generations", 8),
        max_completion_length=sampling_cfg.get("max_new_tokens", 256),
        learning_rate=opt_cfg.get("policy_lr", 1e-6),
        warmup_steps=opt_cfg.get("warmup_steps", 50),
        gradient_accumulation_steps=1,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        logging_steps=train_cfg.get("log_every", 10),
        eval_steps=train_cfg.get("eval_every", 100),
        save_steps=train_cfg.get("save_every", 500),
        seed=seed,
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
        # DiffuGRPO-specific args (set via diffu_grpo_config)
    )

    # Algorithm
    algorithm = train_cfg.get("algorithm", "diffu_grpo")
    print(f"Algorithm: {algorithm}")

    trainer = build_trainer(
        algorithm=algorithm,
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        grpo_config=grpo_config,
        cfg=cfg,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
