"""
Unified training entry point for all cc_rl methods.

Supports:
  --method diffu_grpo  : official DiffuGRPOTrainer (baseline)
  --method stage1      : CWGRPOTrainer (confidence-weighted)
  --method stage2      : ValueCreditTrainer (state-value credit)
  --method stage3      : QCreditTrainer (Q-value credit)
"""
import sys
import os
import warnings

# Clear expired HF token so public datasets/models load without auth errors
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

# TRL 1.6.0 compat: patch is_rich_available which moved to transformers.utils
import trl.import_utils as _trl_iu
if not hasattr(_trl_iu, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_iu.is_rich_available = _ira
    except ImportError:
        _trl_iu.is_rich_available = lambda: False

# Register d1 code on path
_D1_PATH = os.path.join(os.path.dirname(__file__), "../../d1/diffu-grpo")
_D1_PATH = os.path.abspath(_D1_PATH)
if _D1_PATH not in sys.path:
    sys.path.insert(0, _D1_PATH)

# Register cc_rl package
_CC_RL_SRC = os.path.join(os.path.dirname(__file__), "../src")
_CC_RL_SRC = os.path.abspath(_CC_RL_SRC)
if _CC_RL_SRC not in sys.path:
    sys.path.insert(0, _CC_RL_SRC)

import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModel
try:
    from peft import LoraConfig
except Exception:
    LoraConfig = None

# d1 imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    correctness_reward_func_math,
    boxed_and_answer_tags_format_reward,
)
from data_utils import (
    get_gsm8k_questions,
    get_math_questions,
    set_random_seed,
)

# cc_rl imports
from cc_rl.algorithms.stage1_cw_grpo import CWGRPOTrainer
from cc_rl.algorithms.stage2_value_credit import ValueCreditTrainer
from cc_rl.algorithms.stage3_q_credit import QCreditTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="diffu_grpo",
                        choices=["diffu_grpo", "stage1", "stage2", "stage3"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="cc_rl_run")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon_clip", type=float, default=0.2)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--use_peft", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    # Credit params
    parser.add_argument("--credit_alpha", type=float, default=1.0)
    parser.add_argument("--credit_eps", type=float, default=1e-6)
    parser.add_argument("--credit_clip_min", type=float, default=0.25)
    parser.add_argument("--credit_clip_max", type=float, default=4.0)
    parser.add_argument("--critic_lr", type=float, default=5e-6)
    parser.add_argument("--critic_loss_coef", type=float, default=0.5)
    parser.add_argument("--value_hidden_size", type=int, default=1024)
    parser.add_argument("--value_mlp_layers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset + rewards
    if args.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif args.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset = dataset.shuffle(seed=args.seed)

    print(f"[train_experiment] Method: {args.method}")
    print(f"[train_experiment] Model: {args.model_path}")
    print(f"[train_experiment] Dataset: {args.dataset}, {len(dataset)} examples")
    print(f"[train_experiment] Max steps: {args.max_steps}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_experiment] Loading model to {device}...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.config.use_cache = False
    print(f"[train_experiment] Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    peft_config = None
    if args.use_peft:
        if LoraConfig is None:
            raise RuntimeError("peft is required for LoRA but failed to import")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        print(f"[train_experiment] Using LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    # Build DiffuGRPOConfig
    grpo_config = DiffuGRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        model_path=args.model_path,
        dataset=args.dataset,
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        diffusion_steps=args.diffusion_steps,
        block_length=args.block_length,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        epsilon=args.epsilon_clip,
        num_iterations=args.num_iterations,
        generation_batch_size=args.num_generations,
        remasking="low_confidence",
        random_masking=True,
        p_mask_prompt=0.15,
        mask_id=126336,
        gradient_checkpointing=False,  # LLaDA does not support gradient checkpointing
        remove_unused_columns=False,
        log_completions=False,  # Disabled: TRL 1.6.0 changed print_prompt_completions_sample signature
        report_to=[],  # no wandb by default
    )

    # Common trainer kwargs
    trainer_kwargs = dict(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Credit kwargs (Stage 1+)
    credit_kwargs = dict(
        credit_alpha=args.credit_alpha,
        credit_eps=args.credit_eps,
        credit_clip_min=args.credit_clip_min,
        credit_clip_max=args.credit_clip_max,
    )

    method = args.method
    # TRL 1.6.0 compat: add all missing GRPOConfig fields to DiffuGRPOConfig
    trl_defaults = {
        "steps_per_generation": grpo_config.gradient_accumulation_steps,
        "num_generations_eval": None,
        "generation_kwargs": None,
        "chat_template_kwargs": None,
        "max_tool_calling_iterations": None,
        "disable_dropout": False,
        "cast_lm_head_to_fp32": False,
        "shuffle_dataset": True,
        "pad_to_multiple_of": None,
        "vllm_mode": "colocate",
        "vllm_model_impl": "vllm",
        "vllm_enable_sleep_mode": False,
        "vllm_structured_outputs_regex": None,
        "vllm_server_base_url": None,
        "vllm_server_host": "0.0.0.0",
        "vllm_server_port": 8000,
        "vllm_server_timeout": 240.0,
        "vllm_group_port": 51216,
        "vllm_max_model_length": None,
        "vllm_tensor_parallel_size": 1,
        "delta": None,
        "epsilon_high": None,
        "scale_rewards": "group",
        "loss_type": "grpo",
        "mask_truncated_completions": False,
        "top_entropy_quantile": 1.0,
        "importance_sampling_level": "token",
        "multi_objective_aggregation": "sum_then_normalize",
        "use_bias_correction_kl": False,
        "num_completions_to_print": None,
        "log_unique_prompts": False,
        "log_completions_hub_repo": None,
        "use_transformers_paged": False,
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": "sequence_mask",
        "vllm_importance_sampling_clip_max": 3.0,
        "vllm_importance_sampling_clip_min": None,
        "off_policy_mask_threshold": None,
        "vllm_importance_sampling_cap": None,
        "sapo_temperature_neg": 1.05,
        "sapo_temperature_pos": 1.0,
        "vespo_k_pos": 2.0,
        "vespo_lambda_pos": 3.0,
        "vespo_k_neg": 3.0,
        "vespo_lambda_neg": 2.0,
    }
    for attr, val in trl_defaults.items():
        if not hasattr(grpo_config, attr):
            setattr(grpo_config, attr, val)

    print(f"[train_experiment] Building trainer: {method}")
    if method == "diffu_grpo":
        trainer = DiffuGRPOTrainer(**trainer_kwargs)
    elif method == "stage1":
        trainer = CWGRPOTrainer(**trainer_kwargs, **credit_kwargs)
    elif method == "stage2":
        trainer = ValueCreditTrainer(
            **trainer_kwargs,
            **credit_kwargs,
            critic_lr=args.critic_lr,
            critic_loss_coef=args.critic_loss_coef,
            value_hidden_size=args.value_hidden_size,
            value_mlp_layers=args.value_mlp_layers,
        )
    elif method == "stage3":
        trainer = QCreditTrainer(
            **trainer_kwargs,
            **credit_kwargs,
            critic_lr=args.critic_lr,
            critic_loss_coef=args.critic_loss_coef,
            value_hidden_size=args.value_hidden_size,
            value_mlp_layers=args.value_mlp_layers,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # TRL 1.6.0 compat: max_prompt_length was removed from GRPOTrainer instance
    # but d1's DiffuGRPOTrainer._generate_and_score_completions still uses it
    if not hasattr(trainer, "max_prompt_length"):
        trainer.max_prompt_length = grpo_config.max_prompt_length

    print("[train_experiment] Starting training...")
    trainer.train()
    print(f"[train_experiment] Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
