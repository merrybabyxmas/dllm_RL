#!/usr/bin/env python3
"""
Official experiment runner.
Datasets: gsm8k, mbpp, humaneval, svamp, countdown, spider
Methods:  baseline, delta_v_only

- baseline    : zero-shot eval, no training
- delta_v_only: ValueCreditTrainer with delta-V credit (no confidence weighting), 1 epoch

Usage:
  python experiments/run_experiment.py --dataset mbpp     --method baseline
  python experiments/run_experiment.py --dataset gsm8k    --method delta_v_only --gen_length 256
  python experiments/run_experiment.py --dataset countdown --method delta_v_only --gen_length 128
"""
from __future__ import annotations

import argparse
import ast as _ast
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter as _Counter

# Reduce CUDA memory fragmentation — recommended by PyTorch when near GPU limit
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_D1_DIFFU_GRPO  = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
_SRC_PATH       = os.path.join(_PROJECT_ROOT, "src")
_DATA_DIR       = os.path.join(_PROJECT_ROOT, "data")
_MODEL_PATH     = "/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct"
_MASK_ID        = 126336

for _p in [_D1_DIFFU_GRPO, _SRC_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# TRL 1.6.0 compat
import trl.import_utils as _trl_iu
if not hasattr(_trl_iu, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_iu.is_rich_available = _ira
    except ImportError:
        _trl_iu.is_rich_available = lambda: False

# ---------------------------------------------------------------------------
# Official d1 imports
# ---------------------------------------------------------------------------
from diffu_grpo_trainer import DiffuGRPOTrainer  # noqa: E402
from diffu_grpo_config  import DiffuGRPOConfig   # noqa: E402
from data_utils import get_gsm8k_questions        # noqa: E402
from reward_func import (                         # noqa: E402
    correctness_reward_func,
    int_reward_func,
    soft_format_reward_func,
)

# ---------------------------------------------------------------------------
# cc_rl imports
# ---------------------------------------------------------------------------
from cc_rl.algorithms.stage1_cw_grpo    import CWGRPOTrainer        # noqa: E402
from cc_rl.algorithms.stage2_value_credit import ValueCreditTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# HF / PEFT
# ---------------------------------------------------------------------------
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from trl.models import unwrap_model_for_generation


# ===========================================================================
# Reproducibility
# ===========================================================================

def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ===========================================================================
# Reward helpers for MBPP and Spider (d1 conversational format)
# ===========================================================================

_CODE_BLOCK_RE      = re.compile(r"<answer>\s*```python\s*(.*?)```\s*</answer>", re.DOTALL)
_CODE_BLOCK_BARE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
_XML_ANSWER_RE      = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SQL_ANSWER_RE      = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_HASH_NUM_RE        = re.compile(r"####\s*([\d\.\-\+]+)")


def _get_content(completion) -> str:
    """Extract text from d1-style conversational completion."""
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def _extract_code(text: str):
    m = _CODE_BLOCK_RE.search(text)
    if m: return m.group(1).strip()
    m = _CODE_BLOCK_BARE_RE.search(text)
    if m: return m.group(1).strip()
    m = _XML_ANSWER_RE.search(text)
    if m: return m.group(1).strip()
    return None


def _run_mbpp_tests(code: str, test_list: list, setup: str = "") -> float:
    full = (setup + "\n\n" if setup else "") + code + "\n\n" + "\n".join(test_list)
    try:
        res = subprocess.run(
            [sys.executable, "-c", full],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return 1.0 if res.returncode == 0 else 0.0
    except Exception:
        return 0.0


def mbpp_reward_func(prompts, completions, test_list, test_setup_code, **kwargs) -> list[float]:
    rewards = []
    for c, tl, ts in zip(completions, test_list, test_setup_code):
        text = _get_content(c)
        code = _extract_code(text)
        rewards.append(0.0 if code is None else _run_mbpp_tests(code, tl, ts))
    return rewards


def _normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("( ", "(").replace(" )", ")")
    return sql


def spider_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for c, a in zip(completions, answer):
        text  = _get_content(c)
        m     = _SQL_ANSWER_RE.search(text)
        pred  = _normalize_sql(m.group(1)) if m else ""
        gold  = _normalize_sql(a)
        if not pred:
            rewards.append(0.0)
        elif pred == gold:
            rewards.append(1.0)
        else:
            gt, pt = set(gold.split()), set(pred.split())
            if not pt:
                rewards.append(0.0)
            else:
                prec = len(gt & pt) / len(pt)
                rec  = len(gt & pt) / len(gt) if gt else 0.0
                rewards.append(2 * prec * rec / (prec + rec + 1e-9))
    return rewards


def _run_humaneval_tests(code: str, test: str, entry_point: str) -> float:
    """Execute generated code + HumanEval check() function."""
    full = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    try:
        res = subprocess.run(
            [sys.executable, "-c", full],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return 1.0 if res.returncode == 0 else 0.0
    except Exception:
        return 0.0


def humaneval_reward_func(prompts, completions, test, entry_point, **kwargs) -> list[float]:
    rewards = []
    for c, t, ep in zip(completions, test, entry_point):
        text = _get_content(c)
        code = _extract_code(text) or text  # fall back to raw text if no tags
        rewards.append(_run_humaneval_tests(code, t, ep))
    return rewards


def _extract_number(text: str) -> Optional[float]:
    """Extract the final numeric answer (after #### or as last number)."""
    m = _HASH_NUM_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Fallback: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None


def svamp_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for c, a in zip(completions, answer):
        text = _get_content(c)
        pred = _extract_number(text)
        try:
            gold = float(a)
        except (ValueError, TypeError):
            rewards.append(0.0)
            continue
        if pred is None:
            rewards.append(0.0)
        elif abs(pred - gold) < 1e-3:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Countdown reward
# ---------------------------------------------------------------------------
_COUNTDOWN_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _safe_eval_expr(expr: str) -> Optional[float]:
    """Evaluate a pure-arithmetic expression safely via ast.parse."""
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    # Only allow: numbers, unary minus, binary +/-/*//, parentheses
    allowed_ops = (
        _ast.Expression, _ast.BinOp, _ast.UnaryOp,
        _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv,
        _ast.USub, _ast.UAdd,
        _ast.Constant,
    )
    for node in _ast.walk(tree):
        if not isinstance(node, allowed_ops):
            return None
    try:
        result = eval(compile(tree, "<expr>", "eval"))
        return float(result)
    except Exception:
        return None


def _extract_used_numbers(expr: str) -> Optional[list]:
    """Extract the numeric literals used in the expression."""
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    nums = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            nums.append(node.value)
    return nums


def countdown_reward_func(prompts, completions, nums, target, **kwargs) -> list[float]:
    """
    reward=1.0 if model uses each given number exactly once with +,-,*,/
    and the expression evaluates to target. else 0.0.
    """
    rewards = []
    for c, given_nums, tgt in zip(completions, nums, target):
        text = _get_content(c)
        m    = _COUNTDOWN_ANSWER_RE.search(text)
        if m is None:
            rewards.append(0.0)
            continue
        expr = m.group(1).strip()
        result = _safe_eval_expr(expr)
        if result is None:
            rewards.append(0.0)
            continue
        used = _extract_used_numbers(expr)
        if used is None:
            rewards.append(0.0)
            continue
        # All given_nums are ints; compare as ints via Counter
        try:
            given_ints = [int(n) for n in given_nums]
            used_ints  = [int(n) for n in used]
        except (ValueError, TypeError):
            rewards.append(0.0)
            continue
        if _Counter(used_ints) != _Counter(given_ints):
            rewards.append(0.0)
            continue
        # Check if result equals target (allow small float tolerance)
        try:
            tgt_f = float(tgt)
        except (ValueError, TypeError):
            rewards.append(0.0)
            continue
        rewards.append(1.0 if abs(result - tgt_f) < 1e-3 else 0.0)
    return rewards


# ===========================================================================
# Dataset loaders
# ===========================================================================

def load_gsm8k(seed: int, max_train: int) -> tuple:
    """Returns (train_hf_dataset, eval_examples_list)."""
    import random as _random
    full_train = get_gsm8k_questions("train")   # HF Dataset, 7473 examples
    full_eval  = get_gsm8k_questions("test")    # HF Dataset, 1319 examples

    # Cap training data
    if len(full_train) > max_train:
        rng = _random.Random(seed)
        idx = list(range(len(full_train)))
        rng.shuffle(idx)
        full_train = full_train.select(idx[:max_train])

    eval_examples = [full_eval[i] for i in range(len(full_eval))]
    return full_train, eval_examples


def load_mbpp(seed: int, max_train: int) -> tuple:
    """
    Standard MBPP split: tasks 11-510 → train, 511-600 → test.
    Returns (train_hf_dataset, eval_examples_list).
    """
    import random as _random
    path = os.path.join(_DATA_DIR, "mbpp.jsonl")
    examples = [json.loads(l) for l in open(path)]

    train_ex = [e for e in examples if 11  <= e["task_id"] <= 510]
    eval_ex  = [e for e in examples if 511 <= e["task_id"] <= 600]

    if len(train_ex) > max_train:
        rng = _random.Random(seed)
        rng.shuffle(train_ex)
        train_ex = train_ex[:max_train]

    # GSM8K system prompt format (reasoning + answer tags)
    _SYS = (
        "Respond in the following format:\n"
        "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
    )

    def _to_prompt(e):
        tests = "\n".join(e["test_list"])
        return {
            "prompt": [{"role": "user", "content":
                f"{_SYS}\n\nYou are a Python expert. Solve the following task.\n\n"
                f"Task: {e['text']}\n\nYour function must pass:\n{tests}\n\n"
                "Write your solution inside <answer>```python\n...\n```</answer> tags."
            }],
            "test_list":      e["test_list"],
            "test_setup_code": e.get("test_setup_code", ""),
        }

    train_rows = [_to_prompt(e) for e in train_ex]
    eval_rows  = [_to_prompt(e) for e in eval_ex]

    train_hf = Dataset.from_list(train_rows)
    return train_hf, eval_rows


def load_spider(seed: int, max_train: int) -> tuple:
    """
    sql_create_context: 78,577 examples. 90/10 split (shuffle with seed).
    Training capped at max_train (default 10,000 for practical reasons).
    Returns (train_hf_dataset, eval_examples_list).
    """
    import random as _random
    path = os.path.join(_DATA_DIR, "sql_create_context.json")
    with open(path) as f:
        raw = json.load(f)

    rng = _random.Random(seed)
    rng.shuffle(raw)

    split     = int(len(raw) * 0.9)
    train_raw = raw[:split]    # ~70,720
    eval_raw  = raw[split:]    # ~7,857

    # Cap training
    if len(train_raw) > max_train:
        train_raw = train_raw[:max_train]

    # Cap eval (1000 for practical timing)
    MAX_EVAL = 1000
    if len(eval_raw) > MAX_EVAL:
        eval_raw = eval_raw[:MAX_EVAL]

    _SYS = (
        "Respond in the following format:\n"
        "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
    )

    def _to_prompt(e):
        return {
            "prompt": [{"role": "user", "content":
                f"{_SYS}\n\nConvert the natural language question to SQL.\n\n"
                f"Schema:\n{e['context']}\n\nQuestion: {e['question']}\n\n"
                "Write only the SQL inside <answer>...</answer> tags."
            }],
            "answer": e["answer"].strip(),
        }

    train_rows = [_to_prompt(e) for e in train_raw]
    eval_rows  = [_to_prompt(e) for e in eval_raw]

    train_hf = Dataset.from_list(train_rows)
    return train_hf, eval_rows


def load_humaneval(seed: int, max_train: int) -> tuple:
    """
    HumanEval: 164 examples total (no official train split).
    Use all 164 for training (diffusion RL cycles through them) and eval.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")

    _SYS = (
        "Respond in the following format:\n"
        "<reasoning>\n...\n</reasoning>\n<answer>\n```python\n...\n```\n</answer>"
    )

    def _to_prompt(e):
        return {
            "prompt": [{"role": "user", "content": (
                f"{_SYS}\n\nComplete the following Python function. "
                "Write the complete function inside <answer>```python\n...\n```</answer> tags.\n\n"
                f"```python\n{e['prompt']}\n```"
            )}],
            "test":        e["test"],
            "entry_point": e["entry_point"],
        }

    import random as _random
    examples = [_to_prompt(e) for e in ds]
    if len(examples) > max_train:
        rng = _random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_train]

    from datasets import Dataset
    train_hf = Dataset.from_list(examples)
    eval_rows = [_to_prompt(e) for e in ds]  # all 164 for eval
    return train_hf, eval_rows


def load_svamp(seed: int, max_train: int) -> tuple:
    """
    SVAMP: 1,000 challenge math word problems. 800 train / 200 eval.
    ChilleD/SVAMP has 700 train + 300 test splits → combine for a full 1000-pool.
    """
    from datasets import load_dataset, Dataset as HFDataset
    import random as _random

    ds_train = load_dataset("ChilleD/SVAMP", split="train")   # 700
    ds_test  = load_dataset("ChilleD/SVAMP", split="test")    # 300
    examples = list(ds_train) + list(ds_test)                 # 1000
    rng = _random.Random(seed)
    rng.shuffle(examples)

    train_raw = examples[:min(max_train, 800)]
    eval_raw  = examples[800:1000]

    _SYS = "Solve the math problem step by step. Give the final numeric answer after ####."

    def _to_prompt(e):
        problem = e["Body"].strip() + " " + e["Question"].strip()
        return {
            "prompt": [
                {"role": "system", "content": _SYS},
                {"role": "user",   "content": problem},
            ],
            "answer": str(e["Answer"]),
        }

    train_rows = [_to_prompt(e) for e in train_raw]
    eval_rows  = [_to_prompt(e) for e in eval_raw]

    train_hf = HFDataset.from_list(train_rows)
    return train_hf, eval_rows


def load_countdown(seed: int, max_train: int) -> tuple:
    """
    Countdown: Jiayi-Pan/Countdown-Tasks-3to4 from HuggingFace.
    Each example: {"nums": [1,2,5,10], "target": 791}
    90/10 train/test split (shuffle with seed).
    """
    from datasets import load_dataset, Dataset as HFDataset
    import random as _random

    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    examples = list(ds)

    rng = _random.Random(seed)
    rng.shuffle(examples)

    split_idx  = int(len(examples) * 0.9)
    train_raw  = examples[:split_idx]
    eval_raw   = examples[split_idx:]

    if len(train_raw) > max_train:
        train_raw = train_raw[:max_train]

    MAX_EVAL = 500
    if len(eval_raw) > MAX_EVAL:
        eval_raw = eval_raw[:MAX_EVAL]

    _SYS = (
        "You are given a list of numbers and a target value. "
        "Use each number exactly once with the operations +, -, *, / "
        "to reach the target. Write your solution inside <answer>expression</answer> tags.\n"
        "Example: if nums=[1,2,5] and target=7, write <answer>1+1*2+5</answer> or <answer>5+2</answer> etc."
    )

    def _to_prompt(e):
        nums_str = str(e["nums"])
        return {
            "prompt": [{"role": "user", "content":
                f"{_SYS}\n\nNumbers: {nums_str}\nTarget: {e['target']}\n\n"
                "Write your arithmetic expression inside <answer>expression</answer> tags."
            }],
            "nums":   e["nums"],
            "target": e["target"],
        }

    train_rows = [_to_prompt(e) for e in train_raw]
    eval_rows  = [_to_prompt(e) for e in eval_raw]

    train_hf = HFDataset.from_list(train_rows)
    return train_hf, eval_rows


DATASET_LOADERS = {
    "gsm8k":     load_gsm8k,
    "mbpp":      load_mbpp,
    "spider":    load_spider,
    "humaneval": load_humaneval,
    "svamp":     load_svamp,
    "countdown": load_countdown,
}

# Reward functions passed to the official trainer
TRAIN_REWARD_FUNCS = {
    "gsm8k":     [correctness_reward_func, int_reward_func, soft_format_reward_func],
    "mbpp":      [mbpp_reward_func],
    "spider":    [spider_reward_func],
    "humaneval": [humaneval_reward_func],
    "svamp":     [svamp_reward_func],
    "countdown": [countdown_reward_func],
}


# ===========================================================================
# Eval logic (greedy generation + scoring)
# ===========================================================================

def extract_answer_text(text: str, dataset: str) -> str:
    """Best-effort answer extraction for scoring."""
    m = _XML_ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def score_completion(text: str, example: dict, dataset: str) -> float:
    if dataset == "gsm8k":
        pred = _extract_number(text)
        gt_str = example.get("answer", "")
        m = _HASH_NUM_RE.search(gt_str)
        try:
            gt_f = float(m.group(1)) if m else float(gt_str.split("####")[-1].strip())
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if pred is not None and abs(pred - gt_f) < 1e-3 else 0.0
    elif dataset == "mbpp":
        code = _extract_code(text)
        if code is None:
            return 0.0
        return _run_mbpp_tests(code, example["test_list"], example.get("test_setup_code", ""))
    elif dataset == "spider":
        m    = _SQL_ANSWER_RE.search(text)
        pred = _normalize_sql(m.group(1)) if m else ""
        gold = _normalize_sql(example["answer"])
        return 1.0 if pred == gold else 0.0
    elif dataset == "humaneval":
        code = _extract_code(text) or text
        return _run_humaneval_tests(code, example["test"], example["entry_point"])
    elif dataset == "svamp":
        pred = _extract_number(text)
        if pred is None:
            return 0.0
        try:
            gold = float(example["answer"])
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if abs(pred - gold) < 1e-3 else 0.0
    elif dataset == "countdown":
        m = _COUNTDOWN_ANSWER_RE.search(text)
        if m is None:
            return 0.0
        expr = m.group(1).strip()
        result = _safe_eval_expr(expr)
        if result is None:
            return 0.0
        used = _extract_used_numbers(expr)
        if used is None:
            return 0.0
        given_nums = example.get("nums", [])
        try:
            given_ints = [int(n) for n in given_nums]
            used_ints  = [int(n) for n in used]
        except (ValueError, TypeError):
            return 0.0
        if _Counter(used_ints) != _Counter(given_ints):
            return 0.0
        tgt = example.get("target", None)
        if tgt is None:
            return 0.0
        try:
            tgt_f = float(tgt)
        except (ValueError, TypeError):
            return 0.0
        return 1.0 if abs(result - tgt_f) < 1e-3 else 0.0
    return 0.0


@torch.no_grad()
def run_eval(model, tokenizer, eval_examples, dataset, args_cfg, device, log_fn):
    """
    Greedy evaluation loop. Returns mean reward.
    """
    model.eval()
    scores = []

    for i, example in enumerate(eval_examples):
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = enc["input_ids"].to(device)[:, -args_cfg.max_prompt_length:]

        with unwrap_model_for_generation(model, model.device) as unwrapped:
            try:
                full_ids = unwrapped.generate(prompt_ids) if hasattr(unwrapped, 'generate') else None
            except Exception:
                full_ids = None

        # Use the trainer's generate method via a temp DiffuGRPOTrainer instance
        # For eval, we just call the model's built-in diffusion sampling directly
        if full_ids is None:
            # Fallback: call the sampling loop manually (copied from d1)
            full_ids = _diffusion_generate(
                model=unwrapped,
                prompt_ids=prompt_ids,
                gen_length=args_cfg.max_completion_length,
                block_length=args_cfg.block_length,
                steps=args_cfg.diffusion_steps,
                mask_id=_MASK_ID,
            )

        prompt_len  = prompt_ids.size(1)
        comp_text   = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
        score       = score_completion(comp_text, example, dataset)
        scores.append(score)

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_examples):
            log_fn(f"  eval {i+1}/{len(eval_examples)}  running_mean={np.mean(scores):.4f}")

    model.train()
    return float(np.mean(scores))


@torch.no_grad()
def _diffusion_generate(model, prompt_ids, gen_length, block_length, steps, mask_id):
    """Minimal diffusion sampling (greedy, temperature=0) for eval."""
    import torch.nn.functional as F
    device = prompt_ids.device
    bs     = prompt_ids.size(0)
    x      = torch.full((bs, prompt_ids.size(1) + gen_length), mask_id,
                        dtype=torch.long, device=device)
    x[:, :prompt_ids.size(1)] = prompt_ids.clone()

    num_blocks      = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    with torch.cuda.amp.autocast(enabled=True):
        for nb in range(num_blocks):
            start = prompt_ids.size(1) + nb * block_length
            end   = prompt_ids.size(1) + (nb + 1) * block_length

            block_mask = x[:, start:end] == mask_id
            # Num tokens to transfer per step
            mask_num = block_mask.sum(dim=1, keepdim=True)
            base = mask_num // steps_per_block
            rem  = mask_num % steps_per_block
            num_transfer = base.expand(-1, steps_per_block).clone()
            idx_mask = torch.arange(steps_per_block, device=device).unsqueeze(0) < rem
            num_transfer[idx_mask] += 1
            num_transfer = num_transfer.to(torch.int64)

            for i in range(steps_per_block):
                mask_index = x == mask_id
                logits = model(x).logits
                x0     = torch.argmax(logits, dim=-1)
                p      = F.softmax(logits.float(), dim=-1)
                x0_p   = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, end:] = -float("inf")

                x0   = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

                transfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(bs):
                    n = num_transfer[j, i].item()
                    if n > 0:
                        _, sel = torch.topk(conf[j], k=int(n))
                        transfer[j, sel] = True
                x[transfer] = x0[transfer]

    return x


# ===========================================================================
# EvalCallback for periodic eval during training
# ===========================================================================

class FullEvalCallback(TrainerCallback):
    """Run greedy eval on a fixed eval set periodically during training."""

    def __init__(self, trainer_ref, eval_examples, tokenizer, dataset, eval_every, log_fn):
        super().__init__()
        self.trainer_ref   = trainer_ref
        self.eval_examples = eval_examples
        self.tokenizer     = tokenizer
        self.dataset       = dataset
        self.eval_every    = eval_every
        self.log_fn        = log_fn
        self.history       = []  # (step, mean_score)

    @torch.no_grad()
    def _run_eval(self, global_step: int) -> float:
        trainer = self.trainer_ref
        args    = trainer.args
        device  = trainer.accelerator.device

        trainer.model.eval()
        scores = []

        for example in self.eval_examples:
            prompt_text = self.tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )
            prompt_ids = enc["input_ids"].to(device)[:, -args.max_prompt_length:]

            with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as unwrapped:
                full_ids = trainer.generate(
                    model=unwrapped,
                    prompt=prompt_ids,
                    steps=args.diffusion_steps,
                    gen_length=args.max_completion_length,
                    block_length=args.block_length,
                    temperature=0.0,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                )

            comp_text = self.tokenizer.decode(
                full_ids[0, prompt_ids.size(1):], skip_special_tokens=True
            )
            scores.append(score_completion(comp_text, example, self.dataset))

        trainer.model.train()
        mean_s = float(np.mean(scores))
        self.history.append((global_step, mean_s))
        self.log_fn(
            f"  [EvalCallback step={global_step}]  mean_score={mean_s:.4f}"
            f"  (n={len(self.eval_examples)})"
        )
        return mean_s

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_every == 0:
            self._run_eval(state.global_step)
        return control


# ===========================================================================
# LoRA helpers
# ===========================================================================

def _patch_llada_config(config):
    """LLaDAConfig doesn't define use_cache; patch it so modeling_llada.py can read it."""
    if not hasattr(config, "use_cache"):
        config.update({"use_cache": False})


def build_base_model(model_path: str, device):
    """Load LLaDA-8B base model (no LoRA). Returns (base_model, tokenizer, peft_config).
    TRL 1.x expects the BASE (unwrapped) model + peft_config; it applies LoRA internally.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    _patch_llada_config(base.config)

    peft_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05,
    )
    return base, tokenizer, peft_config


# Keep old name as alias for any remaining callers
build_lora_model = build_base_model


def build_baseline_model(model_path: str, device):
    """Load base model only (no LoRA, no training). For zero-shot eval."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    _patch_llada_config(model.config)
    model.eval()
    return model, tokenizer


# ===========================================================================
# Config builder
# ===========================================================================

def build_config(method: str, dataset: str, output_dir: str, seed: int,
                 num_train_examples: int, gen_length: int = 256) -> DiffuGRPOConfig:
    """
    Build DiffuGRPOConfig for 1-epoch training.
    max_steps=-1 means use num_train_epochs instead.
    gen_length controls max_completion_length; block_length=32 is fixed.
    """
    cfg = DiffuGRPOConfig(
        output_dir=output_dir,
        # 1-epoch training
        max_steps=-1,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_generations=4,
        # generation_batch_size matches total effective batch to avoid sub-batching issues
        generation_batch_size=4,
        learning_rate=1e-6,
        beta=0.04,
        epsilon=0.2,
        max_completion_length=gen_length,
        max_prompt_length=256,
        diffusion_steps=64,
        block_length=32,
        temperature=0.9,
        mask_id=_MASK_ID,
        remasking="low_confidence",
        cfg_scale=0.0,
        eval_strategy="no",
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        seed=seed,
        dataloader_drop_last=True,
        report_to=[],
    )
    # TRL 1.6.0 GRPOTrainer accesses many GRPOConfig attrs that DiffuGRPOConfig
    # (which extends TrainingArguments, not GRPOConfig) doesn't define.
    # Patch all missing attrs with safe defaults.
    _GRPO_DEFAULTS = {
        "steps_per_generation":         cfg.num_generations,
        "max_tool_calling_iterations":  None,
        "num_generations_eval":         None,
        "disable_dropout":              False,
        "cast_lm_head_to_fp32":        False,
        "shuffle_dataset":              True,
        "pad_to_multiple_of":           None,
        "generation_kwargs":            None,
        "chat_template_kwargs":         None,
        "vllm_mode":                    "colocate",
        "vllm_model_impl":              "vllm",
        "vllm_enable_sleep_mode":       False,
        "vllm_structured_outputs_regex": None,
        "vllm_server_base_url":         None,
        "vllm_server_host":             "0.0.0.0",
        "vllm_server_port":             8000,
        "vllm_server_timeout":          240.0,
        "vllm_group_port":              51216,
        "vllm_max_model_length":        None,
        "vllm_tensor_parallel_size":    1,
        "delta":                        None,
        "epsilon_high":                 None,
        "importance_sampling_level":    "token",
        "multi_objective_aggregation":  "sum_then_normalize",
        "scale_rewards":                "group",
        "loss_type":                    "dapo",
        "mask_truncated_completions":   False,
        "top_entropy_quantile":         1.0,
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": "sequence_mask",
        "vllm_importance_sampling_clip_max": 3.0,
        "vllm_importance_sampling_clip_min": None,
        "off_policy_mask_threshold":    None,
        "use_bias_correction_kl":       False,
        "num_completions_to_print":     None,
        "log_unique_prompts":           False,
        "log_completions_hub_repo":     None,
        "use_transformers_paged":       False,
        "use_liger_kernel":             False,
        "sapo_temperature_neg":         1.05,
        "sapo_temperature_pos":         1.0,
        "vespo_k_pos": 2.0, "vespo_lambda_pos": 3.0,
        "vespo_k_neg": 3.0, "vespo_lambda_neg": 2.0,
    }
    for attr, default in _GRPO_DEFAULTS.items():
        if not hasattr(cfg, attr):
            object.__setattr__(cfg, attr, default)
    return cfg


# ===========================================================================
# Method runners
# ===========================================================================

def run_baseline(dataset: str, eval_examples: list, tokenizer, output_dir: str,
                 log_fn, gen_length: int = 256) -> dict:
    """Zero-shot eval on pretrained LLaDA-8B (no training, no LoRA)."""
    log_fn(f"[baseline] Zero-shot eval on {len(eval_examples)} examples ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_baseline_model(_MODEL_PATH, device)

    # Build a minimal trainer-like object just for generate()
    # Use _diffusion_generate directly
    model.eval()
    scores = []
    t0     = time.time()

    for i, example in enumerate(eval_examples):
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = enc["input_ids"].to(device)[:, -256:]  # max_prompt_length=256

        full_ids  = _diffusion_generate(model, prompt_ids,
                                        gen_length=gen_length, block_length=32,
                                        steps=64, mask_id=_MASK_ID)
        comp_text = tokenizer.decode(full_ids[0, prompt_ids.size(1):], skip_special_tokens=True)
        scores.append(score_completion(comp_text, example, dataset))

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_examples):
            log_fn(f"  {i+1}/{len(eval_examples)}  running_mean={np.mean(scores):.4f}")

    elapsed   = time.time() - t0
    mean_score = float(np.mean(scores))

    result = {
        "method":      "baseline",
        "dataset":     dataset,
        "mean_score":  mean_score,
        "n_eval":      len(eval_examples),
        "eval_time_s": elapsed,
        "eval_history": [],
        "train_time_s": 0.0,
        "n_train":      0,
    }
    log_fn(f"[baseline] FINAL mean_score={mean_score:.4f}  time={elapsed/3600:.2f}h")
    return result


def run_training(method: str, dataset: str, train_dataset, eval_examples: list,
                 tokenizer, peft_config, output_dir: str, seed: int, log_fn,
                 max_value_states: int = 4, gen_length: int = 256) -> dict:
    """Train for 1 epoch then eval. method ∈ {diffu_grpo, stage2}.
    TRL 1.x: pass BASE model + peft_config; trainer applies LoRA internally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load base model (NOT wrapped with get_peft_model — TRL 1.x does this itself)
    model, tok2, peft_config = build_base_model(_MODEL_PATH, device)
    # tokenizer already passed in; use it (tok2 is the same, just ignore it)

    n_train = len(train_dataset)
    log_fn(f"[{method}] Training on {n_train} examples for 1 epoch ...")
    log_fn(f"[{method}] Expected steps: {n_train} (per_device_bs=1)")

    cfg = build_config(method, dataset, os.path.join(output_dir, "checkpoints"),
                       seed, n_train, gen_length=gen_length)

    reward_funcs = TRAIN_REWARD_FUNCS[dataset]

    _CW_KWARGS = dict(credit_alpha=1.0, credit_eps=1e-6,
                      credit_clip_min=0.25, credit_clip_max=4.0)
    _DV_KWARGS = dict(value_hidden_size=256, value_mlp_layers=2,
                      critic_lr=5e-6, critic_loss_coef=0.5, delta_v_gate=0.01,
                      max_value_states=max_value_states)

    extra_kwargs: dict = {}
    if method == "cw_grpo":
        # confidence weighting only, no value head
        extra_kwargs.update(_CW_KWARGS)
    elif method == "delta_v_only":
        # delta-V only, no confidence weighting
        extra_kwargs.update(_CW_KWARGS)           # needed by CWGRPOTrainer base
        extra_kwargs.update(_DV_KWARGS)
        extra_kwargs["use_confidence_weight"] = False
    elif method == "stage2":
        # delta-V + confidence weighting (full method)
        extra_kwargs.update(_CW_KWARGS)
        extra_kwargs.update(_DV_KWARGS)
        extra_kwargs["use_confidence_weight"] = True

    trainer_cls = {
        "diffu_grpo":   DiffuGRPOTrainer,
        "cw_grpo":      CWGRPOTrainer,
        "delta_v_only": ValueCreditTrainer,
        "stage2":       ValueCreditTrainer,
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

    # -----------------------------------------------------------------------
    # TRL 1.6.0 → d1 compatibility patches (applied once after init)
    # d1's DiffuGRPOTrainer was written for TRL 0.15.x; TRL 1.6.0 renamed
    # / removed several instance attributes that d1 accesses directly.
    # -----------------------------------------------------------------------
    # 1. max_prompt_length: removed from GRPOTrainer instance
    if not hasattr(trainer, "max_prompt_length") or trainer.max_prompt_length is None:
        trainer.max_prompt_length = cfg.max_prompt_length
    # 2. _buffered_inputs: was a list, now None in 1.6.0
    if not hasattr(trainer, "_buffered_inputs") or trainer._buffered_inputs is None:
        ga = max(1, cfg.gradient_accumulation_steps)
        trainer._buffered_inputs = [None] * ga
    # 3. epsilon: renamed to epsilon_low in TRL 1.6.0
    if not hasattr(trainer, "epsilon"):
        trainer.epsilon = getattr(trainer, "epsilon_low", cfg.epsilon)
    # 4. log_completions: might not be set in 1.6.0 init path
    if not hasattr(trainer, "log_completions"):
        trainer.log_completions = getattr(cfg, "log_completions", False)

    # Periodic eval callback: every max(100, n_train//10) steps
    eval_every = max(100, n_train // 10)
    # Use a 100-example subset for periodic eval (fast)
    import random as _rnd
    _rnd.seed(seed)
    periodic_eval = _rnd.sample(eval_examples, min(100, len(eval_examples)))
    eval_cb = FullEvalCallback(
        trainer_ref=trainer,
        eval_examples=periodic_eval,
        tokenizer=tokenizer,
        dataset=dataset,
        eval_every=eval_every,
        log_fn=log_fn,
    )
    trainer.add_callback(eval_cb)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    log_fn(f"[{method}] Training done in {train_time/3600:.2f}h")

    # Full final eval
    log_fn(f"[{method}] Running full eval on {len(eval_examples)} examples ...")
    model.eval()
    scores = []
    t1     = time.time()

    for i, example in enumerate(eval_examples):
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = enc["input_ids"].to(device)[:, -cfg.max_prompt_length:]

        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator) as unwrapped:
            full_ids = trainer.generate(
                model=unwrapped,
                prompt=prompt_ids,
                steps=cfg.diffusion_steps,
                gen_length=cfg.max_completion_length,
                block_length=cfg.block_length,
                temperature=0.0,
                cfg_scale=cfg.cfg_scale,
                remasking=cfg.remasking,
                mask_id=cfg.mask_id,
            )

        comp_text = tokenizer.decode(full_ids[0, prompt_ids.size(1):], skip_special_tokens=True)
        scores.append(score_completion(comp_text, example, dataset))

        if (i + 1) % 50 == 0 or (i + 1) == len(eval_examples):
            log_fn(f"  eval {i+1}/{len(eval_examples)}  mean={np.mean(scores):.4f}")

    eval_time  = time.time() - t1
    mean_score = float(np.mean(scores))

    result = {
        "method":       method,
        "dataset":      dataset,
        "mean_score":   mean_score,
        "n_eval":       len(eval_examples),
        "n_train":      n_train,
        "train_time_s": train_time,
        "eval_time_s":  eval_time,
        "eval_history": [(int(s), float(v)) for s, v in eval_cb.history],
    }
    log_fn(f"[{method}] FINAL mean_score={mean_score:.4f}  "
           f"train={train_time/3600:.2f}h  eval={eval_time/3600:.2f}h")
    return result


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   choices=["gsm8k", "mbpp", "spider", "humaneval", "svamp", "countdown"],
                   required=True)
    p.add_argument("--method",
                   choices=["baseline", "diffu_grpo", "cw_grpo", "delta_v_only", "stage2"],
                   required=True)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROJECT_ROOT, "experiments", "outputs", "official_9exp"))
    p.add_argument("--model_path", type=str, default=_MODEL_PATH)
    p.add_argument("--max_train_examples", type=int, default=100_000,
                   help="Cap training data size")
    p.add_argument("--max_value_states", type=int, default=4,
                   help="Max block boundaries to evaluate V at per trajectory (stage2 only)."
                        " 2 = very fast; 4 = default; num_blocks = full.")
    p.add_argument("--gen_length", type=int, default=256,
                   choices=[128, 256, 512],
                   help="Max completion length (tokens). block_length=32 fixed.")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    exp_dir = os.path.join(args.output_dir, args.dataset, f"gl{args.gen_length}", args.method)
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "run.log")
    log_f    = open(log_path, "a", buffering=1)

    def log_fn(msg: str):
        ts  = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    log_fn("=" * 70)
    log_fn(f"EXPERIMENT: dataset={args.dataset}  method={args.method}  gen_length={args.gen_length}")
    log_fn(f"seed={args.seed}  max_train={args.max_train_examples}")
    log_fn("=" * 70)

    # Load dataset
    log_fn(f"Loading dataset: {args.dataset} ...")
    train_ds, eval_examples = DATASET_LOADERS[args.dataset](
        seed=args.seed, max_train=args.max_train_examples
    )
    log_fn(f"  train={len(train_ds)}  eval={len(eval_examples)}")

    # Load tokenizer (always needed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t_start = time.time()

    if args.method == "baseline":
        result = run_baseline(
            dataset=args.dataset,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            output_dir=exp_dir,
            log_fn=log_fn,
            gen_length=args.gen_length,
        )
    else:
        # peft_config is built inside run_training (base model loaded there too)
        result = run_training(
            method=args.method,
            dataset=args.dataset,
            train_dataset=train_ds,
            eval_examples=eval_examples,
            tokenizer=tokenizer,
            peft_config=None,  # built internally
            output_dir=exp_dir,
            seed=args.seed,
            log_fn=log_fn,
            max_value_states=args.max_value_states,
            gen_length=args.gen_length,
        )

    total_time = time.time() - t_start
    result["total_time_s"] = total_time
    result["gen_length"]   = args.gen_length

    out_path = os.path.join(exp_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    log_fn(f"\nResult saved → {out_path}")
    log_fn(f"Total time: {total_time/3600:.2f}h")
    log_fn("=" * 70)
    log_f.close()


if __name__ == "__main__":
    main()
