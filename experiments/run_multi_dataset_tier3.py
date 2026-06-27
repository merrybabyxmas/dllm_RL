#!/usr/bin/env python3
"""
Multi-dataset Tier 3: compare Diffu-GRPO / Stage1 / Stage2 on
  - GSM8K     (word problems, XML answer format)
  - Countdown (arithmetic expression, <answer> tag format)
  - MATH-500  (hard math, \\boxed{} format)
  - MBPP      (Python code generation, unit test execution)
  - Spider    (text-to-SQL, exact match; uses sql-create-context format)

Each dataset: 64 train / 32 eval / 300 steps / 4 rollouts.
Loads local files when HF Hub is unavailable.

Data files expected:
  data/mbpp.jsonl              - from google-research/google-research mbpp
  data/sql_create_context.json - from b-mc2/sql-create-context (Spider-format)

Usage:
  cd /home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl
  python experiments/run_multi_dataset_tier3.py
  python experiments/run_multi_dataset_tier3.py --datasets mbpp spider
"""
from __future__ import annotations

import argparse, copy, json, math, os, random, re, subprocess, sys, time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: d1 package and cc_rl package
# ---------------------------------------------------------------------------
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_D1_DIFFU_GRPO = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
_D1_DATASET    = "/home/dongwoo43/papers/paper_dllm/d1/dataset"
_EVAL_BASELINES = "/home/dongwoo43/papers/paper_dllm/d1/eval/eval_baselines"
_SRC_PATH      = os.path.join(_PROJECT_ROOT, "src")

for _p in [_D1_DIFFU_GRPO, _SRC_PATH]:
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
from diffu_grpo_trainer import DiffuGRPOTrainer   # noqa: E402
from diffu_grpo_config import DiffuGRPOConfig      # noqa: E402

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

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Hyper-parameters (same as run_tier3.py)
# ---------------------------------------------------------------------------
N_TRAIN               = 64
N_EVAL                = 32
MAX_STEPS             = 300
EVAL_EVERY            = 30
NUM_GENERATIONS       = 4
LORA_R                = 8
LORA_ALPHA            = 16
TEMPERATURE           = 0.9
DIFFUSION_STEPS       = 64
BLOCK_LENGTH          = 32
MAX_COMPLETION_LENGTH = 256
MAX_PROMPT_LENGTH     = 256
LEARNING_RATE         = 1e-6
CRITIC_LR             = 5e-6
MODEL_PATH            = "/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct"
MASK_ID               = 126336
_DATA_DIR             = os.path.join(_PROJECT_ROOT, "data")


# ===========================================================================
# Dataset loaders
# ===========================================================================

def load_gsm8k(seed, n_train, n_eval):
    """Load GSM8K from d1 data_utils."""
    # Import d1's data_utils for GSM8K
    d1_data_utils_path = os.path.join(_D1_DIFFU_GRPO, "data_utils.py")
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("d1_data_utils", d1_data_utils_path)
    _d1du = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_d1du)
    full_hf = _d1du.get_gsm8k_questions("train")
    # full_hf is an HF Dataset; convert to list of dicts
    full = [full_hf[i] for i in range(len(full_hf))]
    rng = random.Random(seed)
    idx = list(range(len(full)))
    rng.shuffle(idx)
    train = [full[i] for i in idx[:n_train]]
    eval_ = [full[i] for i in idx[n_train:n_train + n_eval]]
    return train, eval_


def load_countdown(seed, n_train, n_eval):
    path = os.path.join(_D1_DATASET, "countdown_cd3_test.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            nums = [int(x) for x in rec["input"].split(",")]
            target = int(rec["output"])
            prompt_text = (
                f"Using only the numbers {nums}, create an arithmetic expression that "
                f"evaluates to exactly {target}. You must use all numbers exactly once. "
                f"You may use +, -, *, /. Wrap your final expression inside "
                f"<answer>...</answer> tags."
            )
            examples.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "nums": nums,
                "target": target,
            })
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n_train], examples[n_train:n_train + n_eval]


def load_math500(seed, n_train, n_eval):
    import glob
    questions = {}
    for fp in glob.glob(os.path.join(_EVAL_BASELINES, "math500_*.json")):
        data = json.load(open(fp))
        for g in data.get("generations", []):
            q = g.get("question", "")
            if q and q not in questions:
                questions[q] = g.get("ground_truth", "")

    examples = []
    for q, a in questions.items():
        prompt_text = (
            "Solve the following math problem step by step. "
            "Put your final answer inside \\boxed{...}.\n\n" + q
        )
        examples.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": a,
        })

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n_train], examples[n_train:n_train + n_eval]


def load_mbpp(seed, n_train, n_eval):
    """Load MBPP from data/mbpp.jsonl (official Google Research repo).

    Download:
      curl -L https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl \
           -o data/mbpp.jsonl
    """
    path = os.path.join(_DATA_DIR, "mbpp.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MBPP data not found at {path}. "
            "Run: curl -L https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl -o data/mbpp.jsonl"
        )
    examples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            tests = "\n".join(rec["test_list"])
            setup = rec.get("test_setup_code", "").strip()
            prompt_text = (
                "You are a Python coding expert. Solve the following programming task.\n\n"
                f"Task: {rec['text']}\n\n"
                f"Your function must pass these tests:\n{tests}\n\n"
                "Write your solution inside <answer>```python\n...\n```</answer> tags."
            )
            examples.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "test_list": rec["test_list"],
                "test_setup_code": setup,
                "task_id": rec["task_id"],
            })
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n_train], examples[n_train:n_train + n_eval]


def load_spider(seed, n_train, n_eval):
    """Load Spider-format SQL data from data/sql_create_context.json.

    Uses b-mc2/sql-create-context (Spider-compatible question+schema+SQL format).
    Download:
      python -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('https://huggingface.co/datasets/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json').read()); json.dump(d,open('data/sql_create_context.json','w'))"
    """
    path = os.path.join(_DATA_DIR, "sql_create_context.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spider data not found at {path}. "
            "See docstring for download instructions."
        )
    with open(path) as f:
        raw = json.load(f)
    examples = []
    for rec in raw:
        prompt_text = (
            "Convert the following natural language question into a SQL query.\n\n"
            f"Schema:\n{rec['context']}\n\n"
            f"Question: {rec['question']}\n\n"
            "Write only the SQL query inside <answer>...</answer> tags."
        )
        examples.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": rec["answer"].strip(),
        })
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n_train], examples[n_train:n_train + n_eval]


# ===========================================================================
# Reward functions  (signature: fn(example, completions) -> List[float])
# These are the "internal" versions used by eval and smoke tests.
# ===========================================================================

# ---- GSM8K -----------------------------------------------------------------
_XML_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_REASON_FMT_RE = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)

def gsm8k_reward(example, completions):
    answer = example.get("answer", "")
    rewards = []
    for c in completions:
        m = _XML_ANSWER_RE.search(c)
        extracted = m.group(1).strip() if m else ""
        r = 2.0 if extracted == answer else 0.0
        r += 0.5 if extracted.isdigit() else 0.0
        r += 0.5 if _REASON_FMT_RE.search(c, re.DOTALL) else 0.0
        rewards.append(r)
    return rewards

def gsm8k_eval_reward(completion, example):
    return gsm8k_reward(example, [completion])[0]


# ---- Countdown -------------------------------------------------------------
_CD_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def _validate_eq(eq, nums):
    try:
        found = [int(n) for n in re.findall(r"\d+", eq)]
        return sorted(found) == sorted(nums)
    except Exception:
        return False

def _eval_eq(eq):
    try:
        allowed = re.compile(r"^[\d+\-*/().\s]+$")
        if not allowed.match(eq):
            return None
        return eval(eq, {"__builtins__": None}, {})
    except Exception:
        return None

def countdown_reward(example, completions):
    target = example["target"]
    nums   = example["nums"]
    rewards = []
    for c in completions:
        m = _CD_ANSWER_RE.search(c)
        if m is None:
            rewards.append(0.0); continue
        eq = m.group(1).strip()
        if not _validate_eq(eq, nums):
            rewards.append(0.1); continue
        result = _eval_eq(eq)
        if result is not None and abs(result - target) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.1)
    return rewards

def countdown_eval_reward(completion, example):
    return countdown_reward(example, [completion])[0]


# ---- MATH-500 --------------------------------------------------------------
def _last_boxed(text):
    """Extract content of last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{") - 1  # index of '{'
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i].strip()
    return None

def _normalize_math(s):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\!", "").replace("\\ ", " ")
    return s

def math500_reward(example, completions):
    gt = _normalize_math(example.get("answer", ""))
    rewards = []
    for c in completions:
        pred = _last_boxed(c)
        if pred is None:
            rewards.append(0.0); continue
        pred_n = _normalize_math(pred)
        rewards.append(1.0 if pred_n == gt else 0.0)
    return rewards

def math500_eval_reward(completion, example):
    return math500_reward(example, [completion])[0]


# ---- MBPP ------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(
    r"<answer>\s*```python\s*(.*?)```\s*</answer>", re.DOTALL
)
_CODE_BLOCK_BARE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

def _extract_code(completion):
    m = _CODE_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip()
    m = _CODE_BLOCK_BARE_RE.search(completion)
    if m:
        return m.group(1).strip()
    m = _XML_ANSWER_RE.search(completion)
    if m:
        return m.group(1).strip()
    return None

def _run_mbpp_tests(code, test_list, setup=""):
    """Execute code + unit tests in a subprocess. Returns 1.0 if all pass, else 0.0."""
    full = (setup + "\n\n" if setup else "") + code + "\n\n" + "\n".join(test_list)
    try:
        result = subprocess.run(
            [sys.executable, "-c", full],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0

def mbpp_reward(example, completions):
    test_list = example["test_list"]
    setup     = example.get("test_setup_code", "")
    rewards   = []
    for c in completions:
        code = _extract_code(c)
        if code is None:
            rewards.append(0.0)
        else:
            rewards.append(_run_mbpp_tests(code, test_list, setup))
    return rewards

def mbpp_eval_reward(completion, example):
    return mbpp_reward(example, [completion])[0]


# ---- Spider (text-to-SQL, exact normalized match) --------------------------
_SQL_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def _normalize_sql(sql):
    sql = sql.strip().rstrip(";").lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("( ", "(").replace(" )", ")")
    return sql

def spider_reward(example, completions):
    gold = _normalize_sql(example.get("answer", ""))
    rewards = []
    for c in completions:
        m = _SQL_ANSWER_RE.search(c)
        pred = _normalize_sql(m.group(1)) if m else ""
        if not pred:
            rewards.append(0.0); continue
        if pred == gold:
            rewards.append(1.0); continue
        # Partial token-F1 as soft reward
        gold_toks = set(gold.split())
        pred_toks = set(pred.split())
        if not pred_toks:
            rewards.append(0.0); continue
        prec = len(gold_toks & pred_toks) / len(pred_toks)
        rec  = len(gold_toks & pred_toks) / len(gold_toks) if gold_toks else 0.0
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        rewards.append(round(f1, 3))
    return rewards

def spider_eval_reward(completion, example):
    return spider_reward(example, [completion])[0]


# ===========================================================================
# Dataset config registry
# ===========================================================================
DATASET_CONFIGS = {
    "gsm8k": {
        "name":        "GSM8K",
        "loader":      load_gsm8k,
        "reward_fn":   gsm8k_reward,
        "eval_reward": gsm8k_eval_reward,
        "max_reward":  3.0,
    },
    "countdown": {
        "name":        "Countdown",
        "loader":      load_countdown,
        "reward_fn":   countdown_reward,
        "eval_reward": countdown_eval_reward,
        "max_reward":  1.0,
    },
    "math500": {
        "name":        "MATH-500",
        "loader":      load_math500,
        "reward_fn":   math500_reward,
        "eval_reward": math500_eval_reward,
        "max_reward":  1.0,
    },
    "mbpp": {
        "name":        "MBPP",
        "loader":      load_mbpp,
        "reward_fn":   mbpp_reward,
        "eval_reward": mbpp_eval_reward,
        "max_reward":  1.0,
    },
    "spider": {
        "name":        "Spider",
        "loader":      load_spider,
        "reward_fn":   spider_reward,
        "eval_reward": spider_eval_reward,
        "max_reward":  1.0,
    },
}


# ===========================================================================
# Official reward function wrappers
# (DiffuGRPOTrainer signature: fn(prompts, completions, **kwargs) -> List[float])
# Extra dataset columns arrive as **kwargs key-per-column, each a list of values.
# ===========================================================================

def make_gsm8k_reward_fn():
    def fn(prompts, completions, answer, **kwargs):
        rewards = []
        for c, a in zip(completions, answer):
            rewards.append(gsm8k_reward({"answer": a}, [c])[0])
        return rewards
    return fn


def make_countdown_reward_fn():
    def fn(prompts, completions, nums, target, **kwargs):
        rewards = []
        for c, n, t in zip(completions, nums, target):
            rewards.append(countdown_reward({"nums": n, "target": t}, [c])[0])
        return rewards
    return fn


def make_math500_reward_fn():
    def fn(prompts, completions, answer, **kwargs):
        rewards = []
        for c, a in zip(completions, answer):
            rewards.append(math500_reward({"answer": a}, [c])[0])
        return rewards
    return fn


def make_mbpp_reward_fn():
    def fn(prompts, completions, test_list, test_setup_code, **kwargs):
        rewards = []
        for c, tl, ts in zip(completions, test_list, test_setup_code):
            rewards.append(
                mbpp_reward({"test_list": tl, "test_setup_code": ts}, [c])[0]
            )
        return rewards
    return fn


def make_spider_reward_fn():
    def fn(prompts, completions, answer, **kwargs):
        rewards = []
        for c, a in zip(completions, answer):
            rewards.append(spider_reward({"answer": a}, [c])[0])
        return rewards
    return fn


_REWARD_FN_FACTORIES = {
    "gsm8k":    make_gsm8k_reward_fn,
    "countdown": make_countdown_reward_fn,
    "math500":  make_math500_reward_fn,
    "mbpp":     make_mbpp_reward_fn,
    "spider":   make_spider_reward_fn,
}


# ===========================================================================
# Convert examples to HF Dataset
# ===========================================================================

def examples_to_hf_dataset(examples, ds_key):
    """Convert our example dicts to HF Dataset with all needed columns."""
    rows = []
    for ex in examples:
        row = {"prompt": ex["prompt"]}
        if ds_key in ("gsm8k", "math500", "spider"):
            row["answer"] = ex.get("answer", "")
        elif ds_key == "countdown":
            row["nums"] = ex["nums"]
            row["target"] = ex["target"]
        elif ds_key == "mbpp":
            row["test_list"] = ex["test_list"]
            row["test_setup_code"] = ex.get("test_setup_code", "")
        rows.append(row)
    return Dataset.from_list(rows)


# ===========================================================================
# LoRA reset
# ===========================================================================

def reset_lora_weights(model):
    for _name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for key in module.lora_A:
                nn.init.kaiming_uniform_(module.lora_A[key].weight, a=math.sqrt(5))
            for key in module.lora_B:
                nn.init.zeros_(module.lora_B[key].weight)


# ===========================================================================
# EvalCallback
# ===========================================================================

class EvalCallback(TrainerCallback):
    """
    Runs greedy diffusion generation on a fixed eval set every `eval_every`
    training steps and records mean reward.

    Stores results in self.eval_history as List[(step, mean_reward)].
    """

    def __init__(self, trainer_ref, eval_examples, tokenizer, eval_reward_fn, eval_every=30):
        super().__init__()
        self.trainer_ref    = trainer_ref
        self.eval_examples  = eval_examples
        self.tokenizer      = tokenizer
        self.eval_reward_fn = eval_reward_fn   # fn(completion_str, example) -> float
        self.eval_every     = eval_every
        self.eval_history   = []               # List[(step, mean_reward)]

    @torch.no_grad()
    def _run_eval(self, global_step):
        trainer = self.trainer_ref
        args    = trainer.args
        device  = trainer.accelerator.device

        all_rewards = []
        model = trainer.model
        model.eval()

        for example in self.eval_examples:
            # Tokenize prompt
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
                    temperature=0.0,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                )  # [1, prompt_len + comp_len]

            prompt_length   = prompt_ids.size(1)
            completion_text = self.tokenizer.decode(
                full_ids[0, prompt_length:], skip_special_tokens=True
            )
            all_rewards.append(self.eval_reward_fn(completion_text, example))

        model.train()
        mean_r = float(np.mean(all_rewards)) if all_rewards else 0.0
        self.eval_history.append((global_step, mean_r))
        return mean_r

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0 and state.global_step > 0:
            mean_r = self._run_eval(state.global_step)
            print(
                f"  [EvalCallback step={state.global_step:4d}]"
                f"  eval_reward={mean_r:.4f}"
            )
        return control


# ===========================================================================
# Build DiffuGRPOConfig
# ===========================================================================

def build_config(method, ds_key, output_base, seed):
    return DiffuGRPOConfig(
        output_dir=os.path.join(output_base, ds_key, method),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        num_generations=NUM_GENERATIONS,
        generation_batch_size=1,
        learning_rate=LEARNING_RATE,
        beta=0.04,
        epsilon=0.2,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        diffusion_steps=DIFFUSION_STEPS,
        block_length=BLOCK_LENGTH,
        temperature=TEMPERATURE,
        mask_id=MASK_ID,
        remasking="low_confidence",
        cfg_scale=0.0,
        eval_strategy="no",
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        seed=seed,
        dataloader_drop_last=False,
        report_to=[],
    )


# ===========================================================================
# Single-method runner (official trainer)
# ===========================================================================

def run_method_official(
    method,
    model,
    tokenizer,
    train_dataset,
    eval_examples,
    peft_config,
    ds_key,
    eval_reward_fn,
    output_base,
    seed,
):
    """
    Train one method for MAX_STEPS steps using the official trainer hierarchy.

    Returns dict with keys:
        method, final_eval_reward, eval_history, train_time_s,
        grad_steps_nonzero (approximated as MAX_STEPS), ema_expvar_final,
        stage2_adv_count, training_rewards
    """
    label = {
        "diffu_grpo": "A: Diffu-GRPO",
        "stage1":     "B: Stage 1 (CW)",
        "stage2":     "C: Stage 2 (delta-V)",
    }[method]
    print(f"\n[Method {label}]")

    # Reset LoRA to fresh state
    reset_lora_weights(model)
    print("  LoRA weights re-initialized to identity.")

    cfg = build_config(method, ds_key, output_base, seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Official reward function wrapper for this dataset
    reward_fn_official = _REWARD_FN_FACTORIES[ds_key]()

    # Trainer-specific extra kwargs
    extra_kwargs = {}
    if method in ("stage1", "stage2"):
        extra_kwargs.update(
            credit_alpha=1.0,
            credit_eps=1e-6,
            credit_clip_min=0.25,
            credit_clip_max=4.0,
        )
    if method == "stage2":
        extra_kwargs.update(
            value_hidden_size=1024,
            value_mlp_layers=2,
            critic_lr=CRITIC_LR,
            critic_loss_coef=0.5,
            delta_v_gate=0.01,
        )

    # Select trainer class
    trainer_cls = {
        "diffu_grpo": DiffuGRPOTrainer,
        "stage1":     CWGRPOTrainer,
        "stage2":     ValueCreditTrainer,
    }[method]

    trainer = trainer_cls(
        model=model,
        reward_funcs=[reward_fn_official],
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
        eval_reward_fn=eval_reward_fn,
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

    torch.cuda.empty_cache()

    return {
        "method":             method,
        "final_eval_reward":  final_r,
        "eval_history":       eval_cb.eval_history,
        "train_time_s":       train_time,
        "grad_steps_nonzero": MAX_STEPS,   # trainer handles internally
        "ema_expvar_final":   0.0,         # not exposed by official trainer
        "stage2_adv_count":   0,           # not exposed by official trainer
        "training_rewards":   [],
    }


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROJECT_ROOT, "experiments", "outputs", "multi_dataset_tier3"))
    p.add_argument("--model_path", type=str, default=MODEL_PATH)
    p.add_argument("--datasets",   nargs="+",
                   choices=list(DATASET_CONFIGS.keys()),
                   default=["gsm8k", "countdown", "math500"],
                   help="Which datasets to run")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run data loader + reward function smoke tests (no GPU needed)")
    p.add_argument("--methods",    nargs="+",
                   choices=["diffu_grpo", "stage1", "stage2"],
                   default=["diffu_grpo", "stage1", "stage2"])
    return p.parse_args()


def run_smoke_tests(datasets, seed=42):
    """Verify data loaders and reward functions without loading the model."""
    import traceback
    print("=" * 60)
    print("SMOKE TESTS (no GPU / model required)")
    print("=" * 60)
    all_pass = True
    for ds_key in datasets:
        cfg = DATASET_CONFIGS[ds_key]
        name = cfg["name"]
        print(f"\n[{name}]")
        try:
            train, eval_ = cfg["loader"](seed, 8, 4)
            assert len(train) == 8 and len(eval_) == 4, "wrong split size"
            ex = train[0]
            assert "prompt" in ex and isinstance(ex["prompt"], list), "missing prompt"
            print(f"  loader: OK  (8 train / 4 eval loaded)")

            # Reward function test with a dummy completion
            dummy_completions = ["dummy answer"] * 4
            rewards = cfg["reward_fn"](ex, dummy_completions)
            assert len(rewards) == 4, "reward list length mismatch"
            assert all(isinstance(r, float) for r in rewards), "non-float reward"
            print(f"  reward_fn: OK  (dummy rewards={rewards[:2]})")

            # Eval reward test
            er = cfg["eval_reward"]("dummy", ex)
            assert isinstance(er, float), "eval_reward not float"
            print(f"  eval_reward: OK  (dummy eval={er:.3f})")

            # Official reward fn wrapper test
            official_fn = _REWARD_FN_FACTORIES[ds_key]()
            prompts_dummy = [ex["prompt"]] * 2
            comps_dummy   = ["dummy"] * 2
            if ds_key in ("gsm8k", "math500", "spider"):
                off_r = official_fn(prompts_dummy, comps_dummy, answer=[ex.get("answer", "")] * 2)
            elif ds_key == "countdown":
                off_r = official_fn(prompts_dummy, comps_dummy,
                                    nums=[ex["nums"]] * 2, target=[ex["target"]] * 2)
            elif ds_key == "mbpp":
                off_r = official_fn(prompts_dummy, comps_dummy,
                                    test_list=[ex["test_list"]] * 2,
                                    test_setup_code=[ex.get("test_setup_code", "")] * 2)
            assert len(off_r) == 2 and all(isinstance(v, float) for v in off_r), \
                f"official reward fn bad output: {off_r}"
            print(f"  official_reward_fn: OK  (dummy={off_r})")

            # HF dataset conversion test
            hf_ds = examples_to_hf_dataset([ex], ds_key)
            assert "prompt" in hf_ds.column_names, "missing prompt column"
            print(f"  hf_dataset: OK  (columns={hf_ds.column_names})")

            # Dataset-specific functional tests
            if ds_key == "mbpp":
                fake_completion = (
                    "<answer>```python\ndef stub(): pass\n```</answer>"
                )
                r = mbpp_reward(ex, [fake_completion])
                print(f"  mbpp code exec: OK  (stub reward={r[0]:.1f})")

            elif ds_key == "spider":
                gold_sql = ex["answer"]
                perfect = f"<answer>{gold_sql}</answer>"
                r_perfect = spider_reward(ex, [perfect])
                r_wrong   = spider_reward(ex, ["<answer>SELECT 1</answer>"])
                assert r_perfect[0] == 1.0, f"perfect SQL reward should be 1.0, got {r_perfect[0]}"
                assert r_wrong[0] < 1.0, "wrong SQL should be < 1.0"
                print(f"  spider exact match: OK  (perfect={r_perfect[0]:.1f}, wrong={r_wrong[0]:.3f})")

            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()
            all_pass = False

    print(f"\n{'='*60}")
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}\n")
    if not all_pass:
        sys.exit(1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke_test:
        run_smoke_tests(args.datasets, args.seed)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Multi-Dataset Tier 3 (300 steps x 3 methods) — Official Trainer")
    print(f"Datasets : {args.datasets}")
    print(f"Methods  : {args.methods}")
    print(f"Config   : {N_TRAIN} train / {N_EVAL} eval / {MAX_STEPS} steps / {NUM_GENERATIONS} rollouts")
    print(f"Model    : LLaDA-8B-Instruct + LoRA(r={LORA_R})")
    print("=" * 60)

    print(f"\nLoading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ...")
    t0 = time.time()
    base_model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    base_model.config.use_cache = False
    print(f"Base model loaded in {time.time()-t0:.1f}s")

    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    all_results = {}

    for ds_key in args.datasets:
        cfg_info = DATASET_CONFIGS[ds_key]
        ds_name  = cfg_info["name"]
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*60}")

        print(f"Loading {ds_name} data...")
        train_examples, eval_examples = cfg_info["loader"](args.seed, N_TRAIN, N_EVAL)
        print(f"  {len(train_examples)} train / {len(eval_examples)} eval loaded")

        # Convert train examples to HF Dataset for the official trainer
        train_dataset = examples_to_hf_dataset(train_examples, ds_key)

        eval_reward_fn = cfg_info["eval_reward"]

        ds_results = {}
        for method in args.methods:
            result = run_method_official(
                method=method,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_examples=eval_examples,
                peft_config=peft_config,
                ds_key=ds_key,
                eval_reward_fn=eval_reward_fn,
                output_base=args.output_dir,
                seed=args.seed,
            )
            ds_results[method] = result

        all_results[ds_key] = ds_results

        # Per-dataset summary
        print(f"\n--- {ds_name} Summary (step {MAX_STEPS}) ---")
        print(f"{'Method':<25} {'eval_reward':>12}  {'time(s)':>10}  notes")
        print("-" * 65)
        grpo_r = ds_results.get("diffu_grpo", {}).get("final_eval_reward", float("nan"))
        for m in args.methods:
            r = ds_results[m]
            notes = ""
            diff = ""
            if m != "diffu_grpo" and not math.isnan(grpo_r):
                delta = (r["final_eval_reward"] - grpo_r) / (grpo_r + 1e-9) * 100
                diff = f"({delta:+.1f}%)"
            print(f"  {m:<23} {r['final_eval_reward']:>12.3f}  "
                  f"{r['train_time_s']:>10.1f}  {notes} {diff}")

    # Cross-dataset summary
    print(f"\n{'='*60}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    header = f"{'Method':<20}" + "".join(f"  {DATASET_CONFIGS[d]['name']:>10}" for d in args.datasets)
    print(header)
    print("-" * len(header))
    for method in args.methods:
        row = f"  {method:<18}"
        for ds_key in args.datasets:
            r = all_results[ds_key].get(method, {}).get("final_eval_reward", float("nan"))
            row += f"  {r:>10.3f}"
        print(row)

    # Save results
    out = {
        "seed":     args.seed,
        "n_train":  N_TRAIN,
        "n_eval":   N_EVAL,
        "max_steps": MAX_STEPS,
        "datasets": args.datasets,
        "methods":  args.methods,
        "results":  {
            ds: {
                m: {
                    k: (
                        [(int(s), float(v)) for s, v in r[k]]
                        if k == "eval_history"
                        else r[k]
                    )
                    for k, _ in r.items()
                    if k != "training_rewards"
                }
                for m, r in ds_res.items()
            }
            for ds, ds_res in all_results.items()
        },
    }
    out_path = os.path.join(args.output_dir, "multi_dataset_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
