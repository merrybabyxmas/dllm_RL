"""
Standalone GRPO training script for LLaDA-8B-Instruct.

Bypasses TRL entirely.  Implements a plain PyTorch training loop for three
methods:
  diffu_grpo : standard GRPO (group-relative policy optimisation)
  stage1     : GRPO + per-token confidence-weighted advantages
  stage2     : Stage 1 + value head (delta-V advantages)

Core diffusion primitives are copied verbatim from
  /home/dongwoo43/papers/paper_dllm/d1/diffu-grpo/diffu_grpo_trainer.py
to avoid any TRL dependency.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python experiments/train_standalone.py \\
      --method diffu_grpo \\
      --model_path /path/to/LLaDA-8B-Instruct \\
      --output_dir experiments/outputs/diffu_grpo_3000step \\
      --max_steps 3000

Environment: Python 3.10+, PyTorch 2.1+, transformers, peft, datasets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_compat(seed) -> None:
    """Python 3.13-safe seed setter.  Converts tensor seeds to Python int."""
    seed_val: int = int(seed.item()) if hasattr(seed, "item") else int(seed)
    random.seed(seed_val)
    np.random.seed(seed_val % (2 ** 32))
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)


# ---------------------------------------------------------------------------
# GSM8K data loading  (self-contained, no d1 import needed)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split: str = "train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]}],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


# ---------------------------------------------------------------------------
# Reward functions  (copied from d1/diffu-grpo/reward_func.py, no external dep)
# ---------------------------------------------------------------------------

def _extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(
    prompts, completions, answer, **kwargs
) -> List[float]:
    responses = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    extracted = [_extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]


def int_reward_func(completions, **kwargs) -> List[float]:
    responses = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    extracted = [_extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    import re
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    import re
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    def _count_xml(text: str) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
        return count

    responses = [c if isinstance(c, str) else c[0]["content"] for c in completions]
    return [_count_xml(r) for r in responses]


# ---------------------------------------------------------------------------
# Diffusion primitives  (copied from d1/diffu-grpo/diffu_grpo_trainer.py)
# ---------------------------------------------------------------------------

def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample from categorical via Gumbel-Max trick."""
    if temperature == 0.0:
        return logits
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(
    mask_index: torch.Tensor,  # [batch, block_len]  bool
    steps: int,
) -> torch.Tensor:
    """Precompute how many tokens to reveal at each denoising step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)      # [batch, 1]
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,           # [bs, prompt_len]
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> torch.Tensor:
    """
    Masked-diffusion generation for LLaDA-style models.

    Returns x: [bs, prompt_len + gen_length] with final token ids.
    """
    bs = prompt.shape[0]
    dtype = model.dtype
    device = prompt.device

    x = torch.full((bs, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id           # [bs, total_len]

    assert gen_length % block_length == 0, \
        f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    for num_block in range(num_blocks):
        start_idx = prompt.shape[1] + num_block * block_length
        end_idx   = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            torch.cuda.empty_cache()
            mask_index = x == mask_id

            with torch.cuda.amp.autocast(enabled=True):
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                del logits_with_noise

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
                else:
                    raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

                x0_p[:, end_idx:] = -float("inf")

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(bs):
                    n_tok = int(num_transfer[j, i].item())
                    if n_tok > 0:
                        _, sel = torch.topk(confidence[j], k=n_tok)
                        transfer_index[j, sel] = True

                x[transfer_index] = x0[transfer_index]
                del x0, confidence, transfer_index

    return x


@torch.no_grad()
def generate_with_confidence(
    model: nn.Module,
    prompt: torch.Tensor,           # [bs, prompt_len]
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Like generate() but also returns per-token confidence.

    Returns
    -------
    x               : [bs, prompt_len + gen_length]
    token_confidence: [bs, prompt_len + gen_length]
        0.0 for prompt positions; softmax prob at reveal time for generated tokens.
    """
    bs = prompt.shape[0]
    dtype = model.dtype
    device = prompt.device

    x = torch.full((bs, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    token_confidence = torch.zeros(bs, prompt.shape[1] + gen_length, device=device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    for num_block in range(num_blocks):
        start_idx = prompt.shape[1] + num_block * block_length
        end_idx   = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = x[:, start_idx:end_idx] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            torch.cuda.empty_cache()
            mask_index = x == mask_id

            with torch.cuda.amp.autocast(enabled=True):
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1
                    )
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)

                x0_p[:, end_idx:] = -float("inf")

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(bs):
                    n_tok = int(num_transfer[j, i].item())
                    if n_tok > 0:
                        _, sel = torch.topk(confidence[j], k=n_tok)
                        transfer_index[j, sel] = True

                # Record confidence at reveal time
                for j in range(bs):
                    revealed = transfer_index[j] & mask_index[j]
                    positions = revealed.nonzero(as_tuple=True)[0]
                    if len(positions) > 0:
                        confs = x0_p[j, positions].clamp(0.0, 1.0)
                        token_confidence[j, positions] = confs.float()

                x[transfer_index] = x0[transfer_index]

    return x, token_confidence


def forward_process(
    batch: torch.Tensor,        # [b, l]
    prompt_index: torch.Tensor, # [l]  bool
    mask_id: int,
    seed=None,
    p_mask_prompt: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply forward masking to a batch of (prompt + completion) sequences.

    Prompt tokens are masked with probability p_mask_prompt.
    Completion tokens are always masked.

    Returns (noisy_batch, p_mask) both shape [b, l].
    """
    if seed is not None:
        set_seed_compat(seed)

    b, l = batch.shape
    t_p = torch.ones(b, device=batch.device) * p_mask_prompt

    random_matrix = torch.rand((b, l), device=batch.device)

    is_mask_prompt      = prompt_index.unsqueeze(0) & (random_matrix < t_p.unsqueeze(1))
    is_mask_completion  = ~prompt_index.unsqueeze(0)
    is_mask = is_mask_prompt | is_mask_completion

    noisy_batch = torch.where(is_mask, mask_id, batch)

    p_mask = torch.where(
        prompt_index.unsqueeze(0),
        t_p.unsqueeze(1),
        torch.ones_like(t_p).unsqueeze(1),
    )
    return noisy_batch, p_mask


def get_logits(
    model: nn.Module,
    batch: torch.Tensor,
    prompt_index: torch.Tensor,
    cfg_scale: float,
    mask_id: int,
) -> torch.Tensor:
    """Run model forward and apply classifier-free guidance if cfg_scale > 0."""
    if cfg_scale > 0.0:
        prompt_index_2d = prompt_index.unsqueeze(0).expand(batch.shape[0], -1)
        un_batch = batch.clone()
        un_batch[prompt_index_2d] = mask_id
        full = torch.cat([batch, un_batch])
        logits = model(full).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(batch).logits
    return logits


def _get_per_token_logps(
    model: nn.Module,
    input_ids: torch.Tensor,   # [num_iter, batch, seq_len]
    logits_to_keep: int,
    mask_seeds: List,
    cfg_scale: float = 0.0,
    mask_id: int = 126336,
    p_mask_prompt: float = 0.3,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities for completion tokens.

    Applies a random forward masking (forward_process) to the prompt+completion
    sequence before each model forward pass, as in d1's GRPO implementation.

    Returns
    -------
    per_token_logps : [num_iter, batch, logits_to_keep]  float32
    """
    num_iterations, batch_size, seq_len = input_ids.size()
    device = input_ids.device
    per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

    assert len(mask_seeds) == num_iterations, \
        f"Expected {num_iterations} mask seeds, got {len(mask_seeds)}"

    prompt_length = seq_len - logits_to_keep
    prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
    prompt_index[:prompt_length] = True

    all_perturbed = []
    all_expanded  = []
    for iter_idx, mask_seed in enumerate(mask_seeds):
        expanded_input = input_ids[iter_idx]       # [batch, seq_len]
        perturbed, _ = forward_process(
            expanded_input, prompt_index, mask_id,
            seed=mask_seed, p_mask_prompt=p_mask_prompt,
        )
        all_perturbed.append(perturbed)
        all_expanded.append(expanded_input)

    # Single batched forward pass over all iterations
    perturbed_batch = torch.cat(all_perturbed, dim=0)   # [num_iter*batch, seq_len]
    expanded_batch  = torch.cat(all_expanded,  dim=0)   # [num_iter*batch, seq_len]

    logits = get_logits(model, perturbed_batch, prompt_index, cfg_scale, mask_id)
    # logits: [num_iter*batch, seq_len, vocab_size]

    completion_logits  = logits[:, -logits_to_keep:, :]  # [N, L, V]
    completion_targets = expanded_batch[:, -logits_to_keep:]  # [N, L]

    flat_logits  = completion_logits.reshape(-1, completion_logits.size(-1))
    flat_targets = completion_targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

    completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
    per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

    del perturbed_batch, logits, all_perturbed, all_expanded
    torch.cuda.empty_cache()

    return per_token_logps.to(torch.float32)


# ---------------------------------------------------------------------------
# Responsibility weights  (inline, no cc_rl import needed)
# ---------------------------------------------------------------------------

def compute_responsibility_weights_batch(
    confidences: torch.Tensor,      # [batch, completion_len]
    completion_mask: torch.Tensor,  # [batch, completion_len]  int {0,1}
    alpha: float = 1.0,
    eps: float = 1e-6,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> torch.Tensor:
    """
    Vectorized responsibility weight computation.

    rho_t = clip( (c_t + eps)^{-alpha}, clip_min, clip_max )
    then normalize to unit mean within each trajectory (masked mean).

    Returns
    -------
    weights : [batch, completion_len]  float32, normalized per sequence
    """
    # rho_t = (c_t + eps)^{-alpha}
    rho = (confidences.float() + eps).pow(-alpha)
    rho = rho.clamp(clip_min, clip_max)

    # Normalize per trajectory using masked mean
    mask_f = completion_mask.float()
    mean_rho = (rho * mask_f).sum(1, keepdim=True) / mask_f.sum(1, keepdim=True).clamp(min=1.0)
    rho_norm = rho / (mean_rho + 1e-8)

    return rho_norm


# ---------------------------------------------------------------------------
# Value head for Stage 2
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """Mean-pooled MLP value head.  Attaches to LLaDA's hidden states."""

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int = 1024,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_size = hidden_size
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_size, mlp_hidden_size))
            layers.append(nn.GELU())
            in_size = mlp_hidden_size
        layers.append(nn.Linear(in_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,        # [batch, seq, hidden]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq]
    ) -> torch.Tensor:
        """Returns [batch] scalar value estimates."""
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = hidden_states.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenise_prompt(tokenizer, prompt_msgs: List[dict], max_prompt_length: int) -> torch.Tensor:
    """
    Apply chat template and tokenise a single prompt (list of dicts).

    Returns 1-D LongTensor of token ids (no batch dim).
    """
    # LLaDA's tokenizer supports apply_chat_template
    text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].squeeze(0)

    # Truncate from the left if too long
    if ids.shape[0] > max_prompt_length:
        ids = ids[-max_prompt_length:]
    return ids


# ---------------------------------------------------------------------------
# Training step: rollout + loss computation
# ---------------------------------------------------------------------------

def rollout_and_compute_advantages(
    model: nn.Module,
    tokenizer,
    batch_examples: List[dict],
    cfg,
    device: torch.device,
    value_head: Optional[ValueHead] = None,
) -> dict:
    """
    Perform rollout for one training step.

    Returns a dict with everything needed to compute the training loss:
      prompt_ids        : [num_gen, prompt_len]
      completion_ids    : [num_gen, completion_len]
      completion_mask   : [num_gen, completion_len]
      advantages        : [num_gen] or [num_gen, completion_len] for Stage 1/2
      old_logps         : [1, num_gen, completion_len]  (for PPO ratio)
      ref_logps         : [1, num_gen, completion_len]  or None
      mask_seed         : int
      rewards           : [num_gen]
      confidence_weights: [num_gen, completion_len]  or None
    """
    num_gen = cfg.num_generations
    assert len(batch_examples) == 1, "batch_size=1 expected (one prompt per step)"
    example = batch_examples[0]

    # Tokenise prompt
    prompt_ids_1d = tokenise_prompt(tokenizer, example["prompt"], cfg.max_prompt_length)
    prompt_ids_1d = prompt_ids_1d.to(device)
    # Repeat for num_generations  -> [num_gen, prompt_len]
    prompt_ids = prompt_ids_1d.unsqueeze(0).expand(num_gen, -1).clone()

    # ---- Generate completions ------------------------------------------------
    model.eval()
    with torch.no_grad():
        if cfg.method in ("stage1", "stage2"):
            # Track confidence at reveal time
            full_ids, token_conf = generate_with_confidence(
                model=model,
                prompt=prompt_ids,
                steps=cfg.diffusion_steps,
                gen_length=cfg.max_completion_length,
                block_length=cfg.block_length,
                temperature=cfg.temperature,
                cfg_scale=cfg.cfg_scale,
                remasking=cfg.remasking,
                mask_id=cfg.mask_id,
            )
        else:
            full_ids = generate(
                model=model,
                prompt=prompt_ids,
                steps=cfg.diffusion_steps,
                gen_length=cfg.max_completion_length,
                block_length=cfg.block_length,
                temperature=cfg.temperature,
                cfg_scale=cfg.cfg_scale,
                remasking=cfg.remasking,
                mask_id=cfg.mask_id,
            )
            token_conf = None

    prompt_length  = prompt_ids.shape[1]
    completion_ids = full_ids[:, prompt_length:]    # [num_gen, completion_len]

    # Build completion mask (everything up to and including first EOS)
    eos_id = tokenizer.eos_token_id
    is_eos = completion_ids == eos_id
    eos_idx = torch.full((num_gen,), completion_ids.shape[1], dtype=torch.long, device=device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_indices = torch.arange(completion_ids.shape[1], device=device).unsqueeze(0)
    completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()  # [num_gen, completion_len]

    # ---- Decode and compute rewards ------------------------------------------
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    rewards_list = []
    prompts_for_reward = [example["prompt"]] * num_gen
    completions_for_reward = completions_text

    reward_fns = [correctness_reward_func, int_reward_func, soft_format_reward_func]
    answers_for_reward = [example.get("answer", "")] * num_gen

    total_rewards = np.zeros(num_gen)
    for rfn in reward_fns:
        try:
            r = rfn(
                prompts=prompts_for_reward,
                completions=completions_for_reward,
                answer=answers_for_reward,
            )
            total_rewards += np.array(r, dtype=np.float32)
        except Exception:
            pass

    rewards = torch.tensor(total_rewards, dtype=torch.float32, device=device)

    # ---- GRPO advantages -----------------------------------------------------
    # Group = all num_gen completions for the same prompt
    mu    = rewards.mean()
    sigma = rewards.std()
    group_adv = (rewards - mu) / (sigma + 1e-8)  # [num_gen]

    # ---- Per-token confidence weights (Stage 1 / 2) -------------------------
    conf_weights_completion = None
    if cfg.method in ("stage1", "stage2") and token_conf is not None:
        # token_conf: [num_gen, prompt_len + completion_len]
        conf_completion = token_conf[:, prompt_length:]  # [num_gen, completion_len]
        conf_weights_completion = compute_responsibility_weights_batch(
            conf_completion,
            completion_mask,
            alpha=cfg.credit_alpha,
            eps=cfg.credit_eps,
            clip_min=cfg.credit_clip_min,
            clip_max=cfg.credit_clip_max,
        )  # [num_gen, completion_len]

    # ---- Stage 2: value head for delta-V advantages --------------------------
    value_estimates = None
    if cfg.method == "stage2" and value_head is not None:
        input_ids_full = full_ids  # [num_gen, total_len]
        with torch.no_grad():
            outputs = model(input_ids_full, output_hidden_states=True)
        hidden = outputs.hidden_states[-1].detach()  # [num_gen, total_len, hidden]
        # Attention mask over full sequence (all tokens are real)
        attn_mask = torch.ones(num_gen, full_ids.shape[1], device=device)
        value_estimates = value_head(hidden, attn_mask)  # [num_gen]

    # ---- Random masking seed for logp computation ---------------------------
    mask_seed = int(torch.randint(0, 2 ** 12, (1,)).item())

    # ---- Compute old log-probs (no grad) ------------------------------------
    model.eval()
    with torch.no_grad():
        full_ids_3d = full_ids.unsqueeze(0)  # [1, num_gen, total_len]
        old_logps = _get_per_token_logps(
            model=model,
            input_ids=full_ids_3d,
            logits_to_keep=completion_ids.shape[1],
            mask_seeds=[mask_seed],
            cfg_scale=cfg.cfg_scale,
            mask_id=cfg.mask_id,
            p_mask_prompt=cfg.p_mask_prompt,
        )  # [1, num_gen, completion_len]

        # Ref log-probs via disabled LoRA adapter
        ref_logps = None
        if cfg.beta != 0.0:
            with model.disable_adapter():
                ref_logps = _get_per_token_logps(
                    model=model,
                    input_ids=full_ids_3d,
                    logits_to_keep=completion_ids.shape[1],
                    mask_seeds=[mask_seed],
                    cfg_scale=cfg.cfg_scale,
                    mask_id=cfg.mask_id,
                    p_mask_prompt=cfg.p_mask_prompt,
                )  # [1, num_gen, completion_len]

    return {
        "prompt_ids":          prompt_ids,           # [num_gen, prompt_len]
        "completion_ids":      completion_ids,        # [num_gen, completion_len]
        "completion_mask":     completion_mask,       # [num_gen, completion_len]
        "group_adv":           group_adv,             # [num_gen]
        "old_logps":           old_logps,             # [1, num_gen, completion_len]
        "ref_logps":           ref_logps,             # [1, num_gen, completion_len] or None
        "mask_seed":           mask_seed,
        "rewards":             rewards,               # [num_gen]
        "confidence_weights":  conf_weights_completion,  # [num_gen, completion_len] or None
        "value_estimates":     value_estimates,       # [num_gen] or None
        "completions_text":    completions_text,
    }


def compute_policy_loss(
    model: nn.Module,
    rollout: dict,
    cfg,
    value_head: Optional[ValueHead] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PPO-clip policy loss from rollout data.

    For diffu_grpo : uniform per-token advantages (group_adv broadcast)
    For stage1     : per-token advantages weighted by confidence rho_t
    For stage2     : stage1 + value-baseline corrected group advantage
                     + value-head MSE loss term

    Returns (loss, metrics_dict).
    """
    device = rollout["completion_ids"].device

    completion_ids  = rollout["completion_ids"]       # [G, L]
    completion_mask = rollout["completion_mask"]      # [G, L]
    prompt_ids      = rollout["prompt_ids"]           # [G, P]
    group_adv       = rollout["group_adv"]            # [G]
    old_logps       = rollout["old_logps"]            # [1, G, L]
    ref_logps       = rollout["ref_logps"]            # [1, G, L] or None
    mask_seed       = rollout["mask_seed"]
    conf_weights    = rollout["confidence_weights"]   # [G, L] or None
    value_estimates = rollout["value_estimates"]      # [G] or None
    rewards         = rollout["rewards"]              # [G]

    G, L = completion_ids.shape
    logits_to_keep = L

    # Full token sequence for logp computation
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # [G, P+L]
    full_ids_3d = full_ids.unsqueeze(0)                        # [1, G, P+L]

    # ---- New log-probs (with gradient) --------------------------------------
    model.train()
    new_logps = _get_per_token_logps(
        model=model,
        input_ids=full_ids_3d,
        logits_to_keep=logits_to_keep,
        mask_seeds=[mask_seed],
        cfg_scale=cfg.cfg_scale,
        mask_id=cfg.mask_id,
        p_mask_prompt=cfg.p_mask_prompt,
    )  # [1, G, L]
    new_logps = new_logps.squeeze(0)   # [G, L]
    old_logps = old_logps.squeeze(0)   # [G, L]

    # ---- Advantage construction ---------------------------------------------
    # Stage 2: learned value baseline corrects the group advantage.
    # We compute the residual (r - V(s)) then normalise it to zero-mean / unit-std,
    # matching the normalisation that GRPO already applied to group_adv.
    # If all residuals are identical (std -> 0), fall back to group_adv so the
    # loss is exactly zero (no policy update) rather than NaN / inf.
    if cfg.method == "stage2" and value_estimates is not None:
        residual = rewards.float() - value_estimates.float().detach()  # [G]
        res_std  = residual.std()
        if res_std > 1e-6:
            effective_adv = (residual - residual.mean()) / (res_std + 1e-8)  # [G]
        else:
            # All residuals identical: no useful gradient signal
            effective_adv = group_adv  # fall back to GRPO advantage
    else:
        effective_adv = group_adv  # [G]

    # Build per-token advantage tensor
    if cfg.method in ("stage1", "stage2") and conf_weights is not None:
        # weighted_adv[g, t] = effective_adv[g] * conf_weights[g, t]
        # (conf_weights already normalized to unit mean within each trajectory)
        weighted_adv = effective_adv.unsqueeze(1) * conf_weights  # [G, L]
    else:
        # Standard GRPO: broadcast scalar advantage to all completion tokens
        weighted_adv = effective_adv.unsqueeze(1).expand(G, L)    # [G, L]

    # ---- PPO-clip loss -------------------------------------------------------
    # ratio = exp(new_logps - old_logps)
    ratio1 = torch.exp(new_logps - old_logps)                       # [G, L]
    ratio2 = torch.clamp(ratio1, 1 - cfg.epsilon, 1 + cfg.epsilon)  # [G, L]

    per_token_loss1 = ratio1 * weighted_adv
    per_token_loss2 = ratio2 * weighted_adv
    per_token_loss  = -torch.min(per_token_loss1, per_token_loss2)  # [G, L]

    # ---- KL penalty ---------------------------------------------------------
    if cfg.beta != 0.0 and ref_logps is not None:
        ref_lp = ref_logps.squeeze(0)  # [G, L]
        # Reverse KL: KL(pi || pi_ref) approximated by Schulman's formula
        per_token_kl = (
            torch.exp(ref_lp - new_logps) - (ref_lp - new_logps) - 1
        )
        per_token_loss = per_token_loss + cfg.beta * per_token_kl
    else:
        per_token_kl = None

    # Mask and sum
    mask_f = completion_mask.float()
    policy_loss = (per_token_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    # ---- Value-head loss (Stage 2) ------------------------------------------
    # The value head uses detached hidden states from the backbone so its
    # gradient does NOT flow through the policy graph.  We keep the two losses
    # on completely separate computation graphs so each optimizer only sees its
    # own gradients.  total_loss -> policy optimizer; value_loss -> value optimizer.
    value_loss_val = 0.0
    explained_var  = 0.0
    value_loss: Optional[torch.Tensor] = None
    if cfg.method == "stage2" and value_head is not None and value_estimates is not None:
        with torch.no_grad():
            outputs_vh = model(full_ids, output_hidden_states=True)
        # Detach so no backbone grad flows into value optimizer
        hidden_det = outputs_vh.hidden_states[-1].detach()
        attn_mask  = torch.ones(G, full_ids.shape[1], device=device)
        v_pred   = value_head(hidden_det, attn_mask)   # [G], has grad through value_head only
        v_target = rewards.float().detach()
        value_loss = F.mse_loss(v_pred, v_target)       # separate graph from policy_loss

        # Explained variance (diagnostic only, no grad needed)
        var_target    = v_target.var().item()
        res_var       = (v_target - v_pred.detach()).var().item()
        explained_var = 1.0 - res_var / (var_target + 1e-8)
        value_loss_val = value_loss.item()

    # policy_loss is the only thing the policy optimizer should differentiate
    total_loss = policy_loss

    # ---- Metrics (all detached before converting to Python scalars) ----------
    with torch.no_grad():
        is_clipped    = (per_token_loss1.detach() < per_token_loss2.detach()).float()
        clip_fraction = (is_clipped * mask_f).sum() / mask_f.sum().clamp(min=1.0)

        mean_conf = 0.0
        if conf_weights is not None:
            mean_conf = float((conf_weights * mask_f).sum() / mask_f.sum().clamp(min=1.0))

        mean_kl = 0.0
        if per_token_kl is not None:
            mean_kl = float(
                (per_token_kl.detach() * mask_f).sum() / mask_f.sum().clamp(min=1.0)
            )

    metrics = {
        "loss":             float(total_loss.detach()),
        "policy_loss":      float(policy_loss.detach()),
        "clip_fraction":    float(clip_fraction.detach()),
        "mean_reward":      float(rewards.mean()),
        "reward_std":       float(rewards.std()),
        "group_adv_mean":   float(group_adv.mean()),
        "group_adv_std":    float(group_adv.std()),
        "mean_confidence":  mean_conf,
        "mean_kl":          mean_kl,
        "value_loss":       value_loss_val,
        "explained_var":    explained_var,
    }

    return total_loss, value_loss, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone GRPO for LLaDA")

    # Method
    p.add_argument("--method", type=str, default="diffu_grpo",
                   choices=["diffu_grpo", "stage1", "stage2"])

    # Model / paths
    p.add_argument("--model_path", type=str,
                   default="/home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct")
    p.add_argument("--dataset", type=str, default="gsm8k")
    p.add_argument("--output_dir", type=str, default="experiments/outputs/standalone")

    # Training
    p.add_argument("--max_steps",        type=int,   default=3000)
    p.add_argument("--num_generations",  type=int,   default=4)
    p.add_argument("--batch_size",       type=int,   default=1)
    p.add_argument("--learning_rate",    type=float, default=1e-6)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)

    # Diffusion generation
    p.add_argument("--diffusion_steps",         type=int,   default=64)
    p.add_argument("--block_length",            type=int,   default=32)
    p.add_argument("--max_completion_length",   type=int,   default=256)
    p.add_argument("--max_prompt_length",       type=int,   default=256)
    p.add_argument("--temperature",             type=float, default=0.0)
    p.add_argument("--cfg_scale",               type=float, default=0.0)
    p.add_argument("--remasking",               type=str,   default="low_confidence")
    p.add_argument("--mask_id",                 type=int,   default=126336)
    p.add_argument("--p_mask_prompt",           type=float, default=0.3)

    # PPO / GRPO
    p.add_argument("--beta",    type=float, default=0.04)
    p.add_argument("--epsilon", type=float, default=0.2)

    # Stage 1: credit assignment
    p.add_argument("--credit_alpha",    type=float, default=1.0)
    p.add_argument("--credit_eps",      type=float, default=1e-6)
    p.add_argument("--credit_clip_min", type=float, default=0.25)
    p.add_argument("--credit_clip_max", type=float, default=4.0)

    # Stage 2: value head
    p.add_argument("--value_hidden_size", type=int,   default=1024)
    p.add_argument("--critic_lr",         type=float, default=5e-6)

    # LoRA
    p.add_argument("--lora_r",     type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)

    # Logging
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps",    type=int, default=500)
    p.add_argument("--seed",          type=int, default=42)

    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}, method={cfg.method}")
    print(f"[init] output_dir={cfg.output_dir}")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"[init] Loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"[init] Loading model from {cfg.model_path}")
    model = AutoModel.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Value head (Stage 2)
    # ------------------------------------------------------------------
    value_head: Optional[ValueHead] = None
    value_optimizer: Optional[torch.optim.AdamW] = None
    if cfg.method == "stage2":
        hidden_size = model.config.hidden_size
        value_head = ValueHead(
            hidden_size=hidden_size,
            mlp_hidden_size=cfg.value_hidden_size,
            n_layers=2,
        ).to(device)
        value_optimizer = torch.optim.AdamW(
            value_head.parameters(),
            lr=cfg.critic_lr,
            weight_decay=0.0,
        )
        print(f"[init] ValueHead: hidden={hidden_size} -> {cfg.value_hidden_size} -> 1")

    # ------------------------------------------------------------------
    # Policy optimizer (LoRA params only)
    # ------------------------------------------------------------------
    policy_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(policy_params, lr=cfg.learning_rate, weight_decay=0.0)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("[init] Loading GSM8K dataset...")
    dataset = get_gsm8k_questions("train")
    dataset_size = len(dataset)
    print(f"[init] Dataset size: {dataset_size}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_records: List[dict] = []
    step_times: List[float] = []

    print(f"\n{'='*60}")
    print(f"Starting {cfg.method} training for {cfg.max_steps} steps")
    print(f"{'='*60}\n")

    for step in range(1, cfg.max_steps + 1):
        t0 = time.time()

        # Sample one example
        idx = random.randint(0, dataset_size - 1)
        example = dataset[idx]
        batch = [example]

        # ----- Rollout (generation + reward) ---------------------------------
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
            print(f"[step {step}] OOM during rollout, skipping.")
            torch.cuda.empty_cache()
            continue

        # ----- Loss computation ----------------------------------------------
        model.train()
        optimizer.zero_grad()
        if value_optimizer is not None:
            value_optimizer.zero_grad()

        try:
            loss, value_loss_tensor, metrics = compute_policy_loss(
                model=model,
                rollout=rollout,
                cfg=cfg,
                value_head=value_head,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[step {step}] OOM during loss computation, skipping.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            if value_optimizer is not None:
                value_optimizer.zero_grad()
            continue

        # ----- Backward pass (policy) -----------------------------------------
        # loss = policy_loss only (value_loss_tensor is on a separate graph)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_params, cfg.max_grad_norm)
        optimizer.step()

        # ----- Backward pass (value head, Stage 2 only) -----------------------
        # value_loss_tensor has gradients only through the value head MLP
        # (backbone hidden states were detached inside compute_policy_loss).
        if value_optimizer is not None and value_loss_tensor is not None:
            value_optimizer.zero_grad()
            value_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            value_optimizer.step()

        step_time = time.time() - t0
        step_times.append(step_time)

        # ----- Logging -------------------------------------------------------
        if step % cfg.logging_steps == 0 or step == 1:
            avg_time = sum(step_times[-cfg.logging_steps:]) / len(step_times[-cfg.logging_steps:])
            record = {
                "step":             step,
                "loss":             metrics["loss"],
                "policy_loss":      metrics["policy_loss"],
                "mean_reward":      metrics["mean_reward"],
                "reward_std":       metrics["reward_std"],
                "group_adv_mean":   metrics["group_adv_mean"],
                "group_adv_std":    metrics["group_adv_std"],
                "clip_fraction":    metrics["clip_fraction"],
                "mean_confidence":  metrics["mean_confidence"],
                "mean_kl":          metrics["mean_kl"],
                "value_loss":       metrics["value_loss"],
                "explained_var":    metrics["explained_var"],
                "step_time_s":      avg_time,
            }
            log_records.append(record)

            # Print to stdout
            print(
                f"[step {step:5d}/{cfg.max_steps}] "
                f"loss={metrics['loss']:.4f} "
                f"reward={metrics['mean_reward']:.3f}±{metrics['reward_std']:.3f} "
                f"adv={metrics['group_adv_mean']:.3f} "
                f"clip={metrics['clip_fraction']:.3f} "
                f"kl={metrics['mean_kl']:.4f} "
                f"time={avg_time:.1f}s"
            )
            if cfg.method in ("stage1", "stage2"):
                print(f"         conf_weight_mean={metrics['mean_confidence']:.4f}")
            if cfg.method == "stage2":
                print(f"         value_loss={metrics['value_loss']:.4f}  "
                      f"explained_var={metrics['explained_var']:.4f}")

            # Append to JSONL log
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # ----- Checkpoint ----------------------------------------------------
        if step % cfg.save_steps == 0 or step == cfg.max_steps:
            ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            if value_head is not None:
                torch.save(value_head.state_dict(),
                           os.path.join(ckpt_dir, "value_head.pt"))
            print(f"[checkpoint] Saved to {ckpt_dir}")

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save final summary
    # ------------------------------------------------------------------
    summary_path = os.path.join(cfg.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "method":     cfg.method,
                "max_steps":  cfg.max_steps,
                "total_steps_completed": step,
                "final_loss":   log_records[-1]["loss"] if log_records else None,
                "final_reward": log_records[-1]["mean_reward"] if log_records else None,
            },
            f,
            indent=2,
        )

    print(f"\nTraining complete.  Logs -> {log_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
