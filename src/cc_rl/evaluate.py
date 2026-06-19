"""
Evaluation entry point for cc_rl.

Runs a trained model checkpoint on an evaluation dataset and computes
accuracy metrics (exact match, pass@k).

Usage
-----
python -m cc_rl.evaluate \
    --model_path outputs/gsm8k_stage1/checkpoint-3000 \
    --dataset gsm8k \
    --split test \
    --output_dir outputs/eval_stage1 \
    --batch_size 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

_DIFFU_GRPO_PATH = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
if _DIFFU_GRPO_PATH not in sys.path:
    sys.path.insert(0, _DIFFU_GRPO_PATH)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    gen_length: int = 256,
    steps: int = 64,
    block_length: int = 64,
    temperature: float = 0.0,
    mask_id: int = 126336,
    batch_size: int = 8,
    device: torch.device = None,
) -> List[str]:
    """
    Generate completions for a list of prompt strings using the diffusion model.

    Parameters
    ----------
    model      : LLaDA / MDLM model.
    tokenizer  : Matching tokenizer.
    prompts    : List of prompt strings (already formatted for the model).
    gen_length : Number of tokens to generate per prompt.
    steps      : Number of denoising steps.
    block_length: Block size for block-wise generation.
    temperature: Sampling temperature (0 = greedy).
    mask_id    : Vocabulary index of the [MASK] token.
    batch_size : Inference batch size.
    device     : Device to run on.

    Returns
    -------
    List of decoded completion strings.
    """
    if device is None:
        device = next(model.parameters()).device

    import numpy as np
    import torch.nn.functional as F

    def _get_num_transfer_tokens(mask_index, n_steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // n_steps
        remainder = mask_num % n_steps
        result = base.expand(-1, n_steps).clone()
        idx = torch.arange(n_steps, device=mask_index.device)
        result[idx.unsqueeze(0) < remainder] += 1
        return result.to(torch.int64)

    all_completions = []
    model.eval()

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        # Tokenize
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = enc["input_ids"].to(device)   # [bs, prompt_len]
        bs = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        total_len = prompt_len + gen_length

        x = torch.full((bs, total_len), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_len] = prompt_ids.clone()

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for block_idx in range(num_blocks):
            start_idx = prompt_len + block_idx * block_length
            end_idx = prompt_len + (block_idx + 1) * block_length
            block_mask = x[:, start_idx:end_idx] == mask_id
            n_transfer = _get_num_transfer_tokens(block_mask, steps_per_block)

            for step_i in range(steps_per_block):
                mask_index = x == mask_id
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(x).logits                        # [bs, total_len, vocab]
                    x0 = torch.argmax(logits, dim=-1)
                    p = F.softmax(logits.float(), dim=-1)
                    x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    x0_p[:, end_idx:] = -float("inf")
                    x0 = torch.where(mask_index, x0, x)
                    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
                    transfer = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(bs):
                        n = n_transfer[j, step_i].item()
                        if n > 0:
                            _, sel = torch.topk(confidence[j], k=int(n))
                            transfer[j, sel] = True
                    x[transfer] = x0[transfer]

        # Decode completions
        completions = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)
        all_completions.extend(completions)

    return all_completions


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    dataset_name: str = "gsm8k",
    split: str = "test",
    output_dir: str = "outputs/eval",
    batch_size: int = 16,
    gen_length: int = 256,
    steps: int = 64,
    block_length: int = 64,
    max_examples: Optional[int] = None,
    precision: str = "bf16",
) -> Dict:
    """
    Run full evaluation on a dataset.

    Returns
    -------
    dict with keys: accuracy, n_correct, n_total
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        precision, torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    # Load dataset
    if dataset_name == "gsm8k":
        from cc_rl.data.gsm8k import get_gsm8k_dataset, SYSTEM_PROMPT
        ds = get_gsm8k_dataset(split=split, max_examples=max_examples)
    elif dataset_name in ("math500", "math_500"):
        from cc_rl.data.math500 import get_math500_dataset
        ds = get_math500_dataset(split=split, max_examples=max_examples)
        SYSTEM_PROMPT = ""
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Format prompts
    prompts_text = []
    for example in ds:
        prompt = example["prompt"]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompts_text.append(text)

    gold_answers = [ex["answer"] for ex in ds]

    # Generate
    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_text,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        batch_size=batch_size,
        device=device,
    )

    # Score
    if dataset_name == "gsm8k":
        from cc_rl.rewards.exact_match import reward_gsm8k
        reward_fn = reward_gsm8k
    else:
        from cc_rl.rewards.math_normalize import reward_math500
        reward_fn = reward_math500

    rewards = [reward_fn(c, g) for c, g in zip(completions, gold_answers)]
    n_correct = sum(r > 0.5 for r in rewards)
    n_total = len(rewards)
    accuracy = n_correct / n_total

    # Save results
    results = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "model_path": model_path,
        "dataset": dataset_name,
        "split": split,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save per-example predictions
    preds_path = os.path.join(output_dir, "predictions.jsonl")
    with open(preds_path, "w") as f:
        for i, (c, g, r) in enumerate(zip(completions, gold_answers, rewards)):
            json.dump({"idx": i, "completion": c, "gold": g, "reward": r}, f)
            f.write("\n")

    print(f"Accuracy: {accuracy:.4f} ({n_correct}/{n_total})")
    print(f"Results saved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--precision", type=str, default="bf16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gen_length=args.gen_length,
        steps=args.steps,
        max_examples=args.max_examples,
        precision=args.precision,
    )
