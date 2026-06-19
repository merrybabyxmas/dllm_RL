"""
Exact-match reward functions for GSM8K and general arithmetic evaluation.

The primary reward is binary: 1.0 if the predicted final number matches
the gold answer, 0.0 otherwise.

Answer extraction prioritizes the #### delimiter format used by GSM8K;
falls back to the last number appearing in the text.
"""
from __future__ import annotations

import re
from typing import List, Optional


def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from a model completion.

    Extraction priority:
    1. After #### delimiter (GSM8K convention): #### 42
    2. Last number in the text (fallback).

    Parameters
    ----------
    text : Model completion string.

    Returns
    -------
    str representation of the extracted number, or None if not found.
    Commas are stripped so "1,234" becomes "1234".
    """
    if not text:
        return None

    # Priority 1: #### answer format (GSM8K standard)
    match = re.search(r'####\s*([+-]?\d[\d,]*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '').strip()

    # Priority 2: last number in the text
    numbers = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()

    return None


def reward_gsm8k(completion: str, gold_answer: str) -> float:
    """
    Compute binary reward for GSM8K: 1.0 if predicted == gold, else 0.0.

    Parameters
    ----------
    completion  : Model-generated completion string.
    gold_answer : Ground-truth answer string (may contain reasoning + ####).

    Returns
    -------
    1.0 if correct, 0.0 if incorrect or unparseable.
    """
    pred = extract_final_number(completion)
    gold = extract_final_number(gold_answer)

    if pred is None or gold is None:
        return 0.0

    try:
        return 1.0 if float(pred) == float(gold) else 0.0
    except ValueError:
        # Non-numeric parse edge case
        return 1.0 if pred.strip() == gold.strip() else 0.0


def reward_gsm8k_batch(
    prompts: List,
    completions: List,
    step: int = 0,
    run_name: str = "",
    **kwargs,
) -> List[float]:
    """
    Batch reward function compatible with DiffuGRPOTrainer's reward_funcs API.

    Parameters
    ----------
    prompts     : List of prompts (unused but required by trainer API).
    completions : List of completion strings or chat completion dicts.
    step        : Training step (unused, required by trainer API).
    run_name    : Run name (unused, required by trainer API).
    **kwargs    : Must include 'answer' key with list of gold answers.

    Returns
    -------
    List of float rewards.
    """
    answers = kwargs.get("answer", [])
    rewards = []

    for i, completion in enumerate(completions):
        # Handle both string and chat completion dict formats
        if isinstance(completion, list):
            comp_text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            comp_text = completion.get("content", "")
        else:
            comp_text = str(completion)

        gold = answers[i] if i < len(answers) else ""
        rewards.append(reward_gsm8k(comp_text, gold))

    return rewards
