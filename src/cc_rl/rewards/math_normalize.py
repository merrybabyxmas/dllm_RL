"""
Reward functions for MATH-500 evaluation.

MATH-500 answers may be in LaTeX boxed format: \\boxed{<answer>}.
We extract and normalize before comparing.
"""
from __future__ import annotations

import re
from typing import List, Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    r"""
    Extract the content of the innermost \\boxed{} command.

    Parameters
    ----------
    text : LaTeX-formatted solution string.

    Returns
    -------
    String content inside \\boxed{}, or None if not found.
    """
    # Find all \boxed{ ... } occurrences
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # take last boxed answer
    return None


def normalize_math_answer(text: str) -> str:
    """
    Normalize a math answer string for comparison.

    Normalization steps:
    - Strip whitespace
    - Remove trailing zeros after decimal point
    - Normalize fraction formats
    - Lowercase
    """
    if not text:
        return ""
    s = text.strip()
    # Remove commas in numbers
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    # Try to parse as float for numeric normalization
    try:
        val = float(s)
        # Remove unnecessary trailing zeros
        if val == int(val):
            return str(int(val))
        return f"{val:.10f}".rstrip('0').rstrip('.')
    except ValueError:
        pass
    return s.lower().strip()


def reward_math500(completion: str, gold_answer: str) -> float:
    """
    Compute binary reward for MATH-500.

    Extracts \\boxed{} answer from both completion and gold, then
    normalizes and compares.

    Parameters
    ----------
    completion  : Model-generated solution string.
    gold_answer : Ground-truth solution string.

    Returns
    -------
    1.0 if answers match after normalization, 0.0 otherwise.
    """
    pred_boxed = extract_boxed_answer(completion)
    gold_boxed = extract_boxed_answer(gold_answer)

    # Fall back to the raw string if no boxed found
    pred = normalize_math_answer(pred_boxed if pred_boxed else completion)
    gold = normalize_math_answer(gold_boxed if gold_boxed else gold_answer)

    if not pred or not gold:
        return 0.0

    return 1.0 if pred == gold else 0.0


def reward_math500_batch(
    prompts: List,
    completions: List,
    step: int = 0,
    run_name: str = "",
    **kwargs,
) -> List[float]:
    """
    Batch reward function for MATH-500, compatible with DiffuGRPOTrainer API.
    """
    answers = kwargs.get("answer", [])
    rewards = []

    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            comp_text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            comp_text = completion.get("content", "")
        else:
            comp_text = str(completion)

        gold = answers[i] if i < len(answers) else ""
        rewards.append(reward_math500(comp_text, gold))

    return rewards
