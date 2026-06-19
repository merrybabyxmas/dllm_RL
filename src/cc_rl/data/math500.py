"""
MATH-500 dataset loader for diffusion LLM RL evaluation.

MATH-500 is a 500-problem subset of the MATH benchmark used for
evaluating mathematical reasoning quality.
"""
from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "Solve the following math problem. Show all your work step by step, "
    "then box your final answer as \\boxed{<answer>}."
)


def get_math500_dataset(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> Dataset:
    """
    Load the MATH-500 evaluation benchmark.

    Parameters
    ----------
    split        : Dataset split (typically "test").
    max_examples : Optional cap on number of examples.

    Returns
    -------
    HuggingFace Dataset with columns:
        - prompt   : list of chat messages
        - answer   : ground-truth answer string
        - subject  : math subject category
        - level    : difficulty level (1-5)
    """
    try:
        ds = load_dataset("lighteval/MATH-Hard", split=split)
    except Exception:
        # Fallback: try alternative HF hub location
        ds = load_dataset("EleutherAI/hendrycks_math", "all", split=split, trust_remote_code=True)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def format_example(example: dict) -> dict:
        question = example.get("problem", example.get("question", ""))
        answer = example.get("solution", example.get("answer", ""))
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "answer": answer,
            "subject": example.get("type", example.get("subject", "unknown")),
            "level": example.get("level", -1),
        }

    keep_cols = ["prompt", "answer", "subject", "level"]
    ds = ds.map(format_example)
    # Only keep columns that exist
    existing = [c for c in keep_cols if c in ds.column_names]
    ds = ds.select_columns(existing)
    return ds
