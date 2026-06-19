"""
GSM8K dataset loader for diffusion LLM RL training.

Formats the dataset into chat-style prompt/answer pairs compatible with
the DiffuGRPOTrainer's expected input format.
"""
from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "Solve the problem. Show your reasoning and give the final answer after ####."
)


def get_gsm8k_dataset(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> Dataset:
    """
    Load and format the GSM8K dataset.

    Parameters
    ----------
    split        : "train" or "test"
    max_examples : Optional cap on number of examples (for smoke tests).

    Returns
    -------
    HuggingFace Dataset with columns:
        - prompt   : list of chat messages (system + user)
        - answer   : ground-truth answer string (contains #### <number>)
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def format_example(example: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    return ds.map(format_example, remove_columns=ds.column_names)
