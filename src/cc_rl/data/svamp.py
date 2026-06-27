"""
SVAMP dataset loader for diffusion LLM RL training.

SVAMP: 1,000 challenge math word problems (arithmetic).
Harder than standard elementary benchmarks due to structural variations.
Faster than GSM8K (~800 train examples vs 7,473).

Source: ChilleD/SVAMP on HuggingFace
Fields: ID, Body, Question, Equation, Answer (float), Type
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

from datasets import load_dataset, Dataset

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "Give the final numeric answer after ####."
)


def get_svamp_dataset(
    seed: int = 42,
    train_size: int = 800,
    eval_size: int = 200,
) -> Tuple[Dataset, Dataset]:
    """
    Load and format SVAMP with an 800/200 train/eval split.

    Returns
    -------
    (train_ds, eval_ds) : HF Datasets with columns:
        prompt  : chat messages (user)
        answer  : str — the numeric answer (e.g. "8.0")
    """
    ds = load_dataset("ChilleD/SVAMP", split="train")  # all 1000 examples in "train"

    examples = list(ds)
    rng = random.Random(seed)
    rng.shuffle(examples)

    train_raw = examples[:train_size]
    eval_raw  = examples[train_size:train_size + eval_size]

    def format_example(ex: dict) -> dict:
        body     = ex["Body"].strip()
        question = ex["Question"].strip()
        answer   = str(ex["Answer"])

        problem_text = body + " " + question

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": problem_text},
            ],
            "answer": answer,
        }

    train_ds = Dataset.from_list([format_example(e) for e in train_raw])
    eval_ds  = Dataset.from_list([format_example(e) for e in eval_raw])
    return train_ds, eval_ds
