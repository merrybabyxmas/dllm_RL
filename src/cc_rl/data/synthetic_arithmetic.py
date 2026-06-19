"""
Synthetic arithmetic dataset generator for controlled experiments.

Generates simple integer arithmetic problems (addition, subtraction,
multiplication) with known correct answers.  Useful for verifying that
training is working without needing the full GSM8K dataset.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from datasets import Dataset


SYSTEM_PROMPT = "Solve the arithmetic problem. Answer with just the number."


def generate_arithmetic_examples(
    n: int = 1000,
    ops: str = "+-*",
    max_val: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate n arithmetic problem/answer pairs.

    Parameters
    ----------
    n       : Number of examples.
    ops     : String of operator characters to use (e.g., "+-", "+-*").
    max_val : Maximum absolute value of operands.
    seed    : Random seed for reproducibility.

    Returns
    -------
    List of dicts with keys: prompt, answer
    """
    rng = random.Random(seed)
    examples = []
    op_list = list(ops)

    for _ in range(n):
        a = rng.randint(-max_val, max_val)
        b = rng.randint(-max_val, max_val)
        op = rng.choice(op_list)

        if op == "+":
            result = a + b
            expr = f"{a} + {b}"
        elif op == "-":
            result = a - b
            expr = f"{a} - {b}"
        elif op == "*":
            result = a * b
            expr = f"{a} * {b}"
        elif op == "/" and b != 0:
            # Integer division only
            result = a // b
            expr = f"{a} // {b}"
        else:
            # Default to addition if division by zero
            result = a + b
            expr = f"{a} + {b}"

        question = f"What is {expr}?"
        answer = str(result)
        examples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "answer": answer,
            "question": question,
        })

    return examples


def get_synthetic_arithmetic_dataset(
    n: int = 1000,
    ops: str = "+-*",
    max_val: int = 100,
    seed: int = 42,
    split_ratio: float = 0.9,
    split: str = "train",
) -> Dataset:
    """
    Create a HuggingFace Dataset of synthetic arithmetic problems.

    Parameters
    ----------
    n           : Total number of examples.
    ops         : Operators to include.
    max_val     : Max operand absolute value.
    seed        : Random seed.
    split_ratio : Fraction of data for training (remainder for eval).
    split       : "train" or "eval".

    Returns
    -------
    HuggingFace Dataset.
    """
    all_examples = generate_arithmetic_examples(n=n, ops=ops, max_val=max_val, seed=seed)
    n_train = int(n * split_ratio)
    if split == "train":
        examples = all_examples[:n_train]
    else:
        examples = all_examples[n_train:]

    return Dataset.from_list(examples)
