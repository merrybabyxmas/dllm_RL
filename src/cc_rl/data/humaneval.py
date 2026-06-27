"""
HumanEval dataset loader for diffusion LLM RL training.

HumanEval: 164 Python programming problems with unit tests.
Since HumanEval has no official train split, we use all 164 for both
training (diffusion RL) and evaluation (zero-shot greedy decoding).
"""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset, Dataset

SYSTEM_PROMPT = (
    "Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n<answer>\n```python\n...\n```\n</answer>"
)


def get_humaneval_dataset(
    split: str = "test",
    max_examples: Optional[int] = None,
) -> Dataset:
    """
    Load and format HumanEval.

    Parameters
    ----------
    split        : "test" (only split available; we use it for both train and eval)
    max_examples : Optional cap.

    Returns
    -------
    HF Dataset with columns:
        prompt          : chat messages (user)
        test            : string — the check() function with assertions
        entry_point     : string — function name to call in check()
        canonical_solution : string — reference solution (for debugging)
    """
    ds = load_dataset("openai/openai_humaneval", split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def format_example(ex: dict) -> dict:
        return {
            "prompt": [
                {"role": "user", "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    "Complete the following Python function. "
                    "Write the complete function (including the signature) "
                    "inside <answer>```python\n...\n```</answer> tags.\n\n"
                    f"```python\n{ex['prompt']}\n```"
                )},
            ],
            "test":               ex["test"],
            "entry_point":        ex["entry_point"],
            "canonical_solution": ex["canonical_solution"],
        }

    return ds.map(format_example, remove_columns=ds.column_names)
