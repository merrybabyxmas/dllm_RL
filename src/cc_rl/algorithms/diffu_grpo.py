"""
Thin wrapper around DiffuGRPOTrainer for use within cc_rl.

Adds sys.path manipulation so the d1/diffu-grpo directory is importable,
and re-exports DiffuGRPOTrainer as WrappedDiffuGRPOTrainer for import
consistency.

Also patches TRL 1.6.0 compatibility: `is_rich_available` moved from
`trl.import_utils` to `transformers.utils` in newer TRL versions.
"""
from __future__ import annotations

import sys
import os

# TRL 1.6.0 compat: patch missing is_rich_available in trl.import_utils
import trl.import_utils as _trl_import_utils
if not hasattr(_trl_import_utils, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_import_utils.is_rich_available = _ira
    except ImportError:
        _trl_import_utils.is_rich_available = lambda: False

# Add the d1/diffu-grpo directory to the Python path so that
# `diffu_grpo_trainer` and related modules are importable.
_DIFFU_GRPO_PATH = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
if _DIFFU_GRPO_PATH not in sys.path:
    sys.path.insert(0, _DIFFU_GRPO_PATH)

from diffu_grpo_trainer import DiffuGRPOTrainer  # noqa: E402


# Re-export with a stable name for internal use
WrappedDiffuGRPOTrainer = DiffuGRPOTrainer

__all__ = ["WrappedDiffuGRPOTrainer", "DiffuGRPOTrainer"]
