"""
Toy tabular critics for unit testing Stage 2 (delta-V) and Stage 3 (Q-V)
advantage assignment without requiring a real neural model.

TabularValue maps state strings to scalar values.
TabularQ maps (state, action) tuples to scalar Q-values.

Missing keys return 0.0 by default (equivalent to uninitiated value estimates).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


class TabularValue:
    """
    Finite-state tabular value function V(s).

    Parameters
    ----------
    table : dict mapping state -> float

    Usage
    -----
    value_fn = TabularValue({"s0": 0.25, "s1": 0.80})
    v = value_fn("s0")  # -> 0.25
    v = value_fn("unknown")  # -> 0.0
    """

    def __init__(self, table: Dict[Any, float]) -> None:
        self.table = table

    def __call__(self, state: Any) -> float:
        """Return V(state), defaulting to 0.0 for unseen states."""
        return self.table.get(state, 0.0)

    def __repr__(self) -> str:
        return f"TabularValue(n_states={len(self.table)})"


class TabularQ:
    """
    Finite-state tabular action-value function Q(s, a).

    Parameters
    ----------
    table : dict mapping (state, action) tuple -> float

    Usage
    -----
    q_fn = TabularQ({("s0", "+"): 0.80, ("s0", "*"): 0.05})
    q = q_fn("s0", "+")  # -> 0.80
    q = q_fn("s0", "?")  # -> 0.0  (unknown action)
    """

    def __init__(self, table: Dict[Tuple[Any, Any], float]) -> None:
        self.table = table

    def __call__(self, state: Any, action: Any) -> float:
        """Return Q(state, action), defaulting to 0.0 for unseen pairs."""
        return self.table.get((state, action), 0.0)

    def __repr__(self) -> str:
        return f"TabularQ(n_pairs={len(self.table)})"
