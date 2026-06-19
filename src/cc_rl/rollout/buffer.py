"""
Rollout buffer for storing and sampling trajectory records.

Supports fixed-size circular buffer semantics and group-aware sampling
(ensures all samples for a prompt_id are in the same batch).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from cc_rl.rollout.trajectory import TrajectoryRecord


class RolloutBuffer:
    """
    Fixed-capacity rollout buffer storing TrajectoryRecord objects.

    Usage
    -----
    buffer = RolloutBuffer(capacity=1024)
    buffer.add(trajectory)
    batch = buffer.sample_group(prompt_id="q1", n=8)
    """

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._records: List[TrajectoryRecord] = []
        self._ptr = 0  # circular write pointer
        self._full = False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, record: TrajectoryRecord) -> None:
        """Insert one TrajectoryRecord into the buffer (circular)."""
        if self._full:
            self._records[self._ptr] = record
        else:
            self._records.append(record)
        self._ptr = (self._ptr + 1) % self.capacity
        if self._ptr == 0:
            self._full = True

    def add_batch(self, records: List[TrajectoryRecord]) -> None:
        for r in records:
            self.add(r)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def is_empty(self) -> bool:
        return len(self._records) == 0

    def all(self) -> List[TrajectoryRecord]:
        """Return all records currently stored."""
        return list(self._records)

    def sample_group(self, prompt_id: str) -> List[TrajectoryRecord]:
        """Return all records matching a given prompt_id."""
        return [r for r in self._records if r.prompt_id == prompt_id]

    def group_ids(self) -> List[str]:
        """Return the unique prompt_ids present in the buffer."""
        seen = []
        for r in self._records:
            if r.prompt_id not in seen:
                seen.append(r.prompt_id)
        return seen

    def clear(self) -> None:
        self._records = []
        self._ptr = 0
        self._full = False
