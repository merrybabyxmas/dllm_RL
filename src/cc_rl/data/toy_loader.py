"""
Toy trajectory loader for unit-testing credit assignment algorithms.

Loads JSONL fixture files containing small, manually-crafted trajectories
with known structure, enabling exact numerical verification of advantage
computation without requiring a real model.

Expected JSONL format per line:
{
  "prompt_id": "toy_2_plus_3",
  "sample_id": 1,
  "prompt": "2 + 3 = ?",
  "reward": 1,
  "steps": [
    {"state": "[2 □ 3 = □]", "action": "+", "next_state": "[2 + 3 = □]",
     "confidence": 0.55},
    {"state": "[2 + 3 = □]", "action": "5", "next_state": "[2 + 3 = 5]",
     "confidence": 0.90, "done": true}
  ]
}
"""
from __future__ import annotations

import json
from typing import Dict, List

from cc_rl.rollout.trajectory import TrajectoryRecord, TrajectoryStep


def load_toy_2_plus_3(
    path: str = "tests/fixtures/toy_2_plus_3.jsonl",
) -> List[TrajectoryRecord]:
    """
    Load toy "2 + 3 = ?" trajectories from a JSONL fixture file.

    Parameters
    ----------
    path : Path to the JSONL fixture file.

    Returns
    -------
    List of TrajectoryRecord objects, one per line in the file.
    """
    trajectories: List[TrajectoryRecord] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            steps: List[TrajectoryStep] = []
            for i, s in enumerate(d["steps"]):
                step = TrajectoryStep(
                    prompt_id=d["prompt_id"],
                    sample_id=d["sample_id"],
                    step_idx=i,
                    state=s["state"],
                    action=s["action"],
                    next_state=s["next_state"],
                    confidence=s["confidence"],
                    old_logprob=0.0,
                    done=s.get("done", False),
                )
                steps.append(step)

            traj = TrajectoryRecord(
                prompt_id=d["prompt_id"],
                sample_id=d["sample_id"],
                prompt_text=d["prompt"],
                final_text="",
                reward=float(d["reward"]),
                steps=steps,
            )
            trajectories.append(traj)

    return trajectories


def collect_advantages_by_sample_action(
    trajectories: List[TrajectoryRecord],
) -> Dict[int, Dict[str, float]]:
    """
    Extract final_advantage values indexed by sample_id and action.

    Useful for comparing against expected values in unit tests.

    Parameters
    ----------
    trajectories : List of TrajectoryRecord with step.final_advantage filled.

    Returns
    -------
    dict[sample_id][action] = final_advantage
    """
    result: Dict[int, Dict[str, float]] = {}
    for traj in trajectories:
        sid = traj.sample_id
        result[sid] = {}
        for step in traj.steps:
            result[sid][step.action] = step.final_advantage
    return result
