"""
Core data structures for diffusion LLM RL trajectory recording.

TrajectoryStep captures a single denoising decision (action) at a single
diffusion step, along with associated credit-assignment metadata.

TrajectoryRecord aggregates all steps for one sampled completion.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrajectoryStep:
    """
    One step (one token-action decision) in a diffusion trajectory.

    Attributes
    ----------
    prompt_id     : Unique identifier for the prompt (shared across group).
    sample_id     : Unique identifier for this particular completion sample.
    step_idx      : Index of this step within the trajectory (0-based).
    state         : Diffusion state before action (e.g., masked sequence repr).
    action        : Token/action taken at this step.
    next_state    : Diffusion state after action.
    confidence    : Model confidence in the action (softmax prob of chosen token).
    old_logprob   : Log-probability under the old (generation-time) policy.
    ref_logprob   : Log-probability under the reference model (for KL).
    reward        : Scalar reward (typically only non-None at terminal step).
    done          : Whether this is the final step of the trajectory.
    group_advantage       : GRPO group-relative advantage assigned to this trajectory.
    responsibility_weight : Confidence-derived credit weight rho_t.
    final_advantage       : Combined advantage = group_advantage * responsibility_weight
                            (or delta-V / Q-V variant weighted by rho_t).
    """
    prompt_id: str
    sample_id: int
    step_idx: int
    state: Any
    action: Any
    next_state: Any
    confidence: float
    old_logprob: float
    ref_logprob: Optional[float] = None
    reward: Optional[float] = None
    done: bool = False
    group_advantage: Optional[float] = None
    responsibility_weight: Optional[float] = None
    final_advantage: Optional[float] = None


@dataclass
class TrajectoryRecord:
    """
    Full record for one sampled completion from a prompt.

    Attributes
    ----------
    prompt_id   : Identifier shared by all samples in the same GRPO group.
    sample_id   : Unique per-sample identifier within the group.
    prompt_text : Raw prompt string.
    final_text  : Final decoded completion string.
    reward      : Terminal scalar reward for this sample.
    steps       : Ordered list of TrajectoryStep objects.
    metadata    : Arbitrary extra info (e.g., diffusion step count, block config).
    """
    prompt_id: str
    sample_id: int
    prompt_text: str
    final_text: str
    reward: float
    steps: list  # list[TrajectoryStep]
    metadata: dict = field(default_factory=dict)
