"""
Advantage assignment functions for Stages 1, 2, and 3.

Stage 1 (CW-GRPO)   : group-relative z-score * per-token responsibility weight
Stage 2 (Value Cr.) : delta-V local advantage * per-token responsibility weight
Stage 3 (Q Cr.)     : (Q(s,a) - V(s)) * per-token responsibility weight

All stages share:
  - assign_group_advantages() for GRPO baseline normalization
  - compute_responsibility_weights() for confidence -> weight mapping
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, List

from cc_rl.credit.responsibility import compute_responsibility_weights
from cc_rl.rollout.trajectory import TrajectoryRecord


# ---------------------------------------------------------------------------
# Shared: GRPO group-relative advantage
# ---------------------------------------------------------------------------

def assign_group_advantages(
    trajectories: List[TrajectoryRecord],
    eps: float = 1e-6,
) -> None:
    """
    Compute group-relative z-score advantages per prompt_id and write them
    into traj.group_advantage and step.group_advantage for every step.

    A_i = (r_i - mu_g) / (sigma_g + eps)

    where mu_g, sigma_g are the mean and std of rewards within the group
    sharing prompt_id.

    Parameters
    ----------
    trajectories : List of TrajectoryRecord objects (mixed prompt_ids allowed).
    eps          : Smoothing term for numerical stability.

    Mutates
    -------
    traj.group_advantage and step.group_advantage for every traj/step.
    """
    # Group by prompt_id
    groups: dict = defaultdict(list)
    for traj in trajectories:
        groups[traj.prompt_id].append(traj)

    for prompt_id, group in groups.items():
        rewards = [t.reward for t in group]
        n = len(rewards)
        mu = sum(rewards) / n
        var = sum((r - mu) ** 2 for r in rewards) / n
        sigma = var ** 0.5

        for traj in group:
            adv = (traj.reward - mu) / (sigma + eps)
            traj.group_advantage = adv
            for step in traj.steps:
                step.group_advantage = adv


# ---------------------------------------------------------------------------
# Stage 1: CW-GRPO
# ---------------------------------------------------------------------------

def assign_confidence_weighted_advantages(
    trajectories: List[TrajectoryRecord],
    alpha: float = 1.0,
    eps: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 999.0,
    normalize: bool = True,
) -> None:
    """
    Stage 1: Confidence-Weighted GRPO.

    final_advantage_t = group_advantage * rho_t

    where rho_t = compute_responsibility_weights(confidences)[t].

    Requires assign_group_advantages() to have been called first so that
    traj.group_advantage is populated.

    Mutates
    -------
    step.responsibility_weight and step.final_advantage for every step.
    """
    for traj in trajectories:
        confidences = [step.confidence for step in traj.steps]
        weights = compute_responsibility_weights(
            confidences,
            alpha=alpha,
            eps=eps,
            clip_min=clip_min,
            clip_max=clip_max,
            normalize=normalize,
        )
        for step, w in zip(traj.steps, weights):
            step.responsibility_weight = w
            step.final_advantage = traj.group_advantage * w


# ---------------------------------------------------------------------------
# Stage 2: Delta-V credit
# ---------------------------------------------------------------------------

def assign_delta_v_advantages(
    trajectories: List[TrajectoryRecord],
    value_fn: Callable,
    alpha: float = 1.0,
    eps: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 999.0,
    normalize: bool = True,
) -> None:
    """
    Stage 2: State-Value Confidence Credit.

    For non-terminal steps:
        local_adv_t = V(s_{t+1}) - V(s_t)          [TD(0) Bellman residual]
    For terminal step (done=True):
        local_adv_t = r - V(s_t)                    [actual vs. predicted value]

    final_advantage_t = local_adv_t * rho_t

    Does NOT require assign_group_advantages() to be called first — it uses
    the value function as a per-step baseline instead of the group mean.

    Parameters
    ----------
    value_fn : Callable(state) -> float  (e.g., TabularValue or neural ValueHead)
    """
    for traj in trajectories:
        confidences = [step.confidence for step in traj.steps]
        weights = compute_responsibility_weights(
            confidences,
            alpha=alpha,
            eps=eps,
            clip_min=clip_min,
            clip_max=clip_max,
            normalize=normalize,
        )
        for step, w in zip(traj.steps, weights):
            v_s = value_fn(step.state)
            if step.done:
                # Terminal: use actual reward as return
                local_adv = traj.reward - v_s
            else:
                # Non-terminal: one-step TD residual
                v_next = value_fn(step.next_state)
                local_adv = v_next - v_s
            step.responsibility_weight = w
            step.final_advantage = local_adv * w


# ---------------------------------------------------------------------------
# Stage 3: Q - V credit
# ---------------------------------------------------------------------------

def assign_q_minus_v_advantages(
    trajectories: List[TrajectoryRecord],
    value_fn: Callable,
    q_fn: Callable,
    alpha: float = 1.0,
    eps: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 999.0,
    normalize: bool = True,
) -> None:
    """
    Stage 3: Q-Value Confidence Credit.

    local_adv_t = Q(s_t, a_t) - V(s_t)           [advantage = policy gradient]

    final_advantage_t = local_adv_t * rho_t

    Parameters
    ----------
    value_fn : Callable(state) -> float
    q_fn     : Callable(state, action) -> float
    """
    for traj in trajectories:
        confidences = [step.confidence for step in traj.steps]
        weights = compute_responsibility_weights(
            confidences,
            alpha=alpha,
            eps=eps,
            clip_min=clip_min,
            clip_max=clip_max,
            normalize=normalize,
        )
        for step, w in zip(traj.steps, weights):
            q_sa = q_fn(step.state, step.action)
            v_s = value_fn(step.state)
            step.responsibility_weight = w
            step.final_advantage = (q_sa - v_s) * w
