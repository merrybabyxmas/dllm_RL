"""
Stage 2 diagnostic metrics for oracle-V credit assignment.

All computation is pure Python — no GPU, no torch required.

Math notation (matching paper):
  For block b (transition s_b -> s_{b+1}):
    delta_v_b  = oracle_V[b+1] - oracle_V[b]

  All tokens revealed in block b receive block-level delta_v:
    raw_adv_t  = delta_v_b

  Responsibility weight (unit-mean normalized within trajectory):
    rho_raw_t = clip((conf_t + eps)^{-alpha}, rho_clip_min, rho_clip_max)
    rho_t     = rho_raw_t / mean(rho_raw)

  Final token advantage:
    A_t = delta_v_b * rho_t

Design note on normalization:
  We do NOT z-score normalize delta_v values. Z-score normalization with a
  trajectory-level negative mean (common in wrong trajectories) would flip
  zero-delta blocks to POSITIVE values — e.g., coherent_continuation tokens
  sitting in a flat V=0.05 block would be rewarded after (0.0 - mu)/sigma
  when mu < 0. This inverts the causal credit signal. Instead, we preserve the
  raw sign of delta_v and rely on unit-mean rho normalization for scale control.

Diagnostic metrics:
  BEPR = |mean A(causal_error)| / (|mean A(coherent_continuation)| + eps)
         Blunt Error Penalization Ratio — should be >> 1
  COP  = mean |A(coherent_continuation)|  — should be small
  CBR  = fraction of correct_substep tokens with A_t > 0  — should be high
  DVSA = fraction of tokens (with non-zero delta_v) where sign(A) == sign(delta_v)
  FPR  = mean |A(formatting)|  — should be small
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from cc_rl.data.synthetic_causal_math import Trajectory, TokenAnnotation


# ---------------------------------------------------------------------------
# Responsibility weight (mirror of cc_rl.credit.responsibility but pure-Python)
# ---------------------------------------------------------------------------

def _compute_rho(
    confidences: List[float],
    alpha: float,
    eps: float,
    clip_min: float,
    clip_max: float,
) -> List[float]:
    """
    Compute normalized responsibility weights.

    rho_raw_t = clip((conf_t + eps)^{-alpha}, clip_min, clip_max)
    rho_t     = rho_raw_t / mean(rho_raw)    [unit-mean normalization]

    Low-confidence tokens get high rho (they were the pivotal decisions).
    Clipping prevents extreme values from dominating; normalization keeps
    the per-trajectory policy gradient magnitude stable.
    """
    if not confidences:
        return []

    raw: List[float] = []
    for c in confidences:
        r = (c + eps) ** (-alpha)
        r = max(clip_min, min(clip_max, r))
        raw.append(r)

    mean_rho = sum(raw) / len(raw)
    if mean_rho > 0:
        return [r / mean_rho for r in raw]
    return raw


# ---------------------------------------------------------------------------
# Core: compute delta-V advantages for a single trajectory
# ---------------------------------------------------------------------------

def compute_delta_v_advantages(
    trajectory: Trajectory,
    rho_alpha: float = 1.0,
    rho_eps: float = 1e-6,
    rho_clip_min: float = 0.25,
    rho_clip_max: float = 4.0,
) -> List[Tuple[TokenAnnotation, float]]:
    """
    Given a trajectory with oracle V values, compute block delta-V advantages
    with confidence-based responsibility weighting.

    Algorithm:
      1. For each block b in [1, T]:
           delta_v_b = oracle_V[b] - oracle_V[b-1]
         Assign delta_v_b to ALL tokens revealed in that block transition.

      2. Compute per-token rho from confidence (unit-mean normalized within trajectory).

      3. A_t = delta_v_b * rho_t   (raw delta_v, sign preserved)

    Parameters
    ----------
    trajectory   : Trajectory with populated block_states.
    rho_alpha    : Exponent for inverse-confidence weighting.
    rho_eps      : Stability offset in (conf + eps)^{-alpha}.
    rho_clip_min : Clip floor for rho before normalization.
    rho_clip_max : Clip ceil for rho before normalization.

    Returns
    -------
    List of (TokenAnnotation, advantage_value) pairs in token order.
    """
    states = trajectory.block_states
    assert len(states) >= 2, (
        f"Trajectory {trajectory.trajectory_id} has {len(states)} block states; need >= 2"
    )

    # --- Step 1: collect (token, raw_delta_v) for each block transition ---
    token_list: List[TokenAnnotation] = []
    raw_delta_v: List[float] = []

    for b in range(1, len(states)):
        dv = states[b].oracle_v - states[b - 1].oracle_v
        for tok in states[b].tokens_revealed:
            token_list.append(tok)
            raw_delta_v.append(dv)

    if not token_list:
        return []

    # --- Step 2: compute rho (unit-mean normalized within this trajectory) ---
    confidences = [tok.confidence for tok in token_list]
    rhos = _compute_rho(confidences, rho_alpha, rho_eps, rho_clip_min, rho_clip_max)

    # --- Step 3: A_t = delta_v_b * rho_t ---
    advantages = [dv * rho for dv, rho in zip(raw_delta_v, rhos)]

    return list(zip(token_list, advantages))


# ---------------------------------------------------------------------------
# Aggregate metrics over a list of trajectories
# ---------------------------------------------------------------------------

def compute_metrics(
    trajectories: List[Trajectory],
    rho_alpha: float = 1.0,
    rho_eps: float = 1e-6,
    rho_clip_min: float = 0.25,
    rho_clip_max: float = 4.0,
) -> dict:
    """
    Compute all Stage 2 diagnostic metrics over a list of trajectories.

    Returns
    -------
    dict with keys:
      BEPR              : float  — Blunt Error Penalization Ratio
      COP               : float  — Coherent continuation abs advantage (mean)
      CBR               : float  — Correct substep positive advantage rate
      DVSA              : float  — Delta-V sign accuracy
      FPR               : float  — Formatting token abs advantage (mean)
      adv_by_role       : dict[role -> (mean_adv, std_adv)]
      delta_v_by_role   : dict[role -> (mean_delta_v, std_delta_v)]
      rho_by_role       : dict[role -> mean_rho]
      n_tokens_by_role  : dict[role -> int]
    """
    role_advs: Dict[str, List[float]] = {}
    role_dvs: Dict[str, List[float]] = {}
    role_rhos: Dict[str, List[float]] = {}

    dvsa_correct = 0
    dvsa_total = 0

    for traj in trajectories:
        states = traj.block_states

        # Collect raw delta_v per token (for DVSA and role stats)
        all_raw_dv: List[float] = []
        all_toks: List[TokenAnnotation] = []
        for b in range(1, len(states)):
            dv = states[b].oracle_v - states[b - 1].oracle_v
            for tok in states[b].tokens_revealed:
                all_toks.append(tok)
                all_raw_dv.append(dv)

        if not all_toks:
            continue

        # Compute rho for this trajectory
        all_confs = [tok.confidence for tok in all_toks]
        all_rhos = _compute_rho(all_confs, rho_alpha, rho_eps, rho_clip_min, rho_clip_max)

        # Advantages: A_t = delta_v_b * rho_t
        all_advs = [dv * rho for dv, rho in zip(all_raw_dv, all_rhos)]

        for i, tok in enumerate(all_toks):
            role = tok.role
            if role not in role_advs:
                role_advs[role] = []
                role_dvs[role] = []
                role_rhos[role] = []

            role_advs[role].append(all_advs[i])
            role_dvs[role].append(all_raw_dv[i])
            role_rhos[role].append(all_rhos[i])

            # DVSA: count tokens with non-zero delta_v
            raw_dv = all_raw_dv[i]
            if abs(raw_dv) > 1e-9:
                dvsa_total += 1
                # Sign of advantage must match sign of raw delta_v
                # A_t = delta_v * rho; rho > 0 always, so sign(A_t) == sign(delta_v)
                if (all_advs[i] > 0) == (raw_dv > 0):
                    dvsa_correct += 1

    # -----------------------------------------------------------------------
    # Aggregate per-role statistics
    # -----------------------------------------------------------------------
    def _stats(values: List[float]) -> Tuple[float, float]:
        if not values:
            return (0.0, 0.0)
        mu = sum(values) / len(values)
        var = sum((x - mu) ** 2 for x in values) / len(values)
        return (mu, math.sqrt(var))

    adv_by_role: Dict[str, Tuple[float, float]] = {}
    delta_v_by_role: Dict[str, Tuple[float, float]] = {}
    rho_by_role: Dict[str, float] = {}
    n_tokens_by_role: Dict[str, int] = {}

    for role in role_advs:
        adv_by_role[role] = _stats(role_advs[role])
        delta_v_by_role[role] = _stats(role_dvs[role])
        rho_by_role[role] = sum(role_rhos[role]) / len(role_rhos[role])
        n_tokens_by_role[role] = len(role_advs[role])

    # -----------------------------------------------------------------------
    # BEPR: |mean A(causal_error)| / (|mean A(coherent_continuation)| + eps)
    # -----------------------------------------------------------------------
    mean_adv_causal   = adv_by_role.get("causal_error", (0.0, 0.0))[0]
    mean_adv_coherent = adv_by_role.get("coherent_continuation", (0.0, 0.0))[0]
    bepr = abs(mean_adv_causal) / (abs(mean_adv_coherent) + 1e-8)

    # -----------------------------------------------------------------------
    # COP: mean |A(coherent_continuation)|
    # -----------------------------------------------------------------------
    cop_vals = role_advs.get("coherent_continuation", [])
    cop = sum(abs(a) for a in cop_vals) / (len(cop_vals) + 1e-8)

    # -----------------------------------------------------------------------
    # CBR: fraction of correct_substep tokens with A_t > 0
    # -----------------------------------------------------------------------
    substep_vals = role_advs.get("correct_substep", [])
    cbr = sum(1 for a in substep_vals if a > 0) / (len(substep_vals) + 1e-8)

    # -----------------------------------------------------------------------
    # DVSA: delta-V sign accuracy (over non-zero-delta tokens only)
    # -----------------------------------------------------------------------
    dvsa = dvsa_correct / (dvsa_total + 1e-8)

    # -----------------------------------------------------------------------
    # FPR: mean |A(formatting)|
    # -----------------------------------------------------------------------
    fmt_vals = role_advs.get("formatting", [])
    fpr = sum(abs(a) for a in fmt_vals) / (len(fmt_vals) + 1e-8)

    return {
        "BEPR": bepr,
        "COP":  cop,
        "CBR":  cbr,
        "DVSA": dvsa,
        "FPR":  fpr,
        "adv_by_role":      adv_by_role,
        "delta_v_by_role":  delta_v_by_role,
        "rho_by_role":      rho_by_role,
        "n_tokens_by_role": n_tokens_by_role,
    }


# ---------------------------------------------------------------------------
# Tier 1 pass/fail criteria
# ---------------------------------------------------------------------------

TIER1_PASS_CRITERIA: Dict[str, Tuple[float, str]] = {
    # metric: (threshold, direction)   direction: ">=" or "<="
    "BEPR": (5.0,  ">="),
    "COP":  (0.05, "<="),
    "CBR":  (0.90, ">="),
    "FPR":  (0.10, "<="),
    "DVSA": (0.90, ">="),
}


def check_tier1(metrics: dict) -> Tuple[bool, Dict[str, Tuple[float, bool]]]:
    """
    Check Tier 1 pass criteria against computed metrics.

    Returns
    -------
    (all_passed, {criterion: (value, passed)})
    """
    results: Dict[str, Tuple[float, bool]] = {}
    all_passed = True

    for criterion, (threshold, direction) in TIER1_PASS_CRITERIA.items():
        value = metrics.get(criterion, 0.0)
        if direction == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        results[criterion] = (value, passed)
        if not passed:
            all_passed = False

    return all_passed, results


# ---------------------------------------------------------------------------
# Per-type metric computation (used by the diagnostic runner)
# ---------------------------------------------------------------------------

def compute_metrics_by_type(
    trajectories: List[Trajectory],
    rho_alpha: float = 1.0,
    **rho_kwargs,
) -> Dict[str, dict]:
    """
    Compute metrics separately for each error type (A-E) and for correct trajectories.

    For per-type BEPR/COP/CBR we include all correct trajectories alongside each
    wrong type, because BEPR requires both causal_error and coherent_continuation
    tokens to be present in the same evaluation set.

    Returns dict mapping type label -> metrics dict.
    """
    by_type: Dict[str, List[Trajectory]] = {t: [] for t in ("A", "B", "C", "D", "E", "correct")}
    for traj in trajectories:
        if traj.is_correct:
            by_type["correct"].append(traj)
        elif traj.error_type is not None:
            by_type[traj.error_type].append(traj)

    type_metrics: Dict[str, dict] = {}
    for t in ("A", "B", "C", "D", "E"):
        subset = by_type["correct"] + by_type[t]
        if subset:
            type_metrics[t] = compute_metrics(subset, rho_alpha=rho_alpha, **rho_kwargs)
        else:
            type_metrics[t] = {}

    type_metrics["correct"] = compute_metrics(
        by_type["correct"], rho_alpha=rho_alpha, **rho_kwargs
    )
    return type_metrics
