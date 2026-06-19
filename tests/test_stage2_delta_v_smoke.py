"""
Stage 2 Tier 0 pure-math smoke tests — no model, no GPU, no torch required.

Tests the oracle-V delta-V credit assignment math using synthetic causal math
trajectories. Every test is deterministic (seed=42).

Reference math for advantage computation:
  delta_v_b  = oracle_V[b+1] - oracle_V[b]            (per-block)
  delta_v_norm_t = (delta_v_t - mu) / (sigma + 1e-8)  (trajectory-level normalization)
  rho_t      = clip((conf + eps)^{-alpha}, lo, hi) / mean(rho)
  A_t        = delta_v_norm_t * rho_t
"""
from __future__ import annotations

import math
import pytest

from cc_rl.data.synthetic_causal_math import (
    Trajectory,
    TokenAnnotation,
    BlockState,
    make_dataset,
    ORACLE_V,
)
from cc_rl.diagnostic.metrics import (
    compute_delta_v_advantages,
    compute_metrics,
    check_tier1,
    TIER1_PASS_CRITERIA,
)


# ---------------------------------------------------------------------------
# Fixture: one canonical trajectory of each relevant type
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Full dataset, 8 pairs per type (80 trajectories total)."""
    return make_dataset(seed=42, n_per_type=8)


@pytest.fixture(scope="module")
def correct_trajs(dataset):
    return [t for t in dataset if t.is_correct]


@pytest.fixture(scope="module")
def wrong_trajs(dataset):
    return [t for t in dataset if not t.is_correct]


@pytest.fixture(scope="module")
def type_a_wrong(dataset):
    return [t for t in dataset if t.error_type == "A"]


@pytest.fixture(scope="module")
def type_c_wrong(dataset):
    return [t for t in dataset if t.error_type == "C"]


@pytest.fixture(scope="module")
def type_d_wrong(dataset):
    return [t for t in dataset if t.error_type == "D"]


@pytest.fixture(scope="module")
def type_e_correct(dataset):
    return [t for t in dataset if t.is_correct and t.trajectory_id.startswith("E")]


@pytest.fixture(scope="module")
def all_metrics(dataset):
    return compute_metrics(dataset)


# ---------------------------------------------------------------------------
# Helper: get token-advantage pairs filtered by role
# ---------------------------------------------------------------------------

def _tok_advs_by_role(tok_adv_pairs, role: str):
    return [a for tok, a in tok_adv_pairs if tok.role == role]


# ---------------------------------------------------------------------------
# Test 1: correct_branch tokens must get positive delta-V advantage
# ---------------------------------------------------------------------------

def test_correct_branch_positive_advantage(correct_trajs):
    """
    In every correct trajectory, the cumulative oracle-V goes up (0.30 -> 1.0).
    After trajectory-level normalization, the mean delta_v for correct_branch
    tokens must be positive.
    """
    all_correct_branch_advs = []
    for traj in correct_trajs:
        tok_adv = compute_delta_v_advantages(traj)
        advs = _tok_advs_by_role(tok_adv, "correct_branch")
        all_correct_branch_advs.extend(advs)

    assert len(all_correct_branch_advs) > 0, "No correct_branch tokens found"
    mean_adv = sum(all_correct_branch_advs) / len(all_correct_branch_advs)
    assert mean_adv > 0.0, (
        f"Expected positive mean advantage for correct_branch tokens, got {mean_adv:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: causal_error tokens must get negative delta-V advantage
# ---------------------------------------------------------------------------

def test_causal_error_negative_advantage(wrong_trajs):
    """
    causal_error tokens are located in blocks where oracle_V drops (after_wrong_op
    or after_arithmetic_slip). After normalization, their advantage must be negative.
    """
    all_causal_advs = []
    for traj in wrong_trajs:
        tok_adv = compute_delta_v_advantages(traj)
        advs = _tok_advs_by_role(tok_adv, "causal_error")
        all_causal_advs.extend(advs)

    assert len(all_causal_advs) > 0, "No causal_error tokens found"
    mean_adv = sum(all_causal_advs) / len(all_causal_advs)
    assert mean_adv < 0.0, (
        f"Expected negative mean advantage for causal_error tokens, got {mean_adv:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: coherent_continuation tokens must have near-zero absolute advantage
# ---------------------------------------------------------------------------

def test_coherent_continuation_near_zero(wrong_trajs):
    """
    After the wrong branch is taken, the model coherently follows through.
    Oracle V stays near 0.05 (already near zero) so delta_v for these blocks
    is approximately zero. Their normalized advantage should be |A| < 0.5
    (lenient threshold since normalization can amplify if std is small).

    More importantly, the MEAN |A| should be < COP threshold (0.05 on full dataset).
    We test the per-trajectory bound here, not the aggregate.
    """
    all_coherent_advs = []
    for traj in wrong_trajs:
        tok_adv = compute_delta_v_advantages(traj)
        advs = _tok_advs_by_role(tok_adv, "coherent_continuation")
        all_coherent_advs.extend(advs)

    assert len(all_coherent_advs) > 0, "No coherent_continuation tokens found"
    mean_abs = sum(abs(a) for a in all_coherent_advs) / len(all_coherent_advs)

    # The COP threshold from TIER1 is 0.05, but per-trajectory normalization
    # can scale things. We assert that coherent |A| is much smaller than causal |A|.
    all_causal_advs = []
    for traj in wrong_trajs:
        tok_adv = compute_delta_v_advantages(traj)
        advs = _tok_advs_by_role(tok_adv, "causal_error")
        all_causal_advs.extend(advs)

    mean_causal_abs = sum(abs(a) for a in all_causal_advs) / (len(all_causal_advs) + 1e-8)

    # coherent |A| must be at least 5x smaller than causal |A|
    assert mean_abs * 5.0 < mean_causal_abs, (
        f"coherent_continuation mean |A| = {mean_abs:.4f} is not << "
        f"causal_error mean |A| = {mean_causal_abs:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: correct_substep in failed trajectory must have positive advantage (Type D)
# ---------------------------------------------------------------------------

def test_correct_substep_in_failed_traj_positive(type_d_wrong):
    """
    Type D: the first block is a correct substep even in the failing trajectory.
    oracle_V goes 0.30 -> 0.70 in that block (after_correct_substep > initial).
    delta_v = +0.40 (positive) -> after normalization, A_t for correct_substep > 0.
    """
    all_substep_advs = []
    for traj in type_d_wrong:
        tok_adv = compute_delta_v_advantages(traj)
        advs = _tok_advs_by_role(tok_adv, "correct_substep")
        # For Type D wrong, the FIRST block has correct_substep tokens
        # The SECOND block also has correct_substep tokens mixed with causal_error
        # We only want the first block (all correct_substep before causal_error appears)
        # Extract from first block explicitly
        b1_toks = traj.block_states[1].tokens_revealed  # block 0->1
        b1_advs = [a for tok, a in tok_adv if tok in b1_toks and tok.role == "correct_substep"]
        all_substep_advs.extend(b1_advs)

    assert len(all_substep_advs) > 0, "No correct_substep tokens found in Type D block 1"
    mean_adv = sum(all_substep_advs) / len(all_substep_advs)
    assert mean_adv > 0.0, (
        f"Expected positive advantage for correct_substep in failed traj (Type D), "
        f"got mean A = {mean_adv:.4f}. "
        f"oracle_V should go 0.30->0.70 (+0.40) for this block."
    )


# ---------------------------------------------------------------------------
# Test 5: formatting tokens must have low absolute advantage (Type E)
# ---------------------------------------------------------------------------

def test_formatting_tokens_low_weight(type_e_correct):
    """
    Type E formatting trap: tokens like '$', '.', '00' are in a block where
    oracle_V is UNCHANGED (0.75 -> 0.75, delta_v = 0.0).
    delta_v_norm for formatting = (0.0 - mu) / sigma.
    If the trajectory has other blocks with non-zero delta_v, after normalization
    the formatting block's delta_v is negative (pulled toward mean), so A_t ≈ 0 or slightly negative.
    The KEY is that |A(formatting)| << |A(correct_branch)| in the same trajectory.
    """
    fmt_abs_advs = []
    content_abs_advs = []
    for traj in type_e_correct:
        tok_adv = compute_delta_v_advantages(traj)
        fmt_abs_advs.extend([abs(a) for tok, a in tok_adv if tok.role == "formatting"])
        content_abs_advs.extend([abs(a) for tok, a in tok_adv if tok.role == "correct_branch"])

    assert len(fmt_abs_advs) > 0, "No formatting tokens found in Type E"
    assert len(content_abs_advs) > 0, "No correct_branch tokens found in Type E"

    mean_fmt = sum(fmt_abs_advs) / len(fmt_abs_advs)
    mean_content = sum(content_abs_advs) / len(content_abs_advs)

    # Formatting absolute advantage must be less than content absolute advantage
    assert mean_fmt < mean_content, (
        f"Formatting tokens should have lower |A| than content tokens. "
        f"|A(format)|={mean_fmt:.4f}, |A(content)|={mean_content:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: BEPR must exceed threshold on all trajectory types
# ---------------------------------------------------------------------------

def test_bepr_exceeds_threshold(dataset):
    """
    BEPR = |mean A(causal_error)| / (|mean A(coherent_continuation)| + eps)
    Must be >= 5.0 on the full dataset.
    """
    metrics = compute_metrics(dataset)
    bepr = metrics["BEPR"]
    threshold = TIER1_PASS_CRITERIA["BEPR"][0]
    assert bepr >= threshold, (
        f"BEPR = {bepr:.3f} is below threshold {threshold}. "
        f"causal_error tokens are not being penalized strongly enough relative to "
        f"coherent_continuation tokens."
    )


# ---------------------------------------------------------------------------
# Test 7: CBR must exceed threshold
# ---------------------------------------------------------------------------

def test_cbr_exceeds_threshold(dataset):
    """
    CBR (Correct substep Benefit Rate) = fraction of correct_substep tokens with A_t > 0.
    Must be >= 0.90. This is the key property: even in failing trajectories, correct
    intermediate steps should receive positive credit.
    """
    metrics = compute_metrics(dataset)
    cbr = metrics["CBR"]
    threshold = TIER1_PASS_CRITERIA["CBR"][0]
    assert cbr >= threshold, (
        f"CBR = {cbr:.3f} is below threshold {threshold}. "
        f"Correct substep tokens should mostly receive positive advantage."
    )


# ---------------------------------------------------------------------------
# Test 8: DVSA must exceed threshold
# ---------------------------------------------------------------------------

def test_dvsa_exceeds_threshold(dataset):
    """
    DVSA (Delta-V Sign Accuracy) = fraction of tokens where sign(A_t) == sign(oracle delta_V).
    Must be >= 0.90. This checks that the normalization doesn't flip credit signs.
    """
    metrics = compute_metrics(dataset)
    dvsa = metrics["DVSA"]
    threshold = TIER1_PASS_CRITERIA["DVSA"][0]
    assert dvsa >= threshold, (
        f"DVSA = {dvsa:.3f} is below threshold {threshold}. "
        f"Normalized advantages should preserve the sign of oracle delta-V."
    )


# ---------------------------------------------------------------------------
# Test 9: Type E formatting trap — format vs content
# ---------------------------------------------------------------------------

def test_type_e_formatting_trap(type_e_correct):
    """
    In Type E correct trajectories:
      - Block 1 (correct arithmetic): delta_v = 0.75 - 0.30 = +0.45  [content tokens]
      - Block 2 (formatting "$7.00"): delta_v = 0.75 - 0.75 = 0.00   [formatting tokens]
      - Block 3 (terminal answer):    delta_v = 1.00 - 0.75 = +0.25  [content tokens]

    After normalization: formatting block has delta_v = 0 → normalized toward negative
    (pulled below mean). So |A(formatting)| should be substantially smaller than
    |A(correct_branch)|.
    """
    for traj in type_e_correct:
        tok_adv = compute_delta_v_advantages(traj)

        fmt_vals = [abs(a) for tok, a in tok_adv if tok.role == "formatting"]
        cb_vals = [abs(a) for tok, a in tok_adv if tok.role == "correct_branch"]

        if not fmt_vals or not cb_vals:
            continue

        mean_fmt = sum(fmt_vals) / len(fmt_vals)
        mean_cb = sum(cb_vals) / len(cb_vals)

        assert mean_fmt < mean_cb, (
            f"Trajectory {traj.trajectory_id}: "
            f"formatting |A|={mean_fmt:.4f} should be < content |A|={mean_cb:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 10: mathematical consistency — advantage sum relates to reward signal
# ---------------------------------------------------------------------------

def test_advantage_sum_monotone_with_reward():
    """
    Correct trajectories (reward=1) must have higher mean advantage than wrong
    trajectories (reward=0). This is the ultimate sanity check: credit assignment
    should distinguish good from bad trajectories.
    """
    dataset = make_dataset(seed=42, n_per_type=8)
    correct_mean_advs = []
    wrong_mean_advs = []

    for traj in dataset:
        tok_adv = compute_delta_v_advantages(traj)
        if not tok_adv:
            continue
        mean_adv = sum(a for _, a in tok_adv) / len(tok_adv)
        if traj.is_correct:
            correct_mean_advs.append(mean_adv)
        else:
            wrong_mean_advs.append(mean_adv)

    assert correct_mean_advs and wrong_mean_advs
    mean_correct = sum(correct_mean_advs) / len(correct_mean_advs)
    mean_wrong = sum(wrong_mean_advs) / len(wrong_mean_advs)

    assert mean_correct > mean_wrong, (
        f"Correct trajectories should have higher mean advantage than wrong ones. "
        f"mean_correct={mean_correct:.4f}, mean_wrong={mean_wrong:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 11: rho weights are always positive (numerical stability)
# ---------------------------------------------------------------------------

def test_rho_weights_always_positive():
    """Responsibility weights must always be positive (clip_min > 0 ensures this)."""
    from cc_rl.diagnostic.metrics import _compute_rho

    # Edge cases: very high confidence (rho approaches clip_min)
    # and very low confidence (rho approaches clip_max)
    confs = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    rhos = _compute_rho(confs, alpha=1.0, eps=1e-6, clip_min=0.25, clip_max=4.0)

    assert all(r > 0 for r in rhos), f"All rho must be positive, got {rhos}"
    assert all(math.isfinite(r) for r in rhos), f"All rho must be finite, got {rhos}"


# ---------------------------------------------------------------------------
# Test 12: block state oracle V consistency
# ---------------------------------------------------------------------------

def test_oracle_v_consistency():
    """
    Verify that oracle_V in block_states follows the expected oracle V rules:
    - initial state always has V=0.30
    - terminal correct trajectory ends with V=1.0
    - terminal wrong trajectory ends with V=0.0
    """
    dataset = make_dataset(seed=42, n_per_type=8)
    for traj in dataset:
        states = traj.block_states
        # Initial state
        assert states[0].oracle_v == pytest.approx(ORACLE_V["initial"]), (
            f"{traj.trajectory_id}: initial oracle_v = {states[0].oracle_v}"
        )
        # Terminal state
        expected_terminal = ORACLE_V["terminal_correct"] if traj.is_correct else ORACLE_V["terminal_wrong"]
        assert states[-1].oracle_v == pytest.approx(expected_terminal), (
            f"{traj.trajectory_id}: terminal oracle_v = {states[-1].oracle_v}, "
            f"expected {expected_terminal}"
        )
