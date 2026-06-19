"""
Stage 1 (CW-GRPO) smoke tests on the toy "2 + 3 = ?" fixture.

Verifies that confidence-weighted GRPO advantages are computed correctly.

Reference computation (all values derived analytically):

Group (4 samples, 1 reward=1, 3 reward=0):
  mu = (1 + 0 + 0 + 0) / 4 = 0.25
  var = ((1-0.25)^2 + 3*(0-0.25)^2) / 4 = (0.5625 + 0.1875) / 4 = 0.1875
  sigma = sqrt(0.1875) = 0.433...
  A1 = (1 - 0.25) / 0.433 = +1.732...
  A2 = (0 - 0.25) / 0.433 = -0.577...
  A3 = A4 = -0.577...

Sample 1 (reward=1, A=+1.732, steps: conf=[0.55, 0.90]):
  rho_raw = [1/0.55, 1/0.90] = [1.818, 1.111]
  mean_rho = (1.818 + 1.111) / 2 = 1.465
  w = [1.818/1.465, 1.111/1.465] = [1.241, 0.758]
  A("+") = 1.732051 * 1.241379 = 2.150132...
  A("5") = 1.732051 * 0.758621 = 1.313970...
  (Note: spec incorrectly states 1.312; correct value is 1.314 to 3dp)

Sample 2 (reward=0, A=-0.577, steps: conf=[0.20, 0.95]):
  rho_raw = [1/0.20, 1/0.95] = [5.0, 1.053]
  mean_rho = (5.0 + 1.053) / 2 = 3.026
  w = [5.0/3.026, 1.053/3.026] = [1.652, 0.348]
  A("*") = -0.577 * 1.652 = -0.954
  A("6") = -0.577 * 0.348 = -0.201

Sample 3 (reward=0, A=-0.577, steps: conf=[0.60, 0.35]):
  rho_raw = [1/0.60, 1/0.35] = [1.667, 2.857]
  mean_rho = (1.667 + 2.857) / 2 = 2.262
  w = [1.667/2.262, 2.857/2.262] = [0.737, 1.263]
  A("+") = -0.577 * 0.737 = -0.425
  A("6") = -0.577 * 1.263 = -0.729

Sample 4 (reward=0, A=-0.577, steps: conf=[0.25, 0.80]):
  rho_raw = [1/0.25, 1/0.80] = [4.0, 1.25]
  mean_rho = (4.0 + 1.25) / 2 = 2.625
  w = [4.0/2.625, 1.25/2.625] = [1.524, 0.476]
  A("-") = -0.577 * 1.524 = -0.880
  A("-1") = -0.577 * 0.476 = -0.275
"""
import pytest
import os

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "toy_2_plus_3.jsonl")


def test_stage1_cw_grpo_toy():
    """Full Stage 1 pipeline on toy fixture with exact numerical verification."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3, collect_advantages_by_sample_action
    from cc_rl.credit.advantages import assign_group_advantages, assign_confidence_weighted_advantages

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)

    # Step 1: GRPO group-relative advantages
    assign_group_advantages(trajectories, eps=1e-12)

    # Step 2: Confidence-weighted per-token advantages
    assign_confidence_weighted_advantages(
        trajectories,
        alpha=1.0,
        eps=0.0,
        clip_min=0.0,
        clip_max=999.0,
        normalize=True,
    )

    adv = collect_advantages_by_sample_action(trajectories)

    # -------------------------------------------------------------------
    # Core assertions from spec (must pass)
    # -------------------------------------------------------------------
    # Sample 2: wrong operator "*" should be strongly penalized
    assert adv[2]["*"] == pytest.approx(-0.954, abs=1e-3), \
        f"adv[2]['*'] = {adv[2]['*']:.4f}, expected -0.954"
    assert adv[2]["6"] == pytest.approx(-0.201, abs=1e-3), \
        f"adv[2]['6'] = {adv[2]['6']:.4f}, expected -0.201"

    # The confident wrong answer "6" should get 4x+ less penalty than the
    # uncertain wrong operator "*"
    assert abs(adv[2]["*"]) > 4.0 * abs(adv[2]["6"]), (
        f"|adv[2]['*']| = {abs(adv[2]['*']):.4f} should be > 4x |adv[2]['6']| = {abs(adv[2]['6']):.4f}"
    )

    # -------------------------------------------------------------------
    # Additional spec assertions
    # -------------------------------------------------------------------
    # Sample 1: correct answer, both should be positive
    assert adv[1]["+"] == pytest.approx(2.151, abs=1e-3), \
        f"adv[1]['+'] = {adv[1]['+']:.4f}, expected 2.151"
    # Exact: A1=1.732051, w("5")=1/0.90/(mean([1/0.55,1/0.90]))=0.75862
    # A("5") = 1.732051 * 0.75862 = 1.31397  (spec says 1.312, which is a rounding error in spec)
    assert adv[1]["5"] == pytest.approx(1.314, abs=1e-3), \
        f"adv[1]['5'] = {adv[1]['5']:.4f}, expected 1.314"
    assert adv[1]["+"] > 0 and adv[1]["5"] > 0, "Correct sample must have positive advantages"

    # Sample 3: wrong final answer "6" (low confidence) gets more penalty than "+" (higher conf)
    assert adv[3]["+"] == pytest.approx(-0.425, abs=1e-3), \
        f"adv[3]['+'] = {adv[3]['+']:.4f}, expected -0.425"
    assert adv[3]["6"] == pytest.approx(-0.729, abs=1e-3), \
        f"adv[3]['6'] = {adv[3]['6']:.4f}, expected -0.729"

    # Sample 4: subtraction operator (very low conf=0.25) gets high weight
    assert adv[4]["-"] == pytest.approx(-0.880, abs=1e-3), \
        f"adv[4]['-'] = {adv[4]['-']:.4f}, expected -0.880"
    assert adv[4]["-1"] == pytest.approx(-0.275, abs=1e-3), \
        f"adv[4]['-1'] = {adv[4]['-1']:.4f}, expected -0.275"

    # Sanity: low-confidence decisions get higher magnitude penalties
    assert abs(adv[4]["-"]) > abs(adv[4]["-1"]), \
        "Low-confidence '-' (0.25) should have higher magnitude than high-confidence '-1' (0.80)"


def test_stage1_group_advantage_sanity():
    """Group advantages should sum to zero for equal-sized groups."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    from cc_rl.credit.advantages import assign_group_advantages

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)
    assign_group_advantages(trajectories, eps=1e-12)

    group_advs = [t.group_advantage for t in trajectories]
    total = sum(group_advs)
    # In a 4-sample group with mean subtracted, sum = 0
    assert total == pytest.approx(0.0, abs=1e-6), \
        f"Group advantages sum to {total}, expected 0.0"


def test_stage1_correct_sample_positive_advantage():
    """The correct sample (sample_id=1) must have positive group advantage."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    from cc_rl.credit.advantages import assign_group_advantages

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)
    assign_group_advantages(trajectories, eps=1e-12)

    for traj in trajectories:
        if traj.sample_id == 1:
            assert traj.group_advantage > 0, \
                f"Correct sample should have positive advantage, got {traj.group_advantage}"
        else:
            assert traj.group_advantage < 0, \
                f"Wrong sample {traj.sample_id} should have negative advantage"
