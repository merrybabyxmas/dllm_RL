"""
Stage 2 (Value Credit) smoke tests on the toy "2 + 3 = ?" fixture.

Verifies that delta-V per-token advantages weighted by responsibility are
computed correctly with a tabular value function.

Reference computation (alpha=1.0, eps=0.0, clip_min=0.0, clip_max=999.0):

Value table:
  V("[2 □ 3 = □]") = 0.25
  V("[2 + 3 = □]") = 0.80
  V("[2 * 3 = □]") = 0.05
  V("[2 - 3 = □]") = 0.00

Sample 2 (reward=0, conf=[0.20, 0.95]):
  Step 1 (non-terminal "*"): delta_V = V("[2*3=□]") - V("[2□3=□]") = 0.05 - 0.25 = -0.20
  Step 2 (terminal "6"):    local_adv = reward - V(s_T) = 0 - 0.05 = -0.05
  weights (normalized): [1.652, 0.348]
  A("*") = -0.20 * 1.652 = -0.330
  A("6") = -0.05 * 0.348 = -0.0174  ≈ -0.017

Sample 3 (reward=0, conf=[0.60, 0.35]):
  Step 1 ("+"): delta_V = V("[2+3=□]") - V("[2□3=□]") = 0.80 - 0.25 = +0.55
  Step 2 ("6", terminal): local_adv = 0 - V("[2+3=□]") = 0 - 0.80 = -0.80
  weights (normalized): [0.737, 1.263]
  A("+") = 0.55 * 0.737 = 0.405
  A("6") = -0.80 * 1.263 = -1.010
"""
import pytest
import os

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "toy_2_plus_3.jsonl")


def test_stage2_value_credit_toy():
    """Full Stage 2 pipeline on toy fixture with exact numerical verification."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3, collect_advantages_by_sample_action
    from cc_rl.credit.advantages import assign_delta_v_advantages
    from cc_rl.critics.tabular_toy import TabularValue

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)

    value_fn = TabularValue({
        "[2 □ 3 = □]": 0.25,
        "[2 + 3 = □]": 0.80,
        "[2 * 3 = □]": 0.05,
        "[2 - 3 = □]": 0.00,
    })

    assign_delta_v_advantages(
        trajectories,
        value_fn,
        alpha=1.0,
        eps=0.0,
        clip_min=0.0,
        clip_max=999.0,
        normalize=True,
    )

    adv = collect_advantages_by_sample_action(trajectories)

    # -------------------------------------------------------------------
    # Core assertions from spec
    # -------------------------------------------------------------------
    assert adv[2]["*"] == pytest.approx(-0.330, abs=1e-3), \
        f"adv[2]['*'] = {adv[2]['*']:.4f}, expected -0.330"
    assert adv[2]["6"] == pytest.approx(-0.017, abs=1e-3), \
        f"adv[2]['6'] = {adv[2]['6']:.4f}, expected -0.017"

    assert adv[3]["+"] == pytest.approx(+0.405, abs=1e-3), \
        f"adv[3]['+'] = {adv[3]['+']:.4f}, expected +0.405"
    assert adv[3]["6"] == pytest.approx(-1.010, abs=1e-3), \
        f"adv[3]['6'] = {adv[3]['6']:.4f}, expected -1.010"

    # -------------------------------------------------------------------
    # Semantic acceptance criteria
    # -------------------------------------------------------------------
    # Wrong operator "*" transitions to a bad state (V=0.05), should be penalized
    assert adv[2]["*"] < adv[2]["6"], \
        "Operator '*' leading to bad state should be penalized more than its final token"

    # Confident wrong token "6" at terminal (near-zero disadvantage since V≈0 penalty small)
    assert abs(adv[2]["6"]) < 0.05, \
        "High-confidence final wrong token should get near-zero penalty (terminal advantage ≈ 0)"

    # Correct operator "+" transitions to a good state (V=0.80), should be rewarded
    assert adv[3]["+"] > 0.0, \
        "Operator '+' leading to good state should have positive advantage"

    # Low-confidence wrong terminal "6" in good state should be heavily penalized
    assert adv[3]["6"] < -1.0, \
        "Low-confidence wrong answer '6' in promising state should be strongly penalized"


def test_stage2_tabular_value_unknown_state():
    """Unknown states should default to V=0.0."""
    from cc_rl.critics.tabular_toy import TabularValue
    v = TabularValue({"s0": 1.0})
    assert v("unknown_state") == 0.0
    assert v("s0") == 1.0


def test_stage2_terminal_uses_reward():
    """Terminal steps use (reward - V(s)) not (V(next) - V(s))."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    from cc_rl.credit.advantages import assign_delta_v_advantages
    from cc_rl.critics.tabular_toy import TabularValue

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)

    # Value function that makes V(terminal) = 0.5 for sample 1
    # Terminal state for sample 1 is "[2 + 3 = 5]"
    value_fn = TabularValue({
        "[2 □ 3 = □]": 0.5,
        "[2 + 3 = □]": 0.5,
        "[2 + 3 = 5]": 0.5,
    })

    assign_delta_v_advantages(
        trajectories,
        value_fn,
        alpha=1.0,
        eps=1e-8,
        clip_min=0.0,
        clip_max=999.0,
        normalize=True,
    )

    # Sample 1: terminal step "5", reward=1, V=0.5 -> local_adv = 1.0 - 0.5 = 0.5
    for traj in trajectories:
        if traj.sample_id == 1:
            terminal_step = traj.steps[-1]
            assert terminal_step.done, "Last step should be terminal"
            # local_adv = reward - V(s_terminal) = 1 - 0.5 = 0.5
            # Weighted by rho (normalized), but since both weights sum to unit mean,
            # the terminal step's final_advantage should reflect the +0.5 signal
            assert terminal_step.final_advantage > 0.0, \
                f"Terminal correct step should have positive advantage, got {terminal_step.final_advantage}"
