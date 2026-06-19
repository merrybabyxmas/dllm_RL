"""
Stage 3 (Q-Credit) smoke tests on the toy "2 + 3 = ?" fixture.

Verifies that (Q(s,a) - V(s)) per-token advantages weighted by responsibility
weights are computed correctly with tabular value and Q functions.

Reference computation (alpha=1.0, eps=0.0, clip_min=0.0, clip_max=999.0):

Value table:
  V("[2 □ 3 = □]") = 0.25
  V("[2 + 3 = □]") = 0.80
  V("[2 * 3 = □]") = 0.05

Q table:
  Q("[2 □ 3 = □]", "+") = 0.80
  Q("[2 □ 3 = □]", "*") = 0.05
  Q("[2 □ 3 = □]", "-") = 0.00
  Q("[2 * 3 = □]", "6") = 0.05
  Q("[2 * 3 = □]", "5") = 0.02
  Q("[2 + 3 = □]", "5") = 1.00
  Q("[2 + 3 = □]", "6") = 0.00

Sample 2 (conf=[0.20, 0.95], weights=[1.652, 0.348]):
  Step 1 ("*"): A_raw = Q("[2□3=□]","*") - V("[2□3=□]") = 0.05 - 0.25 = -0.20
  Step 2 ("6"): A_raw = Q("[2*3=□]","6") - V("[2*3=□]") = 0.05 - 0.05 = 0.00
  A("*") = -0.20 * 1.652 = -0.330
  A("6") = 0.00 * 0.348 = 0.000

Sample 3 (conf=[0.60, 0.35], weights=[0.737, 1.263]):
  Step 1 ("+"): A_raw = Q("[2□3=□]","+") - V("[2□3=□]") = 0.80 - 0.25 = +0.55
  Step 2 ("6"): A_raw = Q("[2+3=□]","6") - V("[2+3=□]") = 0.00 - 0.80 = -0.80
  A("+") = 0.55 * 0.737 = 0.405
  A("6") = -0.80 * 1.263 = -1.010
"""
import pytest
import os

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "toy_2_plus_3.jsonl")


def test_stage3_q_credit_toy():
    """Full Stage 3 pipeline on toy fixture with exact numerical verification."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3, collect_advantages_by_sample_action
    from cc_rl.credit.advantages import assign_q_minus_v_advantages
    from cc_rl.critics.tabular_toy import TabularValue, TabularQ

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)

    value_fn = TabularValue({
        "[2 □ 3 = □]": 0.25,
        "[2 + 3 = □]": 0.80,
        "[2 * 3 = □]": 0.05,
    })

    q_fn = TabularQ({
        ("[2 □ 3 = □]", "+"): 0.80,
        ("[2 □ 3 = □]", "*"): 0.05,
        ("[2 □ 3 = □]", "-"): 0.00,
        ("[2 * 3 = □]", "6"): 0.05,
        ("[2 * 3 = □]", "5"): 0.02,
        ("[2 + 3 = □]", "5"): 1.00,
        ("[2 + 3 = □]", "6"): 0.00,
    })

    assign_q_minus_v_advantages(
        trajectories,
        value_fn,
        q_fn,
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
    assert adv[2]["6"] == pytest.approx(0.000, abs=1e-6), \
        f"adv[2]['6'] = {adv[2]['6']:.6f}, expected 0.000"

    assert adv[3]["+"] == pytest.approx(+0.405, abs=1e-3), \
        f"adv[3]['+'] = {adv[3]['+']:.4f}, expected +0.405"
    assert adv[3]["6"] == pytest.approx(-1.010, abs=1e-3), \
        f"adv[3]['6'] = {adv[3]['6']:.4f}, expected -1.010"


def test_stage3_correct_action_positive_advantage():
    """Good Q-value action (Q > V) always yields positive advantage."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    from cc_rl.credit.advantages import assign_q_minus_v_advantages
    from cc_rl.critics.tabular_toy import TabularValue, TabularQ

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)

    value_fn = TabularValue({
        "[2 □ 3 = □]": 0.25,
        "[2 + 3 = □]": 0.80,
        "[2 * 3 = □]": 0.05,
    })

    q_fn = TabularQ({
        ("[2 □ 3 = □]", "+"): 0.80,   # Q > V -> positive advantage
        ("[2 □ 3 = □]", "*"): 0.05,
        ("[2 □ 3 = □]", "-"): 0.00,
        ("[2 * 3 = □]", "6"): 0.05,
        ("[2 + 3 = □]", "5"): 1.00,   # Q >> V -> strongly positive
        ("[2 + 3 = □]", "6"): 0.00,
    })

    assign_q_minus_v_advantages(
        trajectories,
        value_fn,
        q_fn,
        alpha=1.0,
        eps=1e-8,
        clip_min=0.0,
        clip_max=999.0,
        normalize=True,
    )

    # Sample 1: "+" action has Q=0.80 > V=0.25 -> positive advantage
    for traj in trajectories:
        if traj.sample_id == 1:
            plus_step = traj.steps[0]  # action="+"
            assert plus_step.action == "+", f"Expected '+', got {plus_step.action}"
            assert plus_step.final_advantage > 0.0, \
                f"Good action '+' (Q=0.80 > V=0.25) should have positive advantage"

            # "5" is the correct terminal: Q("[2+3=□]","5") = 1.00 > V = 0.80
            five_step = traj.steps[1]
            assert five_step.action == "5"
            assert five_step.final_advantage > 0.0, \
                "Correct terminal '5' should have positive advantage"


def test_stage3_neutral_action_zero_advantage():
    """Action where Q(s,a) = V(s) gives exactly zero advantage regardless of rho."""
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    from cc_rl.credit.advantages import assign_q_minus_v_advantages
    from cc_rl.critics.tabular_toy import TabularValue, TabularQ

    trajectories = load_toy_2_plus_3(FIXTURE_PATH)

    # Set Q = V for all actions -> zero advantages everywhere
    value_fn = TabularValue({
        "[2 □ 3 = □]": 0.5,
        "[2 + 3 = □]": 0.5,
        "[2 * 3 = □]": 0.5,
        "[2 - 3 = □]": 0.5,
    })
    q_fn = TabularQ({
        ("[2 □ 3 = □]", "+"): 0.5,
        ("[2 □ 3 = □]", "*"): 0.5,
        ("[2 □ 3 = □]", "-"): 0.5,
        ("[2 + 3 = □]", "5"): 0.5,
        ("[2 + 3 = □]", "6"): 0.5,
        ("[2 * 3 = □]", "6"): 0.5,
        ("[2 - 3 = □]", "-1"): 0.5,
    })

    assign_q_minus_v_advantages(
        trajectories, value_fn, q_fn,
        alpha=1.0, eps=0.0, clip_min=0.0, clip_max=999.0, normalize=True,
    )

    for traj in trajectories:
        for step in traj.steps:
            assert step.final_advantage == pytest.approx(0.0, abs=1e-9), \
                f"Q=V should give zero advantage, got {step.final_advantage}"


def test_stage3_tabular_q_unknown_pair():
    """Q function returns 0.0 for unseen (state, action) pairs."""
    from cc_rl.critics.tabular_toy import TabularQ
    q = TabularQ({("s0", "a"): 1.0})
    assert q("s0", "a") == 1.0
    assert q("s0", "b") == 0.0   # unknown action
    assert q("s1", "a") == 0.0   # unknown state
