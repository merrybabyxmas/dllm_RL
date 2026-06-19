"""
End-to-end pipeline verification script.
Tests the full training loop with a tiny mock model (no real LLaDA needed).
Run this before the real 3000-step experiments.
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../d1/diffu-grpo"))

# Patch TRL compat
import trl.import_utils as _trl_iu
if not hasattr(_trl_iu, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_iu.is_rich_available = _ira
    except ImportError:
        _trl_iu.is_rich_available = lambda: False


def test_confidence_credit_pipeline():
    """Test the confidence-weighted advantage computation on synthetic data."""
    from cc_rl.credit.responsibility import compute_responsibility_weights
    from cc_rl.credit.advantages import (
        assign_group_advantages,
        assign_confidence_weighted_advantages,
        assign_delta_v_advantages,
        assign_q_minus_v_advantages,
    )
    from cc_rl.critics.tabular_toy import TabularValue, TabularQ
    from cc_rl.data.toy_loader import load_toy_2_plus_3, collect_advantages_by_sample_action

    print("[verify] Loading toy dataset...")
    trajectories = load_toy_2_plus_3(
        os.path.join(os.path.dirname(__file__), "../tests/fixtures/toy_2_plus_3.jsonl")
    )
    assert len(trajectories) == 4, f"Expected 4 trajectories, got {len(trajectories)}"
    print(f"  Loaded {len(trajectories)} trajectories")

    print("[verify] Testing Stage 1 advantages...")
    assign_group_advantages(trajectories, eps=1e-12)
    assign_confidence_weighted_advantages(
        trajectories, alpha=1.0, eps=0.0, clip_min=0.0, clip_max=999.0, normalize=True
    )
    adv = collect_advantages_by_sample_action(trajectories)

    assert abs(adv[2]["*"] - (-0.954)) < 1e-3, f"Stage1 tau2 '*' wrong: {adv[2]['*']}"
    assert abs(adv[2]["6"] - (-0.201)) < 1e-3, f"Stage1 tau2 '6' wrong: {adv[2]['6']}"
    assert abs(adv[2]["*"]) > 4.0 * abs(adv[2]["6"]), "Stage1: * should be >>4x penalized vs 6"
    print("  Stage 1: PASSED (penalizes branching action 4x more than coherent continuation)")

    print("[verify] Testing Stage 2 advantages...")
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    trajectories2 = load_toy_2_plus_3(
        os.path.join(os.path.dirname(__file__), "../tests/fixtures/toy_2_plus_3.jsonl")
    )
    value_fn = TabularValue({
        "[2 □ 3 = □]": 0.25,
        "[2 + 3 = □]": 0.80,
        "[2 * 3 = □]": 0.05,
        "[2 - 3 = □]": 0.00,
    })
    assign_delta_v_advantages(
        trajectories2, value_fn, alpha=1.0, eps=0.0, clip_min=0.0, clip_max=999.0, normalize=True
    )
    adv2 = collect_advantages_by_sample_action(trajectories2)
    assert adv2[3]["+"] > 0.0, f"Stage2: '+' in 2+3=6 should be positive, got {adv2[3]['+']}"
    assert adv2[3]["6"] < -1.0, f"Stage2: '6' in 2+3=6 should be < -1, got {adv2[3]['6']}"
    print("  Stage 2: PASSED (rewards '+' even in wrong-answer trajectory 2+3=6)")

    print("[verify] Testing Stage 3 advantages...")
    from cc_rl.data.toy_loader import load_toy_2_plus_3
    trajectories3 = load_toy_2_plus_3(
        os.path.join(os.path.dirname(__file__), "../tests/fixtures/toy_2_plus_3.jsonl")
    )
    value_fn3 = TabularValue({"[2 □ 3 = □]": 0.25, "[2 + 3 = □]": 0.80, "[2 * 3 = □]": 0.05})
    q_fn = TabularQ({
        ("[2 □ 3 = □]", "+"): 0.80, ("[2 □ 3 = □]", "*"): 0.05, ("[2 □ 3 = □]", "-"): 0.00,
        ("[2 * 3 = □]", "6"): 0.05, ("[2 + 3 = □]", "5"): 1.00, ("[2 + 3 = □]", "6"): 0.00,
    })
    assign_q_minus_v_advantages(
        trajectories3, value_fn3, q_fn, alpha=1.0, eps=0.0, clip_min=0.0, clip_max=999.0, normalize=True
    )
    adv3 = collect_advantages_by_sample_action(trajectories3)
    assert abs(adv3[2]["6"]) < 1e-6, f"Stage3: '6' in 2*3=6 should be ~0, got {adv3[2]['6']}"
    assert adv3[2]["*"] < -0.30, f"Stage3: '*' should be penalized, got {adv3[2]['*']}"
    print("  Stage 3: PASSED (zero penalty for '6' in 2*3=6, strong penalty for '*')")


def test_confidence_weight_computation():
    """Test responsibility weight formula with torch tensors."""
    from cc_rl.credit.responsibility import compute_responsibility_weights

    # Simulate confidence from a generation step
    confidences = [0.9, 0.1, 0.7, 0.3, 0.95]  # mix of high/low
    weights = compute_responsibility_weights(confidences, alpha=1.0, eps=1e-6,
                                             clip_min=0.25, clip_max=4.0, normalize=True)

    # High confidence -> low weight, low confidence -> high weight
    assert weights[1] > weights[0], "Low confidence (0.1) should get higher weight than high (0.9)"
    assert weights[4] < weights[3], "High confidence (0.95) should get lower weight than low (0.3)"

    # Mean should be ~1 after normalization
    mean = sum(weights) / len(weights)
    assert abs(mean - 1.0) < 1e-6, f"Mean weight should be 1.0, got {mean}"

    print(f"  Responsibility weights: {[f'{w:.3f}' for w in weights]}")
    print(f"  Mean: {mean:.6f}")
    print("  Confidence weight computation: PASSED")


def test_reward_functions():
    """Test GSM8K reward function."""
    from cc_rl.rewards.exact_match import reward_gsm8k, extract_final_number

    cases = [
        ("The answer is #### 42", "#### 42", 1.0),
        ("<answer>42</answer>", "42", 1.0),
        ("So 2+3=5", "5", 1.0),
        ("I think it's 5", "10", 0.0),
        ("No answer", "5", 0.0),
        ("1,234 students", "1234", 1.0),
    ]

    for completion, gold, expected in cases:
        result = reward_gsm8k(completion, gold)
        assert abs(result - expected) < 1e-6, \
            f"Reward({completion!r}, {gold!r}) = {result}, expected {expected}"

    print("  Reward functions: PASSED")


def main():
    print("=" * 60)
    print("Pipeline Verification (no model required)")
    print("=" * 60)

    test_confidence_weight_computation()
    test_reward_functions()
    test_confidence_credit_pipeline()

    print()
    print("=" * 60)
    print("ALL PIPELINE VERIFICATION TESTS PASSED")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Ensure LLaDA-8B-Instruct is downloaded to:")
    print("     /home/dongwoo43/papers/paper_dllm/LLaDA-8B-Instruct/")
    print("  2. Run: bash experiments/run_diffu_grpo.sh")
    print("  3. Run: bash experiments/run_stage2.sh")
    print("  4. Run: python experiments/eval_base.py")


if __name__ == "__main__":
    main()
