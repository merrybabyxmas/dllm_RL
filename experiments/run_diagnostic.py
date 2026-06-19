#!/usr/bin/env python3
"""
Stage 2 fast diagnostic runner.

Tier 1 oracle-V diagnostic: verifies delta-V credit assignment math using
synthetic causal math trajectories. Runs in < 5 minutes on CPU, no GPU needed.

Usage:
  python experiments/run_diagnostic.py --mode oracle_v
  python experiments/run_diagnostic.py --mode oracle_v --seed 42 --n_per_type 10
  python experiments/run_diagnostic.py --mode oracle_v --verbose
  python experiments/run_diagnostic.py --mode oracle_v --rho_alpha 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure src is on the path when running from experiments/ or project root
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from cc_rl.data.synthetic_causal_math import (
    Trajectory,
    make_dataset,
    ORACLE_V,
)
from cc_rl.diagnostic.metrics import (
    compute_delta_v_advantages,
    compute_metrics,
    compute_metrics_by_type,
    check_tier1,
    TIER1_PASS_CRITERIA,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pass_str(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _sign(v: float) -> str:
    return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"


def _fmt_threshold(criterion: str) -> str:
    thresh, direction = TIER1_PASS_CRITERIA[criterion]
    return f"[threshold {direction} {thresh}]"


# ---------------------------------------------------------------------------
# Per-trajectory breakdown
# ---------------------------------------------------------------------------

def _print_trajectory_breakdown(trajectories: List[Trajectory], rho_alpha: float) -> None:
    print("\nPer-trajectory breakdown:")
    print(f"  {'ID':<35} {'reward':>6}  {'mean_A':>8}  {'n_tokens':>8}")
    print("  " + "-" * 65)
    for traj in trajectories:
        tok_adv = compute_delta_v_advantages(traj, rho_alpha=rho_alpha)
        if not tok_adv:
            continue
        mean_a = sum(a for _, a in tok_adv) / len(tok_adv)
        n = len(tok_adv)
        print(f"  {traj.trajectory_id:<35} {traj.reward:>6.1f}  {_sign(mean_a):>8}  {n:>8}")


# ---------------------------------------------------------------------------
# Per-type breakdown
# ---------------------------------------------------------------------------

def _print_per_type_breakdown(
    trajectories: List[Trajectory],
    type_metrics: Dict[str, dict],
) -> None:
    print("\nPer-type breakdown:")

    # Type A
    m = type_metrics.get("A", {})
    bepr = m.get("BEPR", 0.0)
    cop  = m.get("COP", 0.0)
    cbr  = m.get("CBR", 0.0)
    print(f"  Type A (op error):         BEPR={bepr:5.1f}  COP={cop:.2f}  CBR={cbr:.2f}")

    # Type B
    m = type_metrics.get("B", {})
    bepr = m.get("BEPR", 0.0)
    cop  = m.get("COP", 0.0)
    cbr  = m.get("CBR", 0.0)
    print(f"  Type B (qty error):        BEPR={bepr:5.1f}  COP={cop:.2f}  CBR={cbr:.2f}")

    # Type C — also report correct_substep rate
    m = type_metrics.get("C", {})
    bepr = m.get("BEPR", 0.0)
    cop  = m.get("COP", 0.0)
    cbr  = m.get("CBR", 0.0)
    substep_advs_by_role = m.get("adv_by_role", {})
    cs_mean = substep_advs_by_role.get("correct_substep", (0.0, 0.0))[0]
    # correct_substep_rate: fraction > 0 — approximate from CBR
    n_cs = m.get("n_tokens_by_role", {}).get("correct_substep", 0)
    print(f"  Type C (arith error):      BEPR={bepr:5.1f}  COP={cop:.2f}  CBR={cbr:.2f}  "
          f"cs_mean_A={_sign(cs_mean)}")

    # Type D — correct substep in failed traj
    m = type_metrics.get("D", {})
    bepr = m.get("BEPR", 0.0)
    cop  = m.get("COP", 0.0)
    cbr  = m.get("CBR", 0.0)
    print(f"  Type D (substep+wrong):    BEPR={bepr:5.1f}  COP={cop:.2f}  CBR={cbr:.2f}")

    # Type E — formatting trap
    m = type_metrics.get("E", {})
    bepr = m.get("BEPR", 0.0)
    fpr  = m.get("FPR", 0.0)
    adv_by_role = m.get("adv_by_role", {})
    fmt_abs = abs(adv_by_role.get("formatting", (0.0, 0.0))[0])
    cb_abs  = abs(adv_by_role.get("correct_branch", (0.0, 0.0))[0])
    fmt_vs_content = fmt_abs / (cb_abs + 1e-8)
    print(f"  Type E (format trap):      BEPR={bepr:5.1f}  FPR={fpr:.2f}  "
          f"format_vs_content={fmt_vs_content:.2f}")


# ---------------------------------------------------------------------------
# Main diagnostic function
# ---------------------------------------------------------------------------

def run_oracle_v_diagnostic(
    seed: int,
    n_per_type: int,
    rho_alpha: float,
    verbose: bool,
) -> bool:
    """
    Tier 1: oracle-V diagnostic.

    Returns True if all Tier 1 criteria pass.
    """
    print("=" * 50)
    print("Stage 2 Tier 1 Oracle-V Diagnostic")
    print("=" * 50)

    # -----------------------------------------------------------------------
    # 1. Generate dataset
    # -----------------------------------------------------------------------
    t0 = time.time()
    trajectories = make_dataset(seed=seed, n_per_type=n_per_type)
    n_correct = sum(1 for t in trajectories if t.is_correct)
    n_wrong   = len(trajectories) - n_correct

    print(f"Dataset: {len(trajectories)} trajectories ({n_correct} correct, {n_wrong} wrong), "
          f"5 types x {n_per_type} pairs x 2")
    print(f"Config:  seed={seed}, n_per_type={n_per_type}, rho_alpha={rho_alpha}")

    # -----------------------------------------------------------------------
    # 2. Compute oracle delta-V advantages for all trajectories
    # -----------------------------------------------------------------------
    metrics = compute_metrics(trajectories, rho_alpha=rho_alpha)
    type_metrics = compute_metrics_by_type(trajectories, rho_alpha=rho_alpha)
    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # 3. Per-role advantage statistics
    # -----------------------------------------------------------------------
    print("\nPer-role advantage statistics:")
    header = f"  {'Role':<26} {'n_tokens':>8}  {'mean_adv':>10}  {'std_adv':>8}  {'mean_rho':>8}"
    print(header)
    print("  " + "-" * 70)

    role_order = [
        "correct_branch",
        "correct_substep",
        "causal_error",
        "coherent_continuation",
        "formatting",
        "neutral",
    ]
    adv_by_role   = metrics["adv_by_role"]
    rho_by_role   = metrics["rho_by_role"]
    n_tok_by_role = metrics["n_tokens_by_role"]

    for role in role_order:
        if role not in adv_by_role:
            continue
        mean_a, std_a = adv_by_role[role]
        mean_rho = rho_by_role.get(role, 0.0)
        n = n_tok_by_role.get(role, 0)
        print(f"  {role:<26} {n:>8}  {_sign(mean_a):>10}  {std_a:>8.3f}  {mean_rho:>8.3f}")

    # Also print any unlisted roles
    for role in sorted(adv_by_role.keys()):
        if role not in role_order:
            mean_a, std_a = adv_by_role[role]
            mean_rho = rho_by_role.get(role, 0.0)
            n = n_tok_by_role.get(role, 0)
            print(f"  {role:<26} {n:>8}  {_sign(mean_a):>10}  {std_a:>8.3f}  {mean_rho:>8.3f}")

    # -----------------------------------------------------------------------
    # 4. Diagnostic metrics table
    # -----------------------------------------------------------------------
    print("\nDiagnostic metrics:")
    all_passed, criteria_results = check_tier1(metrics)

    bepr_val, bepr_pass = criteria_results["BEPR"]
    cop_val,  cop_pass  = criteria_results["COP"]
    cbr_val,  cbr_pass  = criteria_results["CBR"]
    fpr_val,  fpr_pass  = criteria_results["FPR"]
    dvsa_val, dvsa_pass = criteria_results["DVSA"]

    print(f"  BEPR (causal/coherent ratio):         {bepr_val:6.2f}  "
          f"{_fmt_threshold('BEPR')}  {_pass_str(bepr_pass)}")
    print(f"  COP  (coherent abs advantage):        {cop_val:6.4f}  "
          f"{_fmt_threshold('COP')}  {_pass_str(cop_pass)}")
    print(f"  CBR  (correct substep positive rate): {cbr_val:6.2f}  "
          f"{_fmt_threshold('CBR')}  {_pass_str(cbr_pass)}")
    print(f"  FPR  (formatting abs advantage):      {fpr_val:6.4f}  "
          f"{_fmt_threshold('FPR')}  {_pass_str(fpr_pass)}")
    print(f"  DVSA (delta-V sign accuracy):         {dvsa_val:6.2f}  "
          f"{_fmt_threshold('DVSA')}  {_pass_str(dvsa_pass)}")

    # -----------------------------------------------------------------------
    # 5. Per-type breakdown
    # -----------------------------------------------------------------------
    _print_per_type_breakdown(trajectories, type_metrics)

    # -----------------------------------------------------------------------
    # 6. Verbose: per-trajectory details
    # -----------------------------------------------------------------------
    if verbose:
        _print_trajectory_breakdown(trajectories, rho_alpha)

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    n_passed = sum(1 for _, passed in criteria_results.values() if passed)
    n_total  = len(criteria_results)
    print(f"\nOverall: {_pass_str(all_passed)} ({n_passed}/{n_total} criteria met)  "
          f"[elapsed: {elapsed:.2f}s]")

    if not all_passed:
        print("\nFailed criteria:")
        for criterion, (value, passed) in criteria_results.items():
            if not passed:
                thresh, direction = TIER1_PASS_CRITERIA[criterion]
                print(f"  {criterion}: {value:.4f} (need {direction} {thresh})")

    return all_passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 fast diagnostic runner for delta-V credit assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["oracle_v"],
        default="oracle_v",
        help="Diagnostic mode. Currently only 'oracle_v' (Tier 1) is supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    parser.add_argument(
        "--n_per_type",
        type=int,
        default=8,
        help="Number of (correct, wrong) trajectory pairs per error type.",
    )
    parser.add_argument(
        "--rho_alpha",
        type=float,
        default=1.0,
        help="Exponent for inverse-confidence responsibility weight: rho=(conf+eps)^{-alpha}.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trajectory breakdown.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "oracle_v":
        passed = run_oracle_v_diagnostic(
            seed=args.seed,
            n_per_type=args.n_per_type,
            rho_alpha=args.rho_alpha,
            verbose=args.verbose,
        )
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        return 1

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
