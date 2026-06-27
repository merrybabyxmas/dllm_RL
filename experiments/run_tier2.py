#!/usr/bin/env python3
"""
Stage 2 Tier 2 Learned-V Replay Diagnostic.

Two modes:
  --mode toy      : CPU-only, no LLaDA. Validates replay buffer + K-step value
                    training using synthetic oracle-V data from the synthetic
                    causal math dataset.
  --mode backbone : GPU, LLaDA-8B-Instruct (skeleton only, prints instructions).

Toy mode design:
  Feature vector per block-state transition:
    [block_idx/num_blocks, oracle_v_prev, is_causal_error, is_coherent_cont,
     is_correct_branch, is_formatting, is_correct_substep, reward]
  (8-dim float32)
  Target: oracle_v_current (scalar).
  The oracle_v of the CURRENT block-state is the prediction target;
  oracle_v_prev (= oracle_v of the PREVIOUS state) is included as a feature
  to give the MLP a strong prior — we then validate that the network learns
  the RESIDUAL structure from token-role features.

  Gaussian noise (std=0.1) is added to the oracle_v_prev feature to simulate
  imperfect hidden-state quality.

Architecture: tiny MLP  8 -> 64 -> 64 -> 1  with GELU, trained with Huber loss.

Tier 2 pass criteria (toy mode):
  explained_variance           >= 0.20
  delta_v_sign_accuracy        >= 0.80
  correct_branch_positive_rate >= 0.80
  coherent_continuation_abs    <= 0.10

Usage:
  python experiments/run_tier2.py --mode toy
  python experiments/run_tier2.py --mode toy --seed 42 --n_trajectories 80
  python experiments/run_tier2.py --mode backbone   # prints instructions
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or experiments/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from cc_rl.data.synthetic_causal_math import (
    BlockState,
    Trajectory,
    make_dataset,
    ORACLE_V,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_ROLE_FLAGS = [
    "causal_error",
    "coherent_continuation",
    "correct_branch",
    "formatting",
    "correct_substep",
]


def extract_features(
    traj: Trajectory,
    rng: random.Random,
    noise_std: float = 0.1,
) -> List[Tuple[np.ndarray, float, str, float]]:
    """
    Extract (feature_vec, oracle_v_current, dominant_role, oracle_v_prev) tuples
    for every block-state transition in a trajectory.

    Feature vector (8-dim):
      [0] block_idx / num_blocks             (position in sequence)
      [1] oracle_v_prev + Gaussian noise     (noisy prior value estimate)
      [2] is_causal_error                    (any token in block has this role)
      [3] is_coherent_continuation
      [4] is_correct_branch
      [5] is_formatting
      [6] is_correct_substep
      [7] reward                             (terminal reward of trajectory)

    Returns list of (feat, target_v, dominant_role, true_oracle_v_prev).
    """
    states = traj.block_states
    num_blocks = len(states) - 1
    if num_blocks == 0:
        return []

    results = []
    for b in range(1, len(states)):
        prev_state: BlockState = states[b - 1]
        curr_state: BlockState = states[b]

        oracle_v_prev = float(prev_state.oracle_v)
        oracle_v_curr = float(curr_state.oracle_v)

        # Role flags: 1.0 if ANY token in the transition block has this role
        role_counts: Dict[str, int] = {}
        for tok in curr_state.tokens_revealed:
            role_counts[tok.role] = role_counts.get(tok.role, 0) + 1

        role_flags = []
        for role in _ROLE_FLAGS:
            role_flags.append(1.0 if role_counts.get(role, 0) > 0 else 0.0)

        # Dominant role = role with the most tokens (or "neutral" if empty)
        dominant_role = "neutral"
        if role_counts:
            dominant_role = max(role_counts, key=lambda r: role_counts[r])

        # Noisy oracle_v_prev feature
        noisy_v_prev = oracle_v_prev + rng.gauss(0.0, noise_std)
        noisy_v_prev = float(np.clip(noisy_v_prev, -0.3, 1.3))  # soft clip

        feat = np.array(
            [
                float(b - 1) / max(num_blocks, 1),  # block_idx/num_blocks (0-indexed b-1)
                noisy_v_prev,
                *role_flags,
                float(traj.reward),
            ],
            dtype=np.float32,
        )
        assert feat.shape == (8,), f"Expected 8-dim feature, got {feat.shape}"

        results.append((feat, oracle_v_curr, dominant_role, oracle_v_prev))

    return results


# ---------------------------------------------------------------------------
# Replay buffer (CPU, simple circular list)
# ---------------------------------------------------------------------------

class ToyReplayBuffer:
    """
    Simple replay buffer storing (feature_vec, oracle_v) pairs on CPU.

    Uses a list with wrap-around to implement a circular buffer without
    depending on deque's random-access limitations.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = capacity
        self._feats: List[np.ndarray] = []
        self._targets: List[float] = []
        self._ptr: int = 0  # write pointer

    def push(self, feat: np.ndarray, target: float) -> None:
        if len(self._feats) < self.capacity:
            self._feats.append(feat.copy())
            self._targets.append(float(target))
        else:
            self._feats[self._ptr] = feat.copy()
            self._targets[self._ptr] = float(target)
            self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = min(batch_size, len(self._feats))
        indices = random.sample(range(len(self._feats)), n)
        feats   = np.stack([self._feats[i] for i in indices])
        targets = np.array([self._targets[i] for i in indices], dtype=np.float32)
        return torch.from_numpy(feats), torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self._feats)

    def all_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ALL stored samples as tensors (for final evaluation)."""
        feats   = np.stack(self._feats)
        targets = np.array(self._targets, dtype=np.float32)
        return torch.from_numpy(feats), torch.from_numpy(targets)


# ---------------------------------------------------------------------------
# Tiny MLP value model: 8 -> 64 -> 64 -> 1
# ---------------------------------------------------------------------------

class TinyValueMLP(nn.Module):
    """
    Small MLP value head for toy-mode Tier 2 validation.

    Architecture: Linear(8,64) -> GELU -> Linear(64,64) -> GELU -> Linear(64,1)
    Output is a scalar (predicted oracle V for this block state).
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 8] -> [B] scalar predictions."""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_tier2_metrics(
    model: TinyValueMLP,
    all_feats: torch.Tensor,          # [N, 8]
    all_targets: torch.Tensor,         # [N]
    all_roles: List[str],              # len N
    all_oracle_v_prev: List[float],    # len N — the TRUE (unnoised) oracle_v_prev
) -> Dict[str, float]:
    """
    Compute all Tier 2 metrics using the trained model.

    Metrics:
      value_mse                  : MSE(V_pred, oracle_V)
      explained_variance         : 1 - Var(oracle_V - V_pred) / Var(oracle_V)
      delta_v_sign_accuracy      : sign accuracy of predicted delta_V vs oracle delta_V
                                   (computed from adjacent (b, b+1) pairs)
      rank_correlation           : Spearman rho between V_pred and oracle_V
      correct_branch_positive_rate: fraction of correct_branch transitions where
                                   V_pred > oracle_v_prev (predicted positive delta_V)
      coherent_continuation_abs  : mean |V_pred - oracle_v_prev| for coherent_cont
                                   transitions (i.e., mean |predicted delta_V|)
    """
    model.eval()
    preds = model(all_feats)  # [N]

    preds_np    = preds.numpy()
    targets_np  = all_targets.numpy()
    prev_np     = np.array(all_oracle_v_prev, dtype=np.float32)

    # --- value_mse ---
    value_mse = float(np.mean((preds_np - targets_np) ** 2))

    # --- explained_variance: 1 - Var(residual) / Var(target) ---
    residuals = targets_np - preds_np
    var_res    = float(np.var(residuals))
    var_tgt    = float(np.var(targets_np))
    explained_variance = float(1.0 - var_res / (var_tgt + 1e-8))

    # --- rank_correlation (Spearman) ---
    corr_result      = spearmanr(preds_np, targets_np)
    rank_correlation = float(corr_result.statistic)

    # --- delta_v_sign_accuracy ---
    # Predicted delta_V_b = V_pred(b+1) - V_pred(b)
    # Oracle  delta_V_b  = oracle_V(b+1) - oracle_V(b)
    # We have oracle_v_prev stored for each sample, so:
    #   oracle delta_V = targets_np[i] - prev_np[i]
    #   pred  delta_V  = preds_np[i] - (what the model would predict for prev state)
    # Since we don't have paired samples indexed by trajectory here,
    # use a simpler proxy: sign(preds_np[i] - prev_np[i]) vs sign(targets_np[i] - prev_np[i]).
    # This is equivalent to checking if the model predicts the right DIRECTION of change
    # from the ORACLE previous value — a valid proxy for delta-V sign accuracy.
    oracle_delta_v = targets_np - prev_np   # [N]
    pred_delta_v   = preds_np   - prev_np   # [N]  (using oracle prev, not noisy)

    # Only count non-trivial transitions (|oracle_delta_v| > 0.01)
    nontrivial = np.abs(oracle_delta_v) > 0.01
    if nontrivial.sum() > 0:
        signs_match = (np.sign(pred_delta_v[nontrivial]) == np.sign(oracle_delta_v[nontrivial]))
        delta_v_sign_accuracy = float(signs_match.mean())
    else:
        delta_v_sign_accuracy = float("nan")

    # --- correct_branch_positive_rate ---
    # For correct_branch transitions, oracle delta_V >= 0 (V should increase or stay).
    # We check: does the model predict V_pred > oracle_v_prev?
    cb_mask = np.array([r == "correct_branch" for r in all_roles])
    if cb_mask.sum() > 0:
        cb_pred_delta = pred_delta_v[cb_mask]
        cb_positive_rate = float((cb_pred_delta > 0).mean())
    else:
        cb_positive_rate = float("nan")

    # --- coherent_continuation_abs ---
    # For coherent_continuation transitions, oracle delta_V is close to 0 or negative.
    # We measure how much absolute credit the model gives these: mean |pred_delta_V|.
    cc_mask = np.array([r == "coherent_continuation" for r in all_roles])
    if cc_mask.sum() > 0:
        cc_abs_adv = float(np.mean(np.abs(pred_delta_v[cc_mask])))
    else:
        cc_abs_adv = float("nan")

    return {
        "value_mse":                    value_mse,
        "explained_variance":           explained_variance,
        "delta_v_sign_accuracy":        delta_v_sign_accuracy,
        "rank_correlation":             rank_correlation,
        "correct_branch_positive_rate": cb_positive_rate,
        "coherent_continuation_abs":    cc_abs_adv,
    }


# ---------------------------------------------------------------------------
# Tier 2 toy-mode runner
# ---------------------------------------------------------------------------

def run_toy_mode(args: argparse.Namespace) -> bool:
    """
    Toy-mode Tier 2 diagnostic: CPU-only, no LLaDA.

    Steps:
      1. Generate synthetic dataset (n_trajectories trajectories via make_dataset).
      2. Extract (feature_vec, oracle_v) pairs for all block-state transitions.
      3. Push ALL pairs into replay buffer (capacity 2000).
      4. Train tiny MLP for n_epochs x n_grad_steps_per_epoch gradient steps.
      5. Compute and report metrics with PASS/FAIL for each criterion.

    Returns True if all Tier 2 criteria pass.
    """
    seed_everything(args.seed)
    rng = random.Random(args.seed)

    print("=" * 50)
    print("Stage 2 Tier 2 Learned-V Replay Diagnostic")
    print("=" * 50)
    print(f"Mode: toy (CPU-only, no LLaDA required)")

    # ------------------------------------------------------------------
    # 1. Generate synthetic dataset
    # ------------------------------------------------------------------
    # make_dataset(n_per_type=N) yields 2*5*N trajectories total.
    # We want n_trajectories total: n_per_type = n_trajectories // 10
    n_per_type = max(1, args.n_trajectories // 10)
    trajectories = make_dataset(seed=args.seed, n_per_type=n_per_type)
    n_traj = len(trajectories)

    # Count transitions
    all_samples: List[Tuple[np.ndarray, float, str, float]] = []
    for traj in trajectories:
        samples = extract_features(traj, rng=rng, noise_std=args.noise_std)
        all_samples.extend(samples)

    n_transitions = len(all_samples)
    print(f"Dataset: {n_traj} trajectories, {n_transitions} block-state transitions")
    print()

    # ------------------------------------------------------------------
    # 2. Build replay buffer — push ALL pairs
    # ------------------------------------------------------------------
    replay = ToyReplayBuffer(capacity=args.replay_capacity)
    all_feats_list: List[np.ndarray] = []
    all_targets_list: List[float]    = []
    all_roles_list: List[str]        = []
    all_prev_v_list: List[float]     = []

    for feat, target_v, dominant_role, oracle_v_prev in all_samples:
        replay.push(feat, target_v)
        all_feats_list.append(feat)
        all_targets_list.append(target_v)
        all_roles_list.append(dominant_role)
        all_prev_v_list.append(oracle_v_prev)

    # Full evaluation tensors (kept on CPU throughout)
    eval_feats   = torch.from_numpy(np.stack(all_feats_list))           # [N, 8]
    eval_targets = torch.from_numpy(np.array(all_targets_list, dtype=np.float32))  # [N]

    # ------------------------------------------------------------------
    # 3. Build tiny MLP + optimizer
    # ------------------------------------------------------------------
    model    = TinyValueMLP(input_dim=8, hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=0.0)

    # ------------------------------------------------------------------
    # 4. Training loop: n_epochs x n_grad_steps_per_epoch
    # ------------------------------------------------------------------
    print("Training progress:")

    log_epochs = {10, 50, args.n_epochs}  # epochs to print

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_losses = []

        for _ in range(args.n_grad_steps_per_epoch):
            feats_b, targets_b = replay.sample(args.batch_size)
            preds_b = model(feats_b)
            loss = F.huber_loss(preds_b, targets_b, delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch in log_epochs:
            # Compute quick metrics on full eval set
            metrics = compute_tier2_metrics(
                model, eval_feats, eval_targets, all_roles_list, all_prev_v_list
            )
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(
                f"  Epoch {epoch:3d}/{args.n_epochs}: "
                f"mse={metrics['value_mse']:.3f} "
                f"expvar={metrics['explained_variance']:.3f} "
                f"dvsa={metrics['delta_v_sign_accuracy']:.3f}"
            )

    # ------------------------------------------------------------------
    # 5. Final metrics
    # ------------------------------------------------------------------
    final_metrics = compute_tier2_metrics(
        model, eval_feats, eval_targets, all_roles_list, all_prev_v_list
    )

    # Tier 2 pass criteria
    CRITERIA = {
        "explained_variance":           (0.20, ">="),
        "delta_v_sign_accuracy":        (0.80, ">="),
        "correct_branch_positive_rate": (0.80, ">="),
        "coherent_continuation_abs":    (0.10, "<="),
    }

    print()
    print("Final metrics:")
    n_pass = 0
    n_crit = len(CRITERIA)

    def _pf(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    # Print in fixed order
    mse_val = final_metrics["value_mse"]
    ev_val  = final_metrics["explained_variance"]
    dvsa    = final_metrics["delta_v_sign_accuracy"]
    rc_val  = final_metrics["rank_correlation"]
    cbpr    = final_metrics["correct_branch_positive_rate"]
    cc_abs  = final_metrics["coherent_continuation_abs"]

    ev_pass   = ev_val   >= CRITERIA["explained_variance"][0]
    dvsa_pass = dvsa     >= CRITERIA["delta_v_sign_accuracy"][0]
    cbpr_pass = cbpr     >= CRITERIA["correct_branch_positive_rate"][0]
    cc_pass   = cc_abs   <= CRITERIA["coherent_continuation_abs"][0]

    for passed in [ev_pass, dvsa_pass, cbpr_pass, cc_pass]:
        if passed:
            n_pass += 1

    print(f"  value_mse:                    {mse_val:.4f}")
    print(f"  explained_variance:           {ev_val:.4f}  "
          f"[threshold >= {CRITERIA['explained_variance'][0]:.2f}]  {_pf(ev_pass)}")
    print(f"  delta_v_sign_accuracy:        {dvsa:.4f}  "
          f"[threshold >= {CRITERIA['delta_v_sign_accuracy'][0]:.2f}]  {_pf(dvsa_pass)}")
    print(f"  rank_correlation:             {rc_val:.4f}")
    print(f"  correct_branch_positive_rate: {cbpr:.4f}  "
          f"[threshold >= {CRITERIA['correct_branch_positive_rate'][0]:.2f}]  {_pf(cbpr_pass)}")
    print(f"  coherent_continuation_abs:    {cc_abs:.4f}  "
          f"[threshold <= {CRITERIA['coherent_continuation_abs'][0]:.2f}]  {_pf(cc_pass)}")

    all_passed = (n_pass == n_crit)
    print()
    print(f"Overall: {_pf(all_passed)} ({n_pass}/{n_crit} criteria met)")

    return all_passed


# ---------------------------------------------------------------------------
# Backbone mode (stub)
# ---------------------------------------------------------------------------

def run_backbone_mode() -> None:
    print("backbone mode: run with --mode backbone (requires GPU, ~15 min)")
    print()
    print("  Backbone mode loads LLaDA-8B-Instruct and uses real hidden states")
    print("  from generate_with_confidence() as features for value head training.")
    print("  Not yet implemented in this diagnostic script.")
    print()
    print("  To run a full backbone evaluation, use train_standalone_v2.py with")
    print("  --method stage2 and check ema_expvar in the training logs.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 Tier 2 Learned-V Replay Diagnostic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["toy", "backbone"], default="toy",
                   help="Diagnostic mode.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n_trajectories", type=int,   default=80,
                   help="Target number of synthetic trajectories (rounded to nearest 10).")
    p.add_argument("--noise_std",      type=float, default=0.1,
                   help="Gaussian noise std added to oracle_v_prev feature.")
    p.add_argument("--replay_capacity",type=int,   default=2000,
                   help="Replay buffer capacity (max stored samples).")
    p.add_argument("--n_epochs",       type=int,   default=100,
                   help="Number of training epochs.")
    p.add_argument("--n_grad_steps_per_epoch", type=int, default=8,
                   help="Gradient steps per epoch.")
    p.add_argument("--batch_size",     type=int,   default=32,
                   help="Mini-batch size drawn from replay buffer.")
    p.add_argument("--learning_rate",  type=float, default=3e-3,
                   help="AdamW learning rate for tiny MLP.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "backbone":
        run_backbone_mode()
        return 0

    # toy mode
    passed = run_toy_mode(args)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
