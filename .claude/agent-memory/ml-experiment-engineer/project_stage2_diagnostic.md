---
name: project_stage2_diagnostic
description: Stage 2 delta-V credit assignment fast diagnostic suite — oracle-V Tier 0/1 tests, all passing
metadata:
  type: project
---

Stage 2 fast diagnostic (oracle-V, Tier 0/1) is implemented and passing 5/5 criteria.

**Why:** Verify delta-V credit assignment math before running full GSM8K training. Runs in <1s on CPU.

**Files:**
- `src/cc_rl/data/synthetic_causal_math.py` — 5-type synthetic dataset with oracle V
- `src/cc_rl/diagnostic/__init__.py` — package init
- `src/cc_rl/diagnostic/metrics.py` — BEPR/COP/CBR/DVSA/FPR computation
- `tests/test_stage2_delta_v_smoke.py` — 12 Tier 0 pure-math tests
- `experiments/run_diagnostic.py` — CLI runner

**Results (seed=42, n_per_type=8, rho_alpha=1.0):**
- BEPR=34.18 (>= 5.0) PASS — causal errors penalized 34x more than coherent continuations
- COP=0.030  (<= 0.05) PASS — coherent continuation near-zero advantage
- CBR=1.00   (>= 0.90) PASS — all correct substep tokens get positive advantage
- FPR=0.000  (<= 0.10) PASS — formatting tokens get zero advantage (delta_v=0 block)
- DVSA=1.00  (>= 0.90) PASS — sign accuracy is perfect (rho always positive)

**Critical design decisions:**
1. Do NOT z-score normalize delta_v values. Z-score normalization flips zero-delta blocks
   (coherent_continuation staying at V=0.05) to POSITIVE when trajectory mean is negative.
   Use raw delta_v * rho instead.
2. For Type C wrong (arithmetic slip): split "4 * 8 = | 36" into two blocks.
   Tokens "4 * 8 =" get delta_v = +0.35 (correct_substep); "36" gets -0.55 (causal_error).
3. For Type D wrong (correct substep + wrong final): split second step's summation
   structure ("15 + 30 + 10 =") from the wrong result ("65") into separate blocks.
   Label summation structure as "neutral" (zero delta_v), result as causal_error.
4. Operator detection in Type A/B wrong: use positional index (tok_i==1 is always the
   operator in step tokens), not string matching (tokens have spaces like " + ").

**How to apply:** Run `python experiments/run_diagnostic.py --mode oracle_v` before starting Stage 2 training to verify the credit assignment math is sane.
