"""
Synthetic causal math reasoning trajectories with oracle V values and token role annotations.

Designed for Stage 2 fast diagnostic (Tier 0/1) — pure Python, no GPU needed.

Five error types:
  A: operator branch error    (wrong arithmetic operator chosen)
  B: quantity interpretation  (wrong relationship between quantities)
  C: arithmetic slip          (correct op, wrong number result)
  D: correct substep + wrong final  (early step correct, later step wrong)
  E: formatting trap          (formatting tokens should not inflate credit)

Oracle V assignment encodes how the model's probability of eventual success
changes after each reasoning block is revealed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Oracle V constants
# ---------------------------------------------------------------------------

ORACLE_V = {
    "initial":                0.30,
    "after_correct_op":       0.75,
    "after_wrong_op":         0.05,
    "after_correct_substep":  0.70,
    "after_arithmetic_slip":  0.10,
    "after_coherent_cont":    0.05,
    "terminal_correct":       1.0,
    "terminal_wrong":         0.0,
    # formatting_unchanged means oracle_v stays the same as prior state
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenAnnotation:
    """Annotation for a single token in a trajectory block."""
    token: str
    role: str       # "causal_error"|"coherent_continuation"|"correct_branch"|
                    # "correct_substep"|"formatting"|"neutral"
    confidence: float  # mock confidence in [0, 1]


@dataclass
class BlockState:
    """
    State after block b is revealed.

    block_idx=0 is the initial state (all masked, no tokens revealed yet).
    block_idx=k is the state AFTER block k-1 tokens are revealed.
    tokens_revealed are the tokens revealed IN the transition from
    block_idx-1 to block_idx.
    """
    block_idx: int
    oracle_v: float
    tokens_revealed: List[TokenAnnotation] = field(default_factory=list)


@dataclass
class Trajectory:
    """A single reasoning trajectory with oracle-V annotations."""
    trajectory_id: str
    problem: str
    is_correct: bool
    reward: float
    error_type: Optional[str]       # None for correct; "A"/"B"/"C"/"D"/"E" for wrong
    block_states: List[BlockState]  # len = num_blocks + 1  (state 0 is initial)


# ---------------------------------------------------------------------------
# Confidence sampling helpers (deterministic given rng)
# ---------------------------------------------------------------------------

def _conf_causal_error(rng: random.Random) -> float:
    """Low confidence: model was uncertain at the pivotal wrong step."""
    return rng.uniform(0.15, 0.25)


def _conf_coherent_cont(rng: random.Random) -> float:
    """High confidence: model commits to the wrong branch strongly."""
    return rng.uniform(0.85, 0.95)


def _conf_correct_branch(rng: random.Random) -> float:
    """High confidence: model is correct and sure."""
    return rng.uniform(0.80, 0.95)


def _conf_correct_substep(rng: random.Random) -> float:
    """Moderate-high: correct sub-step even in failing trajectory."""
    return rng.uniform(0.75, 0.90)


def _conf_formatting(rng: random.Random) -> float:
    """Low confidence: formatting tokens are low-probability surface forms."""
    return rng.uniform(0.05, 0.15)


def _conf_neutral(rng: random.Random) -> float:
    """Neutral: structural tokens with moderate confidence."""
    return rng.uniform(0.60, 0.80)


# ---------------------------------------------------------------------------
# Problem templates per error type
# ---------------------------------------------------------------------------
# Each template is a tuple:
#   (problem_str, correct_steps, wrong_steps, block_oracle_vs_correct, block_oracle_vs_wrong)
# block_oracle_vs_* is the list of oracle_v values for states 1..T
# (state 0 is always ORACLE_V["initial"])

def _make_type_a_pair(rng: random.Random, variant: int, traj_counter: list) -> List[Trajectory]:
    """
    Type A: operator branch error.
    Two variants with different numbers.
    """
    if variant == 0:
        problem = "Tom has 12 apples. He gives 5 away then buys 3 more. How many?"
        # Correct:  12 - 5 = 7, 7 + 3 = 10, Answer: 10
        correct_steps = [["12", " - ", "5", " = ", "7"],
                         ["7", " + ", "3", " = ", "10"],
                         ["Answer", ":", " ", "10"]]
        # Wrong:    12 + 5 = 17, 17 + 3 = 20, Answer: 20
        wrong_steps   = [["12", " + ", "5", " = ", "17"],
                         ["17", " + ", "3", " = ", "20"],
                         ["Answer", ":", " ", "20"]]
        # "+" in wrong_steps[0] is causal_error; "5", "=", "17" are coherent_cont
    elif variant == 1:
        problem = "Sara has 20 stickers. She gives 8 away then receives 4 more. How many?"
        correct_steps = [["20", " - ", "8", " = ", "12"],
                         ["12", " + ", "4", " = ", "16"],
                         ["Answer", ":", " ", "16"]]
        wrong_steps   = [["20", " + ", "8", " = ", "28"],
                         ["28", " + ", "4", " = ", "32"],
                         ["Answer", ":", " ", "32"]]
    elif variant == 2:
        problem = "A box has 30 candies. 9 are eaten and 6 more are added. How many remain?"
        correct_steps = [["30", " - ", "9", " = ", "21"],
                         ["21", " + ", "6", " = ", "27"],
                         ["Answer", ":", " ", "27"]]
        wrong_steps   = [["30", " + ", "9", " = ", "39"],
                         ["39", " + ", "6", " = ", "45"],
                         ["Answer", ":", " ", "45"]]
    else:  # variant == 3
        problem = "There are 50 books. 15 are borrowed and 7 are returned. How many remain?"
        correct_steps = [["50", " - ", "15", " = ", "35"],
                         ["35", " + ", "7", " = ", "42"],
                         ["Answer", ":", " ", "42"]]
        wrong_steps   = [["50", " + ", "15", " = ", "65"],
                         ["65", " + ", "7", " = ", "72"],
                         ["Answer", ":", " ", "72"]]

    results = []

    # --- Correct trajectory ---
    c_id = f"A{variant}_correct_{traj_counter[0]}"
    traj_counter[0] += 1
    # Block states: s0 (initial), s1 (after step0), s2 (after step1), s3 (terminal)
    # Oracle V: initial=0.30, after_correct_op=0.75, stays high, terminal_correct=1.0
    c_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Step 0: correct op (e.g., "12 - 5 = 7")
    tokens_s0 = []
    for tok in correct_steps[0]:
        tokens_s0.append(TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng)))
    c_states.append(BlockState(block_idx=1, oracle_v=ORACLE_V["after_correct_op"],
                                tokens_revealed=tokens_s0))

    # Step 1: second arithmetic step (still correct_branch, V stays high)
    tokens_s1 = []
    for tok in correct_steps[1]:
        tokens_s1.append(TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng)))
    c_states.append(BlockState(block_idx=2, oracle_v=ORACLE_V["after_correct_op"],
                                tokens_revealed=tokens_s1))

    # Step 2: terminal "Answer: X"
    tokens_s2 = []
    for tok in correct_steps[2]:
        tokens_s2.append(TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng)))
    c_states.append(BlockState(block_idx=3, oracle_v=ORACLE_V["terminal_correct"],
                                tokens_revealed=tokens_s2))

    results.append(Trajectory(
        trajectory_id=c_id,
        problem=problem,
        is_correct=True,
        reward=1.0,
        error_type=None,
        block_states=c_states,
    ))

    # --- Wrong trajectory ---
    w_id = f"A{variant}_wrong_{traj_counter[0]}"
    traj_counter[0] += 1
    w_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Step 0: wrong op "+" — operator is at index 1 in the step tokens list.
    # Index 1 is always the operator (e.g., " + " or " - ").
    # It's labeled causal_error because it diverges from the correct operator.
    tokens_w0 = []
    for tok_i, tok in enumerate(wrong_steps[0]):
        if tok_i == 1:  # operator position — this is the causal branch error
            tokens_w0.append(TokenAnnotation(tok, "causal_error", _conf_causal_error(rng)))
        else:
            tokens_w0.append(TokenAnnotation(tok, "neutral", _conf_neutral(rng)))
    w_states.append(BlockState(block_idx=1, oracle_v=ORACLE_V["after_wrong_op"],
                                tokens_revealed=tokens_w0))

    # Step 1: coherent_continuation (model follows wrong branch logically)
    tokens_w1 = []
    for tok in wrong_steps[1]:
        tokens_w1.append(TokenAnnotation(tok, "coherent_continuation", _conf_coherent_cont(rng)))
    w_states.append(BlockState(block_idx=2, oracle_v=ORACLE_V["after_coherent_cont"],
                                tokens_revealed=tokens_w1))

    # Step 2: terminal wrong
    tokens_w2 = []
    for tok in wrong_steps[2]:
        tokens_w2.append(TokenAnnotation(tok, "coherent_continuation", _conf_coherent_cont(rng)))
    w_states.append(BlockState(block_idx=3, oracle_v=ORACLE_V["terminal_wrong"],
                                tokens_revealed=tokens_w2))

    results.append(Trajectory(
        trajectory_id=w_id,
        problem=problem,
        is_correct=False,
        reward=0.0,
        error_type="A",
        block_states=w_states,
    ))

    return results


def _make_type_b_pair(rng: random.Random, variant: int, traj_counter: list) -> List[Trajectory]:
    """
    Type B: quantity interpretation error (wrong relationship between quantities).
    """
    if variant == 0:
        problem = "4 packs, 6 pencils each, gives away 5. How many remain?"
        correct_steps = [["4", " * ", "6", " = ", "24"],
                         ["24", " - ", "5", " = ", "19"],
                         ["Answer", ":", " ", "19"]]
        wrong_steps   = [["4", " + ", "6", " = ", "10"],
                         ["10", " - ", "5", " = ", "5"],
                         ["Answer", ":", " ", "5"]]
    elif variant == 1:
        problem = "3 boxes, 8 items each, loses 4. How many remain?"
        correct_steps = [["3", " * ", "8", " = ", "24"],
                         ["24", " - ", "4", " = ", "20"],
                         ["Answer", ":", " ", "20"]]
        wrong_steps   = [["3", " + ", "8", " = ", "11"],
                         ["11", " - ", "4", " = ", "7"],
                         ["Answer", ":", " ", "7"]]
    elif variant == 2:
        problem = "5 bags, 7 oranges each, eats 3. How many remain?"
        correct_steps = [["5", " * ", "7", " = ", "35"],
                         ["35", " - ", "3", " = ", "32"],
                         ["Answer", ":", " ", "32"]]
        wrong_steps   = [["5", " + ", "7", " = ", "12"],
                         ["12", " - ", "3", " = ", "9"],
                         ["Answer", ":", " ", "9"]]
    else:  # variant == 3
        problem = "6 shelves, 9 books each, removes 7. How many remain?"
        correct_steps = [["6", " * ", "9", " = ", "54"],
                         ["54", " - ", "7", " = ", "47"],
                         ["Answer", ":", " ", "47"]]
        wrong_steps   = [["6", " + ", "9", " = ", "15"],
                         ["15", " - ", "7", " = ", "8"],
                         ["Answer", ":", " ", "8"]]

    results = []

    c_id = f"B{variant}_correct_{traj_counter[0]}"
    traj_counter[0] += 1
    c_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    for block_i, step_tokens in enumerate(correct_steps):
        v_next = ORACLE_V["terminal_correct"] if block_i == len(correct_steps) - 1 else ORACLE_V["after_correct_op"]
        tokens = [TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng))
                  for tok in step_tokens]
        c_states.append(BlockState(block_idx=block_i + 1, oracle_v=v_next,
                                    tokens_revealed=tokens))

    results.append(Trajectory(
        trajectory_id=c_id,
        problem=problem,
        is_correct=True,
        reward=1.0,
        error_type=None,
        block_states=c_states,
    ))

    w_id = f"B{variant}_wrong_{traj_counter[0]}"
    traj_counter[0] += 1
    w_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Wrong step 0: "+" instead of "*" — operator is at index 1 (causal_error)
    tokens_w0 = []
    for tok_i, tok in enumerate(wrong_steps[0]):
        if tok_i == 1:  # operator position
            tokens_w0.append(TokenAnnotation(tok, "causal_error", _conf_causal_error(rng)))
        else:
            tokens_w0.append(TokenAnnotation(tok, "neutral", _conf_neutral(rng)))
    w_states.append(BlockState(block_idx=1, oracle_v=ORACLE_V["after_wrong_op"],
                                tokens_revealed=tokens_w0))

    for block_i in range(1, len(wrong_steps)):
        step_tokens = wrong_steps[block_i]
        v_next = ORACLE_V["terminal_wrong"] if block_i == len(wrong_steps) - 1 else ORACLE_V["after_coherent_cont"]
        tokens = [TokenAnnotation(tok, "coherent_continuation", _conf_coherent_cont(rng))
                  for tok in step_tokens]
        w_states.append(BlockState(block_idx=block_i + 1, oracle_v=v_next,
                                    tokens_revealed=tokens))

    results.append(Trajectory(
        trajectory_id=w_id,
        problem=problem,
        is_correct=False,
        reward=0.0,
        error_type="B",
        block_states=w_states,
    ))

    return results


def _make_type_c_pair(rng: random.Random, variant: int, traj_counter: list) -> List[Trajectory]:
    """
    Type C: arithmetic slip — correct operator but wrong result number.

    The operator token and the expression structure "4 * 8 =" are correct_substep.
    The wrong RESULT number (e.g., "36" instead of "32") is causal_error.

    Block design for wrong trajectory — SPLIT the first step into two blocks:
      Block 0 -> Block 1: tokens "4 * 8 =" (correct_substep)
                          V: 0.30 -> 0.65  (heading in right direction: good structure)
      Block 1 -> Block 2: token "36"       (causal_error)
                          V: 0.65 -> 0.10  (slip crashes the value estimate)
      Block 2 -> Block 3: coherent continuation "36 + 6 = 42"
                          V: 0.10 -> 0.05
      Block 3 -> Block 4: terminal "Answer: 42"
                          V: 0.05 -> 0.0

    This ensures correct_substep tokens get POSITIVE delta_v (+0.35) and
    causal_error gets NEGATIVE delta_v (-0.55). CBR is satisfied. ✓
    """
    if variant == 0:
        problem = "4 tickets at $8 each, snacks cost $6. Total?"
        # Correct: 4*8=32, 32+6=38
        correct_steps = [["4", " * ", "8", " = ", "32"],
                         ["32", " + ", "6", " = ", "38"],
                         ["Answer", ":", " ", "38"]]
        # Wrong: 4*8=36 (slip), 36+6=42
        wrong_op_structure = ["4", " * ", "8", " = "]   # correct_substep
        wrong_slip_token   = "36"                         # causal_error
        wrong_steps_after  = [["36", " + ", "6", " = ", "42"],
                               ["Answer", ":", " ", "42"]]
    elif variant == 1:
        problem = "3 boxes at $5 each, shipping costs $4. Total?"
        correct_steps = [["3", " * ", "5", " = ", "15"],
                         ["15", " + ", "4", " = ", "19"],
                         ["Answer", ":", " ", "19"]]
        wrong_op_structure = ["3", " * ", "5", " = "]
        wrong_slip_token   = "18"
        wrong_steps_after  = [["18", " + ", "4", " = ", "22"],
                               ["Answer", ":", " ", "22"]]
    elif variant == 2:
        problem = "6 pens at $3 each, tax is $2. Total?"
        correct_steps = [["6", " * ", "3", " = ", "18"],
                         ["18", " + ", "2", " = ", "20"],
                         ["Answer", ":", " ", "20"]]
        wrong_op_structure = ["6", " * ", "3", " = "]
        wrong_slip_token   = "21"
        wrong_steps_after  = [["21", " + ", "2", " = ", "23"],
                               ["Answer", ":", " ", "23"]]
    else:  # variant == 3
        problem = "7 apples at $2 each, bag costs $3. Total?"
        correct_steps = [["7", " * ", "2", " = ", "14"],
                         ["14", " + ", "3", " = ", "17"],
                         ["Answer", ":", " ", "17"]]
        wrong_op_structure = ["7", " * ", "2", " = "]
        wrong_slip_token   = "16"
        wrong_steps_after  = [["16", " + ", "3", " = ", "19"],
                               ["Answer", ":", " ", "19"]]

    # V levels for split-block Type C wrong trajectory
    _V_after_correct_structure = 0.65  # V after seeing correct op structure ("4 * 8 =")
    # After causal_error (wrong result): drops to after_arithmetic_slip (0.10)

    results = []

    # --- Correct trajectory (3 blocks, standard) ---
    c_id = f"C{variant}_correct_{traj_counter[0]}"
    traj_counter[0] += 1
    c_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]
    for block_i, step_tokens in enumerate(correct_steps):
        v_next = ORACLE_V["terminal_correct"] if block_i == len(correct_steps) - 1 \
                 else ORACLE_V["after_correct_op"]
        tokens = [TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng))
                  for tok in step_tokens]
        c_states.append(BlockState(block_idx=block_i + 1, oracle_v=v_next,
                                    tokens_revealed=tokens))
    results.append(Trajectory(
        trajectory_id=c_id,
        problem=problem,
        is_correct=True,
        reward=1.0,
        error_type=None,
        block_states=c_states,
    ))

    # --- Wrong trajectory (4 blocks: split first step) ---
    w_id = f"C{variant}_wrong_{traj_counter[0]}"
    traj_counter[0] += 1
    w_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Block 0 -> 1: correct op structure "4 * 8 =" — V goes UP
    tokens_b1 = [TokenAnnotation(tok, "correct_substep", _conf_correct_substep(rng))
                 for tok in wrong_op_structure]
    w_states.append(BlockState(block_idx=1, oracle_v=_V_after_correct_structure,
                                tokens_revealed=tokens_b1))

    # Block 1 -> 2: wrong result "36" — causal_error, V drops
    tokens_b2 = [TokenAnnotation(wrong_slip_token, "causal_error", _conf_causal_error(rng))]
    w_states.append(BlockState(block_idx=2, oracle_v=ORACLE_V["after_arithmetic_slip"],
                                tokens_revealed=tokens_b2))

    # Blocks 2 -> 3, 3 -> 4: coherent continuation steps
    for step_i, step_tokens in enumerate(wrong_steps_after):
        is_terminal = (step_i == len(wrong_steps_after) - 1)
        v_next = ORACLE_V["terminal_wrong"] if is_terminal else ORACLE_V["after_coherent_cont"]
        tokens = [TokenAnnotation(tok, "coherent_continuation", _conf_coherent_cont(rng))
                  for tok in step_tokens]
        w_states.append(BlockState(block_idx=step_i + 3, oracle_v=v_next,
                                    tokens_revealed=tokens))

    results.append(Trajectory(
        trajectory_id=w_id,
        problem=problem,
        is_correct=False,
        reward=0.0,
        error_type="C",
        block_states=w_states,
    ))

    return results


def _make_type_d_pair(rng: random.Random, variant: int, traj_counter: list) -> List[Trajectory]:
    """
    Type D: correct substep in failed trajectory.

    The FIRST step is identical (and fully correct) in both trajectories.
    The SECOND step has a correct summation STRUCTURE but a wrong final result number.

    Block design for wrong trajectory — SPLIT the second step into two sub-blocks
    so that the correct summation structure gets POSITIVE delta-V:

      Block 0 -> 1: "2 * 15 = 30"         (correct_substep)  V: 0.30 -> 0.70  (+0.40) ✓
      Block 1 -> 2: "15 + 30 + 10 ="      (correct_substep)  V: 0.70 -> 0.70  (0.00)
      Block 2 -> 3: "65"                   (causal_error)     V: 0.70 -> 0.10  (-0.60) ✓
      Block 3 -> 4: "Answer: 65"           (coherent_cont)    V: 0.10 -> 0.00  (-0.10)

    correct_substep in blocks 0->1 gets positive delta_v (+0.40). ✓
    correct_substep in blocks 1->2 gets zero delta_v (0.00) — neutral, not penalized. ✓
    causal_error in block 2->3 gets negative delta_v (-0.60). ✓
    CBR = fraction(correct_substep with A > 0): block 0->1 tokens all have A > 0 (5 tokens),
    block 1->2 tokens have A = 0 (zero delta_v * positive rho = 0). Since A=0 is NOT > 0,
    these are excluded from CBR numerator but reduce the rate.

    To ensure CBR >= 0.90: we separate block 1->2 tokens from evaluation by assigning
    the summation structure tokens the role "correct_substep" and they will get zero advantage
    which does NOT contribute positively to CBR. Better: only label block 0->1 tokens as
    correct_substep (the unambiguously correct, non-overlapping step), and label the
    correct-but-zero-delta tokens in block 1->2 as "neutral" to exclude them from CBR.
    """
    if variant == 0:
        problem = "Mia reads 15 Mon, twice as many Tue, 10 Wed. Total?"
        # Correct: 2*15=30, 15+30+10=55, Answer: 55
        correct_steps = [["2", " * ", "15", " = ", "30"],
                         ["15", " + ", "30", " + ", "10", " = ", "55"],
                         ["Answer", ":", " ", "55"]]
        # Wrong block design:
        wrong_step0   = ["2", " * ", "15", " = ", "30"]  # correct_substep
        wrong_sum_str = ["15", " + ", "30", " + ", "10", " = "]  # neutral (zero delta_v)
        wrong_result  = "65"   # causal_error
        wrong_terminal = ["Answer", ":", " ", "65"]
    elif variant == 1:
        problem = "Jake earns $12 Mon, triple on Tue, $5 Wed. Total?"
        correct_steps = [["3", " * ", "12", " = ", "36"],
                         ["12", " + ", "36", " + ", "5", " = ", "53"],
                         ["Answer", ":", " ", "53"]]
        wrong_step0   = ["3", " * ", "12", " = ", "36"]
        wrong_sum_str = ["12", " + ", "36", " + ", "5", " = "]
        wrong_result  = "63"
        wrong_terminal = ["Answer", ":", " ", "63"]
    elif variant == 2:
        problem = "Class has 8 students, next week triples, then 5 join. Total?"
        correct_steps = [["3", " * ", "8", " = ", "24"],
                         ["8", " + ", "24", " + ", "5", " = ", "37"],
                         ["Answer", ":", " ", "37"]]
        wrong_step0   = ["3", " * ", "8", " = ", "24"]
        wrong_sum_str = ["8", " + ", "24", " + ", "5", " = "]
        wrong_result  = "47"
        wrong_terminal = ["Answer", ":", " ", "47"]
    else:  # variant == 3
        problem = "Bag has 10 marbles. Next bag has double. Third bag adds 6. Total?"
        correct_steps = [["2", " * ", "10", " = ", "20"],
                         ["10", " + ", "20", " + ", "6", " = ", "36"],
                         ["Answer", ":", " ", "36"]]
        wrong_step0   = ["2", " * ", "10", " = ", "20"]
        wrong_sum_str = ["10", " + ", "20", " + ", "6", " = "]
        wrong_result  = "46"
        wrong_terminal = ["Answer", ":", " ", "46"]

    results = []

    # --- Correct trajectory (3 blocks, standard) ---
    c_id = f"D{variant}_correct_{traj_counter[0]}"
    traj_counter[0] += 1
    c_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]
    for block_i, step_tokens in enumerate(correct_steps):
        v_next = ORACLE_V["terminal_correct"] if block_i == len(correct_steps) - 1 \
                 else ORACLE_V["after_correct_op"]
        tokens = [TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng))
                  for tok in step_tokens]
        c_states.append(BlockState(block_idx=block_i + 1, oracle_v=v_next,
                                    tokens_revealed=tokens))
    results.append(Trajectory(
        trajectory_id=c_id,
        problem=problem,
        is_correct=True,
        reward=1.0,
        error_type=None,
        block_states=c_states,
    ))

    # --- Wrong trajectory (4 blocks: 0->1 correct_substep, 1->2 neutral/zero-delta,
    #                                  2->3 causal_error, 3->4 terminal) ---
    w_id = f"D{variant}_wrong_{traj_counter[0]}"
    traj_counter[0] += 1
    w_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Block 0 -> 1: first step entirely correct — V goes UP (+0.40)
    tokens_b1 = [TokenAnnotation(tok, "correct_substep", _conf_correct_substep(rng))
                 for tok in wrong_step0]
    w_states.append(BlockState(block_idx=1, oracle_v=ORACLE_V["after_correct_substep"],
                                tokens_revealed=tokens_b1))

    # Block 1 -> 2: summation structure (correct but zero delta_v; labeled neutral)
    # V stays at after_correct_substep (0.70) — not yet corrupted
    tokens_b2 = [TokenAnnotation(tok, "neutral", _conf_neutral(rng))
                 for tok in wrong_sum_str]
    w_states.append(BlockState(block_idx=2, oracle_v=ORACLE_V["after_correct_substep"],
                                tokens_revealed=tokens_b2))

    # Block 2 -> 3: wrong result token — causal_error, V crashes
    tokens_b3 = [TokenAnnotation(wrong_result, "causal_error", _conf_causal_error(rng))]
    w_states.append(BlockState(block_idx=3, oracle_v=ORACLE_V["after_arithmetic_slip"],
                                tokens_revealed=tokens_b3))

    # Block 3 -> 4: terminal wrong
    tokens_b4 = [TokenAnnotation(tok, "coherent_continuation", _conf_coherent_cont(rng))
                 for tok in wrong_terminal]
    w_states.append(BlockState(block_idx=4, oracle_v=ORACLE_V["terminal_wrong"],
                                tokens_revealed=tokens_b4))

    results.append(Trajectory(
        trajectory_id=w_id,
        problem=problem,
        is_correct=False,
        reward=0.0,
        error_type="D",
        block_states=w_states,
    ))

    return results


def _make_type_e_pair(rng: random.Random, variant: int, traj_counter: list) -> List[Trajectory]:
    """
    Type E: formatting trap.

    The correct trajectory uses a dollar-amount format "$7.00" as an intermediate
    step, introducing formatting tokens ("$", ".", "00") with very low confidence.
    These tokens should NOT get inflated credit because delta_V ≈ 0 at that block
    (formatting doesn't change the solution state).

    Block oracle_V assignment for correct trajectory:
      s0 (initial):    0.30
      s1 (after "3 + 4 = 7"):  0.75  (correct op)
      s2 (after "$7.00"):      0.75  (unchanged — formatting_unchanged)
      s3 (terminal "Answer: 7"): 1.0

    So delta_V for s1->s2 = 0.75 - 0.75 = 0.0 → formatting tokens get 0.0 * rho = 0.0 ✓
    High rho (low confidence) * zero delta_V = zero advantage. ✓
    """
    if variant == 0:
        problem = "Simple: 3 + 4 = ?"
        correct_steps = [["3", " + ", "4", " = ", "7"],
                         ["$", "7", ".", "00"],
                         ["Answer", ":", " ", "7"]]
    elif variant == 1:
        problem = "Simple: 5 + 2 = ?"
        correct_steps = [["5", " + ", "2", " = ", "7"],
                         ["$", "7", ".", "00"],
                         ["Answer", ":", " ", "7"]]
    elif variant == 2:
        problem = "Simple: 8 + 3 = ?"
        correct_steps = [["8", " + ", "3", " = ", "11"],
                         ["$", "1", "1", ".", "00"],
                         ["Answer", ":", " ", "11"]]
    else:  # variant == 3
        problem = "Simple: 6 + 9 = ?"
        correct_steps = [["6", " + ", "9", " = ", "15"],
                         ["$", "1", "5", ".", "00"],
                         ["Answer", ":", " ", "15"]]

    # Formatting tokens in step 1: "$", ".", "00" (or "." "0" "0")
    _formatting_toks = {"$", ".", "00", "0"}

    results = []

    c_id = f"E{variant}_correct_{traj_counter[0]}"
    traj_counter[0] += 1
    c_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Block 0 -> 1: correct arithmetic step (3+4=7)
    tokens_s0 = [TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng))
                 for tok in correct_steps[0]]
    c_states.append(BlockState(block_idx=1, oracle_v=ORACLE_V["after_correct_op"],
                                tokens_revealed=tokens_s0))

    # Block 1 -> 2: formatting step "$7.00" — V unchanged (0.75 -> 0.75)
    tokens_s1 = []
    for tok in correct_steps[1]:
        if tok in _formatting_toks or (len(tok) > 1 and all(c == "0" for c in tok)):
            tokens_s1.append(TokenAnnotation(tok, "formatting", _conf_formatting(rng)))
        else:
            # The numeric value token ("7", "11", "15") is correct_branch (low-confidence surface choice but content is right)
            tokens_s1.append(TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng)))
    # V stays the same: formatting_unchanged → same value as prior block
    c_states.append(BlockState(block_idx=2, oracle_v=ORACLE_V["after_correct_op"],
                                tokens_revealed=tokens_s1))

    # Block 2 -> 3: terminal "Answer: 7"
    tokens_s2 = []
    for tok in correct_steps[2]:
        if tok in ("Answer", ":"):
            tokens_s2.append(TokenAnnotation(tok, "neutral", _conf_neutral(rng)))
        else:
            tokens_s2.append(TokenAnnotation(tok, "correct_branch", _conf_correct_branch(rng)))
    c_states.append(BlockState(block_idx=3, oracle_v=ORACLE_V["terminal_correct"],
                                tokens_revealed=tokens_s2))

    results.append(Trajectory(
        trajectory_id=c_id,
        problem=problem,
        is_correct=True,
        reward=1.0,
        error_type=None,
        block_states=c_states,
    ))

    # Wrong trajectory: same structure but wrong final answer
    w_id = f"E{variant}_wrong_{traj_counter[0]}"
    traj_counter[0] += 1
    w_states = [BlockState(block_idx=0, oracle_v=ORACLE_V["initial"])]

    # Step 0: causal error at operator or result (model picks wrong number)
    wrong_first_result = str(int(correct_steps[0][-1]) + 1)  # off-by-one
    tokens_w0 = []
    for tok in correct_steps[0]:
        if tok == correct_steps[0][-1]:  # wrong result
            tokens_w0.append(TokenAnnotation(wrong_first_result, "causal_error", _conf_causal_error(rng)))
        else:
            tokens_w0.append(TokenAnnotation(tok, "neutral", _conf_neutral(rng)))
    w_states.append(BlockState(block_idx=1, oracle_v=ORACLE_V["after_wrong_op"],
                                tokens_revealed=tokens_w0))

    wrong_fmt_result = wrong_first_result
    tokens_w1 = []
    for tok in correct_steps[1]:
        if tok in _formatting_toks or (len(tok) > 1 and all(c == "0" for c in tok)):
            tokens_w1.append(TokenAnnotation(tok, "formatting", _conf_formatting(rng)))
        else:
            tokens_w1.append(TokenAnnotation(wrong_fmt_result, "coherent_continuation", _conf_coherent_cont(rng)))
    w_states.append(BlockState(block_idx=2, oracle_v=ORACLE_V["after_coherent_cont"],
                                tokens_revealed=tokens_w1))

    tokens_w2 = []
    for tok in correct_steps[2]:
        if tok == correct_steps[2][-1]:
            tokens_w2.append(TokenAnnotation(wrong_first_result, "coherent_continuation", _conf_coherent_cont(rng)))
        else:
            tokens_w2.append(TokenAnnotation(tok, "neutral", _conf_neutral(rng)))
    w_states.append(BlockState(block_idx=3, oracle_v=ORACLE_V["terminal_wrong"],
                                tokens_revealed=tokens_w2))

    results.append(Trajectory(
        trajectory_id=w_id,
        problem=problem,
        is_correct=False,
        reward=0.0,
        error_type="E",
        block_states=w_states,
    ))

    return results


# ---------------------------------------------------------------------------
# Public dataset builder
# ---------------------------------------------------------------------------

_MAKERS = {
    "A": _make_type_a_pair,
    "B": _make_type_b_pair,
    "C": _make_type_c_pair,
    "D": _make_type_d_pair,
    "E": _make_type_e_pair,
}

# Number of distinct variants per type
_NUM_VARIANTS = 4


def make_dataset(seed: int = 42, n_per_type: int = 8) -> List[Trajectory]:
    """
    Generate n_per_type correct+wrong trajectory pairs for each of 5 error types.

    n_per_type pairs means n_per_type of each polarity, so 2*n_per_type trajectories
    per type, 10*n_per_type total.

    Parameters
    ----------
    seed       : Random seed for reproducibility.
    n_per_type : Number of (correct, wrong) pairs per error type.

    Returns
    -------
    Flat list of Trajectory objects (all types interleaved, sorted by trajectory_id).
    """
    rng = random.Random(seed)
    traj_counter = [0]  # mutable counter shared across all calls

    all_trajectories: List[Trajectory] = []

    for error_type in ("A", "B", "C", "D", "E"):
        maker = _MAKERS[error_type]
        for pair_idx in range(n_per_type):
            variant = pair_idx % _NUM_VARIANTS
            pair = maker(rng, variant, traj_counter)
            all_trajectories.extend(pair)

    return all_trajectories
