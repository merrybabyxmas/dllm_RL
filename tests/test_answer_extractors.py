"""
Tests for reward functions and answer extractors.

Covers:
- extract_final_number (GSM8K)
- reward_gsm8k
- extract_boxed_answer (MATH-500)
- normalize_math_answer
- reward_math500
"""
import pytest
from cc_rl.rewards.exact_match import extract_final_number, reward_gsm8k
from cc_rl.rewards.math_normalize import extract_boxed_answer, normalize_math_answer, reward_math500


class TestExtractFinalNumber:
    """Tests for the GSM8K #### delimiter extractor."""

    def test_hash_format(self):
        assert extract_final_number("The answer is #### 42") == "42"

    def test_hash_format_with_spaces(self):
        assert extract_final_number("#### 100") == "100"

    def test_hash_format_decimal(self):
        assert extract_final_number("Result: #### 3.14") == "3.14"

    def test_hash_format_negative(self):
        assert extract_final_number("Loss: #### -5") == "-5"

    def test_fallback_last_number(self):
        """Without ####, return last number in text."""
        assert extract_final_number("I computed 12 then got 42") == "42"

    def test_comma_stripped(self):
        """Commas in large numbers should be removed."""
        result = extract_final_number("#### 1,234")
        assert result == "1234"

    def test_empty_string(self):
        assert extract_final_number("") is None

    def test_no_numbers(self):
        assert extract_final_number("no numbers here") is None

    def test_multiline(self):
        text = "Step 1: add 5\nStep 2: multiply\n#### 25"
        assert extract_final_number(text) == "25"


class TestRewardGSM8K:
    """Tests for the binary GSM8K reward function."""

    def test_correct_integer(self):
        assert reward_gsm8k("The answer is #### 42", "#### 42") == 1.0

    def test_wrong_integer(self):
        assert reward_gsm8k("The answer is #### 41", "#### 42") == 0.0

    def test_correct_decimal(self):
        assert reward_gsm8k("#### 3.5", "#### 3.5") == 1.0

    def test_format_tolerance(self):
        """Completion without #### but with correct last number."""
        assert reward_gsm8k("I get 42 as the answer", "#### 42") == 1.0

    def test_none_completion(self):
        """No number in completion -> 0.0."""
        assert reward_gsm8k("I don't know", "#### 42") == 0.0

    def test_comma_normalization(self):
        """1,000 == 1000."""
        assert reward_gsm8k("#### 1,000", "#### 1000") == 1.0

    def test_gold_with_reasoning(self):
        """Gold answer with full GSM8K reasoning + #### delimiter."""
        gold = "She has 3 apples + 4 oranges = 7 fruits.\n#### 7"
        assert reward_gsm8k("The answer is #### 7", gold) == 1.0
        assert reward_gsm8k("The answer is #### 8", gold) == 0.0


class TestExtractBoxedAnswer:
    r"""Tests for \boxed{} extraction."""

    def test_simple_boxed(self):
        assert extract_boxed_answer(r"The answer is \boxed{42}") == "42"

    def test_boxed_fraction(self):
        assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_last_boxed(self):
        """Multiple boxed — return the last one."""
        text = r"First \boxed{1} then \boxed{2}"
        assert extract_boxed_answer(text) == "2"

    def test_no_boxed(self):
        assert extract_boxed_answer("no boxed here") is None


class TestNormalizeMathAnswer:
    """Tests for math answer normalization."""

    def test_integer(self):
        assert normalize_math_answer("42") == "42"

    def test_float_trailing_zero(self):
        assert normalize_math_answer("3.50") == "3.5"

    def test_integer_float(self):
        """5.0 should normalize to '5'."""
        assert normalize_math_answer("5.0") == "5"

    def test_comma_removal(self):
        assert normalize_math_answer("1,000") == "1000"


class TestRewardMath500:
    r"""Tests for \boxed{}-aware MATH-500 reward."""

    def test_correct(self):
        assert reward_math500(r"\boxed{42}", r"\boxed{42}") == 1.0

    def test_wrong(self):
        assert reward_math500(r"\boxed{41}", r"\boxed{42}") == 0.0

    def test_normalized_equal(self):
        """5.0 and 5 should be equal."""
        assert reward_math500(r"\boxed{5.0}", r"\boxed{5}") == 1.0

    def test_empty_completion(self):
        assert reward_math500("", r"\boxed{42}") == 0.0
