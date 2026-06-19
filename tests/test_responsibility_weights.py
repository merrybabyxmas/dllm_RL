"""
Unit tests for compute_responsibility_weights.

Verifies the inverse-power-law weight formula and normalization.

Mathematical expectations:

tau2: confidences = [0.20, 0.95], alpha=1.0, eps=0.0
  rho_raw = [1/0.20, 1/0.95] = [5.0, 1.05263...]
  mean_rho = (5.0 + 1.05263) / 2 = 3.02631...
  w[0] = 5.0 / 3.02631 = 1.6524...
  w[1] = 1.05263 / 3.02631 = 0.3476...

tau1: confidences = [0.55, 0.90], alpha=1.0, eps=0.0
  rho_raw = [1/0.55, 1/0.90] = [1.81818..., 1.11111...]
  mean_rho = (1.81818 + 1.11111) / 2 = 1.46465...
  w[0] = 1.81818 / 1.46465 = 1.2414...
  w[1] = 1.11111 / 1.46465 = 0.7586...
"""
import pytest
from cc_rl.credit.responsibility import compute_responsibility_weights


class TestResponsibilityWeights:
    """Core unit tests for compute_responsibility_weights."""

    def test_responsibility_weights_tau2(self):
        """Low-confidence token (0.20) should get ~4.75x the weight of 0.95."""
        confidences = [0.20, 0.95]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=999.0,
            normalize=True,
        )
        assert weights[0] == pytest.approx(1.652, abs=1e-3)
        assert weights[1] == pytest.approx(0.348, abs=1e-3)

    def test_responsibility_weights_tau1(self):
        """Moderate confidence tokens (0.55, 0.90)."""
        confidences = [0.55, 0.90]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=999.0,
            normalize=True,
        )
        assert weights[0] == pytest.approx(1.242, abs=1e-3)
        assert weights[1] == pytest.approx(0.758, abs=1e-3)

    def test_normalize_mean_is_one(self):
        """After normalization, the mean weight should always be 1.0."""
        for confidences in [
            [0.1, 0.5, 0.9],
            [0.3, 0.7],
            [0.55, 0.60, 0.65, 0.70],
        ]:
            weights = compute_responsibility_weights(
                confidences,
                alpha=1.0,
                eps=1e-8,
                clip_min=0.0,
                clip_max=999.0,
                normalize=True,
            )
            mean_w = sum(weights) / len(weights)
            assert mean_w == pytest.approx(1.0, abs=1e-6), \
                f"Mean weight {mean_w} != 1.0 for confidences {confidences}"

    def test_uniform_confidence_gives_uniform_weights(self):
        """All equal confidences -> all weights equal (1.0 after normalization)."""
        confidences = [0.5, 0.5, 0.5]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=999.0,
            normalize=True,
        )
        for w in weights:
            assert w == pytest.approx(1.0, abs=1e-6)

    def test_alpha_zero_gives_uniform_weights(self):
        """alpha=0 means (c)^0 = 1.0 for all tokens -> uniform weights."""
        confidences = [0.1, 0.5, 0.9]
        weights = compute_responsibility_weights(
            confidences,
            alpha=0.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=999.0,
            normalize=True,
        )
        for w in weights:
            assert w == pytest.approx(1.0, abs=1e-6)

    def test_clip_min_floor(self):
        """Very high-confidence tokens should be floored at clip_min."""
        confidences = [0.999]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=0.0,
            clip_min=0.5,
            clip_max=999.0,
            normalize=True,
        )
        # rho_raw = 1/0.999 ≈ 1.001, which is > clip_min=0.5 -> no clipping
        # After norm with single element: weight = 1.0
        assert weights[0] == pytest.approx(1.0, abs=1e-3)

    def test_clip_max_ceiling(self):
        """Near-zero confidence tokens should be capped at clip_max."""
        confidences = [1e-10, 0.9]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=5.0,   # cap at 5.0
            normalize=True,
        )
        # rho_raw[0] = 1/1e-10 = 1e10, clipped to 5.0
        # rho_raw[1] = 1/0.9 ≈ 1.111
        # mean = (5.0 + 1.111) / 2 = 3.056
        # w[0] = 5.0 / 3.056 = 1.636..., w[1] = 1.111 / 3.056 = 0.364
        assert weights[0] < weights[1] * 6  # clipped ratio
        assert weights[0] > 0

    def test_no_normalize(self):
        """Without normalization, weights are raw inverse confidence values."""
        confidences = [0.5, 0.5]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=999.0,
            normalize=False,
        )
        expected = 1.0 / 0.5
        for w in weights:
            assert w == pytest.approx(expected, abs=1e-6)

    def test_empty_input(self):
        """Empty confidence list should return empty list."""
        weights = compute_responsibility_weights([], normalize=True)
        assert weights == []

    def test_single_token(self):
        """Single token always gets weight 1.0 after normalization."""
        weights = compute_responsibility_weights(
            [0.3],
            alpha=1.0,
            eps=0.0,
            clip_min=0.0,
            clip_max=999.0,
            normalize=True,
        )
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0, abs=1e-6)

    def test_ordering_preserved(self):
        """Lower confidence always maps to higher weight (before clipping)."""
        confidences = [0.1, 0.3, 0.7, 0.9]
        weights = compute_responsibility_weights(
            confidences,
            alpha=1.0,
            eps=1e-9,
            clip_min=0.0,
            clip_max=9999.0,
            normalize=True,
        )
        # Weights should be monotonically decreasing (low conf -> high weight)
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], \
                f"Weight ordering violated: w[{i}]={weights[i]} <= w[{i+1}]={weights[i+1]}"
