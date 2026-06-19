"""
Confidence-to-responsibility-weight mapping.

The core credit-assignment hypothesis: tokens where the model was LESS
confident were the pivotal decision points.  We therefore assign HIGHER
responsibility to low-confidence tokens via an inverse-power law:

    rho_t = clip( (c_t + eps)^{-alpha}, clip_min, clip_max )

then optionally normalize so the mean within a trajectory equals 1.0,
preserving the expected magnitude of the policy gradient.

Parameters
----------
alpha    : Exponent controlling how aggressively low-confidence tokens are
           up-weighted.  alpha=1 gives pure inverse weighting; alpha=0 gives
           uniform weighting.
eps      : Small additive offset to prevent division by zero at c_t=0.
clip_min : Floor on rho_t (prevents near-zero-confidence from dominating).
clip_max : Ceiling on rho_t.
normalize: If True, divide each rho_t by mean(rho_t) so the per-trajectory
           mean weight is 1.0 (neutral scale for policy gradient).
"""
from __future__ import annotations

from typing import List


def compute_responsibility_weights(
    confidences: List[float],
    alpha: float = 1.0,
    eps: float = 1e-6,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
    normalize: bool = True,
) -> List[float]:
    """
    Map per-token confidence scores to responsibility weights.

    rho_t = clip( (c_t + eps)^{-alpha}, clip_min, clip_max )

    If normalize=True:  rho_t <- rho_t / mean(rho)

    Parameters
    ----------
    confidences : List of per-token confidence values c_t in [0, 1].
    alpha       : Inverse-power exponent.
    eps         : Stability offset added to each c_t before exponentiation.
    clip_min    : Minimum allowed weight before normalization.
    clip_max    : Maximum allowed weight before normalization.
    normalize   : Whether to normalize weights to unit mean.

    Returns
    -------
    List of responsibility weights rho_t, same length as confidences.

    Examples
    --------
    >>> compute_responsibility_weights([0.20, 0.95], alpha=1.0, eps=0.0,
    ...                                clip_min=0.0, clip_max=999.0)
    [1.6524..., 0.3476...]  # low-confidence token gets ~4.75x more weight
    """
    if not confidences:
        return []

    raw: List[float] = []
    for c in confidences:
        # rho_t = (c_t + eps)^{-alpha}
        rho = (c + eps) ** (-alpha)
        # Clip to valid range
        rho = max(clip_min, min(clip_max, rho))
        raw.append(rho)

    if normalize and len(raw) > 0:
        mean_rho = sum(raw) / len(raw)
        if mean_rho > 0:
            return [r / mean_rho for r in raw]

    return raw
