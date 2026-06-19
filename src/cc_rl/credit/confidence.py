"""
Utilities for extracting per-token confidence scores from a masked diffusion
LLM's generation process.

Confidence c_t is defined as the softmax probability of the model's chosen
token at the denoising step in which that token was first committed:

    c_t = softmax(logits_t)[argmax(logits_t)]

This is the "low_confidence remasking" score from LLaDA / MDLM.  High
confidence means the model was certain; low confidence means it hedged.

The credit assignment in responsibility.py inverts this: uncertain decisions
(low c_t) receive higher responsibility weights.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def extract_token_confidence(
    logits: torch.Tensor,          # [batch, seq_len, vocab_size]
    chosen_ids: torch.Tensor,      # [batch, seq_len]
    mask: Optional[torch.Tensor] = None,  # [batch, seq_len] bool — positions to compute
) -> torch.Tensor:
    """
    Extract softmax confidence of the chosen token at each position.

    Parameters
    ----------
    logits     : Raw model logits [batch, seq_len, vocab_size].
    chosen_ids : Token IDs that were chosen [batch, seq_len].
    mask       : Optional boolean mask; only extract confidence at True positions.
                 Unmasked positions get confidence = 0.0.

    Returns
    -------
    confidence : [batch, seq_len] tensor in [0, 1].
    """
    probs = F.softmax(logits.float(), dim=-1)  # [batch, seq_len, vocab]
    chosen_probs = torch.gather(probs, dim=-1, index=chosen_ids.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        chosen_probs = chosen_probs * mask.float()
    return chosen_probs


def compute_mean_confidence(
    confidences: List[float],
    mask: Optional[List[bool]] = None,
) -> float:
    """
    Compute mean confidence over a list of per-token scores.

    Parameters
    ----------
    confidences : List of per-token confidence values.
    mask        : Optional boolean list; only average over True positions.

    Returns
    -------
    float mean confidence.
    """
    if mask is not None:
        vals = [c for c, m in zip(confidences, mask) if m]
    else:
        vals = confidences
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
