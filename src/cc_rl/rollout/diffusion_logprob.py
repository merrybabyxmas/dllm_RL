"""
Utilities for computing per-token log-probabilities under a masked
diffusion LLM (e.g., LLaDA / MDLM).

The forward process randomly masks tokens; the model predicts the original
token IDs from the masked sequence.  The conditional log-prob of a
completion token x_t given a noised sequence x_tilde is:

    log p_theta(x_t | x_tilde) = log softmax(logits)[x_t]

which equals the negative cross-entropy loss for that position.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


@torch.inference_mode()
def get_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,       # [batch, seq_len]
    completion_mask: torch.Tensor, # [batch, seq_len]  bool/int
    mask_id: int = 126336,
    p_mask: float = 0.15,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute per-token log-probabilities for completion tokens only.

    Parameters
    ----------
    model          : Masked-diffusion LM (e.g., LLaDA).
    input_ids      : Token IDs [batch, seq_len], prompt + completion concatenated.
    completion_mask: Boolean/int mask, True for completion positions [batch, seq_len].
    mask_id        : Vocabulary index of the [MASK] token.
    p_mask         : Probability of masking each completion token during evaluation.
    seed           : Optional RNG seed for reproducible masking.

    Returns
    -------
    per_token_logps : [batch, seq_len] — log p(x_t | x_tilde) at completion positions,
                      0.0 at prompt positions.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = input_ids.device
    batch, seq_len = input_ids.shape

    # Apply random masking to completion tokens only
    completion_bool = completion_mask.bool()
    rand = torch.rand(batch, seq_len, device=device)
    masked_ids = input_ids.clone()
    should_mask = completion_bool & (rand < p_mask)
    masked_ids[should_mask] = mask_id

    # Forward pass
    logits = model(masked_ids).logits  # [batch, seq_len, vocab]

    # Cross-entropy = -log p(x_t | x_tilde) for each position
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = input_ids.reshape(-1)
    loss_flat = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    log_probs = -loss_flat.view(batch, seq_len)  # [batch, seq_len]

    # Zero out prompt positions
    log_probs = log_probs * completion_bool.float()
    return log_probs.to(torch.float32)


@torch.inference_mode()
def get_per_token_logprobs_multiseed(
    model: torch.nn.Module,
    input_ids: torch.Tensor,        # [batch, seq_len]
    completion_mask: torch.Tensor,  # [batch, seq_len]
    seeds: List[int],
    mask_id: int = 126336,
    p_mask: float = 0.15,
) -> torch.Tensor:
    """
    Average per-token log-probs over multiple masking seeds for lower variance.

    Returns
    -------
    [batch, seq_len] mean log-probabilities.
    """
    all_logps = []
    for seed in seeds:
        lp = get_per_token_logprobs(
            model, input_ids, completion_mask, mask_id=mask_id, p_mask=p_mask, seed=seed
        )
        all_logps.append(lp)
    # Stack and mean: [n_seeds, batch, seq_len] -> [batch, seq_len]
    return torch.stack(all_logps, dim=0).mean(0)
