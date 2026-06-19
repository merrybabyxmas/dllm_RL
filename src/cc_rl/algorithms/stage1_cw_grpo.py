"""
Stage 1: Confidence-Weighted GRPO (CW-GRPO).

Extends DiffuGRPOTrainer to apply per-token confidence-derived responsibility
weights to the GRPO advantages before computing the PPO-clip policy loss.

Key modification in compute_loss():
    weighted_adv_t = group_advantage * rho_t
    loss_t = -min(r_t * weighted_adv_t, clip(r_t) * weighted_adv_t)

where rho_t = (c_t + eps)^{-alpha}, normalized to unit mean within each
trajectory.  This up-weights gradient signal at positions where the model
was LESS confident (low c_t -> high rho_t).

The generation step is identical to DiffuGRPO.  Confidence scores are
extracted from the low-confidence remasking probability (softmax prob of
the chosen token at each denoising step).
"""
from __future__ import annotations

import sys
import os
import math
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# TRL 1.6.0 compat patch
import trl.import_utils as _trl_iu
if not hasattr(_trl_iu, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_iu.is_rich_available = _ira
    except ImportError:
        _trl_iu.is_rich_available = lambda: False

# Ensure d1/diffu-grpo is importable
_DIFFU_GRPO_PATH = "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo"
if _DIFFU_GRPO_PATH not in sys.path:
    sys.path.insert(0, _DIFFU_GRPO_PATH)

from diffu_grpo_trainer import DiffuGRPOTrainer  # noqa: E402
from cc_rl.credit.responsibility import compute_responsibility_weights


class CWGRPOTrainer(DiffuGRPOTrainer):
    """
    Stage 1: Confidence-Weighted GRPO trainer.

    Extends DiffuGRPOTrainer with per-token responsibility weighting of
    the PPO advantages.  All other hyperparameters (generation, KL penalty,
    clipping) are inherited unchanged.

    Additional __init__ parameters
    -------------------------------
    credit_alpha    : Exponent for inverse confidence weighting (default 1.0).
    credit_eps      : Stability offset for confidence (default 1e-6).
    credit_clip_min : Minimum weight before normalization (default 0.25).
    credit_clip_max : Maximum weight before normalization (default 4.0).
    """

    def __init__(
        self,
        *args,
        credit_alpha: float = 1.0,
        credit_eps: float = 1e-6,
        credit_clip_min: float = 0.25,
        credit_clip_max: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.credit_alpha = credit_alpha
        self.credit_eps = credit_eps
        self.credit_clip_min = credit_clip_min
        self.credit_clip_max = credit_clip_max

    # ------------------------------------------------------------------
    # Generation with confidence tracking
    # ------------------------------------------------------------------

    def generate_with_confidence(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
    ):
        """
        Extended generate() that also returns per-token confidence scores.

        Returns
        -------
        x               : [batch, prompt_len + gen_length]  final token IDs
        token_confidence: [batch, prompt_len + gen_length]  per-token confidence
                          (0.0 for prompt positions; softmax prob at reveal time
                          for generated positions)
        """
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            total_len = prompt.shape[1] + gen_length
            x = torch.full((bs, total_len), mask_id, dtype=torch.long, device=model.device)
            x[:, :prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id  # [bs, total_len]

            # Per-token confidence: 0 for prompt, filled on reveal
            token_confidence = torch.zeros(bs, total_len, device=model.device)

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            steps_per_block = max(1, steps // num_blocks)

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    mask_index = x == mask_id  # [bs, total_len]

                    with torch.cuda.amp.autocast(enabled=self.args.fp16 if hasattr(self.args, 'fp16') else True):
                        if cfg_scale > 0.0:
                            un_x = x.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            logits = model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            logits = model(x).logits

                        logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                        x0 = torch.argmax(logits_with_noise, dim=-1)

                        if remasking == "low_confidence":
                            p = F.softmax(logits.to(dtype), dim=-1)
                            x0_p = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                            )
                        else:
                            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

                        # Only consider tokens within the current block
                        x0_p[:, end_idx:] = -np.inf

                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

                        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                        for j in range(bs):
                            num_tokens = num_transfer_tokens[j, i].item()
                            if num_tokens > 0:
                                _, select_index = torch.topk(confidence[j], k=int(num_tokens))
                                transfer_index[j, select_index] = True

                        # Record confidence at the moment of reveal
                        for j in range(bs):
                            revealed = transfer_index[j] & mask_index[j]
                            positions = revealed.nonzero(as_tuple=True)[0]
                            if len(positions) > 0:
                                confs = x0_p[j, positions].clamp(0.0, 1.0)
                                token_confidence[j, positions] = confs.float()

                        x[transfer_index] = x0[transfer_index]

            return x, token_confidence

    # ------------------------------------------------------------------
    # Loss computation with confidence weighting
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        PPO-clip loss with confidence-weighted per-token advantages.

        Modification vs. DiffuGRPOTrainer.compute_loss:
            weighted_adv[b, t] = advantages[b] * cw_normalized[b, t]

        where cw_normalized is the responsibility weight rho_t normalized
        to unit mean within each sequence.
        """
        if return_outputs:
            raise ValueError("CWGRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]
        # confidence_weights: [batch, completion_len] — may or may not be present
        confidence_weights = inputs.get("confidence_weights", None)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids_3d = input_ids.unsqueeze(0)  # [1, batch, seq_len]
        per_token_logps = self._get_per_token_logps(
            model, input_ids_3d, logits_to_keep, [this_itr_mask_seed]
        )

        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]  # [batch]

        # Build confidence-weighted advantages: [batch, completion_len]
        if confidence_weights is not None:
            # confidence_weights is [batch, total_seq_len]; slice to completion
            cw = confidence_weights[:, -logits_to_keep:]  # [batch, completion_len]
            # Compute per-trajectory normalization (masked mean)
            mask_f = completion_mask.float()
            mean_cw = (cw * mask_f).sum(1, keepdim=True) / mask_f.sum(1, keepdim=True).clamp(min=1.0)
            cw_normalized = cw / (mean_cw + 1e-8)
            # Expand group advantage and multiply by per-token weight
            weighted_adv = advantages.unsqueeze(1) * cw_normalized  # [batch, completion_len]
        else:
            # No confidence weights: standard GRPO
            weighted_adv = advantages.unsqueeze(1).expand_as(completion_mask)

        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )

        # PPO-clip ratio
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * weighted_adv
        per_token_loss2 = coef_2 * weighted_adv
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Metrics
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss
