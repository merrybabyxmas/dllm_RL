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
_DIFFU_GRPO_PATH = os.environ.get(
    "D1_DIFFU_GRPO_PATH",
    "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo",  # local fallback
)
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

    @torch.no_grad()
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
    # Override rollout to inject confidence_weights into inputs
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self,
        inputs: dict,
    ) -> dict:
        """
        Override d1's rollout to replace generate() with generate_with_confidence()
        and attach per-token responsibility weights to the returned inputs dict.

        All reward scoring, logp computation, and advantage normalization are
        handled by the parent class; we only replace the generation call and
        append confidence_weights so that compute_loss() can use them.
        """
        from trl.models import unwrap_model_for_generation

        device = self.accelerator.device

        # ---- Re-derive prompt tensors (same logic as parent) -----------------
        from trl.data_utils import maybe_apply_chat_template, is_conversational
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        from transformers import Trainer as _Trainer
        prompt_inputs = _Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]

        gen_length   = self.args.max_completion_length
        block_length = self.args.block_length
        steps        = self.args.diffusion_steps
        temperature  = self.args.temperature or 0.0
        cfg_scale    = self.args.cfg_scale

        # ---- Generate with confidence tracking (replaces parent's generate()) -
        all_full_ids       = []
        all_token_conf     = []

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped:
            bs = getattr(self.args, "generation_batch_size", prompt_ids.size(0))
            for i in range(0, prompt_ids.size(0), bs):
                batch_prompt = prompt_ids[i: i + bs]
                full_ids, tok_conf = self.generate_with_confidence(
                    model=unwrapped,
                    prompt=batch_prompt,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                )
                all_full_ids.append(full_ids)
                all_token_conf.append(tok_conf)
                del batch_prompt, full_ids, tok_conf
                torch.cuda.empty_cache()

        prompt_completion_ids = torch.cat(all_full_ids, dim=0)
        token_confidence      = torch.cat(all_token_conf, dim=0)  # [B, total_len]

        # ---- Build completion mask -------------------------------------------
        prompt_length  = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        is_eos         = completion_ids == self.processing_class.eos_token_id
        eos_idx        = torch.full((is_eos.size(0),), is_eos.size(1),
                                    dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx        = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()  # [B, comp_len]

        # ---- Confidence → responsibility weights ----------------------------
        conf_completion = token_confidence[:, prompt_length:]  # [B, comp_len]
        eps      = self.credit_eps
        alpha    = self.credit_alpha
        clip_min = self.credit_clip_min
        clip_max = self.credit_clip_max

        rho      = (conf_completion.float() + eps).pow(-alpha).clamp(clip_min, clip_max)
        mask_f   = completion_mask.float()
        mean_rho = (rho * mask_f).sum(1, keepdim=True) / mask_f.sum(1, keepdim=True).clamp(min=1.0)
        confidence_weights = rho / (mean_rho + 1e-8)  # [B, comp_len]

        # ---- Run parent for everything else (rewards, logps, advantages) ----
        # Monkey-patch self.generate so the parent uses our already-generated ids.
        # Use an offset counter so sub-batching (generation_batch_size < total batch)
        # returns the correct slice for each parent generate() call instead of always
        # returning the first `bs` rows.
        _cached_ids    = prompt_completion_ids  # [total_batch, prompt_len + gen_len]
        _orig_generate = self.generate
        _offset        = [0]

        def _cached_generate(model, prompt, **kwargs):
            bs = prompt.size(0)
            start = _offset[0]
            _offset[0] += bs
            return _cached_ids[start : start + bs]

        self.generate = _cached_generate
        try:
            result = super()._generate_and_score_completions(inputs)
        finally:
            self.generate = _orig_generate
            _offset[0]    = 0  # reset for safety

        # ---- Attach confidence weights to result ----------------------------
        result["confidence_weights"] = confidence_weights  # [B, comp_len]
        return result

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

        # INTEGRATION CHECK: confidence_weights must be injected by the data
        # pipeline (i.e. _generate_and_score_completions must be overridden to
        # call generate_with_confidence and attach the weights to inputs).
        # If this key is absent, Stage 1 silently falls back to standard GRPO.
        # TODO: override _generate_and_score_completions to guarantee injection.
        import warnings as _warnings
        if "confidence_weights" not in inputs or inputs.get("confidence_weights") is None:
            _warnings.warn(
                "CWGRPOTrainer.compute_loss: 'confidence_weights' not found in "
                "inputs — falling back to standard GRPO (no confidence weighting). "
                "Override _generate_and_score_completions to inject confidence weights.",
                stacklevel=2,
            )

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
