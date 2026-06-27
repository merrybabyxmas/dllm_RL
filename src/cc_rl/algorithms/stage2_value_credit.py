"""
Stage 2: Value-Baseline Confidence Credit trainer (delta-V implementation).

Theory (equations referenced in comments below)
-------------------------------------------------
At each sampled block boundary b:

    s_b  = mean_pool( h[-1](x[:p+b*L]) )         (state after completing block b)
    V(s) = value_head( s )                         (scalar value estimate)

Per-segment advantage (with subsampling to max_value_states boundaries):
    delta_v[segment] = V(s_{b_next}) - V(s_{b_cur})   (non-terminal)
    delta_v[terminal] = r - V(s_{B-1})                 (terminal TD(0))

All blocks in the same segment share the segment-level delta_v.
Per-token advantage = delta_v[segment] * rho_t  (confidence weight, Eq. 3)

Key OOM fixes vs. the naive implementation
-------------------------------------------
1. value_hidden_size default reduced to 256 (was 1024): ~4x smaller AdamW states.
2. max_value_states (default 4): subsample block boundaries for delta-V; only
   K+1 forward passes instead of num_blocks passes.
3. All intermediate V tensors are immediately moved to CPU after each forward
   pass; torch.cuda.empty_cache() is called between passes.
4. The final delta_v_per_token is the only GPU tensor stored in the result dict.
5. Policy and value backward passes use completely separate graphs:
   - Value: backbone runs in torch.no_grad(), hidden.detach() before value_head.
   - Policy: separate backbone forward with grad enabled.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# TRL 1.6.0 compat patch
import trl.import_utils as _trl_iu
if not hasattr(_trl_iu, "is_rich_available"):
    try:
        from transformers.utils import is_rich_available as _ira
        _trl_iu.is_rich_available = _ira
    except ImportError:
        _trl_iu.is_rich_available = lambda: False

_DIFFU_GRPO_PATH = os.environ.get(
    "D1_DIFFU_GRPO_PATH",
    "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo",
)
if _DIFFU_GRPO_PATH not in sys.path:
    sys.path.insert(0, _DIFFU_GRPO_PATH)

from cc_rl.algorithms.stage1_cw_grpo import CWGRPOTrainer
from cc_rl.critics.value_head import ValueHead


class ValueCreditTrainer(CWGRPOTrainer):
    """
    Stage 2: Block-level delta-V credit assignment trainer.

    Extends CWGRPOTrainer (Stage 1) to replace the group-level GRPO advantage
    with per-block temporal-difference advantages from a learned value head.

    Additional __init__ parameters
    -------------------------------
    value_hidden_size  : Hidden dim of value MLP. Default 256 (reduced from 1024
                         to cut GPU memory ~4x and avoid OOM on 95 GB GPUs).
    value_mlp_layers   : Number of MLP layers (default 2).
    critic_lr          : Learning rate for value head optimizer (default 5e-6).
    critic_loss_coef   : Kept for API compatibility; value head uses own optimizer.
    delta_v_gate       : Min delta_v std before switching from GRPO to delta-V.
    max_value_states   : Max number of block boundaries to evaluate V at per
                         trajectory. Subsamples evenly to limit forward passes
                         and peak GPU memory. Default 4; set to 2 for tiny
                         datasets (MBPP/HumanEval). Use num_blocks to disable.
    """

    def __init__(
        self,
        *args,
        value_hidden_size: int = 256,
        value_mlp_layers: int = 2,
        critic_lr: float = 5e-6,
        critic_loss_coef: float = 0.5,
        delta_v_gate: float = 0.01,
        max_value_states: int = 4,
        use_confidence_weight: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.critic_loss_coef       = critic_loss_coef
        self.delta_v_gate           = delta_v_gate
        self.max_value_states       = max_value_states
        self.use_confidence_weight  = use_confidence_weight

        try:
            hidden_size = self.model.config.hidden_size
        except AttributeError:
            hidden_size = self.model.base_model.config.hidden_size

        self.value_head = ValueHead(
            hidden_size=hidden_size,
            mlp_hidden_size=value_hidden_size,
            n_layers=value_mlp_layers,
        ).to(self.accelerator.device)

        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=critic_lr,
            weight_decay=0.0,
        )

    # ------------------------------------------------------------------
    # Helper: backbone hidden states for a prefix (no-grad, returns CPU)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _value_at_prefix(
        self,
        model: nn.Module,
        prompt_completion_ids: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        """
        Return V(s) where s is the mean-pooled state of the first prefix_len tokens.
        Result is returned on CPU to free GPU memory immediately.
        """
        prefix_ids = prompt_completion_ids[:, :prefix_len]
        outputs    = model(prefix_ids, output_hidden_states=True)
        hidden     = outputs.hidden_states[-1]          # [batch, prefix_len, H]
        v          = self.value_head(hidden.float())    # value head is fp32
        return v.cpu()

    # ------------------------------------------------------------------
    # Delta-V advantages with subsampling + CPU staging
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_delta_v_advantages(
        self,
        model: nn.Module,
        prompt_completion_ids: torch.Tensor,
        prompt_length: int,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token delta-V advantages using at most max_value_states
        evenly-sampled block boundaries.

        Segments between consecutive sampled boundaries share the segment-level
        delta V. Terminal block (always the last) uses r - V(s_{B-1}).

        Returns
        -------
        delta_v_per_token : [batch, gen_length]  on the original GPU device.
        """
        gen_length   = self.args.max_completion_length
        block_length = self.args.block_length
        num_blocks   = gen_length // block_length
        batch_size   = prompt_completion_ids.size(0)
        device       = prompt_completion_ids.device

        # ---- Select block boundaries to evaluate --------------------------------
        num_samples = min(self.max_value_states, num_blocks)
        if num_samples >= num_blocks:
            eval_blocks = list(range(num_blocks))
        else:
            # Evenly spaced: always include block 0 and block num_blocks-1
            step = (num_blocks - 1) / max(num_samples - 1, 1)
            raw  = {round(i * step) for i in range(num_samples)}
            raw.add(0); raw.add(num_blocks - 1)
            eval_blocks = sorted(raw)[:num_samples]
            # Guarantee terminal is last
            if eval_blocks[-1] != num_blocks - 1:
                eval_blocks[-1] = num_blocks - 1

        # ---- Evaluate V at each selected boundary (results on CPU) --------------
        V_at: dict[int, torch.Tensor] = {}
        self.value_head.eval()
        for b in eval_blocks:
            prefix_len  = prompt_length + (b + 1) * block_length
            V_at[b]     = self._value_at_prefix(model, prompt_completion_ids, prefix_len)
            torch.cuda.empty_cache()
        self.value_head.train()

        # ---- Build per-block delta_v on CPU ------------------------------------
        rewards_cpu          = rewards.float().cpu()
        delta_v_per_block    = torch.zeros(batch_size, num_blocks)  # CPU

        for i, b in enumerate(eval_blocks):
            if b == num_blocks - 1:
                # Terminal block: TD error r - V(s_{B-1})
                delta_v_per_block[:, b] = rewards_cpu - V_at[b]
            else:
                b_next = eval_blocks[i + 1]
                dv     = V_at[b_next] - V_at[b]          # [batch]
                # All blocks in [b .. b_next-1] share this segment-level delta
                delta_v_per_block[:, b:b_next] = dv.unsqueeze(1)

        # ---- Expand to per-token and move to GPU --------------------------------
        delta_v_per_block  = delta_v_per_block.to(device)
        delta_v_per_token  = delta_v_per_block.repeat_interleave(block_length, dim=1)
        assert delta_v_per_token.shape == (batch_size, gen_length), (
            f"Shape mismatch: {delta_v_per_token.shape} vs expected {(batch_size, gen_length)}"
        )
        return delta_v_per_token

    # ------------------------------------------------------------------
    # Override rollout to add delta_v_advantages
    # ------------------------------------------------------------------

    def _generate_and_score_completions(self, inputs: dict) -> dict:
        """
        Extends Stage 1 rollout to attach block-level delta-V advantages.

        Clears GPU cache before the parent generation call to avoid OOM from
        value-head memory overhead, then computes delta-V after generation
        with all intermediate tensors staged on CPU.
        """
        # Explicit cache clear: value head parameters + optimizer states add
        # ~10-20 MB (with default value_hidden_size=256); clearing fragmented
        # cache ensures they don't eat into the generation budget.
        torch.cuda.empty_cache()

        # Step 1: parent (Stage 1) generation + confidence weights + GRPO advantages
        result = super()._generate_and_score_completions(inputs)

        # Step 2: compute block-level delta-V advantages
        prompt_ids     = result["prompt_ids"]
        completion_ids = result["completion_ids"]
        raw_rewards    = result["advantages"]   # group-centered rewards from GRPO

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_length         = prompt_ids.size(1)

        from trl.models import unwrap_model_for_generation
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped:
            delta_v_advantages = self._compute_delta_v_advantages(
                model=unwrapped,
                prompt_completion_ids=prompt_completion_ids,
                prompt_length=prompt_length,
                rewards=raw_rewards,
            )

        result["delta_v_advantages"] = delta_v_advantages   # [batch, comp_len] on GPU
        result["raw_rewards"]        = raw_rewards           # for value head MSE target
        return result

    # ------------------------------------------------------------------
    # Loss: separate value update (own optimizer) + policy PPO-clip
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Stage 2 loss computation.

        Step 1  — Value head update (separate optimizer, no backbone grad):
            hidden = backbone(input).detach()
            v_loss = MSE(value_head(hidden), raw_reward)
            value_optimizer.step()  ← completely isolated from policy graph

        Step 2+  — Policy PPO-clip with delta-V * confidence weights:
            adv = delta_v_advantages * cw_normalized   (Eq. 3)
            policy_loss = PPO-clip(adv.detach())
            ← HF Trainer calls loss.backward() after this returns

        The two backward passes share no computational graph nodes.
        """
        if return_outputs:
            raise ValueError("ValueCreditTrainer does not support return_outputs=True")

        prompt_ids      = inputs["prompt_ids"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        mask_seeds      = inputs["mask_seeds"]

        # ==============================================================
        # Step 1: Value head update (backbone frozen via no_grad + detach)
        # ==============================================================
        raw_rewards = inputs.get("raw_rewards")
        v_loss_val  = 0.0
        if raw_rewards is not None:
            input_ids_full = torch.cat([prompt_ids, completion_ids], dim=1)

            with torch.no_grad():
                backbone_out = model(input_ids_full, output_hidden_states=True)
            # Detach: value_head graph does NOT flow through backbone
            hidden = backbone_out.hidden_states[-1].detach()   # [batch, seq, H]
            del backbone_out
            torch.cuda.empty_cache()

            v_pred  = self.value_head(hidden.float())           # value head is fp32
            target  = raw_rewards.float().to(v_pred.device)
            v_loss  = F.huber_loss(v_pred, target, delta=1.0)   # Huber for stability

            self.value_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.value_optimizer.step()
            v_loss_val = v_loss.item()

            del hidden, v_pred, v_loss
            torch.cuda.empty_cache()

        # ==============================================================
        # Step 2: Per-token policy log-probs (new graph, no value grad)
        # ==============================================================
        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        this_itr_idx       = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids_3d       = input_ids.unsqueeze(0)

        per_token_logps = self._get_per_token_logps(
            model, input_ids_3d, logits_to_keep, [this_itr_mask_seed]
        ).squeeze(0)  # [batch, comp_len]

        # KL penalty
        if self.beta != 0.0:
            ref_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_logps - per_token_logps)
                - (ref_logps - per_token_logps)
                - 1
            )

        # ==============================================================
        # Step 3: Build per-token advantages (delta-V * conf weights)
        # ==============================================================
        delta_v_adv     = inputs.get("delta_v_advantages")    # [batch, comp_len]
        conf_weights    = inputs.get("confidence_weights") if self.use_confidence_weight else None
        grpo_adv        = inputs["advantages"]                 # [batch]

        def _build_adv(scalar_or_pertok, conf_weights):
            """Apply confidence weighting; scalar_or_pertok may be [batch] or [batch, L]."""
            mask_f   = completion_mask.float()
            if scalar_or_pertok.dim() == 1:
                base = scalar_or_pertok.unsqueeze(1).expand_as(mask_f)
            else:
                base = scalar_or_pertok
            if conf_weights is not None:
                cw      = conf_weights[:, -logits_to_keep:]
                mean_cw = (cw * mask_f).sum(1, keepdim=True) / mask_f.sum(1, keepdim=True).clamp(min=1)
                cw_norm = cw / (mean_cw + 1e-8)
                return base * cw_norm
            return base

        using_delta_v = False
        if delta_v_adv is not None and delta_v_adv.std().item() > self.delta_v_gate:
            per_token_adv = _build_adv(delta_v_adv, conf_weights)
            using_delta_v = True
        else:
            per_token_adv = _build_adv(grpo_adv, conf_weights)

        # ==============================================================
        # Step 4: PPO-clip policy loss  (adv treated as constant)
        # ==============================================================
        old_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )

        ratio  = torch.exp(per_token_logps - old_logps)
        ratio_clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        adv_const = per_token_adv.detach()   # treat advantage as constant

        per_token_loss = -torch.min(ratio * adv_const, ratio_clipped * adv_const)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        policy_loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # ==============================================================
        # Step 5: Metrics logging
        # ==============================================================
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_clipped = (ratio < ratio_clipped).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        # Log value_loss and policy_loss separately
        self._metrics[mode].setdefault("value_loss",    []).append(v_loss_val)
        self._metrics[mode].setdefault("policy_loss",   []).append(policy_loss.item())
        self._metrics[mode].setdefault("using_delta_v", []).append(float(using_delta_v))

        return policy_loss
