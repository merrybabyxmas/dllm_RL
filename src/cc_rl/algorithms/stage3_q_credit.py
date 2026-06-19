"""
Stage 3: Q-Value Confidence Credit trainer.

Extends ValueCreditTrainer with an additional Q-head Q(s, a) that estimates
the action-conditional return.  The per-step advantage becomes:

    final_adv_t = (Q(s_t, a_t) - V(s_t)) * rho_t

Q-head architecture: concatenate mean-pooled state repr with the action
token's hidden representation, then pass through an MLP.

Both value head and Q-head are trained via MSE to the terminal reward.
"""
from __future__ import annotations

import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIFFU_GRPO_PATH = os.environ.get("D1_DIFFU_GRPO_PATH", "/home/dongwoo43/papers/paper_dllm/d1/diffu-grpo")
if _DIFFU_GRPO_PATH not in sys.path:
    sys.path.insert(0, _DIFFU_GRPO_PATH)

from cc_rl.algorithms.stage2_value_credit import ValueCreditTrainer
from cc_rl.critics.q_head import QHead


class QCreditTrainer(ValueCreditTrainer):
    """
    Stage 3: Q-Value Confidence Credit trainer.

    Additional __init__ parameters
    -------------------------------
    q_hidden_size : Hidden dimension of the Q-head MLP (default 1024).
    q_mlp_layers  : Number of MLP layers in the Q-head (default 2).
    q_lr          : Learning rate for the Q-head optimizer (default 5e-6).
    q_loss_coef   : Weight on the Q-head MSE loss term (default 0.5).
    """

    def __init__(
        self,
        *args,
        q_hidden_size: int = 1024,
        q_mlp_layers: int = 2,
        q_lr: float = 5e-6,
        q_loss_coef: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.q_loss_coef = q_loss_coef

        hidden_size = self.model.config.hidden_size
        self.q_head = QHead(
            hidden_size=hidden_size,
            mlp_hidden_size=q_hidden_size,
            n_layers=q_mlp_layers,
        ).to(self.accelerator.device)

        self.q_optimizer = torch.optim.AdamW(
            self.q_head.parameters(),
            lr=q_lr,
            weight_decay=0.0,
        )

    # ------------------------------------------------------------------
    # Q-value estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_q_value(
        self,
        input_ids: torch.Tensor,          # [batch, seq_len]
        action_position_ids: torch.Tensor, # [batch]  index of action token
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Estimate Q(s, a) for each sample in the batch.

        The action is identified by its position in the sequence;
        the corresponding hidden state is used as the action embedding.

        Parameters
        ----------
        input_ids          : [batch, seq_len]
        action_position_ids: [batch]  token position of the action token
        attention_mask     : [batch, seq_len]

        Returns
        -------
        q_values : [batch]
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]

        # Extract action-position hidden state as action embedding
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        # Clamp position IDs to valid range
        pos = action_position_ids.clamp(0, hidden.size(1) - 1)
        action_hiddens = hidden[batch_idx, pos, :]  # [batch, hidden]

        return self.q_head(hidden, action_hiddens, attention_mask)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Combined policy + value + Q loss.

        Q-head is trained to predict the terminal reward given state + action
        embeddings.  The policy update uses (Q - V) advantages weighted by
        confidence scores.
        """
        if return_outputs:
            raise ValueError("QCreditTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]

        # ------------------------------------------------------------------
        # Train Q-head (if raw rewards available)
        # ------------------------------------------------------------------
        raw_rewards = inputs.get("raw_rewards")
        action_positions = inputs.get("action_positions")  # [batch] last token position

        if raw_rewards is not None and action_positions is not None:
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            with torch.enable_grad():
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].detach()

                # State representation: mean pool
                batch = hidden.size(0)
                state_repr = hidden.mean(dim=1)                    # [batch, hidden]

                # Action embedding: hidden at action position
                pos = action_positions.clamp(0, hidden.size(1) - 1)
                batch_idx = torch.arange(batch, device=hidden.device)
                action_hiddens = hidden[batch_idx, pos, :]         # [batch, hidden]

                q_pred = self.q_head(hidden, action_hiddens)       # [batch]
                target = raw_rewards.float().to(q_pred.device)
                q_loss = F.mse_loss(q_pred, target) * self.q_loss_coef

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_head.parameters(), 1.0)
            self.q_optimizer.step()

        # Fall through to Stage 2's compute_loss (which trains value head and runs policy)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
