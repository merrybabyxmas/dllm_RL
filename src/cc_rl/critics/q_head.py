"""
Neural Q-head for Stage 3 credit assignment.

QHead estimates Q(s, a) by pooling hidden states (for the state embedding)
and concatenating the action embedding (token logit at the chosen position).

Architecture:
    state_repr = mean_pool(hidden_states[-1])   # [batch, hidden_size]
    action_emb = backbone_embed(action_id)      # [batch, hidden_size]
    q = MLP( cat([state_repr, action_emb]) )    # [batch]
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class QHead(nn.Module):
    """
    Action-conditioned Q-value head.

    Parameters
    ----------
    hidden_size     : Backbone hidden dimension.
    mlp_hidden_size : Intermediate MLP width.
    n_layers        : Number of MLP layers.
    dropout         : Dropout probability.

    Forward signature
    -----------------
    hidden_states : [batch, seq_len, hidden_size]  backbone last hidden layer
    action_hiddens: [batch, hidden_size]  hidden state at the token action position
    attention_mask: [batch, seq_len]  optional masking for state pooling

    Returns
    -------
    q_values : [batch]
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        # Input: concatenation of state pool + action embedding = 2 * hidden_size
        input_size = hidden_size * 2
        layers = []
        in_size = input_size
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_size, mlp_hidden_size))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = mlp_hidden_size
        layers.append(nn.Linear(in_size, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,           # [batch, seq_len, hidden_size]
        action_hiddens: torch.Tensor,           # [batch, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        q_values : [batch]
        """
        # Pool hidden states for state representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            state_repr = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            state_repr = hidden_states.mean(dim=1)          # [batch, hidden]

        # Concatenate state repr and action embedding
        combined = torch.cat([state_repr, action_hiddens], dim=-1)  # [batch, 2*hidden]
        return self.mlp(combined).squeeze(-1)               # [batch]
