"""
Neural value head for Stage 2 credit assignment.

ValueHead attaches to a pretrained language model backbone and produces
scalar state-value estimates by pooling the final hidden states.

Architecture: mean-pooled hidden_states[-1] -> MLP(hidden_size, ..., 1)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    MLP value head operating on mean-pooled hidden states.

    Parameters
    ----------
    hidden_size     : Backbone hidden dimension (e.g., 4096 for LLaDA-8B).
    mlp_hidden_size : Intermediate MLP width.
    n_layers        : Number of linear layers in the MLP (must be >= 1).
    dropout         : Dropout probability.

    Forward input
    -------------
    hidden_states : [batch, seq_len, hidden_size]  last hidden layer of backbone.
    attention_mask: [batch, seq_len]  optional; used for masked mean pooling.

    Forward output
    --------------
    values : [batch]  scalar value estimates.
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

        layers = []
        in_size = hidden_size
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
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [batch, seq_len, hidden_size]
        attention_mask: [batch, seq_len]  (1 = real token, 0 = pad)

        Returns
        -------
        values : [batch]
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()           # [batch, seq, 1]
            pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = hidden_states.mean(dim=1)                    # [batch, hidden]

        return self.mlp(pooled).squeeze(-1)                       # [batch]
