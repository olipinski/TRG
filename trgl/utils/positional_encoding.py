"""Positional encoding for use with the attention layers."""
# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Positional encoding for use with the attention layers."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Positional encoding for use with the attention layers."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode positions of tokens.

        Parameters
        ----------
        x: torch.Tensor
            shape [seq_len, batch_size, embedding_dim]

        Returns
        -------
            Encoded token, with dropout applied if set
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
