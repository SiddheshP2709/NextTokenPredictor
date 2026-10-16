"""Recurrent Neural Network (LSTM) baseline for sequential token prediction.

Provides a comparison baseline to the fixed-window MLP architecture.
Maintains an internal hidden state across arbitrary sequence lengths, addressing
the fixed-window restriction of n-gram models.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class LSTMNextTokenPredictor(nn.Module):
    """LSTM-based Language Model for Next Token Prediction.

    Maintains recurrent memory of previous tokens rather than flattening a fixed context.

    Architecture:
        Token IDs -> Embedding -> LSTM(Layers) -> Linear -> Vocabulary Logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input token IDs of shape (batch_size, sequence_length).
            hidden: Optional tuple (h_0, c_0) for recurrent state.

        Returns:
            Tuple of:
                logits: Unnormalized predictions for the last token (batch_size, vocab_size).
                hidden: Updated hidden and cell states.
        """
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output[:, -1, :])
        return logits, hidden

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
