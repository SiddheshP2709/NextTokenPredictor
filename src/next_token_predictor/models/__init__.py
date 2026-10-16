"""Model architectures for next token prediction."""

from next_token_predictor.models.mlp import (
    Next_Word_Predictor,
    NextTokenPredictor,
    MLPNextTokenPredictor,
)
from next_token_predictor.models.lstm import LSTMNextTokenPredictor

__all__ = [
    "Next_Word_Predictor",
    "NextTokenPredictor",
    "MLPNextTokenPredictor",
    "LSTMNextTokenPredictor",
]
