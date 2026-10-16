"""Legacy utility functions (Maintained for backward compatibility).

Recommended modern interface: import from `next_token_predictor`.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from next_token_predictor.models.mlp import Next_Word_Predictor, NextTokenPredictor
from next_token_predictor.inference.loader import load_checkpoint as load_pretrained_model
from next_token_predictor.inference.generator import TextGenerator

__all__ = [
    "Next_Word_Predictor",
    "NextTokenPredictor",
    "load_pretrained_model",
]