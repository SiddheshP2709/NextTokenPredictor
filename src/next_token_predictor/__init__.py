"""Next Token Predictor: Neural Language Modeling Package.

Provides neural probabilistic language models (Bengio et al. MLP architectures),
lexical code tokenizers, natural language tokenizers, autoregressive generation,
and unified model checkpoint loaders across Natural Language, C++, and Python domains.
"""

from next_token_predictor.models.mlp import Next_Word_Predictor, NextTokenPredictor, MLPNextTokenPredictor
from next_token_predictor.models.lstm import LSTMNextTokenPredictor
from next_token_predictor.tokenization.text_tokenizer import TextTokenizer
from next_token_predictor.tokenization.code_tokenizer import CodeTokenizer
from next_token_predictor.inference.loader import load_checkpoint, load_vocabulary
from next_token_predictor.inference.generator import TextGenerator
from next_token_predictor.utils.device import get_device

__version__ = "1.0.0"

__all__ = [
    "Next_Word_Predictor",
    "NextTokenPredictor",
    "MLPNextTokenPredictor",
    "LSTMNextTokenPredictor",
    "TextTokenizer",
    "CodeTokenizer",
    "load_checkpoint",
    "load_vocabulary",
    "TextGenerator",
    "get_device",
]
