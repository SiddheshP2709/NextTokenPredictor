"""Unit tests for neural language model architectures."""

import pytest
import torch

from next_token_predictor.models.mlp import (
    Next_Word_Predictor,
    NextTokenPredictor,
    MLPNextTokenPredictor,
)
from next_token_predictor.models.lstm import LSTMNextTokenPredictor


def test_next_word_predictor_forward():
    batch_size = 4
    context_size = 5
    vocab_size = 100
    embed_dim = 16
    hidden_dim = 32

    model = Next_Word_Predictor(
        size=context_size,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        activation_fn="relu",
    )

    x = torch.randint(0, vocab_size, (batch_size, context_size))
    logits = model(x)

    assert logits.shape == (batch_size, vocab_size)
    assert not torch.isnan(logits).any()


def test_next_token_predictor_forward():
    batch_size = 4
    context_size = 5
    vocab_size = 250
    embed_dim = 32
    hidden_dim = 64

    model = NextTokenPredictor(
        vocab_size=vocab_size,
        context_size=context_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )

    x = torch.randint(0, vocab_size, (batch_size, context_size))
    logits = model(x)

    assert logits.shape == (batch_size, vocab_size)
    assert model.get_num_params() > 0


def test_mlp_configurable_activations():
    vocab_size = 50
    context_size = 3
    for act in ["relu", "tanh", "sigmoid", "gelu"]:
        model = MLPNextTokenPredictor(
            vocab_size=vocab_size,
            context_size=context_size,
            activation=act,
        )
        x = torch.randint(0, vocab_size, (2, context_size))
        out = model(x)
        assert out.shape == (2, vocab_size)


def test_lstm_forward_pass():
    batch_size = 2
    seq_len = 7
    vocab_size = 80
    embed_dim = 16
    hidden_dim = 32

    model = LSTMNextTokenPredictor(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, (hn, cn) = model(x)

    assert logits.shape == (batch_size, vocab_size)
    assert hn.shape == (1, batch_size, hidden_dim)
    assert cn.shape == (1, batch_size, hidden_dim)
