"""Unit tests for inference, checkpoint loading, and autoregressive generation."""

import os
import pytest
import torch

from next_token_predictor.models.mlp import Next_Word_Predictor
from next_token_predictor.inference.generator import TextGenerator
from next_token_predictor.inference.loader import load_checkpoint, load_vocabulary


def test_generator_deterministic_seed():
    vocab = ["the", "prince", "entered", "room", "and", "smiled", "."]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}

    model = Next_Word_Predictor(size=3, vocab_size=len(vocab), embed_dim=8, hidden_dim=16)
    generator = TextGenerator(model=model, stoi=stoi, itos=itos, context_size=3, domain="text")

    out1 = generator.generate("the prince", max_new_tokens=4, seed=42)
    out2 = generator.generate("the prince", max_new_tokens=4, seed=42)

    assert out1["generated_tokens"] == out2["generated_tokens"]
    assert out1["full_text"] == out2["full_text"]


def test_generator_probability_tracking():
    vocab = ["int", "main", "(", ")", "{", "return", "0", ";", "}"]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}

    model = Next_Word_Predictor(size=2, vocab_size=len(vocab), embed_dim=8, hidden_dim=16)
    generator = TextGenerator(model=model, stoi=stoi, itos=itos, context_size=2, domain="cpp")

    result = generator.generate("int main", max_new_tokens=3, track_probabilities=True)
    assert len(result["steps"]) == 3
    assert len(result["steps"][0]["candidates"]) > 0


def test_load_existing_checkpoints():
    wp_path = "checkpoints/war_and_peace/model2_task1.pt"
    if os.path.exists(wp_path):
        model, device = load_checkpoint(wp_path)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    py_path = "checkpoints/python/python_model1.pt"
    if os.path.exists(py_path):
        model, device = load_checkpoint(py_path)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
