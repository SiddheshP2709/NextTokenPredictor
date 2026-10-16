"""Evaluation metrics for language modeling tasks."""

import math
import torch
import torch.nn.functional as F


def compute_perplexity(loss: float) -> float:
    """Compute perplexity (PPL) from cross-entropy loss.

    Perplexity = exp(loss), representing the effective branching factor of the model.
    A lower perplexity indicates the model is less surprised by the sequence.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 prediction accuracy.

    Args:
        logits: Tensor of shape (batch_size, vocab_size).
        targets: Ground truth indices of shape (batch_size,).

    Returns:
        Accuracy float between 0.0 and 1.0.
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0
