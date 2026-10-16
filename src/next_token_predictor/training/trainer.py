"""Standardized Model Trainer for Next Token Predictor."""

from typing import Callable, Dict, List, Optional
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from next_token_predictor.utils.device import get_device
from next_token_predictor.utils.metrics import compute_perplexity, compute_accuracy


class Trainer:
    """Configurable training manager for neural probabilistic language models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch over the dataset."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.dim() == 3:
                logits = logits[:, -1, :]

            loss = self.loss_fn(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += batch_size

        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
        epoch_ppl = compute_perplexity(epoch_loss)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "perplexity": epoch_ppl,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                loss = self.loss_fn(logits, y_batch)

                batch_size = y_batch.size(0)
                total_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += batch_size

        eval_loss = total_loss / total_samples if total_samples > 0 else 0.0
        eval_acc = total_correct / total_samples if total_samples > 0 else 0.0
        eval_ppl = compute_perplexity(eval_loss)

        return {
            "loss": eval_loss,
            "accuracy": eval_acc,
            "perplexity": eval_ppl,
        }

    def save_checkpoint(self, save_path: str, save_state_dict_only: bool = False) -> None:
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_state_dict_only:
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model, save_path)
