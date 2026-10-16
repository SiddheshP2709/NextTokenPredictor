"""Dataset and Data Loading utilities for n-gram language modeling."""

from typing import List, Tuple
import torch
from torch.utils.data import Dataset


class NGramDataset(Dataset):
    """PyTorch Dataset that constructs context-target pairs from a sequence of token IDs.

    For context size k, maps sequence (t_1, t_2, ..., t_M) into pairs:
        X_i = (t_i, t_{i+1}, ..., t_{i+k-1})
        Y_i = t_{i+k}
    """

    def __init__(self, token_ids: List[int], context_size: int = 5) -> None:
        self.context_size = context_size

        if len(token_ids) <= context_size:
            raise ValueError(
                f"Token sequence length ({len(token_ids)}) must be strictly greater than context_size ({context_size})."
            )

        num_samples = len(token_ids) - context_size
        X = []
        Y = []
        for i in range(num_samples):
            X.append(token_ids[i : i + context_size])
            Y.append(token_ids[i + context_size])

        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
