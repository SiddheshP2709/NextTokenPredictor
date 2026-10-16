"""Device management utility for PyTorch computations."""

from typing import Optional
import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Resolve the optimal compute device available (CUDA, MPS, or CPU).

    Args:
        prefer_cuda: If True, checks for CUDA availability first.

    Returns:
        torch.device instance.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
