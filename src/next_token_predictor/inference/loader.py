"""Unified checkpoint and vocabulary loader across all domains.

Solves PyTorch serialization caveats (such as unpickling models saved with module '__main__')
and provides backward compatibility across text, C++, and Python models.
"""

from typing import Any, Dict, Optional, Tuple, Union
import os
import sys
import pickle
import torch
import torch.nn as nn

from next_token_predictor.models.mlp import (
    Next_Word_Predictor,
    NextTokenPredictor,
    MLPNextTokenPredictor,
)
from next_token_predictor.utils.device import get_device


def _ensure_main_namespace() -> None:
    """Inject model class definitions into '__main__' namespace for unpickling safety.

    PyTorch full-model serialization (`torch.save(model, ...)`) stores class references
    relative to the module where they were created. Notebook training scripts run in '__main__',
    so deserialization in external scripts or Streamlit requires these classes in '__main__'.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        if not hasattr(main_mod, "Next_Word_Predictor"):
            setattr(main_mod, "Next_Word_Predictor", Next_Word_Predictor)
        if not hasattr(main_mod, "NextTokenPredictor"):
            setattr(main_mod, "NextTokenPredictor", NextTokenPredictor)
        if not hasattr(main_mod, "MLPNextTokenPredictor"):
            setattr(main_mod, "MLPNextTokenPredictor", MLPNextTokenPredictor)


def resolve_path(file_path: str) -> str:
    """Resolve file path across common directory layouts."""
    if os.path.exists(file_path):
        return file_path

    # Try relative to repo root
    possible_roots = [
        os.getcwd(),
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        os.path.join(os.getcwd(), "NextTokenPredictor"),
    ]

    candidate_subdirs = [
        "",
        "checkpoints",
        "checkpoints/war_and_peace",
        "checkpoints/python",
        "data/vocabularies",
        "data/vocabularies/war_and_peace",
        "data/vocabularies/python",
    ]

    base_name = os.path.basename(file_path)

    for root in possible_roots:
        # direct join
        cand = os.path.join(root, file_path)
        if os.path.exists(cand):
            return cand

        # search subdirs
        for sub in candidate_subdirs:
            cand = os.path.join(root, sub, base_name)
            if os.path.exists(cand):
                return cand

    return file_path


def load_checkpoint(
    model_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.device]:
    """Load a trained model checkpoint from disk.

    Args:
        model_path: Path to the .pt file.
        device: Target device (defaults to auto-detected GPU/CPU).

    Returns:
        Tuple of (model, device).
    """
    resolved_path = resolve_path(model_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{model_path}' (resolved: '{resolved_path}')"
        )

    if device is None:
        device = get_device()

    _ensure_main_namespace()

    # Attempt loading using torch.serialization with fallback
    try:
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([Next_Word_Predictor, NextTokenPredictor, MLPNextTokenPredictor]):
                model = torch.load(resolved_path, map_location=device, weights_only=False)
        else:
            model = torch.load(resolved_path, map_location=device, weights_only=False)
    except Exception as e:
        # Retry with direct unpickle
        _ensure_main_namespace()
        model = torch.load(resolved_path, map_location=device, weights_only=False)

    model.to(device)
    model.eval()
    return model, device


def load_vocabulary(
    vocab_path: str,
    itos_path: Optional[str] = None,
    max_vocab_size: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary mappings (string-to-index and index-to-string).

    Supports .pkl dictionary files and .csv frequency files.
    """
    resolved_path = resolve_path(vocab_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    if resolved_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(resolved_path, keep_default_na=False)
        if max_vocab_size is not None and len(df) > max_vocab_size:
            df = df.iloc[:max_vocab_size]
        tokens = df["token"].astype(str).tolist()
        stoi = {t: i for i, t in enumerate(tokens)}
        itos = {i: s for s, i in stoi.items()}
        return stoi, itos

    elif resolved_path.endswith(".pkl"):
        with open(resolved_path, "rb") as f:
            stoi = pickle.load(f)

        if itos_path:
            resolved_itos = resolve_path(itos_path)
            if os.path.exists(resolved_itos):
                with open(resolved_itos, "rb") as f:
                    itos = pickle.load(f)
            else:
                itos = {i: s for s, i in stoi.items()}
        else:
            # Check for standard itos naming pattern
            dir_name = os.path.dirname(resolved_path)
            base_name = os.path.basename(resolved_path)
            candidate_itos = base_name.replace("stoi", "itos")
            cand_path = os.path.join(dir_name, candidate_itos)
            if os.path.exists(cand_path):
                with open(cand_path, "rb") as f:
                    itos = pickle.load(f)
            else:
                itos = {i: s for s, i in stoi.items()}

        return stoi, itos

    else:
        raise ValueError(f"Unsupported vocabulary format: {resolved_path}")
