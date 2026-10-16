"""Word-level Tokenizer for Natural Language Corpora (e.g., War and Peace).

Handles lowercase normalization, alphanumeric filtering, explicit punctuation handling
(retaining '.' as sentence boundary markers), and bidirectional string-to-integer mappings.
"""

from typing import Dict, List, Optional, Union
import os
import pickle
import re


class TextTokenizer:
    """Word-level tokenizer with string-to-index (stoi) and index-to-string (itos) mappings."""

    def __init__(
        self,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
        unk_token: str = "<unk>",
        unk_id: int = 0,
    ) -> None:
        self.stoi = stoi or {}
        self.itos = itos or {i: s for s, i in self.stoi.items()}
        self.unk_token = unk_token
        self.unk_id = unk_id

    @classmethod
    def from_pickle(cls, stoi_path: str, itos_path: Optional[str] = None) -> "TextTokenizer":
        """Load vocabulary mappings from serialized pickle files."""
        if not os.path.exists(stoi_path):
            raise FileNotFoundError(f"Vocabulary mapping file not found at: {stoi_path}")

        with open(stoi_path, "rb") as f:
            stoi = pickle.load(f)

        if itos_path and os.path.exists(itos_path):
            with open(itos_path, "rb") as f:
                itos = pickle.load(f)
        else:
            itos = {i: s for s, i in stoi.items()}

        return cls(stoi=stoi, itos=itos)

    @classmethod
    def build_from_text(cls, text: str) -> "TextTokenizer":
        """Build vocabulary from raw plain text.

        Replicates the exact normalization and unique-word ordering from the training pipeline.
        """
        filtered = re.sub(r"-", " ", text)
        filtered = re.sub(r"[^a-zA-Z0-9 \.\n]", "", filtered).lower()

        lines = filtered.split(".")
        words: List[str] = ["."]
        for line in lines:
            for w in line.split():
                if w:
                    words.append(w)

        unique_words: List[str] = []
        seen = set()
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)

        stoi = {s: i for i, s in enumerate(unique_words)}
        itos = {i: s for s, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def save(self, stoi_path: str, itos_path: Optional[str] = None) -> None:
        """Serialize vocabulary mappings to disk."""
        os.makedirs(os.path.dirname(stoi_path), exist_ok=True)
        with open(stoi_path, "wb") as f:
            pickle.dump(self.stoi, f)
        if itos_path:
            os.makedirs(os.path.dirname(itos_path), exist_ok=True)
            with open(itos_path, "wb") as f:
                pickle.dump(self.itos, f)

    def clean_text(self, text: str) -> List[str]:
        """Normalize raw input string into token strings."""
        normalized = text.lower()
        normalized = re.sub(r"[^a-zA-Z0-9 \.]", "", normalized)
        normalized = re.sub(r"\.", " . ", normalized)
        return normalized.split()

    def encode(self, text: str, skip_oov: bool = False) -> List[int]:
        """Convert raw text into a list of token IDs.

        Args:
            text: Input string.
            skip_oov: If True, tokens not present in vocabulary are omitted.
                      If False, unknown tokens map to unk_id.

        Returns:
            List of integer token IDs.
        """
        tokens = self.clean_text(text)
        token_ids: List[int] = []
        for t in tokens:
            if t in self.stoi:
                token_ids.append(self.stoi[t])
            elif not skip_oov:
                token_ids.append(self.unk_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert a sequence of token IDs back into text."""
        words: List[str] = [self.itos.get(idx, self.unk_token) for idx in token_ids]
        text = " ".join(words)
        text = re.sub(r"\s+\.", ".", text)
        return text

    def __len__(self) -> int:
        return len(self.stoi)
