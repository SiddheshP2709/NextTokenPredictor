"""Lexical Code Tokenizer for C++ and Python Source Code.

Implements lexical analysis for programming languages by extracting identifiers,
keywords, numeric constants, operators, and structural delimiters using regular expressions.
"""

from typing import Dict, List, Optional
import os
import pickle
import re
import pandas as pd


class CodeTokenizer:
    """Tokenizer designed for source code syntax in C++, Python, and similar languages.

    Regex pattern matches:
        - Identifier / keyword tokens: [A-Za-z_][A-Za-z0-9_]*
        - Structural delimiters & operators: [{}()\\[\\];.,=+\\-*/<>!&|^%~?:]
    """

    TOKEN_REGEX = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*|[{}()\[\];.,=+\-*/<>!&|^%~?:]"
    )

    def __init__(
        self,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
        unk_token: str = "<unk>",
        unk_id: int = 0,
        language: str = "cpp",
    ) -> None:
        self.stoi = stoi or {}
        self.itos = itos or {i: s for s, i in self.stoi.items()}
        self.unk_token = unk_token
        self.unk_id = unk_id
        self.language = language

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        max_vocab_size: Optional[int] = None,
        language: str = "cpp",
    ) -> "CodeTokenizer":
        """Load vocabulary from CSV file containing token frequency distribution.

        Note: Uses keep_default_na=False to avoid corrupting tokens like 'NULL' or 'nan'.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Vocabulary CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, keep_default_na=False)
        if max_vocab_size is not None and len(df) > max_vocab_size:
            df = df.iloc[:max_vocab_size]

        tokens = df["token"].astype(str).tolist()
        stoi = {t: i for i, t in enumerate(tokens)}
        itos = {i: s for s, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, language=language)

    @classmethod
    def from_pickle(
        cls,
        stoi_path: str,
        itos_path: Optional[str] = None,
        language: str = "cpp",
    ) -> "CodeTokenizer":
        """Load vocabulary from pickled dictionaries."""
        if not os.path.exists(stoi_path):
            raise FileNotFoundError(f"Pickle not found: {stoi_path}")

        with open(stoi_path, "rb") as f:
            stoi = pickle.load(f)

        if itos_path and os.path.exists(itos_path):
            with open(itos_path, "rb") as f:
                itos = pickle.load(f)
        else:
            itos = {i: s for s, i in stoi.items()}

        return cls(stoi=stoi, itos=itos, language=language)

    def save_pickle(self, stoi_path: str, itos_path: Optional[str] = None) -> None:
        """Serialize mappings to pickle."""
        os.makedirs(os.path.dirname(stoi_path), exist_ok=True)
        with open(stoi_path, "wb") as f:
            pickle.dump(self.stoi, f)
        if itos_path:
            os.makedirs(os.path.dirname(itos_path), exist_ok=True)
            with open(itos_path, "wb") as f:
                pickle.dump(self.itos, f)

    def tokenize_code(self, code_str: str) -> List[str]:
        """Extract lexical tokens from code text."""
        return self.TOKEN_REGEX.findall(code_str)

    def encode(self, code_str: str, skip_oov: bool = False) -> List[int]:
        """Convert code text into token IDs."""
        tokens = self.tokenize_code(code_str)
        token_ids: List[int] = []
        for t in tokens:
            if t in self.stoi:
                token_ids.append(self.stoi[t])
            elif not skip_oov:
                token_ids.append(self.unk_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Reconstruct code text from token IDs with basic formatting heuristics."""
        tokens = [self.itos.get(idx, self.unk_token) for idx in token_ids]
        reconstructed = ""
        no_space_before = {";", ",", ")", "]", "}", "."}
        no_space_after = {"(", "[", "{", "."}

        for i, t in enumerate(tokens):
            if i == 0:
                reconstructed += t
            elif t in no_space_before or (i > 0 and tokens[i - 1] in no_space_after):
                reconstructed += t
            else:
                reconstructed += " " + t
        return reconstructed

    def __len__(self) -> int:
        return len(self.stoi)
