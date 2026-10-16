"""Autoregressive text and code generation engine."""

from typing import Any, Dict, List, Optional, Tuple, Union
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGenerator:
    """Autoregressive sequence generator for language and code models.

    Supports:
        - Fixed context sliding window
        - Temperature scaling (controlling entropy / randomness)
        - Top-k truncation filtering
        - Multinomial sampling and greedy decoding
        - Real-time probability tracking for interview demonstrations
    """

    def __init__(
        self,
        model: nn.Module,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        device: Optional[torch.device] = None,
        context_size: Optional[int] = None,
        domain: str = "text",
    ) -> None:
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.domain = domain.lower()

        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
            self.model.to(self.device)

        self.model.eval()

        # Infer context window size from model attributes
        if context_size is not None:
            self.context_size = context_size
        elif hasattr(model, "size"):
            self.context_size = getattr(model, "size")
        elif hasattr(model, "context_size"):
            self.context_size = getattr(model, "context_size")
        elif hasattr(model, "block_size"):
            self.context_size = getattr(model, "block_size")
        else:
            self.context_size = 5

    def tokenize_prompt(self, prompt: str) -> List[str]:
        """Tokenize input prompt according to domain rules."""
        if self.domain in ("cpp", "python", "code"):
            regex = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[{}()\[\];.,=+\-*/<>!&|^%~?:]")
            return regex.findall(prompt)
        else:
            cleaned = prompt.lower()
            cleaned = re.sub(r"[^a-zA-Z0-9 \.]", "", cleaned)
            cleaned = re.sub(r"\.", " . ", cleaned)
            return cleaned.split()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        track_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Autoregressively generate next tokens from prompt.

        Args:
            prompt: Seed string.
            max_new_tokens: Number of future tokens to predict.
            temperature: Sampling temperature (T > 0). Lower is more deterministic, higher is more diverse.
            top_k: If set, restricts sampling to the top k most probable candidates.
            seed: Optional integer seed for deterministic reproducibility.
            track_probabilities: If True, records top candidates at each generation step.

        Returns:
            Dict containing:
                'prompt': original prompt
                'generated_tokens': list of generated token strings
                'full_text': reconstructed output string
                'steps': per-step probability tracking list
        """
        if seed is not None:
            torch.manual_seed(seed)

        tokens = self.tokenize_prompt(prompt)
        token_ids: List[int] = [self.stoi.get(t, 0) for t in tokens]

        generated_token_strings: List[str] = []
        step_logs: List[Dict[str, Any]] = []

        temperature = max(1e-4, float(temperature))

        for step in range(max_new_tokens):
            # Windowing: pad with 0 if shorter than context_size, or slice last context_size tokens
            if len(token_ids) < self.context_size:
                context_ids = [0] * (self.context_size - len(token_ids)) + token_ids
            else:
                context_ids = token_ids[-self.context_size:]

            x = torch.tensor([context_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Extract last position logits (shape: 1, vocab_size)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                logits = logits.squeeze(0)

                # Temperature scaling
                scaled_logits = logits / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[-1]] = -float("Inf")

                # Probability distribution via Softmax
                probs = F.softmax(scaled_logits, dim=-1)

                # Track top candidate probabilities for educational/interview visualization
                if track_probabilities:
                    top_probs, top_indices = torch.topk(probs, min(5, probs.size(-1)))
                    step_logs.append({
                        "step": step + 1,
                        "candidates": [
                            (self.itos.get(idx.item(), f"<{idx.item()}>"), prob.item())
                            for idx, prob in zip(top_indices, top_probs)
                        ],
                    })

                # Sample next token
                next_id = torch.multinomial(probs, num_samples=1).item()

            token_ids.append(next_id)
            next_word = self.itos.get(next_id, f"<{next_id}>")
            generated_token_strings.append(next_word)

        # Reconstruct output text
        if self.domain in ("cpp", "python", "code"):
            full_text = prompt + (" " if prompt and not prompt.endswith(" ") else "") + " ".join(generated_token_strings)
        else:
            full_text = prompt + " " + " ".join(generated_token_strings)
            full_text = re.sub(r"\s+\.", ".", full_text)

        return {
            "prompt": prompt,
            "generated_tokens": generated_token_strings,
            "full_text": full_text,
            "steps": step_logs,
        }
