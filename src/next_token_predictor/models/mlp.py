"""Multi-Layer Perceptron (MLP) architectures for Neural Language Modeling.

Based on the foundational Neural Probabilistic Language Model (Bengio et al., 2003).
Maps an n-gram context of previous tokens to high-dimensional distributed vector
embeddings, concatenates the embeddings, projects them through a non-linear hidden
layer, and computes unnormalized log-probabilities (logits) over the vocabulary.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class Next_Word_Predictor(nn.Module):
    """Bengio-style Neural Language Model for Natural Language Next-Word Prediction.

    Maintains backward compatibility with serialized checkpoint weights trained
    on Leo Tolstoy's War and Peace corpus (model1_task1.pt, model2_task1.pt, model3_task1.pt).

    Mathematical Formulation:
        1. Context tokens: c = (w_1, w_2, ..., w_n) where w_i in {0, ..., |V|-1}
        2. Embedding lookup: e_i = E[w_i] in R^d
        3. Flattened projection: h_0 = concat(e_1, e_2, ..., e_n) in R^(n * d)
        4. Hidden transformation: h_1 = activation(W_1 * h_0 + b_1) in R^H
        5. Logits projection: z = W_2 * h_1 + b_2 in R^|V|
    """

    def __init__(
        self,
        size: int = 5,
        vocab_size: int = 19764,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        activation_fn: str = "relu",
        seed_value: int = 42,
    ) -> None:
        super().__init__()
        self.size = size
        self.hyperpams: Dict[str, Any] = {
            "size": size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "activation_fn": activation_fn,
            "seed_value": seed_value,
        }

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(size * embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        activations = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        self.activation_fn = activations.get(
            str(activation_fn).lower(), torch.relu
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, context_size) containing token IDs.

        Returns:
            Logits tensor of shape (batch_size, vocab_size).
        """
        x = self.embed(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

    def get_num_params(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NextTokenPredictor(nn.Module):
    """Lexical Source Code Next-Token Predictor.

    Maintains backward compatibility with serialized checkpoint weights trained
    on C++ (llama.cpp) and Python (GIANT) codebases.

    Architecture:
        Embedding(V, D) -> Flatten(N * D) -> Linear(N * D, H) -> ReLU -> Linear(H, V)
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, context_size).

        Returns:
            Logits tensor of shape (batch_size, vocab_size).
        """
        x = self.embed(x)
        x = x.view(x.shape[0], -1)
        if hasattr(self, "activation"):
            x = self.activation(self.fc1(x))
        elif hasattr(self, "relu"):
            x = self.relu(self.fc1(x))
        else:
            x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_num_params(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPNextTokenPredictor(nn.Module):
    """Unified and configurable Multi-Layer Perceptron Language Model.

    Combines the flexibility of modern PyTorch modules with configurable activation
    functions, dropout regularization, and layer normalization.
    """

    def __init__(
        self,
        vocab_size: int,
        context_size: int = 5,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.flatten_dim = context_size * embed_dim

        act_layer: nn.Module
        if activation.lower() == "tanh":
            act_layer = nn.Tanh()
        elif activation.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif activation.lower() == "gelu":
            act_layer = nn.GELU()
        else:
            act_layer = nn.ReLU()

        layers = [
            nn.Linear(self.flatten_dim, hidden_dim),
            act_layer,
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, vocab_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits from context token IDs."""
        embeds = self.embedding(x)
        flattened = embeds.view(embeds.size(0), -1)
        return self.network(flattened)

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
