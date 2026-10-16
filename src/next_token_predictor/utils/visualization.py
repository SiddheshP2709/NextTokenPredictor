"""Visualization utilities for learned token embedding spaces."""

from typing import Dict, List, Optional, Tuple
import numpy as np


def reduce_embeddings(
    embedding_matrix: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Project high-dimensional token embeddings into 2D or 3D space.

    Args:
        embedding_matrix: Array of shape (num_tokens, embedding_dim).
        method: Dimensionality reduction technique ('tsne' or 'pca').
        n_components: Target dimension (typically 2).
        random_state: Random seed for reproducibility.

    Returns:
        Array of shape (num_tokens, n_components).
    """
    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, embedding_matrix.shape[0] - 1))
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )

    return reducer.fit_transform(embedding_matrix)


def extract_word_group_embeddings(
    embedding_weight: np.ndarray,
    stoi: Dict[str, int],
    word_groups: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Extract embeddings and labels for specific semantic or syntactic token groups."""
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    groups: List[str] = []

    for group_name, word_list in word_groups.items():
        for word in word_list:
            if word in stoi:
                idx = stoi[word]
                embeddings.append(embedding_weight[idx])
                labels.append(word)
                groups.append(group_name)

    if not embeddings:
        return np.empty((0, embedding_weight.shape[1])), [], []

    return np.array(embeddings), labels, groups
