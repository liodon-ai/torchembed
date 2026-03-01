"""
Categorical embeddings for tabular and structured data.

Includes:
- EntityEmbedding: learned embedding for a single categorical feature with
  automatic dimension sizing (the fast.ai heuristic)
- MultiCategoricalEmbedding: handles an entire table of categorical columns
  at once, concatenating all embeddings into a single output tensor
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Sequence, Tuple


def _auto_dim(num_categories: int) -> int:
    """Automatic embedding dimension heuristic.

    Uses the fast.ai rule: min(600, round(1.6 * num_categories ** 0.56)).
    This tends to outperform the older num_categories // 2 rule.
    """
    return min(600, round(1.6 * num_categories ** 0.56))


class EntityEmbedding(nn.Module):
    """Learned embedding for a single categorical feature.

    Wraps ``nn.Embedding`` with automatic dimension sizing and optional
    dropout. The auto-sizing heuristic (fast.ai rule) empirically works well
    for tabular data without manual tuning.

    Reference:
        Howard & Gugger, "Deep Learning for Coders with fastai and PyTorch"
        https://arxiv.org/abs/2002.04688 (entity embeddings section)

    Args:
        num_categories: Vocabulary size (number of unique categories + 1 for
            unknown). Indices must be in [0, num_categories - 1].
        embed_dim: Embedding dimension. If None, uses the auto-sizing
            heuristic: ``min(600, round(1.6 * num_categories ** 0.56))``.
        dropout: Dropout rate applied to embeddings.
        padding_idx: Index that will always produce a zero embedding
            (e.g. for padding / unknown tokens).

    Example::

        emb = EntityEmbedding(num_categories=50)
        x = torch.randint(0, 50, (32,))    # batch of category indices
        out = emb(x)                        # (32, embed_dim)
    """

    def __init__(
        self,
        num_categories: int,
        embed_dim: Optional[int] = None,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.embed_dim = embed_dim or _auto_dim(num_categories)

        self.embedding = nn.Embedding(
            num_categories, self.embed_dim, padding_idx=padding_idx
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.embedding.weight, std=0.01)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    def forward(self, x: Tensor) -> Tensor:
        """Embed categorical indices.

        Args:
            x: Long tensor of category indices, shape (...,).

        Returns:
            Float tensor of shape (..., embed_dim).
        """
        return self.dropout(self.embedding(x))


class MultiCategoricalEmbedding(nn.Module):
    """Joint embedding for multiple categorical columns.

    Intended for tabular data where each row has several categorical features
    (e.g. country, product category, day of week). Creates one
    ``EntityEmbedding`` per column and concatenates the results.

    Args:
        cardinalities: Sequence of vocabulary sizes, one per categorical column.
            E.g. ``[50, 7, 12]`` for columns with 50, 7, and 12 categories.
        embed_dims: Per-column embedding dimensions. If None, uses the
            auto-sizing heuristic for each column.
        dropout: Dropout rate applied to each embedding independently.
        padding_idx: Shared padding index applied to all columns.

    Properties:
        output_dim: Total output dimension (sum of all embed_dims).

    Example::

        # 3 categorical columns: country (50 cats), weekday (7), month (12)
        emb = MultiCategoricalEmbedding(cardinalities=[50, 7, 12])
        x = torch.stack([
            torch.randint(0, 50, (32,)),
            torch.randint(0, 7, (32,)),
            torch.randint(0, 12, (32,)),
        ], dim=1)                             # (32, 3)
        out = emb(x)                          # (32, output_dim)
        print(emb.output_dim)
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        embed_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        if embed_dims is not None and len(embed_dims) != len(cardinalities):
            raise ValueError(
                f"embed_dims length ({len(embed_dims)}) must match "
                f"cardinalities length ({len(cardinalities)})"
            )

        dims = embed_dims or [None] * len(cardinalities)   # type: ignore[list-item]
        self.embeddings = nn.ModuleList([
            EntityEmbedding(n, d, dropout, padding_idx)
            for n, d in zip(cardinalities, dims)
        ])
        self.output_dim: int = sum(e.embed_dim for e in self.embeddings)  # type: ignore[union-attr]

    def forward(self, x: Tensor) -> Tensor:
        """Embed all categorical columns and concatenate.

        Args:
            x: Long tensor of shape (batch, num_columns) containing
               category indices. Column order must match ``cardinalities``.

        Returns:
            Float tensor of shape (batch, output_dim).
        """
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)

    def embedding_dims(self) -> List[Tuple[int, int]]:
        """Return list of (num_categories, embed_dim) for each column."""
        return [(e.num_categories, e.embed_dim) for e in self.embeddings]  # type: ignore[union-attr]
