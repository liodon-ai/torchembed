"""
torchembed — Modern embedding strategies for PyTorch.

Provides embedding modules missing from ``torch.nn``:
positional (RoPE, ALiBi, sinusoidal), Fourier features,
categorical/entity embeddings, patch/tubelet embeddings,
and temporal/cyclic embeddings.

All classes are ``nn.Module`` subclasses with no required
dependencies beyond PyTorch.
"""

from torchembed.categorical import EntityEmbedding, MultiCategoricalEmbedding
from torchembed.fourier import (
    GaussianFourierProjection,
    LearnedFourierFeatures,
    RandomFourierFeatures,
)
from torchembed.patch import PatchEmbedding, TubeletEmbedding
from torchembed.positional import (
    ALiBiEmbedding,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    SinusoidalEmbedding,
)
from torchembed.temporal import CyclicEmbedding, FrequencyEmbedding, TimestampEmbedding

__all__ = [
    "RotaryEmbedding",
    "ALiBiEmbedding",
    "SinusoidalEmbedding",
    "LearnedPositionalEmbedding",
    "RandomFourierFeatures",
    "LearnedFourierFeatures",
    "GaussianFourierProjection",
    "EntityEmbedding",
    "MultiCategoricalEmbedding",
    "PatchEmbedding",
    "TubeletEmbedding",
    "CyclicEmbedding",
    "TimestampEmbedding",
    "FrequencyEmbedding",
]
