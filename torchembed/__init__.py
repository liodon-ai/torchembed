"""**torchembed** is a single, well-tested, pip-installable home for modern
PyTorch embedding strategies — the ones missing from `torch.nn`. `torch.nn`
gives you `nn.Embedding`, a lookup table, and nothing else; the moment you
work with continuous inputs, transformer positional encodings, coordinates,
time, or tabular data, you're on your own. torchembed covers all of it in
one library, with an optional fused Triton kernel for RoPE on GPU.

Every embedding is a plain `nn.Module`: no required dependencies beyond
PyTorch, no `.cuda()` calls baked in, and no framework lock-in — use one
class or all of them.

## Features

- **Positional embeddings** — `RotaryEmbedding` (RoPE, LLaMA/Mistral-style),
  `ALiBiEmbedding` (long-context extrapolation), `SinusoidalEmbedding`,
  `LearnedPositionalEmbedding`.
- **Fourier features** — `RandomFourierFeatures` (coordinate/kernel
  encoding), `LearnedFourierFeatures`, `GaussianFourierProjection`
  (diffusion timestep embedding).
- **Categorical embeddings** — `EntityEmbedding` and
  `MultiCategoricalEmbedding` for tabular data, with auto-sized embedding
  dimensions.
- **Patch embeddings** — `PatchEmbedding` (ViT) and `TubeletEmbedding`
  (video transformers: VideoMAE, ViViT).
- **Temporal embeddings** — `CyclicEmbedding`, `TimestampEmbedding`,
  `FrequencyEmbedding` for hour/day/month and periodic time series.
- **Fused Triton kernels** — optional GPU-accelerated RoPE forward and
  backward, ~4x faster than plain PyTorch and ~2x faster than
  `torch.compile`, with automatic CPU fallback when triton isn't installed.
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
