# torchembed

**Modern embedding strategies for PyTorch — the ones missing from `torch.nn`.**

[![PyPI version](https://img.shields.io/pypi/v/torchembed.svg)](https://pypi.org/project/torchembed/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`torch.nn` gives you `nn.Embedding` (a lookup table). That's it. The moment you work with continuous inputs, modern transformer architectures, coordinates, time, or tabular data, you're on your own — copy-pasting RoPE implementations across projects.

**torchembed** is a single, well-tested, pip-installable home for all of them.

---

## Table of Contents

- [Installation](#installation)
- [What's included](#whats-included)
- [Quick start](#quick-start)
- [Examples](#examples)
  - [Rotary Embedding (RoPE)](#rotary-embedding-rope--llama--mistral-style)
  - [ALiBi](#alibi--long-context-with-length-extrapolation)
  - [Gaussian Fourier Projection](#gaussian-fourier-projection--diffusion-model-timestep-embedding)
  - [ViT Patch Embedding](#vit-patch-embedding)
  - [Tubelet Embedding](#tubelet-embedding--video-transformers)
  - [Tabular categorical features](#tabular-categorical-features)
  - [Cyclic time features](#cyclic-time-features)
  - [Frequency Embedding](#frequency-embedding--learnable-periodic-decomposition)
  - [Random Fourier Features](#random-fourier-features-for-coordinate-encoding)
- [Design principles](#design-principles)
- [Running tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

```bash
pip install torchembed
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0. No other required dependencies.

---

## What's included


| Module | Class | Use case |
|---|---|---|
| `positional` | `RotaryEmbedding` | Modern LLMs (LLaMA, Mistral, Falcon) |
| `positional` | `ALiBiEmbedding` | Long-context models (BLOOM, MPT) |
| `positional` | `SinusoidalEmbedding` | Classic Transformers |
| `positional` | `LearnedPositionalEmbedding` | BERT, GPT-2 |
| `fourier` | `RandomFourierFeatures` | Kernel approximation, coordinate encoding |
| `fourier` | `LearnedFourierFeatures` | Trainable frequency decomposition |
| `fourier` | `GaussianFourierProjection` | Diffusion models (timestep embedding) |
| `categorical` | `EntityEmbedding` | Tabular categorical features |
| `categorical` | `MultiCategoricalEmbedding` | Multiple categorical columns at once |
| `patch` | `PatchEmbedding` | Vision Transformers (ViT) |
| `patch` | `TubeletEmbedding` | Video Transformers (VideoMAE, ViViT) |
| `temporal` | `CyclicEmbedding` | Hour, day, month (cyclic features) |
| `temporal` | `TimestampEmbedding` | Continuous timestamps |
| `temporal` | `FrequencyEmbedding` | Time series, periodic signals |

---

## Quick start

Import from submodules:

```python
from torchembed.positional import RotaryEmbedding, ALiBiEmbedding, SinusoidalEmbedding, LearnedPositionalEmbedding
from torchembed.fourier import RandomFourierFeatures, LearnedFourierFeatures, GaussianFourierProjection
from torchembed.categorical import EntityEmbedding, MultiCategoricalEmbedding
from torchembed.patch import PatchEmbedding, TubeletEmbedding
from torchembed.temporal import CyclicEmbedding, TimestampEmbedding, FrequencyEmbedding
```

Or import individual classes directly:

```python
from torchembed.positional import RotaryEmbedding
from torchembed.fourier import GaussianFourierProjection
from torchembed.patch import PatchEmbedding
```

---

## Examples

### Rotary Embedding (RoPE) — LLaMA / Mistral style

```python
import torch
from torchembed.positional import RotaryEmbedding

rope = RotaryEmbedding(dim=64)  # head_dim

# Inside your attention layer:
q = torch.randn(batch, heads, seq_len, 64)
k = torch.randn(batch, heads, seq_len, 64)
q, k = rope(q, k)  # apply rotation in-place
```

RoPE has no trainable parameters and preserves vector norms (it's a pure rotation).
The default base of 10,000 matches the original paper; use `base=500_000` for LLaMA 3.

---

### ALiBi — long context with length extrapolation

```python
from torchembed.positional import ALiBiEmbedding

alibi = ALiBiEmbedding(num_heads=8)

# After computing raw attention scores:
attn_scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
attn_scores = alibi(attn_scores)   # adds learned distance penalty
attn_weights = attn_scores.softmax(-1)
```

---

### Gaussian Fourier Projection — diffusion model timestep embedding

```python
from torchembed.fourier import GaussianFourierProjection
import torch.nn as nn

class DiffusionTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fourier = GaussianFourierProjection(embed_dim=embed_dim, scale=16)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t):
        return self.mlp(self.fourier(t))

t_emb = DiffusionTimeEmbedding(embed_dim=256)
t = torch.rand(32)   # normalized timesteps
emb = t_emb(t)       # (32, 256) — condition your UNet on this
```

---

### ViT Patch Embedding

```python
from torchembed.patch import PatchEmbedding

patch_emb = PatchEmbedding(
    image_size=224,
    patch_size=16,
    embed_dim=768,
)

images = torch.randn(4, 3, 224, 224)
tokens = patch_emb(images)    # (4, 196, 768)
print(patch_emb.num_patches)  # 196
```

---

### Tubelet Embedding — Video Transformers

```python
from torchembed.patch import TubeletEmbedding

tubelet_emb = TubeletEmbedding(
    image_size=224,
    patch_size=16,
    tubelet_size=2,
    embed_dim=768,
)

video = torch.randn(2, 3, 16, 224, 224)   # (B, C, T, H, W)
tokens = tubelet_emb(video)                # (2, 1568, 768)
# 1568 = (16/2) * (224/16) * (224/16) = 8 * 14 * 14
```

---

### Tabular categorical features

```python
from torchembed.categorical import MultiCategoricalEmbedding

# A tabular dataset with 3 categorical columns:
# country (50 unique values), day of week (7), product category (120)
emb = MultiCategoricalEmbedding(cardinalities=[50, 7, 120])
print(emb.output_dim)   # sum of auto-sized embed dims

x = torch.stack([country_ids, dow_ids, category_ids], dim=1)   # (batch, 3)
features = emb(x)   # (batch, output_dim)
```

---

### Cyclic time features

```python
from torchembed.temporal import CyclicEmbedding
import torch

hour_enc  = CyclicEmbedding(period=24)
dow_enc   = CyclicEmbedding(period=7)
month_enc = CyclicEmbedding(period=12)

hour   = torch.tensor([0.0, 6.0, 12.0, 18.0])
dow    = torch.tensor([0.0, 1.0, 2.0, 3.0])
month  = torch.tensor([1.0, 4.0, 7.0, 10.0])

time_features = torch.cat([
    hour_enc(hour),    # (4, 2)
    dow_enc(dow),      # (4, 2)
    month_enc(month),  # (4, 2)
], dim=-1)             # (4, 6)
```

---

### Random Fourier Features for coordinate encoding

```python
from torchembed.fourier import RandomFourierFeatures

# Encode 2D spatial coordinates for a neural field / NeRF-style model
rff = RandomFourierFeatures(in_features=2, out_features=256, sigma=1.0)

coords = torch.rand(1024, 2)   # (x, y) pairs in [0, 1]
features = rff(coords)          # (1024, 256)
```

---

### Frequency Embedding — learnable periodic decomposition

```python
from torchembed.temporal import FrequencyEmbedding

# Discover periodic structure in time series automatically
freq_emb = FrequencyEmbedding(embed_dim=32)

t = torch.linspace(0, 100, 512).unsqueeze(0)   # (1, 512) time steps
out = freq_emb(t)                               # (1, 512, 33)
# 33 = 1 linear trend + 32 sinusoidal components
```

---

## Design principles

**Everything is an `nn.Module`.** You can use any embedding as a layer in a larger model, save/load it with `state_dict`, move it across devices, and wrap it with `torch.compile`.

**No required dependencies beyond PyTorch.** `torchembed` has exactly one required dependency: PyTorch itself. We don't pull in transformers, numpy, or anything else.

**Device-agnostic.** No `.cuda()` calls inside the library. Move your model to whatever device you want — the embeddings follow.

**Bring just what you need.** Every embedding class is independent. Use one, use all, use none — no framework lock-in.

---

## Running tests

```bash
pip install torchembed[dev]
pytest
```

---

## Contributing

Contributions welcome! If there's an embedding strategy you find yourself copy-pasting into projects, open a PR. Please include:

- The module with a clear docstring and paper reference
- Tests covering shape, gradients, and key mathematical properties
- An example in the README

---

## License

MIT
