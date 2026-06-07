# torchembed

Modern embedding strategies for PyTorch — the ones missing from `torch.nn`.

## Installation

```bash
pip install torchembed
```

For GPU-accelerated kernels:

```bash
pip install torchembed[triton]
```

Requires Python >= 3.9 and PyTorch >= 2.0.

## Modules

| Module | Classes | Use case |
|---|---|---|
| [positional](positional.md) | `RotaryEmbedding`, `ALiBiEmbedding`, `SinusoidalEmbedding`, `LearnedPositionalEmbedding` | Position encodings for transformers |
| [fourier](fourier.md) | `RandomFourierFeatures`, `LearnedFourierFeatures`, `GaussianFourierProjection` | Kernel approximation, coordinate encoding, diffusion timesteps |
| [categorical](categorical.md) | `EntityEmbedding`, `MultiCategoricalEmbedding` | Tabular categorical features |
| [patch](patch.md) | `PatchEmbedding`, `TubeletEmbedding` | Vision and video transformers |
| [temporal](temporal.md) | `CyclicEmbedding`, `TimestampEmbedding`, `FrequencyEmbedding` | Time series, periodic signals |

## Quick start

```python
from torchembed.positional import RotaryEmbedding, ALiBiEmbedding
from torchembed.fourier import GaussianFourierProjection
from torchembed.categorical import MultiCategoricalEmbedding
from torchembed.patch import PatchEmbedding
from torchembed.temporal import CyclicEmbedding, FrequencyEmbedding

# RoPE — the backbone of modern LLMs
rope = RotaryEmbedding(dim=64)
q_rot, k_rot = rope(q, k)

# ALiBi — long-context extrapolation
alibi = ALiBiEmbedding(num_heads=8)
scores = alibi(scores)

# Diffusion timestep embedding
t_emb = GaussianFourierProjection(embed_dim=256)

# ViT patch embedding
patch_emb = PatchEmbedding(image_size=224, patch_size=16, embed_dim=768)
```

## Triton kernels

torchembed includes optional triton-accelerated kernels for GPU (requires `pip install torchembed[triton]`). Enable with `use_fused=True`:

```python
rope = RotaryEmbedding(dim=64, use_fused=True)
```

The fused kernel combines cos/sin lookup, rotate-half, and element-wise multiplication into a single triton launch, reducing memory traffic. Available for any RoPE dim (32, 64, 128, etc.). Falls back to the vanilla PyTorch implementation on CPU or when triton is unavailable.

## Design principles

- **Everything is an `nn.Module`.** Use any embedding as a layer in a larger model, save/load with `state_dict`, move across devices, wrap with `torch.compile`.
- **No required dependencies beyond PyTorch.** Exactly one required dependency: PyTorch itself. Triton is optional.
- **Device-agnostic.** No `.cuda()` calls inside the library.
- **Bring just what you need.** Every class is independent. No framework lock-in.

## Running tests

```bash
pip install torchembed[dev]
pytest
```

## Building API docs

```bash
pip install torchembed[dev]
make docs   # generates docs/api/
make docs-serve  # live preview at http://localhost:8080
```

The API reference is auto-generated from docstrings using [pdoc](https://pdoc.dev/). Hand-written guides for each module are in this directory.
