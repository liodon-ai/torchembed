# Fourier feature embeddings

Three classes for encoding continuous inputs using Fourier features:

| Class | Description |
|---|---|
| `RandomFourierFeatures` | Fixed random projection for kernel approximation |
| `LearnedFourierFeatures` | Trainable frequency decomposition |
| `GaussianFourierProjection` | Diffusion model timestep embedding |

---

## RandomFourierFeatures

Random Fourier Features (RFF) maps input features to a randomized low-dimensional feature space such that the inner product approximates a shift-invariant kernel (e.g. RBF/Gaussian). Useful for scalable kernel methods and as a fixed encoding for continuous inputs like coordinates.

**Reference:** Rahimi & Recht, "Random Features for Large-Scale Kernel Machines" ([2007](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html))

### Constructor

```python
RandomFourierFeatures(
    in_features: int,
    out_features: int,
    sigma: float = 1.0,
    trainable: bool = False,
)
```

| Param | Description |
|---|---|
| `in_features` | Input dimension. |
| `out_features` | Output dimension (must be even). Typically 256-2048. |
| `sigma` | Bandwidth of the RBF kernel. Smaller = higher frequency. |
| `trainable` | If True, the random projection is learnable. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(..., in_features)` | Input |
| Returns | `(..., out_features)` | `sqrt(2/D) * [cos(Wx), sin(Wx)]` |

### Example (coordinate encoding for NeRF)

```python
from torchembed.fourier import RandomFourierFeatures

rff = RandomFourierFeatures(in_features=2, out_features=256, sigma=2.0)
coords = torch.rand(1024, 2)   # (x, y) positions
features = rff(coords)          # (1024, 256)
```

---

## LearnedFourierFeatures

A trainable variant of random Fourier features where both the frequency matrix and a final linear projection are learned end-to-end. Useful when you want the network to discover the right frequency decomposition for your specific input domain.

### Constructor

```python
LearnedFourierFeatures(
    in_features: int,
    num_frequencies: int,
    out_features: int,
    sigma_init: float = 1.0,
)
```

| Param | Description |
|---|---|
| `in_features` | Input dimension. |
| `num_frequencies` | Number of frequency components. |
| `out_features` | Final output dimension after linear projection. |
| `sigma_init` | Initial scale for frequency initialization. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(..., in_features)` | Input |
| Returns | `(..., out_features)` | `Linear([cos(Wx), sin(Wx)])` |

---

## GaussianFourierProjection

A widely used technique in diffusion models and score-based generative models to embed continuous scalars (e.g. timestep t, noise level sigma) into a high-dimensional space. Uses fixed random frequencies scaled by a learnable or fixed bandwidth.

**Reference:** Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" ([arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456))

### Constructor

```python
GaussianFourierProjection(
    embed_dim: int,
    scale: float = 16.0,
    learnable: bool = False,
)
```

| Param | Description |
|---|---|
| `embed_dim` | Output dimension (must be even). |
| `scale` | Frequency scale. Controls oscillation speed. Typical values: 16-30. |
| `learnable` | If True, frequencies are trainable. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(batch,)` or `(batch, 1)` | Scalar timesteps |
| Returns | `(batch, embed_dim)` | `[sin(Wx), cos(Wx)]` |

### Example (diffusion model timestep embedding)

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
t = torch.rand(32)
emb = t_emb(t)   # (32, 256)
```
