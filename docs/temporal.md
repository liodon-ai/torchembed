# Temporal embeddings

Three classes for time series and event-based models:

| Class | Description |
|---|---|
| `CyclicEmbedding` | Sin/cos encoding of periodic features (hour, day, month) |
| `TimestampEmbedding` | Continuous timestamp embedding with Fourier features + MLP |
| `FrequencyEmbedding` | Learnable periodic decomposition for time series |

---

## CyclicEmbedding

Cyclic encoding for periodic scalar features. Encodes a scalar that cycles over a known period (e.g. hour of day, day of week, month of year) as (sin, cos) pairs. This preserves the topology of the cycle — 11pm and 1am are close, not far apart.

This is a fixed, non-learned transformation that produces a 2D output per input feature.

### Constructor

```python
CyclicEmbedding(
    period: float,
    normalize_input: bool = True,
)
```

| Param | Description |
|---|---|
| `period` | The period of the cycle. E.g. 24 for hours, 7 for days, 12 for months. |
| `normalize_input` | If True, input is assumed to be in [0, period). Default True. |

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(...)` | Scalar values in [0, period) |
| Returns | `(..., 2)` | `[sin(2pi * x / period), cos(2pi * x / period)]` |

### Example

```python
from torchembed.temporal import CyclicEmbedding

hour_enc = CyclicEmbedding(period=24)
dow_enc  = CyclicEmbedding(period=7)
month_enc = CyclicEmbedding(period=12)

hours = torch.tensor([0.0, 6.0, 12.0, 18.0])
dow   = torch.tensor([0.0, 1.0, 2.0, 3.0])
month = torch.tensor([1.0, 4.0, 7.0, 10.0])

time_features = torch.cat([
    hour_enc(hours),    # (4, 2)
    dow_enc(dow),       # (4, 2)
    month_enc(month),   # (4, 2)
], dim=-1)              # (4, 6)
```

---

## TimestampEmbedding

Embedding for raw continuous timestamps. Takes a raw scalar timestamp (e.g. normalized time in [0, 1]) and produces an embedding using a Gaussian Fourier projection followed by an MLP.

### Constructor

```python
TimestampEmbedding(
    embed_dim: int,
    num_frequencies: int = 64,
    scale: float = 10.0,
    mlp_layers: int = 2,
)
```

| Param | Description |
|---|---|
| `embed_dim` | Output embedding dimension (must be even). |
| `num_frequencies` | Number of Fourier frequency components. |
| `scale` | Frequency scale for the Fourier projection. |
| `mlp_layers` | Number of MLP layers after Fourier projection. |

### Forward

```python
forward(t) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `t` | `(batch,)` or `(batch, 1)` | Scalar timestamps |
| Returns | `(batch, embed_dim)` | Embedded timestamps |

---

## FrequencyEmbedding

Learnable frequency decomposition for periodic time series. Decomposes input time values into a bank of learnable sinusoidal oscillators. Each oscillator has a learnable frequency, phase, and amplitude.

**Reference:** Inspired by Time2Vec (Kazemi et al., 2019) ([arxiv.org/abs/1907.05321](https://arxiv.org/abs/1907.05321))

### Constructor

```python
FrequencyEmbedding(
    embed_dim: int,
    learnable_freq: bool = True,
)
```

| Param | Description |
|---|---|
| `embed_dim` | Number of sinusoidal components. Output dim is `embed_dim + 1` (one linear trend is always included). |
| `learnable_freq` | If True, frequencies are learnable. If False, uses log-spaced fixed frequencies. |

### Forward

```python
forward(t) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `t` | `(batch, seq_len)` or `(batch,)` | Time values |
| Returns | `(..., embed_dim + 1)` | Trend + sinusoidal components |

### Example

```python
from torchembed.temporal import FrequencyEmbedding

freq_emb = FrequencyEmbedding(embed_dim=32, learnable_freq=True)
t = torch.linspace(0, 100, 512).unsqueeze(0)
out = freq_emb(t)   # (1, 512, 33)
```
