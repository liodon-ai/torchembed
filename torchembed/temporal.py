"""
Temporal embeddings for time series and event-based models.

Includes:
- TimestampEmbedding: embeds raw Unix timestamps or datetime features
- FrequencyEmbedding: learnable frequency decomposition for periodic signals
- CyclicEmbedding: encodes cyclic features (hour, day, month) with sin/cos
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class CyclicEmbedding(nn.Module):
    """Cyclic encoding for periodic scalar features.

    Encodes a scalar that cycles over a known period (e.g. hour of day,
    day of week, month of year) as (sin, cos) pairs. This preserves the
    topology of the cycle — 11pm and 1am are close, not far apart.

    This is a fixed, non-learned transformation that produces a 2D output
    per input feature. Often stacked with other embeddings.

    Args:
        period: The period of the cycle. E.g. 24 for hours, 7 for days,
            12 for months, 60 for seconds/minutes.
        normalize_input: If True, input is assumed to be already in [0, period).
            If False, the raw value is used directly. Default True.

    Example::

        hour_emb = CyclicEmbedding(period=24)
        hours = torch.tensor([0.0, 6.0, 12.0, 18.0, 23.0])
        out = hour_emb(hours)   # (5, 2)

        # Combine multiple cyclic features
        hour_enc = CyclicEmbedding(24)(hour_of_day)    # (B, 2)
        dow_enc  = CyclicEmbedding(7)(day_of_week)     # (B, 2)
        month_enc = CyclicEmbedding(12)(month)          # (B, 2)
        combined = torch.cat([hour_enc, dow_enc, month_enc], dim=-1)  # (B, 6)
    """

    def __init__(self, period: float, normalize_input: bool = True) -> None:
        super().__init__()
        self.period = period
        self.normalize_input = normalize_input

    def forward(self, x: Tensor) -> Tensor:
        """Encode cyclic scalar as (sin, cos) pair.

        Args:
            x: Scalar tensor of shape (...,) with values in [0, period).

        Returns:
            Tensor of shape (..., 2) with [sin, cos] encoding.
        """
        angle = (2 * math.pi * x) / self.period
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


class TimestampEmbedding(nn.Module):
    """Embedding for raw continuous timestamps or datetime feature vectors.

    Two modes:

    1. **Scalar mode**: Takes a raw scalar timestamp (e.g. Unix time,
       normalized time in [0, 1]) and produces an embedding using
       a Gaussian Fourier projection followed by an MLP.

    2. **Feature mode**: Takes pre-extracted calendar features (hour, day,
       month, etc.) as a vector and projects them with cyclic encodings + MLP.

    For most practical cases, scalar mode is simpler and works well.

    Args:
        embed_dim: Output embedding dimension.
        num_frequencies: Number of Fourier frequency components.
        scale: Frequency scale for the Fourier projection.
        mlp_layers: Number of MLP layers after the Fourier projection.

    Example::

        ts_emb = TimestampEmbedding(embed_dim=64)
        # Normalized timestamps in [0, 1]
        t = torch.rand(32)
        emb = ts_emb(t)    # (32, 64)
    """

    def __init__(
        self,
        embed_dim: int,
        num_frequencies: int = 64,
        scale: float = 10.0,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        # Random Fourier projection (fixed)
        W = torch.randn(num_frequencies) * scale
        self.register_buffer("W", W)
        self.num_frequencies = num_frequencies

        # MLP: Fourier features → embed_dim
        fourier_dim = 2 * num_frequencies
        layers = []
        in_dim = fourier_dim
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(in_dim, embed_dim), nn.SiLU()]
            in_dim = embed_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: Tensor) -> Tensor:
        """Embed a continuous timestamp.

        Args:
            t: Scalar timestamps of shape (batch,) or (batch, 1).
               Works best when normalized to a consistent range.

        Returns:
            Tensor of shape (batch, embed_dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)      # (batch, 1)
        proj = t * self.W.unsqueeze(0) * 2 * math.pi  # (batch, num_freq)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(fourier)


class FrequencyEmbedding(nn.Module):
    """Learnable frequency decomposition for periodic time series.

    Decomposes input time values into a bank of learnable sinusoidal
    oscillators. Each oscillator has a learnable frequency, phase, and
    amplitude. The output is a rich, differentiable representation of
    the temporal structure.

    Well-suited for: forecasting models, time series classification,
    and any model that needs to discover periodic structure automatically.

    Reference:
        Inspired by Neural Basis Expansion Analysis (N-BEATS) and
        Time2Vec (Kazemi et al., 2019) https://arxiv.org/abs/1907.05321

    Args:
        embed_dim: Number of sinusoidal components. Output dimension is
            ``embed_dim + 1`` (one linear trend term is always included).
        learnable_freq: If True, frequencies are learnable. If False,
            uses log-spaced fixed frequencies (like a Fourier basis).

    Example::

        freq_emb = FrequencyEmbedding(embed_dim=32)
        t = torch.linspace(0, 1, 100).unsqueeze(0)   # (1, 100) time steps
        out = freq_emb(t)                              # (1, 100, 33)
    """

    def __init__(self, embed_dim: int, learnable_freq: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        if learnable_freq:
            self.freq = nn.Parameter(torch.randn(embed_dim))
            self.phase = nn.Parameter(torch.zeros(embed_dim))
        else:
            # Log-spaced frequencies covering multiple time scales
            freq = torch.exp(
                torch.linspace(0, math.log(embed_dim), embed_dim)
            )
            self.register_buffer("freq", freq)
            self.register_buffer("phase", torch.zeros(embed_dim))

        self.amp = nn.Parameter(torch.ones(embed_dim + 1))
        self.bias = nn.Parameter(torch.zeros(embed_dim + 1))

    def forward(self, t: Tensor) -> Tensor:
        """Compute frequency embedding for time inputs.

        Args:
            t: Time tensor of shape (batch, seq_len) or (batch,).

        Returns:
            Tensor of shape (..., embed_dim + 1) where the last dimension
            contains sinusoidal components plus one linear trend component.
        """
        # t: (...) → (..., 1)
        t = t.unsqueeze(-1)

        # Linear trend component
        linear = t   # (..., 1)

        # Sinusoidal components
        angles = t * self.freq + self.phase    # (..., embed_dim)
        periodic = torch.sin(angles)

        # Concatenate and scale
        out = torch.cat([linear, periodic], dim=-1)    # (..., embed_dim + 1)
        return self.amp * out + self.bias
    