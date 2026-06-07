"""
Fourier feature embeddings.

Includes:
- RandomFourierFeatures: Random Fourier Features for kernel approximation
- LearnedFourierFeatures: Trainable version of random Fourier features
- GaussianFourierProjection: Diffusion model continuous timestep embedding
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features (RFF) for kernel approximation.

    Maps input features to a randomized low-dimensional feature space such
    that the inner product in the new space approximates a shift-invariant
    kernel (e.g. RBF/Gaussian). Useful for scalable kernel methods and as
    a fixed encoding for continuous inputs like coordinates.

    Reference:
        Rahimi & Recht, "Random Features for Large-Scale Kernel Machines"
        https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html

    Args:
        in_features: Input dimension.
        out_features: Output dimension (number of random features). Should be
            large enough to approximate the kernel well (e.g. 256–2048).
        sigma: Bandwidth of the RBF kernel. Controls the frequency of the
            random features. Smaller = higher frequency.
        trainable: If True, the random projection matrix is a learnable parameter.

    Example::

        rff = RandomFourierFeatures(in_features=2, out_features=256)
        coords = torch.randn(32, 2)   # e.g. 2D spatial coordinates
        features = rff(coords)        # (32, 256)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, got {out_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        # Random projection matrix: (in_features, out_features // 2)
        W = torch.randn(in_features, out_features // 2) / sigma
        if trainable:
            self.W = nn.Parameter(W)
        else:
            self.register_buffer("W", W)

    def forward(self, x: Tensor) -> Tensor:
        """Project input to random Fourier feature space.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features) with cosine and sine features
            concatenated and scaled by sqrt(2 / out_features).
        """
        projection = x @ self.W      # (..., out_features // 2)
        scale = math.sqrt(2.0 / self.out_features)
        return scale * torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)


class LearnedFourierFeatures(nn.Module):
    """Learned Fourier Features.

    A trainable variant of random Fourier features where both the frequency
    matrix and a final linear projection are learned end-to-end. Useful when
    you want the network to discover the right frequency decomposition for
    your specific input domain.

    Args:
        in_features: Input dimension.
        num_frequencies: Number of frequency components (output will be
            2 * num_frequencies before projection).
        out_features: Final output dimension after linear projection.
        sigma_init: Initial scale for frequency initialization.

    Example::

        lff = LearnedFourierFeatures(
            in_features=3, num_frequencies=128, out_features=256
        )
        x = torch.randn(16, 3)
        features = lff(x)    # (16, 256)
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int,
        out_features: int,
        sigma_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.freq = nn.Parameter(
            torch.randn(in_features, num_frequencies) / sigma_init
        )
        self.proj = nn.Linear(2 * num_frequencies, out_features)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Compute learned Fourier features.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features).
        """
        projection = x @ self.freq    # (..., num_frequencies)
        features = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)
        return self.proj(features)


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier Projection for continuous scalar embedding.

    A widely used technique in diffusion models and score-based generative
    models to embed continuous scalars (e.g. timestep t, noise level σ)
    into a high-dimensional space. Uses fixed random frequencies scaled by
    a learnable or fixed bandwidth.

    Reference:
        Song et al., "Score-Based Generative Modeling through Stochastic
        Differential Equations" https://arxiv.org/abs/2011.13456

    Args:
        embed_dim: Output embedding dimension. Must be even.
        scale: Scale factor for the random frequencies. Controls how quickly
            the embedding oscillates. Typical values: 16–30.
        learnable: If True, the random weights are trainable.

    Example::

        gfp = GaussianFourierProjection(embed_dim=256, scale=16)
        t = torch.rand(32)        # continuous timesteps in [0, 1]
        emb = gfp(t)              # (32, 256)

        # Common usage in diffusion models:
        # t_emb = gfp(t)
        # then feed through an MLP to condition the UNet
    """

    def __init__(
        self,
        embed_dim: int,
        scale: float = 16.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        W = torch.randn(embed_dim // 2) * scale
        if learnable:
            self.W = nn.Parameter(W)
        else:
            self.register_buffer("W", W)

    def forward(self, x: Tensor) -> Tensor:
        """Embed a continuous scalar input.

        Args:
            x: Scalar input tensor of shape (batch,) or (batch, 1).
                Values are typically in [0, 1] or [0, T].

        Returns:
            Tensor of shape (batch, embed_dim).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)    # (batch, 1)
        projection = x * self.W.unsqueeze(0) * 2 * math.pi   # (batch, embed_dim//2)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
