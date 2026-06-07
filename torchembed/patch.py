"""
Patch embeddings for vision and video transformers.

Includes:
- PatchEmbedding: ViT-style image patch projection (Dosovitskiy et al., 2020)
- TubeletEmbedding: Video patch projection with temporal stride (VideoMAE, ViViT)
"""

import math
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    """Image-to-patch embedding for Vision Transformers (ViT).

    Splits an image into non-overlapping patches and projects each patch
    into a token embedding using a single convolution (equivalent to splitting
    + linear projection, but faster).

    Reference:
        Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for
        Image Recognition at Scale" https://arxiv.org/abs/2010.11929

    Args:
        image_size: Input image size. Can be an int (square) or (H, W) tuple.
        patch_size: Size of each patch. Can be an int (square) or (pH, pW) tuple.
        in_channels: Number of input image channels. Default 3 (RGB).
        embed_dim: Embedding dimension for each patch token.
        bias: Whether to include bias in the projection convolution.
        norm_layer: Optional normalization layer applied after projection.
            E.g. ``nn.LayerNorm``.
        flatten: If True (default), flatten the spatial patch grid into a
            sequence. If False, return shape (B, C, H_patches, W_patches).

    Properties:
        num_patches: Total number of patches per image.
        grid_size: (H_patches, W_patches) tuple.

    Example::

        patch_emb = PatchEmbedding(image_size=224, patch_size=16, embed_dim=768)
        images = torch.randn(4, 3, 224, 224)
        tokens = patch_emb(images)    # (4, 196, 768)
        print(patch_emb.num_patches)  # 196
    """

    def __init__(
        self,
        image_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()

        image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(
                f"Image size {image_size} must be divisible by patch size {patch_size}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.grid_size: tuple[int, int] = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.num_patches: int = self.grid_size[0] * self.grid_size[1]

        # Conv2d is mathematically equivalent to split-then-linear, but faster
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer if norm_layer is not None else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        fan_in = self.in_channels * self.patch_size[0] * self.patch_size[1]
        nn.init.trunc_normal_(self.proj.weight, std=math.sqrt(2.0 / fan_in))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Project image to patch token sequence.

        Args:
            x: Image tensor of shape (B, C, H, W).

        Returns:
            If ``flatten=True`` (default): shape (B, num_patches, embed_dim).
            If ``flatten=False``: shape (B, embed_dim, H_patches, W_patches).
        """
        B, C, H, W = x.shape
        if H != self.image_size[0] or W != self.image_size[1]:
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match expected "
                f"({self.image_size[0]}x{self.image_size[1]}). "
                "Pass a different image_size to PatchEmbedding or resize your input."
            )

        x = self.proj(x)   # (B, embed_dim, H_patches, W_patches)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)   # (B, num_patches, embed_dim)
        return self.norm(x)


class TubeletEmbedding(nn.Module):
    """Spatiotemporal tubelet embedding for video transformers.

    Extends patch embedding to video by extracting non-overlapping 3D tubelets
    (short video clips over a patch region) using a single Conv3d. Each tubelet
    becomes one token.

    Used in: VideoMAE, ViViT, TimeSformer variants.

    Reference:
        Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners
        for Self-Supervised Video Pre-Training" https://arxiv.org/abs/2203.12602

    Args:
        image_size: Spatial frame size. Int or (H, W).
        patch_size: Spatial patch size. Int or (pH, pW).
        tubelet_size: Number of frames per tubelet (temporal stride).
        in_channels: Input channels (default 3 for RGB video).
        embed_dim: Output embedding dimension.
        bias: Whether to include bias in the projection.
        flatten: If True, return shape (B, num_tubelets, embed_dim).

    Properties:
        num_patches_per_frame: Spatial patches per frame.
        num_tubelets_per_video: Total tubelets for a given number of frames.

    Example::

        tubelet_emb = TubeletEmbedding(
            image_size=224, patch_size=16, tubelet_size=2, embed_dim=768
        )
        video = torch.randn(2, 3, 16, 224, 224)   # (B, C, T, H, W)
        tokens = tubelet_emb(video)                # (2, 1568, 768)
        # 1568 = (16/2) * (224/16) * (224/16) = 8 * 14 * 14
    """

    def __init__(
        self,
        image_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()

        image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.num_patches_per_frame: int = (
            (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        )

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, *patch_size),
            stride=(tubelet_size, *patch_size),
            bias=bias,
        )
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        fan_in = (
            self.in_channels
            * self.tubelet_size
            * self.patch_size[0]
            * self.patch_size[1]
        )
        nn.init.trunc_normal_(self.proj.weight, std=math.sqrt(2.0 / fan_in))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Project video to tubelet token sequence.

        Args:
            x: Video tensor of shape (B, C, T, H, W).

        Returns:
            If ``flatten=True``: shape (B, num_tubelets, embed_dim).
            If ``flatten=False``: shape (B, embed_dim, T//tubelet_size, H//pH, W//pW).
        """
        x = self.proj(x)   # (B, embed_dim, T', H', W')
        if self.flatten:
            B, C, T, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)   # (B, T*H*W, embed_dim)
        return self.norm(x)
