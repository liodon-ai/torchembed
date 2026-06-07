# Patch embeddings

Two classes for vision and video transformers:

| Class | Description |
|---|---|
| `PatchEmbedding` | ViT-style image patch projection |
| `TubeletEmbedding` | Video tubelet projection with temporal stride |

---

## PatchEmbedding

Image-to-patch embedding for Vision Transformers (ViT). Splits an image into non-overlapping patches and projects each patch into a token embedding using a single convolution (equivalent to splitting + linear projection, but faster).

**Reference:** Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ([arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929))

### Constructor

```python
PatchEmbedding(
    image_size: int | tuple = 224,
    patch_size: int | tuple = 16,
    in_channels: int = 3,
    embed_dim: int = 768,
    bias: bool = True,
    norm_layer: Optional[nn.Module] = None,
    flatten: bool = True,
)
```

| Param | Description |
|---|---|
| `image_size` | Input image size. Int (square) or `(H, W)` tuple. |
| `patch_size` | Size of each patch. Int (square) or `(pH, pW)` tuple. |
| `in_channels` | Number of input image channels. Default 3 (RGB). |
| `embed_dim` | Embedding dimension for each patch token. |
| `bias` | Include bias in the projection convolution. |
| `norm_layer` | Optional normalization after projection (e.g. `nn.LayerNorm`). |
| `flatten` | If True, flatten spatial grid to sequence. |

### Properties

- `num_patches`: total number of patches per image
- `grid_size`: `(H_patches, W_patches)` tuple

### Forward

```python
forward(x) -> Tensor
```

| Param | Shape | Description |
|---|---|---|
| `x` | `(B, C, H, W)` | Image tensor |
| Returns | `(B, num_patches, embed_dim)` if flatten=True | Patch token sequence |

### Example

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

## TubeletEmbedding

Spatiotemporal tubelet embedding for video transformers. Extends patch embedding to video by extracting non-overlapping 3D tubelets (short video clips over a patch region) using a single Conv3d. Each tubelet becomes one token.

**Reference:** Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" ([arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602))

### Constructor

```python
TubeletEmbedding(
    image_size: int | tuple = 224,
    patch_size: int | tuple = 16,
    tubelet_size: int = 2,
    in_channels: int = 3,
    embed_dim: int = 768,
    bias: bool = True,
    norm_layer: Optional[nn.Module] = None,
    flatten: bool = True,
)
```

| Param | Description |
|---|---|
| `image_size` | Spatial frame size. Int or `(H, W)`. |
| `patch_size` | Spatial patch size. Int or `(pH, pW)`. |
| `tubelet_size` | Number of frames per tubelet (temporal stride). |
| `in_channels` | Input channels. Default 3 (RGB). |
| `embed_dim` | Output embedding dimension. |
| `bias` | Include bias in the projection. |
| `norm_layer` | Optional normalization. |
| `flatten` | If True, return shape `(B, num_tubelets, embed_dim)`. |

### Example

```python
from torchembed.patch import TubeletEmbedding

tubelet_emb = TubeletEmbedding(
    image_size=224,
    patch_size=16,
    tubelet_size=2,
    embed_dim=768,
)
video = torch.randn(2, 3, 16, 224, 224)
tokens = tubelet_emb(video)    # (2, 1568, 768)
# 1568 = (16/2) * (224/16) * (224/16)
```
