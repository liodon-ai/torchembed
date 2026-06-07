"""Tests for Fourier, categorical, patch, and temporal embedding modules."""

import pytest
import torch
import torch.nn as nn

from torchembed.categorical import EntityEmbedding, MultiCategoricalEmbedding
from torchembed.fourier import (
    GaussianFourierProjection,
    LearnedFourierFeatures,
    RandomFourierFeatures,
)
from torchembed.patch import PatchEmbedding, TubeletEmbedding
from torchembed.temporal import CyclicEmbedding, FrequencyEmbedding, TimestampEmbedding

# ─── Fourier ─────────────────────────────────────────────────────────────────

class TestRandomFourierFeatures:
    def test_output_shape(self):
        rff = RandomFourierFeatures(in_features=3, out_features=128)
        x = torch.randn(32, 3)
        out = rff(x)
        assert out.shape == (32, 128)

    def test_odd_out_features_raises(self):
        with pytest.raises(ValueError, match="even"):
            RandomFourierFeatures(in_features=2, out_features=127)

    def test_fixed_weights_no_grad(self):
        rff = RandomFourierFeatures(in_features=4, out_features=64, trainable=False)
        assert len(list(rff.parameters())) == 0

    def test_trainable_weights_have_grad(self):
        rff = RandomFourierFeatures(in_features=4, out_features=64, trainable=True)
        assert len(list(rff.parameters())) == 1

    def test_output_bounded(self):
        """RFF output should be in [-scale, scale] where scale = sqrt(2/D)."""
        import math
        rff = RandomFourierFeatures(in_features=2, out_features=256)
        x = torch.randn(1000, 2)
        out = rff(x)
        scale = math.sqrt(2.0 / 256)
        assert (out.abs() <= scale + 1e-6).all()

    def test_gradient_flows(self):
        rff = RandomFourierFeatures(in_features=3, out_features=64, trainable=True)
        x = torch.randn(8, 3, requires_grad=True)
        out = rff(x)
        out.sum().backward()
        assert x.grad is not None

    def test_batched_input(self):
        rff = RandomFourierFeatures(in_features=4, out_features=64)
        x = torch.randn(2, 8, 4)
        out = rff(x)
        assert out.shape == (2, 8, 64)


class TestLearnedFourierFeatures:
    def test_output_shape(self):
        lff = LearnedFourierFeatures(in_features=3, num_frequencies=64, out_features=128)  # noqa: E501
        x = torch.randn(16, 3)
        out = lff(x)
        assert out.shape == (16, 128)

    def test_has_learnable_params(self):
        lff = LearnedFourierFeatures(in_features=3, num_frequencies=64, out_features=128)  # noqa: E501
        total = sum(p.numel() for p in lff.parameters())
        assert total > 0

    def test_gradient_flows(self):
        lff = LearnedFourierFeatures(in_features=2, num_frequencies=32, out_features=64)
        x = torch.randn(8, 2, requires_grad=True)
        out = lff(x)
        out.sum().backward()
        assert x.grad is not None


class TestGaussianFourierProjection:
    def test_output_shape(self):
        gfp = GaussianFourierProjection(embed_dim=64)
        t = torch.rand(32)
        out = gfp(t)
        assert out.shape == (32, 64)

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            GaussianFourierProjection(embed_dim=63)

    def test_accepts_2d_input(self):
        gfp = GaussianFourierProjection(embed_dim=32)
        t = torch.rand(16, 1)
        out = gfp(t)
        assert out.shape == (16, 32)

    def test_fixed_weights_no_params(self):
        gfp = GaussianFourierProjection(embed_dim=64, learnable=False)
        assert len(list(gfp.parameters())) == 0

    def test_learnable_weights_have_params(self):
        gfp = GaussianFourierProjection(embed_dim=64, learnable=True)
        assert len(list(gfp.parameters())) == 1

    def test_gradient_flows(self):
        gfp = GaussianFourierProjection(embed_dim=32)
        t = torch.rand(8, requires_grad=True)
        out = gfp(t)
        out.sum().backward()
        assert t.grad is not None


# ─── Categorical ──────────────────────────────────────────────────────────────

class TestEntityEmbedding:
    def test_output_shape(self):
        emb = EntityEmbedding(num_categories=50)
        x = torch.randint(0, 50, (32,))
        out = emb(x)
        assert out.shape == (32, emb.embed_dim)

    def test_auto_dim_sensible(self):
        emb = EntityEmbedding(num_categories=1000)
        assert 1 <= emb.embed_dim <= 600

    def test_custom_dim(self):
        emb = EntityEmbedding(num_categories=50, embed_dim=32)
        assert emb.embed_dim == 32

    def test_has_parameters(self):
        emb = EntityEmbedding(num_categories=20, embed_dim=8)
        assert sum(p.numel() for p in emb.parameters()) == 20 * 8

    def test_padding_idx_zero(self):
        emb = EntityEmbedding(num_categories=10, embed_dim=16, padding_idx=0)
        x = torch.zeros(4, dtype=torch.long)
        out = emb(x)
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_dropout_variant(self):
        emb = EntityEmbedding(num_categories=50, dropout=0.1)
        emb.train()
        x = torch.randint(0, 50, (16,))
        out = emb(x)
        assert out.shape[0] == 16

    def test_gradient_flows(self):
        emb = EntityEmbedding(num_categories=20, embed_dim=8)
        x = torch.randint(0, 20, (8,))
        out = emb(x)
        out.sum().backward()


class TestMultiCategoricalEmbedding:
    def test_output_shape(self):
        emb = MultiCategoricalEmbedding(cardinalities=[50, 7, 12])
        x = torch.stack([
            torch.randint(0, 50, (16,)),
            torch.randint(0, 7, (16,)),
            torch.randint(0, 12, (16,)),
        ], dim=1)
        out = emb(x)
        assert out.shape == (16, emb.output_dim)

    def test_output_dim_is_sum_of_embed_dims(self):
        emb = MultiCategoricalEmbedding(cardinalities=[10, 20, 30], embed_dims=[4, 8, 12])  # noqa: E501
        assert emb.output_dim == 4 + 8 + 12

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError):
            MultiCategoricalEmbedding(cardinalities=[10, 20], embed_dims=[4, 8, 12])

    def test_embedding_dims_property(self):
        emb = MultiCategoricalEmbedding(cardinalities=[5, 10], embed_dims=[2, 4])
        dims = emb.embedding_dims()
        assert dims == [(5, 2), (10, 4)]

    def test_gradient_flows(self):
        emb = MultiCategoricalEmbedding(cardinalities=[10, 5])
        x = torch.stack([
            torch.randint(0, 10, (8,)),
            torch.randint(0, 5, (8,)),
        ], dim=1)
        out = emb(x)
        out.sum().backward()


# ─── Patch ───────────────────────────────────────────────────────────────────

class TestPatchEmbedding:
    def test_output_shape(self):
        emb = PatchEmbedding(image_size=224, patch_size=16, embed_dim=768)
        x = torch.randn(4, 3, 224, 224)
        out = emb(x)
        assert out.shape == (4, 196, 768)

    def test_num_patches(self):
        emb = PatchEmbedding(image_size=224, patch_size=16)
        assert emb.num_patches == 196

    def test_non_square_image(self):
        emb = PatchEmbedding(image_size=(224, 112), patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 224, 112)
        out = emb(x)
        assert out.shape == (2, emb.num_patches, 256)

    def test_indivisible_patch_size_raises(self):
        with pytest.raises(ValueError):
            PatchEmbedding(image_size=224, patch_size=15)

    def test_wrong_image_size_raises(self):
        emb = PatchEmbedding(image_size=224, patch_size=16)
        x = torch.randn(1, 3, 256, 256)
        with pytest.raises(ValueError):
            emb(x)

    def test_no_flatten(self):
        emb = PatchEmbedding(image_size=32, patch_size=8, embed_dim=64, flatten=False)
        x = torch.randn(2, 3, 32, 32)
        out = emb(x)
        assert out.shape == (2, 64, 4, 4)

    def test_with_norm_layer(self):
        emb = PatchEmbedding(
            image_size=32, patch_size=8, embed_dim=64,
            norm_layer=nn.LayerNorm(64)
        )
        x = torch.randn(2, 3, 32, 32)
        out = emb(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flows(self):
        emb = PatchEmbedding(image_size=32, patch_size=8, embed_dim=64)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = emb(x)
        out.sum().backward()
        assert x.grad is not None


class TestTubeletEmbedding:
    def test_output_shape(self):
        emb = TubeletEmbedding(image_size=32, patch_size=8, tubelet_size=2, embed_dim=64)  # noqa: E501
        video = torch.randn(2, 3, 8, 32, 32)   # (B, C, T, H, W)
        out = emb(video)
        # T/2 * H/8 * W/8 = 4 * 4 * 4 = 64 tubelets
        assert out.shape == (2, 64, 64)

    def test_no_flatten(self):
        emb = TubeletEmbedding(image_size=32, patch_size=8, tubelet_size=2,
                                embed_dim=64, flatten=False)
        video = torch.randn(2, 3, 8, 32, 32)
        out = emb(video)
        assert out.shape == (2, 64, 4, 4, 4)

    def test_gradient_flows(self):
        emb = TubeletEmbedding(image_size=16, patch_size=8, tubelet_size=2, embed_dim=32)  # noqa: E501
        video = torch.randn(1, 3, 4, 16, 16, requires_grad=True)
        out = emb(video)
        out.sum().backward()
        assert video.grad is not None


# ─── Temporal ────────────────────────────────────────────────────────────────

class TestCyclicEmbedding:
    def test_output_shape(self):
        emb = CyclicEmbedding(period=24)
        x = torch.tensor([0.0, 6.0, 12.0, 18.0])
        out = emb(x)
        assert out.shape == (4, 2)

    def test_periodicity(self):
        """Values 0 and 24 (one full period) should produce same encoding."""
        emb = CyclicEmbedding(period=24)
        x0 = torch.tensor([0.0])
        x24 = torch.tensor([24.0])
        torch.testing.assert_close(emb(x0), emb(x24), atol=1e-6, rtol=1e-6)

    def test_midpoint_symmetry(self):
        """6 and 18 hours should have equal and opposite sin components."""
        emb = CyclicEmbedding(period=24)
        out6 = emb(torch.tensor([6.0]))
        out18 = emb(torch.tensor([18.0]))
        torch.testing.assert_close(out6[..., 0], -out18[..., 0], atol=1e-6, rtol=1e-6)

    def test_output_range(self):
        """Sin and cos outputs should be in [-1, 1]."""
        emb = CyclicEmbedding(period=12)
        x = torch.linspace(0, 12, 1000)
        out = emb(x)
        assert out.abs().max() <= 1.0 + 1e-6

    def test_batched_input(self):
        emb = CyclicEmbedding(period=7)
        x = torch.randint(0, 7, (8, 4)).float()
        out = emb(x)
        assert out.shape == (8, 4, 2)


class TestTimestampEmbedding:
    def test_output_shape(self):
        emb = TimestampEmbedding(embed_dim=64)
        t = torch.rand(32)
        out = emb(t)
        assert out.shape == (32, 64)

    def test_accepts_2d_input(self):
        emb = TimestampEmbedding(embed_dim=32)
        t = torch.rand(16, 1)
        out = emb(t)
        assert out.shape == (16, 32)

    def test_odd_embed_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            TimestampEmbedding(embed_dim=63)

    def test_gradient_flows(self):
        emb = TimestampEmbedding(embed_dim=32)
        t = torch.rand(8, requires_grad=True)
        out = emb(t)
        out.sum().backward()
        assert t.grad is not None

    def test_has_parameters(self):
        emb = TimestampEmbedding(embed_dim=32)
        assert sum(p.numel() for p in emb.parameters()) > 0


class TestFrequencyEmbedding:
    def test_output_shape(self):
        emb = FrequencyEmbedding(embed_dim=32)
        t = torch.linspace(0, 1, 100).unsqueeze(0)   # (1, 100)
        out = emb(t)
        assert out.shape == (1, 100, 33)   # embed_dim + 1

    def test_output_dim_is_embed_plus_one(self):
        for d in [8, 16, 32, 64]:
            emb = FrequencyEmbedding(embed_dim=d)
            t = torch.rand(4, 10)
            out = emb(t)
            assert out.shape[-1] == d + 1

    def test_learnable_has_params(self):
        emb = FrequencyEmbedding(embed_dim=16, learnable_freq=True)
        assert sum(p.numel() for p in emb.parameters()) > 0

    def test_fixed_freq_has_fewer_params(self):
        emb_learnable = FrequencyEmbedding(embed_dim=16, learnable_freq=True)
        emb_fixed = FrequencyEmbedding(embed_dim=16, learnable_freq=False)
        learned_params = sum(p.numel() for p in emb_learnable.parameters())
        fixed_params = sum(p.numel() for p in emb_fixed.parameters())
        assert fixed_params < learned_params

    def test_gradient_flows(self):
        emb = FrequencyEmbedding(embed_dim=8)
        t = torch.rand(4, 10, requires_grad=True)
        out = emb(t)
        out.sum().backward()
        assert t.grad is not None
