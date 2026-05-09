from __future__ import annotations

import mlx.core as mx
import numpy as np

from mimi_mlx.model import MimiModel, MimiModelConfig
from mimi_mlx.modules import ConvTranspose1d, MimiAttention
from mimi_mlx.quantizer import MimiEuclideanCodebook


def test_depthwise_conv_transpose_uses_native_grouped_layout():
    layer = ConvTranspose1d(4, 4, 3, stride=2, groups=4, bias_enabled=False)

    assert layer.weight.shape == (4, 3, 1)


def test_depthwise_conv_transpose_weight_loader_maps_native_group_layout():
    config = MimiModelConfig(
        sampling_rate=8,
        hidden_size=4,
        num_filters=2,
        upsampling_ratios=(2,),
        kernel_size=3,
        last_kernel_size=3,
        codebook_size=8,
        codebook_dim=4,
        num_codebooks=1,
        num_semantic_quantizers=1,
        upsample_groups=4,
        num_hidden_layers=0,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        frame_rate=2.0,
    )
    model = MimiModel(config)
    checkpoint_weight = mx.array(np.arange(16, dtype=np.float32).reshape(4, 1, 4))

    assert model._assign_weight("upsample.conv.weight", checkpoint_weight)

    assert model.upsample.conv.weight.shape == (4, 4, 1)
    assert np.array_equal(
        np.array(model.upsample.conv.weight),
        checkpoint_weight.transpose(0, 2, 1),
    )


def test_attention_uses_mlx_fast_attention_without_pre_tiling_gqa(monkeypatch):
    config = MimiModelConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        sliding_window=0,
    )
    attention = MimiAttention(config)
    calls: dict[str, object] = {}

    def fake_scaled_dot_product_attention(q, k, v, *, scale, mask=None):
        calls["q_shape"] = q.shape
        calls["k_shape"] = k.shape
        calls["v_shape"] = v.shape
        calls["scale"] = scale
        calls["mask"] = mask
        return mx.zeros(q.shape, dtype=q.dtype)

    monkeypatch.setattr(
        "mimi_mlx.modules.mx.fast.scaled_dot_product_attention",
        fake_scaled_dot_product_attention,
    )

    out = attention(mx.ones((1, 3, 8), dtype=mx.float32))

    assert out.shape == (1, 3, 8)
    assert calls["q_shape"] == (1, 4, 3, 2)
    assert calls["k_shape"] == (1, 2, 3, 2)
    assert calls["v_shape"] == (1, 2, 3, 2)
    assert calls["mask"] == "causal"


def test_mimi_codebook_caches_derived_embedding_until_state_changes():
    codebook = MimiEuclideanCodebook(3, 2)
    codebook.embed_sum = mx.array(
        [[1.0, 3.0], [4.0, 8.0], [10.0, 20.0]],
        dtype=mx.float32,
    )
    codebook.cluster_usage = mx.array([1.0, 2.0, 5.0], dtype=mx.float32)

    first = codebook.embed
    second = codebook.embed

    assert second is first
    assert np.allclose(np.array(first), np.array([[1.0, 3.0], [2.0, 4.0], [2.0, 4.0]]))

    codebook.cluster_usage = mx.array([1.0, 4.0, 5.0], dtype=mx.float32)

    refreshed = codebook.embed
    assert refreshed is not first
    assert np.allclose(np.array(refreshed), np.array([[1.0, 3.0], [1.0, 2.0], [2.0, 4.0]]))
