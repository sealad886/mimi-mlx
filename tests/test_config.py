from __future__ import annotations

import json

from mimi_mlx import MimiCodecConfig


def test_config_loads_huggingface_style_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "sampling_rate": 24000,
                "frame_rate": 12.5,
                "num_quantizers": 32,
                "codebook_size": 2048,
                "audio_channels": 1,
                "upsampling_ratios": [8, 6, 5, 4],
                "model_type": "mimi",
            }
        )
    )

    config = MimiCodecConfig.from_pretrained(config_path)

    assert config.sample_rate == 24_000
    assert config.mimi_sample_rate == 24_000
    assert config.frame_rate == 12.5
    assert config.num_codebooks == 32
    assert config.codebook_size == 2048
    assert config.channels == 1
    assert config.hop_length == 1920


def test_config_defaults_match_kyutai_mimi():
    config = MimiCodecConfig.default()

    assert config.sample_rate == 24_000
    assert config.mimi_sample_rate == 24_000
    assert config.frame_rate == 12.5
    assert config.num_codebooks == 32
    assert config.codebook_size == 2048
    assert config.channels == 1
    assert config.hop_length == 1920
