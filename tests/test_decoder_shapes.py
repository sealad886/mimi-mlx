from __future__ import annotations

import mlx.core as mx
import pytest

from mimi_mlx import MimiCodecConfig, MimiTokenizer


def test_decode_is_explicitly_unimplemented_until_model_stage():
    tokenizer = MimiTokenizer(config=MimiCodecConfig.default())

    with pytest.raises(NotImplementedError, match="MLX Mimi model is not implemented"):
        tokenizer.decode(mx.zeros((1, 10, 32), dtype=mx.int32), sample_rate=24_000)
