from __future__ import annotations

import mlx.core as mx
import pytest

from mimi_mlx import MimiCodecConfig, MimiTokenizer


def test_padded_batch_without_lengths_is_rejected_before_model_execution():
    tokenizer = MimiTokenizer(config=MimiCodecConfig.default())

    with pytest.raises(ValueError, match="lengths are required"):
        tokenizer.encode_batch(mx.zeros((2, 24000)), lengths=None, sample_rate=24_000)
