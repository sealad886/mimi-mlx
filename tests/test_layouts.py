from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mimi_mlx import MimiTokens
from mimi_mlx.layouts import (
    from_upstream_layout,
    to_upstream_layout,
    validate_layout,
)
from mimi_mlx.tokenizer import (
    load_tokens_npy,
    load_tokens_npz,
    save_tokens_npy,
    save_tokens_npz,
)


def test_layout_round_trip_between_canonical_and_upstream():
    codes = mx.array(np.arange(2 * 3 * 4).reshape(2, 3, 4))

    upstream = to_upstream_layout(codes)
    restored = from_upstream_layout(upstream)

    assert upstream.shape == (2, 4, 3)
    assert restored.shape == codes.shape
    assert np.array_equal(np.array(restored), np.array(codes))


def test_layout_validation_rejects_unknown_layout():
    with pytest.raises(ValueError, match="Unsupported token layout"):
        validate_layout("time_batch_codebook")


def test_npz_token_round_trip(tmp_path):
    path = tmp_path / "tokens.npz"
    tokens = MimiTokens(
        codes=mx.array(np.arange(2 * 3 * 4).reshape(2, 3, 4)),
        lengths=mx.array([3, 2]),
        sample_rate=24_000,
        frame_rate=12.5,
        audio_lengths=mx.array([4800, 3200]),
    )

    save_tokens_npz(path, tokens)
    loaded = load_tokens_npz(path)

    assert loaded.sample_rate == 24_000
    assert loaded.frame_rate == 12.5
    assert loaded.layout == "batch_time_codebook"
    assert np.array_equal(np.array(loaded.codes), np.array(tokens.codes))
    assert np.array_equal(np.array(loaded.lengths), np.array(tokens.lengths))
    assert np.array_equal(np.array(loaded.audio_lengths), np.array(tokens.audio_lengths))


def test_npy_code_round_trip(tmp_path):
    path = tmp_path / "tokens.npy"
    codes = mx.array(np.arange(2 * 3 * 4).reshape(2, 3, 4))

    save_tokens_npy(path, codes)
    loaded = load_tokens_npy(path, sample_rate=24_000, frame_rate=12.5)

    assert loaded.codes.shape == (2, 3, 4)
    assert np.array_equal(np.array(loaded.codes), np.array(codes))
    assert np.array_equal(np.array(loaded.lengths), np.array([3, 3]))
