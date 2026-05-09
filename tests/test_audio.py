from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mimi_mlx.audio import normalize_audio_shape, resample_linear


def test_normalize_audio_accepts_single_waveform():
    audio, lengths = normalize_audio_shape(mx.array(np.arange(8, dtype=np.float32)))

    assert audio.shape == (1, 1, 8)
    assert np.array_equal(np.array(lengths), np.array([8]))


def test_normalize_audio_accepts_batched_waveforms():
    audio, lengths = normalize_audio_shape(mx.array(np.zeros((2, 5), dtype=np.float32)))

    assert audio.shape == (2, 1, 5)
    assert np.array_equal(np.array(lengths), np.array([5, 5]))


def test_normalize_audio_accepts_internal_layout():
    audio, lengths = normalize_audio_shape(mx.array(np.zeros((2, 1, 7), dtype=np.float32)))

    assert audio.shape == (2, 1, 7)
    assert np.array_equal(np.array(lengths), np.array([7, 7]))


def test_normalize_audio_rejects_unsupported_channels():
    with pytest.raises(ValueError, match="Expected mono audio"):
        normalize_audio_shape(mx.array(np.zeros((2, 2, 7), dtype=np.float32)))


def test_resample_linear_downsamples_deterministically():
    audio = mx.array(np.linspace(-1.0, 1.0, 48, dtype=np.float32)).reshape(1, 1, 48)

    first = resample_linear(audio, src_rate=48_000, dst_rate=24_000)
    second = resample_linear(audio, src_rate=48_000, dst_rate=24_000)

    assert first.shape == (1, 1, 24)
    assert np.array_equal(np.array(first), np.array(second))
