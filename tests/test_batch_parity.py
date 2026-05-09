from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from mimi_mlx import MimiCodecConfig, MimiTokenizer

ROOT = Path(__file__).resolve().parents[1]
LOCAL_WEIGHTS = ROOT / "fixtures" / "reference" / "hf"


@pytest.fixture(scope="session")
def tokenizer() -> MimiTokenizer:
    if not (LOCAL_WEIGHTS / "model.safetensors").exists():
        pytest.skip("official Mimi weights are not present under fixtures/reference/hf")
    return MimiTokenizer.from_pretrained(LOCAL_WEIGHTS)


def test_encode_batch_without_model_still_rejects_missing_model():
    tokenizer = MimiTokenizer(config=MimiCodecConfig.default())

    with pytest.raises(NotImplementedError, match="model is not loaded"):
        tokenizer.encode_batch(mx.zeros((2, 24000)), lengths=None, sample_rate=24_000)


def test_equal_explicit_lengths_are_encoded_as_one_batch():
    class CountingModel:
        def __init__(self) -> None:
            self.calls = 0
            self.shapes = []

        def encode(self, audio, *, num_codebooks=None):
            self.calls += 1
            self.shapes.append(audio.shape)
            return mx.zeros((audio.shape[0], 2, 3), dtype=mx.int32)

    model = CountingModel()
    tokenizer = MimiTokenizer(config=MimiCodecConfig.default(), model=model)

    tokens = tokenizer.encode_batch(
        mx.zeros((2, 10), dtype=mx.float32),
        lengths=mx.array([6, 6], dtype=mx.int32),
        sample_rate=24_000,
    )

    assert model.calls == 1
    assert model.shapes == [(2, 1, 6)]
    assert tokens.codes.shape == (2, 3, 2)
    assert np.array_equal(np.array(tokens.audio_lengths), np.array([6, 6], dtype=np.int32))


def test_same_length_batch_tokens_match_individual_encode(tokenizer: MimiTokenizer):
    one, sample_rate = sf.read(ROOT / "fixtures/audio/sine_440_025s.wav", dtype="float32")
    two, _ = sf.read(ROOT / "fixtures/audio/noise_low_025s.wav", dtype="float32")
    batch = mx.stack([mx.array(one), mx.array(two)], axis=0)

    batched = tokenizer.encode_batch(batch, sample_rate=sample_rate)
    first = tokenizer.encode(mx.array(one), sample_rate=sample_rate)
    second = tokenizer.encode(mx.array(two), sample_rate=sample_rate)
    first_len = int(np.array(first.lengths)[0])
    second_len = int(np.array(second.lengths)[0])

    assert np.array_equal(np.array(batched.codes[0, :first_len]), np.array(first.codes[0]))
    assert np.array_equal(np.array(batched.codes[1, :second_len]), np.array(second.codes[0]))


def test_variable_length_batch_with_lengths_matches_individual_prefixes(tokenizer: MimiTokenizer):
    long_audio, sample_rate = sf.read(ROOT / "fixtures/audio/sine_440_100s.wav", dtype="float32")
    short_audio = long_audio[:6017]
    padded = np.stack([long_audio, np.pad(short_audio, (0, len(long_audio) - len(short_audio)))])
    lengths = mx.array([len(long_audio), len(short_audio)], dtype=mx.int32)

    batched = tokenizer.encode_batch(mx.array(padded), lengths=lengths, sample_rate=sample_rate)
    first = tokenizer.encode(mx.array(long_audio), sample_rate=sample_rate)
    second = tokenizer.encode(mx.array(short_audio), sample_rate=sample_rate)

    first_len = int(np.array(first.lengths)[0])
    second_len = int(np.array(second.lengths)[0])
    assert np.array_equal(np.array(batched.codes[0, :first_len]), np.array(first.codes[0]))
    assert np.array_equal(np.array(batched.codes[1, :second_len]), np.array(second.codes[0]))
    assert np.array_equal(
        np.array(batched.lengths), np.array([first_len, second_len], dtype=np.int32)
    )
