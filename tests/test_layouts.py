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
        codes=mx.array(np.arange(2 * 3 * 32).reshape(2, 3, 32)),
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


def test_npz_token_file_is_mlx_loadable(tmp_path):
    path = tmp_path / "tokens.npz"
    tokens = MimiTokens(
        codes=mx.array(np.arange(2 * 3 * 32).reshape(2, 3, 32)),
        lengths=mx.array([3, 2]),
        sample_rate=24_000,
        frame_rate=12.5,
        audio_lengths=mx.array([4800, 3200]),
    )

    save_tokens_npz(path, tokens)
    loaded = mx.load(path)

    assert np.array_equal(np.array(loaded["codes"]), np.array(tokens.codes))
    assert np.array_equal(np.array(loaded["lengths"]), np.array(tokens.lengths))
    assert np.array_equal(np.array(loaded["audio_lengths"]), np.array(tokens.audio_lengths))
    assert int(loaded["sample_rate"]) == 24_000
    assert float(loaded["frame_rate"]) == 12.5
    assert bytes(loaded["layout"].tolist()).decode("utf-8") == "batch_time_codebook"


def test_legacy_numpy_npz_token_file_still_loads(tmp_path):
    path = tmp_path / "legacy_tokens.npz"
    np.savez(
        path,
        codes=np.arange(2 * 3 * 32).reshape(2, 3, 32),
        lengths=np.array([3, 2], dtype=np.int32),
        sample_rate=24_000,
        frame_rate=12.5,
        layout="batch_time_codebook",
        audio_lengths=np.array([4800, 3200], dtype=np.int32),
    )

    loaded = load_tokens_npz(path)

    assert loaded.sample_rate == 24_000
    assert loaded.frame_rate == 12.5
    assert loaded.layout == "batch_time_codebook"
    assert np.array_equal(np.array(loaded.codes), np.arange(2 * 3 * 32).reshape(2, 3, 32))
    assert np.array_equal(np.array(loaded.lengths), np.array([3, 2], dtype=np.int32))
    assert np.array_equal(np.array(loaded.audio_lengths), np.array([4800, 3200], dtype=np.int32))


def test_npz_token_file_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "missing_layout.npz"
    np.savez(
        path,
        codes=np.arange(2 * 3 * 32).reshape(2, 3, 32),
        lengths=np.array([3, 2], dtype=np.int32),
        sample_rate=24_000,
        frame_rate=12.5,
    )

    with pytest.raises(ValueError, match="Token archive missing required field: layout"):
        load_tokens_npz(path)


def test_npz_token_file_rejects_non_3d_codes(tmp_path):
    path = tmp_path / "rank2_tokens.npz"
    np.savez(
        path,
        codes=np.arange(2 * 3).reshape(2, 3),
        lengths=np.array([3, 2], dtype=np.int32),
        sample_rate=24_000,
        frame_rate=12.5,
        layout="batch_time_codebook",
    )

    with pytest.raises(ValueError, match=r"Expected token codes shape \[B,T,K=32\]"):
        load_tokens_npz(path)


def test_npz_token_file_rejects_length_batch_mismatch(tmp_path):
    path = tmp_path / "bad_lengths.npz"
    np.savez(
        path,
        codes=np.arange(2 * 3 * 32).reshape(2, 3, 32),
        lengths=np.array([3], dtype=np.int32),
        sample_rate=24_000,
        frame_rate=12.5,
        layout="batch_time_codebook",
    )

    with pytest.raises(ValueError, match="Expected token lengths shape \\[2\\]"):
        load_tokens_npz(path)


def test_npy_code_round_trip(tmp_path):
    path = tmp_path / "tokens.npy"
    codes = mx.array(np.arange(2 * 3 * 32).reshape(2, 3, 32))

    save_tokens_npy(path, codes)
    loaded = load_tokens_npy(path, sample_rate=24_000, frame_rate=12.5)

    assert loaded.codes.shape == (2, 3, 32)
    assert np.array_equal(np.array(loaded.codes), np.array(codes))
    assert np.array_equal(np.array(loaded.lengths), np.array([3, 3]))


def test_npy_code_load_rejects_wrong_codebook_axis(tmp_path):
    path = tmp_path / "upstream_tokens.npy"
    mx.save(path, mx.zeros((1, 32, 4), dtype=mx.int32))

    with pytest.raises(ValueError, match="Expected saved canonical codes shape \\[B,T,K=32\\]"):
        load_tokens_npy(path, sample_rate=24_000, frame_rate=12.5)


def test_npy_save_uses_mlx_save(monkeypatch, tmp_path):
    path = tmp_path / "tokens.npy"
    codes = mx.array(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    calls = {}

    def fake_save(file, arr):
        calls["file"] = file
        calls["arr"] = arr

    monkeypatch.setattr("mimi_mlx.tokenizer.mx.save", fake_save)

    save_tokens_npy(path, codes)

    assert calls == {"file": path, "arr": codes}


def test_npy_load_uses_mlx_load(monkeypatch, tmp_path):
    path = tmp_path / "tokens.npy"
    codes = mx.array(np.arange(2 * 3 * 32).reshape(2, 3, 32))
    calls = {}

    def fake_load(file):
        calls["file"] = file
        return codes

    monkeypatch.setattr("mimi_mlx.tokenizer.mx.load", fake_load)

    loaded = load_tokens_npy(path, sample_rate=24_000, frame_rate=12.5)

    assert calls == {"file": path}
    assert loaded.codes is codes
    assert np.array_equal(np.array(loaded.lengths), np.array([3, 3]))
