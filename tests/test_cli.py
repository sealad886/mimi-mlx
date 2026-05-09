from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LOCAL_WEIGHTS = ROOT / "fixtures" / "reference" / "hf"


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "mimi_mlx.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "encode" in result.stdout
    assert "decode" in result.stdout
    assert "parity" in result.stdout
    assert "benchmark" in result.stdout


@pytest.mark.skipif(
    not (LOCAL_WEIGHTS / "model.safetensors").exists(),
    reason="official Mimi weights are not present under fixtures/reference/hf",
)
def test_cli_encode_decode_and_parity(tmp_path: Path):
    tokens = tmp_path / "tokens.npy"
    recon = tmp_path / "recon.wav"
    audio = ROOT / "fixtures" / "audio" / "sine_440_025s.wav"

    encode = subprocess.run(
        [
            sys.executable,
            "-m",
            "mimi_mlx.cli",
            "encode",
            str(audio),
            "--weights",
            str(LOCAL_WEIGHTS),
            "--output",
            str(tokens),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert encode.returncode == 0, encode.stderr
    assert np.load(tokens).shape == (1, 4, 32)

    decode = subprocess.run(
        [
            sys.executable,
            "-m",
            "mimi_mlx.cli",
            "decode",
            str(tokens),
            "--weights",
            str(LOCAL_WEIGHTS),
            "--output",
            str(recon),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert decode.returncode == 0, decode.stderr
    decoded_audio, sample_rate = sf.read(recon, dtype="float32")
    assert sample_rate == 24_000
    assert decoded_audio.shape[0] == 7680

    parity = subprocess.run(
        [
            sys.executable,
            "-m",
            "mimi_mlx.cli",
            "parity",
            str(audio),
            "--reference",
            "transformers",
            "--weights",
            str(LOCAL_WEIGHTS),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert parity.returncode == 0, parity.stderr
    assert '"ok": true' in parity.stdout


@pytest.mark.skipif(
    not (LOCAL_WEIGHTS / "model.safetensors").exists(),
    reason="official Mimi weights are not present under fixtures/reference/hf",
)
def test_cli_rustymimi_reference_requires_reference_weights():
    audio = ROOT / "fixtures" / "audio" / "sine_440_025s.wav"

    parity = subprocess.run(
        [
            sys.executable,
            "-m",
            "mimi_mlx.cli",
            "parity",
            str(audio),
            "--reference",
            "rustymimi",
            "--weights",
            str(LOCAL_WEIGHTS),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert parity.returncode != 0
    assert "rustymimi parity requires --reference-weights" in parity.stderr


def test_rustymimi_reference_codes_use_channel_first_audio_and_full_codebooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mimi_mlx import cli

    captured: dict[str, object] = {}

    class FakeTokenizer:
        def __init__(self, path: str, *, num_codebooks: int, dtype: str = "f32"):
            captured["path"] = path
            captured["num_codebooks"] = num_codebooks
            captured["dtype"] = dtype

        def encode(self, pcm_data: np.ndarray) -> np.ndarray:
            captured["pcm_shape"] = pcm_data.shape
            captured["pcm_dtype"] = pcm_data.dtype
            return np.zeros((1, captured["num_codebooks"], 4), dtype=np.uint32)

        def reset(self) -> None:
            captured["reset"] = True

    monkeypatch.setitem(sys.modules, "rustymimi", types.SimpleNamespace(Tokenizer=FakeTokenizer))
    reference_weights = tmp_path / "tokenizer.safetensors"
    reference_weights.write_bytes(b"fake")

    codes = cli._rustymimi_reference_codes(
        reference_weights,
        np.zeros(6000, dtype=np.float32),
        num_codebooks=32,
    )

    assert codes.shape == (1, 32, 4)
    assert captured == {
        "path": str(reference_weights),
        "num_codebooks": 32,
        "dtype": "f32",
        "pcm_shape": (1, 1, 6000),
        "pcm_dtype": np.dtype("float32"),
        "reset": True,
    }


@pytest.mark.skipif(
    not (LOCAL_WEIGHTS / "model.safetensors").exists(),
    reason="official Mimi weights are not present under fixtures/reference/hf",
)
def test_cli_benchmark_batching_reports_requested_batch_sizes():
    benchmark = subprocess.run(
        [
            sys.executable,
            "-m",
            "mimi_mlx.cli",
            "benchmark",
            "batching",
            "--weights",
            str(LOCAL_WEIGHTS),
            "--input-dir",
            str(ROOT / "fixtures" / "audio"),
            "--batch-sizes",
            "1,2",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert benchmark.returncode == 0, benchmark.stderr
    payload = json.loads(benchmark.stdout)
    assert payload["command"] == "benchmark batching"
    assert [row["batch_size"] for row in payload["results"]] == [1, 2]
    assert all(row["elapsed_seconds"] > 0 for row in payload["results"])
