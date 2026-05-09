from __future__ import annotations

import subprocess
import sys
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
