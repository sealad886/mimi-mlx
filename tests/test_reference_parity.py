from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from mimi_mlx import MimiTokenizer
from mimi_mlx.layouts import from_upstream_layout, to_upstream_layout
from mimi_mlx.parity import first_token_mismatch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fixtures" / "reference" / "manifest.json"
LOCAL_WEIGHTS = ROOT / "fixtures" / "reference" / "hf"
MANIFEST_DATA = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"fixtures": []}
FIXTURES = MANIFEST_DATA["fixtures"]


@pytest.fixture(scope="session")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("reference fixture manifest is not present")
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="session")
def tokenizer() -> MimiTokenizer:
    if not (LOCAL_WEIGHTS / "model.safetensors").exists():
        pytest.skip("official Mimi weights are not present under fixtures/reference/hf")
    return MimiTokenizer.from_pretrained(LOCAL_WEIGHTS)


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture["id"] for fixture in FIXTURES])
def test_exact_encode_token_parity(tokenizer: MimiTokenizer, fixture: dict):
    audio, sample_rate = sf.read(ROOT / fixture["audio_path"], dtype="float32", always_2d=False)
    reference = np.load(ROOT / fixture["tokens_path"], allow_pickle=False)

    tokens = tokenizer.encode(mx.array(audio), sample_rate=sample_rate)
    actual = np.array(to_upstream_layout(tokens.codes))

    mismatch = first_token_mismatch(reference, actual)
    assert mismatch is None, (
        f"{fixture['id']} first mismatch: frame={mismatch.frame if mismatch else 'n/a'} "
        f"codebook={mismatch.codebook if mismatch else 'n/a'} "
        f"expected={mismatch.expected if mismatch else 'n/a'} "
        f"actual={mismatch.actual if mismatch else 'n/a'}"
    )
    assert actual.shape == tuple(fixture["codes_shape"])


@pytest.mark.parametrize("fixture", FIXTURES, ids=[fixture["id"] for fixture in FIXTURES])
def test_decode_waveform_parity(tokenizer: MimiTokenizer, fixture: dict):
    reference_codes = np.load(ROOT / fixture["tokens_path"], allow_pickle=False)
    reference_recon = np.load(ROOT / fixture["reconstruction_path"], allow_pickle=False)

    decoded = tokenizer.decode(
        from_upstream_layout(mx.array(reference_codes)), sample_rate=fixture["sample_rate"]
    )
    actual = np.array(decoded)

    assert actual.shape == reference_recon.shape
    assert np.max(np.abs(actual - reference_recon)) < 2e-5
    assert np.mean(np.square(actual - reference_recon)) < 1e-10


def test_encode_is_deterministic_across_reset(tokenizer: MimiTokenizer, manifest: dict):
    fixture = manifest["fixtures"][1]
    audio, sample_rate = sf.read(ROOT / fixture["audio_path"], dtype="float32", always_2d=False)

    first = tokenizer.encode(mx.array(audio), sample_rate=sample_rate).codes
    second = tokenizer.encode(mx.array(audio), sample_rate=sample_rate).codes
    tokenizer.reset_state()
    third = tokenizer.encode(mx.array(audio), sample_rate=sample_rate).codes

    assert np.array_equal(np.array(first), np.array(second))
    assert np.array_equal(np.array(first), np.array(third))


@pytest.mark.skipif(
    not (LOCAL_WEIGHTS / "model.safetensors").exists(),
    reason="official Mimi weights are not present under fixtures/reference/hf",
)
def test_rustymimi_exact_token_parity_when_reference_weights_are_available(manifest: dict):
    reference_weights = os.environ.get("MIMI_RUSTYMIMI_WEIGHTS")
    if not reference_weights:
        pytest.skip("MIMI_RUSTYMIMI_WEIGHTS is not set")
    reference_path = Path(reference_weights)
    if not reference_path.exists():
        pytest.skip(f"MIMI_RUSTYMIMI_WEIGHTS does not exist: {reference_path}")

    fixture = manifest["fixtures"][1]
    parity = subprocess.run(
        [
            sys.executable,
            "-m",
            "mimi_mlx.cli",
            "parity",
            str(ROOT / fixture["audio_path"]),
            "--reference",
            "rustymimi",
            "--weights",
            str(LOCAL_WEIGHTS),
            "--reference-weights",
            str(reference_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert parity.returncode == 0, parity.stderr
    payload = json.loads(parity.stdout)
    assert payload["ok"] is True
    assert payload["reference"] == "rustymimi"


def test_first_token_mismatch_reports_upstream_layout_indices():
    expected = np.zeros((1, 32, 4), dtype=np.int32)
    actual = expected.copy()
    actual[0, 7, 2] = 1

    mismatch = first_token_mismatch(expected, actual)

    assert mismatch is not None
    assert mismatch.batch == 0
    assert mismatch.frame == 2
    assert mismatch.codebook == 7
