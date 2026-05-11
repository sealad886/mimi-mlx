from __future__ import annotations

import json
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import mlx.core as mx
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
    assert "encode-dir" in result.stdout
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


def test_encode_directory_converts_audio_once_and_saves_mlx_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from mimi_mlx import cli

    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "tokens"
    input_dir.mkdir()
    first = input_dir / "first.wav"
    second = input_dir / "second.WAV"
    ignored_directory = input_dir / "nested.wav"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    ignored_directory.mkdir()
    reads: list[Path] = []
    encoded_shapes: list[tuple[int, ...]] = []
    saved: list[tuple[Path, mx.array]] = []

    class FakeTokenizer:
        def encode(self, audio: mx.array, *, sample_rate: int):
            assert not isinstance(audio, np.ndarray)
            encoded_shapes.append(audio.shape)
            return types.SimpleNamespace(
                codes=mx.zeros((1, 4, 32), dtype=mx.int32),
                sample_rate=sample_rate,
                frame_rate=12.5,
                layout="batch_time_codebook",
            )

    def fake_read_audio(path: str | Path, *, sample_rate: int | None = None):
        reads.append(Path(path))
        return np.zeros(16, dtype=np.float32), sample_rate or 24_000

    def fake_save_tokens_npy(path: str | Path, codes: mx.array) -> None:
        assert not isinstance(codes, np.ndarray)
        saved.append((Path(path), codes))

    monkeypatch.setattr(cli, "_read_audio", fake_read_audio)
    monkeypatch.setattr(cli.MimiTokenizer, "save_tokens_npy", fake_save_tokens_npy)

    payload = cli._encode_directory(
        FakeTokenizer(),
        input_dir,
        output_dir,
        sample_rate=None,
        prefetch_workers=2,
    )

    assert reads == [first, second]
    assert encoded_shapes == [(16,), (16,)]
    assert [path.name for path, _ in saved] == ["first.npy", "second.npy"]
    assert payload["files"] == 2
    assert payload["outputs"] == [str(output_dir / "first.npy"), str(output_dir / "second.npy")]


def test_prefetched_audio_uses_bounded_thread_executor(monkeypatch: pytest.MonkeyPatch):
    from mimi_mlx import cli

    calls: list[tuple[str, object]] = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            calls.append(("result", self._result.path))
            return self._result

    class FakeExecutor:
        def __init__(self, *, max_workers: int):
            calls.append(("executor", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, path, sample_rate):
            calls.append(("submit", path))
            return FakeFuture(fn(path, sample_rate))

    def fake_read_audio(path: Path, *, sample_rate: int | None = None):
        calls.append(("read", path))
        return np.zeros(4, dtype=np.float32), sample_rate or 24_000

    monkeypatch.setattr(cli, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(cli, "_read_audio", fake_read_audio)

    clips = list(
        cli._iter_prefetched_audio(
            [Path("a.wav"), Path("b.wav"), Path("c.wav")],
            sample_rate=None,
            prefetch_workers=2,
        )
    )

    assert [clip.path for clip in clips] == [Path("a.wav"), Path("b.wav"), Path("c.wav")]
    assert calls == [
        ("executor", 2),
        ("submit", Path("a.wav")),
        ("read", Path("a.wav")),
        ("submit", Path("b.wav")),
        ("read", Path("b.wav")),
        ("result", Path("a.wav")),
        ("submit", Path("c.wav")),
        ("read", Path("c.wav")),
        ("result", Path("b.wav")),
        ("result", Path("c.wav")),
    ]


def test_encode_dir_command_reports_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mimi_mlx import cli

    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "tokens"
    input_dir.mkdir()
    (input_dir / "clip.wav").write_bytes(b"fake")

    monkeypatch.setattr(
        cli,
        "_load_tokenizer",
        lambda weights: object(),
    )
    monkeypatch.setattr(
        cli,
        "_encode_directory",
        lambda tokenizer, input_dir, output_dir, sample_rate, prefetch_workers: {
            "command": "encode-dir",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "files": 1,
            "outputs": [str(output_dir / "clip.npy")],
            "prefetch_workers": prefetch_workers,
        },
    )

    result = cli._encode_dir_command(
        Namespace(
            input_dir=str(input_dir),
            weights="fixtures/reference/hf",
            output_dir=str(output_dir),
            sample_rate=None,
            prefetch_workers=3,
            json=True,
        )
    )

    assert result == 0


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
