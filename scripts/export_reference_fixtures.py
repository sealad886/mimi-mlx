from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import MimiModel

from mimi_mlx.config import DEFAULT_MIMI_REVISION

DEFAULT_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class GeneratedAudio:
    fixture_id: str
    samples: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Mimi reference parity fixtures")
    parser.add_argument("--weights", default="fixtures/reference/hf")
    parser.add_argument("--output-root", default="fixtures")
    parser.add_argument("--revision", default=DEFAULT_MIMI_REVISION)
    parser.add_argument(
        "--allow-missing-speech-source",
        action="store_true",
        help="Regenerate only synthetic fixtures when the LibriSpeech source parquet is absent",
    )
    parser.add_argument(
        "--speech-source-parquet",
        default="fixtures/source/librispeech_asr_dummy/clean/validation-00000-of-00001.parquet",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    audio_dir = output_root / "audio"
    reference_dir = output_root / "reference"
    audio_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    model = load_reference_model(args.weights, revision=args.revision)
    weights_path = resolve_weights_file(args.weights, revision=args.revision)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "upstream": {
            "name": "transformers.MimiModel",
            "repo": "kyutai/mimi",
            "revision": args.revision,
            "weights_sha256": sha256_file(weights_path),
            "token_layout": "batch_codebook_time",
        },
        "fixtures": [],
    }

    for generated in generated_audio(
        Path(args.speech_source_parquet),
        allow_missing_speech_source=args.allow_missing_speech_source,
    ):
        wav_path = audio_dir / f"{generated.fixture_id}.wav"
        sf.write(wav_path, generated.samples, DEFAULT_SAMPLE_RATE, subtype="FLOAT")
        reread, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise RuntimeError(f"Unexpected sample rate for {wav_path}: {sample_rate}")
        if reread.ndim != 1:
            reread = reread[:, 0]

        with torch.no_grad():
            audio = torch.from_numpy(reread)[None, None, :]
            codes = model.encode(audio, return_dict=False)[0].cpu().numpy()
            recon = model.decode(torch.from_numpy(codes), return_dict=False)[0].cpu().numpy()

        tokens_path = reference_dir / f"{generated.fixture_id}.tokens.npy"
        recon_path = reference_dir / f"{generated.fixture_id}.recon.npy"
        np.save(tokens_path, codes)
        np.save(recon_path, recon)

        manifest["fixtures"].append(
            {
                "id": generated.fixture_id,
                "audio_path": str(wav_path),
                "audio_sha256": sha256_file(wav_path),
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "duration_seconds": len(reread) / DEFAULT_SAMPLE_RATE,
                "tokens_path": str(tokens_path),
                "tokens_sha256": sha256_file(tokens_path),
                "reconstruction_path": str(recon_path),
                "reconstruction_sha256": sha256_file(recon_path),
                "layout": "batch_codebook_time",
                "codes_shape": list(codes.shape),
            }
        )

    manifest_path = reference_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")
    return 0


def load_reference_model(weights: str | Path, *, revision: str):
    path = Path(weights)
    if path.exists():
        return MimiModel.from_pretrained(str(path)).eval()
    return MimiModel.from_pretrained(str(weights), revision=revision).eval()


def resolve_weights_file(weights: str | Path, *, revision: str) -> Path:
    path = Path(weights)
    if path.exists():
        weights_path = path / "model.safetensors" if path.is_dir() else path
        if not weights_path.exists():
            raise FileNotFoundError(f"Could not find model.safetensors under {path}")
        return weights_path
    return Path(hf_hub_download(str(weights), "model.safetensors", revision=revision))


def generated_audio(
    speech_source_parquet: Path,
    *,
    allow_missing_speech_source: bool = False,
) -> list[GeneratedAudio]:
    sr = DEFAULT_SAMPLE_RATE
    t025 = np.arange(sr // 4, dtype=np.float32) / sr
    t100 = np.arange(sr, dtype=np.float32) / sr
    rng = np.random.default_rng(1337)

    impulse = np.zeros(sr // 4, dtype=np.float32)
    impulse[len(impulse) // 2] = 0.8

    odd = (
        0.07 * np.sin(2 * np.pi * 330 * np.arange(6017, dtype=np.float32) / sr)
        + 0.02 * np.sin(2 * np.pi * 990 * np.arange(6017, dtype=np.float32) / sr)
    ).astype(np.float32)

    speech_like_t = np.arange(sr // 2, dtype=np.float32) / sr
    envelope = np.clip(np.sin(np.pi * speech_like_t / speech_like_t[-1]), 0.0, 1.0)
    speech_like = (
        envelope
        * (
            0.05 * np.sin(2 * np.pi * 125 * speech_like_t)
            + 0.025 * np.sin(2 * np.pi * 250 * speech_like_t)
            + 0.015 * np.sin(2 * np.pi * 510 * speech_like_t)
        )
    ).astype(np.float32)

    fixtures = [
        GeneratedAudio("silence_025s", np.zeros(sr // 4, dtype=np.float32)),
        GeneratedAudio("sine_440_025s", (0.1 * np.sin(2 * np.pi * 440 * t025)).astype(np.float32)),
        GeneratedAudio("sine_440_100s", (0.1 * np.sin(2 * np.pi * 440 * t100)).astype(np.float32)),
        GeneratedAudio("impulse_center_025s", impulse),
        GeneratedAudio("noise_low_025s", (0.005 * rng.standard_normal(sr // 4)).astype(np.float32)),
        GeneratedAudio(
            "clipped_025s", np.clip(1.2 * np.sin(2 * np.pi * 220 * t025), -1, 1).astype(np.float32)
        ),
        GeneratedAudio("odd_length_6017", odd),
        GeneratedAudio("synthetic_speech_like_050s", speech_like),
    ]
    speech = maybe_load_speech_fixture(
        speech_source_parquet,
        allow_missing=allow_missing_speech_source,
    )
    if speech is not None:
        fixtures.append(speech)
    return fixtures


def maybe_load_speech_fixture(
    source: Path,
    *,
    allow_missing: bool = False,
) -> GeneratedAudio | None:
    if not source.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(
            "Missing LibriSpeech source parquet for real_speech_librispeech_100s: "
            f"{source}. Download reference sources or pass --allow-missing-speech-source "
            "for a synthetic-only fixture set."
        )

    import pyarrow.parquet as pq

    row = pq.read_table(source, columns=["audio"]).slice(0, 1).to_pydict()["audio"][0]
    samples, sample_rate = sf.read(io.BytesIO(row["bytes"]), dtype="float32", always_2d=False)
    if samples.ndim != 1:
        samples = samples[:, 0]
    if sample_rate != DEFAULT_SAMPLE_RATE:
        samples = resample_linear_np(samples, src_rate=sample_rate, dst_rate=DEFAULT_SAMPLE_RATE)
    return GeneratedAudio("real_speech_librispeech_100s", samples[:DEFAULT_SAMPLE_RATE])


def resample_linear_np(samples: np.ndarray, *, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.float32)
    out_length = max(1, int(np.floor(samples.shape[0] * dst_rate / src_rate)))
    positions = np.arange(out_length, dtype=np.float32) * (src_rate / dst_rate)
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, samples.shape[0] - 1)
    frac = positions - left.astype(np.float32)
    return (samples[left] * (1.0 - frac) + samples[right] * frac).astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
