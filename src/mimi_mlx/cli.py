from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from .layouts import to_upstream_layout
from .parity import first_token_mismatch
from .tokenizer import MimiTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mimi-mlx")
    subcommands = parser.add_subparsers(dest="command", required=True)

    encode = subcommands.add_parser("encode", help="Encode audio to Mimi tokens")
    encode.add_argument("input")
    encode.add_argument("--weights", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--sample-rate", type=int)
    encode.add_argument("--json", action="store_true")

    encode_dir = subcommands.add_parser(
        "encode-dir", help="Encode a directory of WAV files to Mimi tokens"
    )
    encode_dir.add_argument("input_dir")
    encode_dir.add_argument("--weights", required=True)
    encode_dir.add_argument("--output-dir", required=True)
    encode_dir.add_argument("--sample-rate", type=int)
    encode_dir.add_argument("--prefetch-workers", type=int, default=2)
    encode_dir.add_argument("--json", action="store_true")

    decode = subcommands.add_parser("decode", help="Decode Mimi tokens to audio")
    decode.add_argument("tokens")
    decode.add_argument("--weights", required=True)
    decode.add_argument("--output", required=True)
    decode.add_argument("--sample-rate", type=int)
    decode.add_argument("--json", action="store_true")

    parity = subcommands.add_parser("parity", help="Compare MLX tokens with a reference backend")
    parity.add_argument("input")
    parity.add_argument("--reference", choices=["rustymimi", "transformers"], required=True)
    parity.add_argument("--weights", required=True)
    parity.add_argument(
        "--reference-weights",
        help="Reference backend weights; required for rustymimi tokenizer checkpoints",
    )
    parity.add_argument("--sample-rate", type=int)
    parity.add_argument("--json", action="store_true")

    benchmark = subcommands.add_parser("benchmark", help="Run Mimi MLX benchmarks")
    benchmark_subcommands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for name in ("encode", "decode", "batching"):
        command = benchmark_subcommands.add_parser(name, help=f"Benchmark {name}")
        command.add_argument("--weights", required=True)
        command.add_argument("--input-dir")
        if name == "batching":
            command.add_argument("--batch-sizes", default="1")
        else:
            command.add_argument("--prefetch-workers", type=int, default=2)
        command.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "encode":
        return _encode_command(args)
    if args.command == "encode-dir":
        return _encode_dir_command(args)
    if args.command == "decode":
        return _decode_command(args)
    if args.command == "parity":
        return _parity_command(args)
    if args.command == "benchmark":
        return _benchmark_command(args)

    parser.error(f"{args.command!r} is not implemented until model parity stages land")
    return 2


def _encode_command(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.weights)
    audio, sample_rate = _read_audio(args.input, sample_rate=args.sample_rate)
    tokens = tokenizer.encode(mx.array(audio), sample_rate=sample_rate)
    MimiTokenizer.save_tokens_npy(args.output, tokens.codes)
    _emit(
        {
            "command": "encode",
            "input": args.input,
            "output": args.output,
            "codes_shape": list(tokens.codes.shape),
            "sample_rate": tokens.sample_rate,
            "frame_rate": tokens.frame_rate,
            "layout": tokens.layout,
        },
        as_json=args.json,
    )
    return 0


def _encode_dir_command(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.weights)
    payload = _encode_directory(
        tokenizer,
        Path(args.input_dir),
        Path(args.output_dir),
        sample_rate=args.sample_rate,
        prefetch_workers=args.prefetch_workers,
    )
    _emit(payload, as_json=args.json)
    return 0


def _decode_command(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.weights)
    sample_rate = args.sample_rate or tokenizer.config.sample_rate
    tokens = MimiTokenizer.load_tokens_npy(
        args.tokens,
        sample_rate=tokenizer.config.sample_rate,
        frame_rate=tokenizer.config.frame_rate,
    )
    audio = tokenizer.decode(tokens.codes, sample_rate=sample_rate)
    audio_np = np.array(audio)
    if audio_np.shape[0] != 1:
        raise SystemExit("CLI decode currently writes one batch item at a time")
    write_audio = audio_np[0].T
    if write_audio.shape[-1] == 1:
        write_audio = write_audio[:, 0]
    sf.write(args.output, write_audio, sample_rate)
    _emit(
        {
            "command": "decode",
            "input": args.tokens,
            "output": args.output,
            "audio_shape": list(audio_np.shape),
            "sample_rate": sample_rate,
        },
        as_json=args.json,
    )
    return 0


def _parity_command(args: argparse.Namespace) -> int:
    reference_weights = None
    if args.reference == "rustymimi":
        reference_weights = _resolve_rustymimi_reference_weights(args.reference_weights)
    tokenizer = _load_tokenizer(args.weights)
    audio, sample_rate = _read_audio(args.input, sample_rate=args.sample_rate)
    if sample_rate != tokenizer.config.sample_rate:
        raise SystemExit(
            "parity requires input audio at the Mimi model sample rate "
            f"({tokenizer.config.sample_rate} Hz); got {sample_rate} Hz"
        )
    tokens = tokenizer.encode(mx.array(audio), sample_rate=sample_rate)
    actual = np.array(to_upstream_layout(tokens.codes))

    if args.reference == "rustymimi":
        expected_np = _rustymimi_reference_codes(
            reference_weights,
            audio,
            num_codebooks=tokenizer.config.num_codebooks,
        )
    else:
        expected_np = _transformers_reference_codes(args.weights, audio)
    mismatch = first_token_mismatch(expected_np, actual)
    ok = mismatch is None
    payload: dict[str, object] = {
        "command": "parity",
        "ok": ok,
        "reference": args.reference,
        "input": args.input,
        "codes_shape": list(actual.shape),
    }
    if reference_weights is not None:
        payload["reference_weights"] = str(reference_weights)
    if mismatch is not None:
        payload["mismatch"] = mismatch.__dict__
    _emit(payload, as_json=args.json)
    return 0 if ok else 1


@dataclass(frozen=True)
class AudioClip:
    path: Path
    audio: np.ndarray
    sample_rate: int


def _load_audio_clip(path: Path, sample_rate: int | None = None) -> AudioClip:
    audio, detected_rate = _read_audio(path, sample_rate=sample_rate)
    return AudioClip(path=path, audio=audio, sample_rate=detected_rate)


def _iter_prefetched_audio(
    paths: Iterable[Path],
    *,
    sample_rate: int | None,
    prefetch_workers: int,
) -> Iterator[AudioClip]:
    if prefetch_workers <= 0:
        raise SystemExit("--prefetch-workers must be positive")

    path_iter = iter(paths)
    with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
        pending: deque[Future[AudioClip]] = deque()

        def submit_next() -> bool:
            try:
                path = next(path_iter)
            except StopIteration:
                return False
            pending.append(executor.submit(_load_audio_clip, path, sample_rate))
            return True

        for _ in range(prefetch_workers):
            if not submit_next():
                break

        while pending:
            future = pending.popleft()
            clip = future.result()
            submit_next()
            yield clip


def _audio_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return audio.size / sample_rate


def _token_frame_count(codes: mx.array) -> int:
    return int(codes.shape[0] * codes.shape[1])


def _find_wav_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input path must be a directory: {input_dir}")
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
    )


def _encode_directory(
    tokenizer: MimiTokenizer,
    input_dir: Path,
    output_dir: Path,
    *,
    sample_rate: int | None,
    prefetch_workers: int,
) -> dict[str, object]:
    clips = _find_wav_files(input_dir)
    if not clips:
        raise SystemExit(f"No .wav files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    outputs: list[str] = []
    total_seconds = 0.0
    token_frames = 0
    for clip in _iter_prefetched_audio(
        clips, sample_rate=sample_rate, prefetch_workers=prefetch_workers
    ):
        tokens = tokenizer.encode(mx.array(clip.audio), sample_rate=clip.sample_rate)
        output_path = output_dir / f"{clip.path.stem}.npy"
        MimiTokenizer.save_tokens_npy(output_path, tokens.codes)
        outputs.append(str(output_path))
        token_frames += _token_frame_count(tokens.codes)
        total_seconds += _audio_seconds(clip.audio, clip.sample_rate)

    elapsed = time.perf_counter() - start
    return {
        "command": "encode-dir",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(outputs),
        "audio_seconds": total_seconds,
        "elapsed_seconds": elapsed,
        "real_time_factor": elapsed / total_seconds if total_seconds else None,
        "token_frames": token_frames,
        "outputs": outputs,
        "prefetch_workers": prefetch_workers,
    }


def _transformers_reference_codes(weights: str, audio: np.ndarray) -> np.ndarray:
    import torch
    from transformers import MimiModel

    model = MimiModel.from_pretrained(str(weights)).eval()
    audio_for_ref = audio
    if audio_for_ref.ndim == 1:
        audio_for_ref = audio_for_ref[None, None, :]
    elif audio_for_ref.ndim == 2:
        audio_for_ref = audio_for_ref[:, None, :]
    with torch.no_grad():
        expected = model.encode(
            torch.from_numpy(audio_for_ref.astype("float32")), return_dict=False
        )[0]
    return expected.cpu().numpy()


def _resolve_rustymimi_reference_weights(reference_weights: str | None) -> Path:
    candidate = reference_weights or os.environ.get("MIMI_RUSTYMIMI_WEIGHTS")
    if not candidate:
        raise SystemExit(
            "rustymimi parity requires --reference-weights pointing at "
            "tokenizer-*.safetensors"
        )
    path = Path(candidate)
    if path.is_dir():
        matches = sorted(path.glob("tokenizer-*.safetensors"))
        if len(matches) == 1:
            return matches[0]
        raise SystemExit(
            f"rustymimi reference directory must contain exactly one "
            f"tokenizer-*.safetensors file: {path}"
        )
    if not path.exists():
        raise SystemExit(f"rustymimi reference weights do not exist: {path}")
    return path


def _rustymimi_reference_codes(
    reference_weights: Path,
    audio: np.ndarray,
    *,
    num_codebooks: int,
) -> np.ndarray:
    import rustymimi

    if audio.ndim == 1:
        pcm_data = audio[None, None, :]
    elif audio.ndim == 2:
        pcm_data = audio[:, None, :]
    elif audio.ndim == 3:
        pcm_data = audio
    else:
        raise ValueError(f"Expected audio with 1, 2, or 3 dimensions, got {audio.shape}")
    tokenizer = rustymimi.Tokenizer(str(reference_weights), num_codebooks=num_codebooks)
    try:
        return np.asarray(tokenizer.encode(np.asarray(pcm_data, dtype=np.float32)))
    finally:
        reset = getattr(tokenizer, "reset", None)
        if callable(reset):
            reset()


def _benchmark_command(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.weights)
    if not args.input_dir:
        raise SystemExit("benchmark requires --input-dir")
    if args.benchmark_command == "batching":
        return _benchmark_batching(tokenizer, args)

    clips = _find_wav_files(Path(args.input_dir))
    if not clips:
        raise SystemExit(f"No .wav files found under {args.input_dir}")

    start = time.perf_counter()
    total_seconds = 0.0
    token_frames = 0
    for clip in _iter_prefetched_audio(
        clips, sample_rate=None, prefetch_workers=args.prefetch_workers
    ):
        if args.benchmark_command == "decode":
            tokens = tokenizer.encode(mx.array(clip.audio), sample_rate=clip.sample_rate)
            decoded = tokenizer.decode(tokens.codes, sample_rate=clip.sample_rate)
            mx.eval(decoded)
            token_frames += _token_frame_count(tokens.codes)
        else:
            tokens = tokenizer.encode(mx.array(clip.audio), sample_rate=clip.sample_rate)
            mx.eval(tokens.codes)
            token_frames += _token_frame_count(tokens.codes)
        total_seconds += _audio_seconds(clip.audio, clip.sample_rate)
    elapsed = time.perf_counter() - start
    payload = {
        "command": f"benchmark {args.benchmark_command}",
        "clips": len(clips),
        "audio_seconds": total_seconds,
        "elapsed_seconds": elapsed,
        "real_time_factor": elapsed / total_seconds if total_seconds else None,
        "token_frames": token_frames,
        "prefetch_workers": args.prefetch_workers,
    }
    _emit(payload, as_json=args.json)
    return 0


def _benchmark_batching(tokenizer: MimiTokenizer, args: argparse.Namespace) -> int:
    clips = _load_same_rate_clips(Path(args.input_dir))
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    results = []
    for batch_size in batch_sizes:
        selected = [clips[index % len(clips)] for index in range(batch_size)]
        min_length = min(audio.shape[0] for audio, _ in selected)
        batch = mx.stack([mx.array(audio[:min_length]) for audio, _ in selected], axis=0)
        sample_rate = selected[0][1]
        start = time.perf_counter()
        tokens = tokenizer.encode_batch(batch, sample_rate=sample_rate)
        mx.eval(tokens.codes)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "batch_size": batch_size,
                "samples_per_clip": min_length,
                "frames": int(tokens.codes.shape[1]),
                "elapsed_seconds": elapsed,
            }
        )
    payload = {"command": "benchmark batching", "results": results}
    _emit_batching(payload, as_json=args.json)
    return 0


def _parse_batch_sizes(batch_sizes: str) -> list[int]:
    try:
        parsed = [int(part) for part in batch_sizes.split(",") if part]
    except ValueError as exc:
        raise SystemExit(f"Invalid --batch-sizes value: {batch_sizes}") from exc
    if not parsed or any(size <= 0 for size in parsed):
        raise SystemExit("--batch-sizes must contain positive integers")
    return parsed


def _load_same_rate_clips(input_dir: Path) -> list[tuple[np.ndarray, int]]:
    clips = []
    for path in _find_wav_files(input_dir):
        audio, sample_rate = _read_audio(path)
        clips.append((audio, sample_rate))
    if not clips:
        raise SystemExit(f"No .wav files found under {input_dir}")
    rates = {sample_rate for _, sample_rate in clips}
    if len(rates) != 1:
        raise SystemExit(f"Batch benchmark requires one sample rate, got {sorted(rates)}")
    return clips


def _load_tokenizer(weights: str) -> MimiTokenizer:
    path = Path(weights)
    if not path.exists():
        raise SystemExit(f"--weights must be a local path for CLI commands: {weights}")
    return MimiTokenizer.from_pretrained(path)


def _read_audio(path: str | Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, detected_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate is not None and sample_rate != detected_rate:
        raise SystemExit(
            f"--sample-rate {sample_rate} does not match detected WAV rate "
            f"{detected_rate} for {path}"
        )
    if audio.ndim == 2:
        channels = audio.shape[1]
        if channels != 1:
            raise SystemExit(f"Expected mono WAV input, got {channels} channels in {path}")
        audio = audio[:, 0]
    return audio, detected_rate


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        status = "ok" if payload.get("ok", True) else "failed"
        print(f"{payload['command']}: {status}")


def _emit_batching(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for result in payload["results"]:
            print(
                f"batch={result['batch_size']} frames={result['frames']} "
                f"elapsed={result['elapsed_seconds']:.4f}s"
            )


if __name__ == "__main__":
    raise SystemExit(main())
