from __future__ import annotations

import argparse
import json
import time
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
    parity.add_argument("--sample-rate", type=int)
    parity.add_argument("--json", action="store_true")

    benchmark = subcommands.add_parser("benchmark", help="Run Mimi MLX benchmarks")
    benchmark_subcommands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    for name in ("encode", "decode", "batching"):
        command = benchmark_subcommands.add_parser(name, help=f"Benchmark {name}")
        command.add_argument("--weights", required=True)
        command.add_argument("--input-dir")
        command.add_argument("--batch-sizes", default="1")
        command.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "encode":
        return _encode_command(args)
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
    if args.reference != "transformers":
        raise SystemExit(
            "rustymimi parity is reference tooling only and is not wired in this CLI yet"
        )

    tokenizer = _load_tokenizer(args.weights)
    audio, sample_rate = _read_audio(args.input, sample_rate=args.sample_rate)
    if sample_rate != tokenizer.config.sample_rate:
        raise SystemExit(
            "parity requires input audio at the Mimi model sample rate "
            f"({tokenizer.config.sample_rate} Hz); got {sample_rate} Hz"
        )
    tokens = tokenizer.encode(mx.array(audio), sample_rate=sample_rate)
    actual = np.array(to_upstream_layout(tokens.codes))

    import torch
    from transformers import MimiModel

    model = MimiModel.from_pretrained(str(args.weights)).eval()
    audio_for_ref = audio
    if audio_for_ref.ndim == 1:
        audio_for_ref = audio_for_ref[None, None, :]
    elif audio_for_ref.ndim == 2:
        audio_for_ref = audio_for_ref[:, None, :]
    with torch.no_grad():
        expected = model.encode(
            torch.from_numpy(audio_for_ref.astype("float32")), return_dict=False
        )[0]
    expected_np = expected.cpu().numpy()
    mismatch = first_token_mismatch(expected_np, actual)
    ok = mismatch is None
    payload: dict[str, object] = {
        "command": "parity",
        "ok": ok,
        "reference": args.reference,
        "input": args.input,
        "codes_shape": list(actual.shape),
    }
    if mismatch is not None:
        payload["mismatch"] = mismatch.__dict__
    _emit(payload, as_json=args.json)
    return 0 if ok else 1


def _benchmark_command(args: argparse.Namespace) -> int:
    tokenizer = _load_tokenizer(args.weights)
    if not args.input_dir:
        raise SystemExit("benchmark requires --input-dir")
    clips = sorted(Path(args.input_dir).glob("*.wav"))
    if not clips:
        raise SystemExit(f"No .wav files found under {args.input_dir}")

    start = time.perf_counter()
    total_seconds = 0.0
    token_frames = 0
    for clip in clips:
        audio, sample_rate = _read_audio(clip)
        if args.benchmark_command == "decode":
            tokens = tokenizer.encode(mx.array(audio), sample_rate=sample_rate)
            decoded = tokenizer.decode(tokens.codes, sample_rate=sample_rate)
            mx.eval(decoded)
            token_frames += int(tokens.codes.shape[1])
        else:
            tokens = tokenizer.encode(mx.array(audio), sample_rate=sample_rate)
            mx.eval(tokens.codes)
            token_frames += int(tokens.codes.shape[1])
        total_seconds += np.asarray(audio).shape[0] / sample_rate
    elapsed = time.perf_counter() - start
    payload = {
        "command": f"benchmark {args.benchmark_command}",
        "clips": len(clips),
        "audio_seconds": total_seconds,
        "elapsed_seconds": elapsed,
        "real_time_factor": elapsed / total_seconds if total_seconds else None,
        "token_frames": token_frames,
    }
    _emit(payload, as_json=args.json)
    return 0


def _load_tokenizer(weights: str) -> MimiTokenizer:
    path = Path(weights)
    if not path.exists():
        raise SystemExit(f"--weights must be a local path for CLI commands: {weights}")
    return MimiTokenizer.from_pretrained(path)


def _read_audio(path: str | Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, detected_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.T
    return audio, sample_rate or detected_rate


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        status = "ok" if payload.get("ok", True) else "failed"
        print(f"{payload['command']}: {status}")


if __name__ == "__main__":
    raise SystemExit(main())
