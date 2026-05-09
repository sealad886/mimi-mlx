from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mimi_mlx import MimiTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLX Mimi batch encode")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tokenizer = MimiTokenizer.from_pretrained(Path(args.weights))
    clips = _load_same_rate_clips(Path(args.input_dir))
    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part]
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
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for result in results:
            print(
                f"batch={result['batch_size']} frames={result['frames']} "
                f"elapsed={result['elapsed_seconds']:.4f}s"
            )
    return 0


def _load_same_rate_clips(input_dir: Path) -> list[tuple[np.ndarray, int]]:
    clips = []
    for path in sorted(input_dir.glob("*.wav")):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim != 1:
            audio = audio[:, 0]
        clips.append((audio, sample_rate))
    if not clips:
        raise SystemExit(f"No .wav files found under {input_dir}")
    rates = {sample_rate for _, sample_rate in clips}
    if len(rates) != 1:
        raise SystemExit(f"Batch benchmark requires one sample rate, got {sorted(rates)}")
    return clips


if __name__ == "__main__":
    raise SystemExit(main())
