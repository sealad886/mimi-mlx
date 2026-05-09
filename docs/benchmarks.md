# Benchmarks

Benchmark entrypoints are runnable and emit machine-readable JSON with elapsed time,
audio seconds, real-time factor, and token-frame counts.

```bash
python scripts/benchmark_encode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --json

python scripts/benchmark_decode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --json

python scripts/benchmark_batching.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
```

The CLI exposes the same benchmark surface:

```bash
mimi-mlx benchmark encode --weights fixtures/reference/hf \
  --input-dir fixtures/audio --json
mimi-mlx benchmark batching --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
```

Current scripts do not yet report peak MLX memory. Add that as a follow-up once MLX
exposes a stable per-process memory counter on the target machines.

## Local Smoke Results

Run on 2026-05-09 against the committed fixture WAVs:

| Command | Audio seconds | Elapsed seconds | Real-time factor |
| --- | ---: | ---: | ---: |
| encode | 4.0007 | 0.2283 | 0.0571 |
| decode | 4.0007 | 0.5154 | 0.1288 |

Batch encode smoke:

| Batch size | Clip samples | Frames | Elapsed seconds |
| ---: | ---: | ---: | ---: |
| 1 | 6000 | 4 | 0.0526 |
| 2 | 6000 | 4 | 0.0768 |
| 4 | 6000 | 4 | 0.0974 |
