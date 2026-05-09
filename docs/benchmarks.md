# Benchmarks

See `docs/development.md` for when to refresh benchmark notes during
development.

Benchmark entrypoints are runnable and emit machine-readable JSON with elapsed
time, audio seconds, real-time factor, token-frame counts, and the configured
prefetch worker count where applicable.

```bash
python scripts/benchmark_encode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json

python scripts/benchmark_decode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json

python scripts/benchmark_batching.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
```

The CLI exposes the same benchmark surface:

```bash
mimi-mlx benchmark encode --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
mimi-mlx benchmark decode --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
mimi-mlx benchmark batching --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
```

Encode and decode benchmarks use the same bounded thread prefetch path as
`encode-dir`: worker threads decode WAV files on CPU while the main thread moves
each clip to MLX once, tokenizes, and keeps token arrays on MLX until save or
evaluation.

Current scripts do not yet report peak MLX memory. Add that as a follow-up once MLX
exposes a stable per-process memory counter on the target machines.

## Local Smoke Results

Run on 2026-05-09 against the committed fixture WAVs after MLX fast-path
optimizations and bounded audio prefetch:

| Command | Prefetch workers | Audio seconds | Elapsed seconds | Real-time factor |
| --- | ---: | ---: | ---: | ---: |
| encode-dir | 2 | 4.0007 | 0.2271 | 0.0568 |
| encode | 2 | 4.0007 | 0.1185 | 0.0296 |
| decode | 2 | 4.0007 | 0.3166 | 0.0791 |

Batch encode smoke:

| Batch size | Clip samples | Frames | Elapsed seconds |
| ---: | ---: | ---: | ---: |
| 1 | 6000 | 4 | 0.0374 |
| 2 | 6000 | 4 | 0.0205 |
| 4 | 6000 | 4 | 0.0280 |
| 8 | 6000 | 4 | 0.0969 |
