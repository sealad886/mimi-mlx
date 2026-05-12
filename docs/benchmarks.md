# Benchmarks

See `docs/development.md` for when to refresh benchmark notes during
development.

Benchmark entrypoints are runnable and emit machine-readable JSON with elapsed
time, audio seconds, real-time factor, token-frame counts, peak MLX memory in
bytes, and the configured prefetch worker count where applicable.

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
each clip to MLX once and keeps token arrays on MLX until evaluation.

Benchmarks run one unmeasured warmup before timing and call
`mx.reset_peak_memory()` immediately before the measured region. The reported
`peak_memory_bytes` value is MLX allocator peak memory from `mx.get_peak_memory()`.
It is not system RSS; MLX active memory also excludes cached buffers.

`benchmark decode` precomputes tokens from the WAV inputs before the timed
region, then measures only `tokenizer.decode(...)` plus `mx.eval(decoded)`.

## Local Smoke Results

Run on 2026-05-12 against the committed fixture WAVs after warmup and peak-memory
instrumentation:

| Command | Prefetch workers | Audio seconds | Elapsed seconds | Real-time factor | Peak MLX memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| encode | 2 | 4.0007 | 0.2778 | 0.0694 | 514,768,904 B |
| decode | 2 | 4.0007 | 0.1181 | 0.0295 | 900,870,796 B |

Batch encode smoke:

| Batch size | Clip samples | Frames | Elapsed seconds | Peak MLX memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6000 | 4 | 0.0241 | 384,650,640 B |
| 2 | 6000 | 4 | 0.0254 | 429,125,820 B |
| 4 | 6000 | 4 | 0.0408 | 515,792,836 B |
| 8 | 6000 | 4 | 0.0286 | 685,669,844 B |
