# Usage Guide

This guide covers the public `mimi-mlx` workflow: installing the package,
downloading model assets, using the CLI, and calling the Python API.

## Installation modes

Create an environment and install the editable package with developer tools:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Reference tooling is optional and intentionally isolated:

```bash
python -m pip install -e ".[dev,reference]"
```

Use the `reference` extra only for workflows that need `transformers`,
PyTorch, `rustymimi`, `datasets[audio]`, or `safetensors` as comparison
backends. The production `mimi_mlx` package stays MLX-native.

## Model assets

CLI commands require `--weights` to point at a local path containing:

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`

Download the official assets:

```bash
python scripts/download_reference_assets.py
```

The default destination, `fixtures/reference/hf/`, is ignored by git because
the safetensors file is large.

Validate the downloaded checkpoint before debugging parity issues:

```bash
python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors --json
```

Expected high-level values are 350 tensors and 96,151,393 parameters for the
`kyutai/mimi` revision pinned in `scripts/download_reference_assets.py`.

## Token layout

`mimi-mlx` uses canonical token layout:

```text
[batch, frames, codebooks]
```

The upstream Transformers and Rust references use:

```text
[batch, codebooks, frames]
```

Convert explicitly when comparing against upstream:

```python
from mimi_mlx.layouts import from_upstream_layout, to_upstream_layout

upstream_codes = to_upstream_layout(tokens.codes)
canonical_codes = from_upstream_layout(upstream_codes)
```

The full Mimi codec uses 32 codebooks, a 2,048-entry codebook size, 24 kHz
audio, and a 12.5 Hz token frame rate.

## Python API

Load the tokenizer from local assets:

```python
import mlx.core as mx
from mimi_mlx import MimiTokenizer

tokenizer = MimiTokenizer.from_pretrained("fixtures/reference/hf")
audio = mx.zeros((24_000,))

tokens = tokenizer.encode(audio, sample_rate=24_000)
decoded = tokenizer.decode(tokens.codes, sample_rate=24_000)
```

Accepted encode input shapes are:

| Shape | Meaning |
| --- | --- |
| `[samples]` | One mono clip. |
| `[batch, samples]` | Batch of mono clips. |
| `[batch, channels, samples]` | Explicit channel dimension. Current Mimi config expects one channel. |

If `sample_rate` differs from the model sample rate, encode and decode use the
package's MLX linear resampler. Parity checks require input audio at the model
sample rate and do not resample.

## Batching

Same-length batches can use `encode` directly:

```python
batch = mx.zeros((4, 24_000))
tokens = tokenizer.encode(batch, sample_rate=24_000)
```

For padded variable-length batches, pass real sample lengths:

```python
batch = mx.zeros((2, 24_000))
lengths = mx.array([12_000, 24_000], dtype=mx.int32)
tokens = tokenizer.encode_batch(batch, lengths=lengths, sample_rate=24_000)
```

Without `lengths`, padded samples are treated as full-length audio. The returned
`MimiTokens.lengths` field stores token-frame lengths; `audio_lengths` stores
input sample lengths when explicit lengths were provided.

## Token files

The CLI reads and writes `.npy` files containing canonical token codes only.
Token arrays are saved with MLX-native IO, so encode paths do not materialize
codes through NumPy before writing them. The Python API also has `.npz` helpers
that preserve token metadata:

```python
from mimi_mlx import MimiTokenizer

MimiTokenizer.save_tokens_npz("/tmp/tokens.npz", tokens)
loaded = MimiTokenizer.load_tokens_npz("/tmp/tokens.npz")
```

Use `.npz` when carrying metadata across processes. Use `.npy` when matching
the current CLI surface.

## CLI commands

The installed console script is `mimi-mlx`; the module entrypoint is equivalent:

```bash
python -m mimi_mlx.cli --help
```

### Encode

```bash
mimi-mlx encode fixtures/audio/sine_440_025s.wav \
  --weights fixtures/reference/hf \
  --output /tmp/sine_440_tokens.npy \
  --json
```

JSON output includes the command name, input path, output path, token shape,
sample rate, frame rate, and token layout.

### Encode Directory

```bash
mimi-mlx encode-dir fixtures/audio \
  --weights fixtures/reference/hf \
  --output-dir /tmp/mimi_tokens \
  --prefetch-workers 2 \
  --json
```

`encode-dir` reads WAV files through a bounded thread prefetcher, converts each
clip to an MLX array once immediately before tokenization, and saves each token
file with MLX-native `.npy` serialization. Worker threads pass paths and decoded
CPU audio only; token tensors stay on the MLX path until they are written.

### Decode

```bash
mimi-mlx decode /tmp/sine_440_tokens.npy \
  --weights fixtures/reference/hf \
  --output /tmp/sine_440_recon.wav \
  --json
```

CLI decode currently writes one batch item at a time. If the token file contains
more than one batch item, use the Python API to decode and write each item.

### Parity

Compare against Transformers:

```bash
mimi-mlx parity fixtures/audio/sine_440_025s.wav \
  --reference transformers \
  --weights fixtures/reference/hf \
  --json
```

Compare against `rustymimi`:

```bash
mimi-mlx parity fixtures/audio/sine_440_025s.wav \
  --reference rustymimi \
  --weights fixtures/reference/hf \
  --reference-weights "$RUSTYMIMI_WEIGHTS" \
  --json
```

For `rustymimi`, `--reference-weights` must point at a Moshi tokenizer
`tokenizer-*.safetensors` checkpoint. The Hugging Face
`kyutai/mimi/model.safetensors` file does not contain the Rust/Candle tensor
names that `rustymimi` expects.

### Benchmarks

```bash
mimi-mlx benchmark encode --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
mimi-mlx benchmark decode --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
mimi-mlx benchmark batching --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
```

Benchmark commands emit elapsed time, audio seconds, real-time factor, and
token-frame counts where applicable. They also report `peak_memory_bytes` using
MLX allocator peak memory after an unmeasured warmup. Encode and decode
benchmarks use the same threaded audio prefetch path as `encode-dir`. Decode
benchmarks precompute tokens before timing so the measured region is decode-only.
See `docs/benchmarks.md` for current local smoke results.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `--weights must be a local path` | Download assets with `python scripts/download_reference_assets.py` and pass the directory path. |
| Missing or mismatched tensors | Run `python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors`. |
| Rust parity asks for reference weights | Download the Moshi tokenizer checkpoint and pass it via `--reference-weights` or `MIMI_RUSTYMIMI_WEIGHTS`. |
| Encode parity mismatch | Confirm token layout, codebook count, input sample rate, and fixture audio checksum. |
| Padded batch output includes padding | Call `encode_batch(..., lengths=...)` with real sample lengths. |
