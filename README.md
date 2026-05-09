# mimi-mlx

`mimi-mlx` is a correctness-first MLX-native port of Kyutai Mimi for Apple
Silicon. It provides local Mimi audio tokenization and reconstruction using the
official Hugging Face `kyutai/mimi` checkpoint without importing PyTorch,
`rustymimi`, `torchaudio`, or upstream Moshi code in the production package.

## Status

- MLX-native encode and decode are implemented for the official `kyutai/mimi`
  checkpoint.
- The committed fixture set has exact encode token parity against
  `transformers.MimiModel`.
- Decode output matches the upstream waveform within the numerical tolerance
  documented in `docs/parity.md`.
- Same-length batches are supported directly. Variable-length padded batches
  must pass explicit `lengths`; otherwise padded samples are treated as real
  input.

## Install

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install reference-only dependencies only when exporting fixtures or comparing
against PyTorch or `rustymimi`:

```bash
python -m pip install -e ".[dev,reference]"
```

## Model assets

The official Mimi weights are not committed. Download them into the ignored
`fixtures/reference/hf/` directory:

```bash
python scripts/download_reference_assets.py
```

Equivalent Hugging Face CLI command:

```bash
HF_HUB_DISABLE_XET=1 hf download kyutai/mimi \
  --revision 89091b3e466eb6a9d11e537bf26b144f194978f7 \
  --local-dir fixtures/reference/hf \
  --include config.json \
  --include preprocessor_config.json \
  --include model.safetensors
```

Validate the downloaded safetensors header:

```bash
python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors
```

## CLI quickstart

Encode a mono WAV file to canonical Mimi tokens (`[batch, frames, codebooks]`):

```bash
mimi-mlx encode fixtures/audio/sine_440_025s.wav \
  --weights fixtures/reference/hf \
  --output /tmp/sine_440_tokens.npy \
  --json
```

Decode the tokens back to audio:

```bash
mimi-mlx decode /tmp/sine_440_tokens.npy \
  --weights fixtures/reference/hf \
  --output /tmp/sine_440_recon.wav \
  --json
```

Encode a WAV directory with bounded CPU prefetch and MLX-native token saves:

```bash
mimi-mlx encode-dir fixtures/audio \
  --weights fixtures/reference/hf \
  --output-dir /tmp/mimi_tokens \
  --prefetch-workers 2 \
  --json
```

Run a reference parity check:

```bash
mimi-mlx parity fixtures/audio/sine_440_025s.wav \
  --reference transformers \
  --weights fixtures/reference/hf \
  --json
```

## Python API quickstart

```python
import mlx.core as mx
from mimi_mlx import MimiTokenizer

tokenizer = MimiTokenizer.from_pretrained("fixtures/reference/hf")
audio = mx.zeros((24_000,))

tokens = tokenizer.encode(audio, sample_rate=24_000)
reconstructed = tokenizer.decode(tokens.codes, sample_rate=24_000)
```

`tokens.codes` uses the project canonical layout `[batch, frames, codebooks]`.
Use `mimi_mlx.layouts.to_upstream_layout` for upstream
`[batch, codebooks, frames]`.

## Verify the checkout

```bash
pytest -q
ruff check .
python -m mimi_mlx.cli --help
```

Once local weights are present, run the parity fixture workflow:

```bash
python scripts/export_reference_fixtures.py --weights fixtures/reference/hf
pytest -q
python scripts/compare_reference.py fixtures/audio/sine_440_025s.wav \
  --weights fixtures/reference/hf --json
```

For Rust parity, `rustymimi` requires a Moshi tokenizer checkpoint rather than
the Hugging Face `kyutai/mimi/model.safetensors` file:

```bash
RUSTYMIMI_WEIGHTS="$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    "kyutai/moshika-mlx-bf16",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
))
PY
)"

mimi-mlx parity fixtures/audio/sine_440_025s.wav \
  --reference rustymimi \
  --weights fixtures/reference/hf \
  --reference-weights "$RUSTYMIMI_WEIGHTS" \
  --json
```

## Documentation

| Document | Purpose |
| --- | --- |
| `docs/usage.md` | Install modes, model assets, CLI commands, Python API, token layout, batching, and troubleshooting. |
| `docs/development.md` | Contributor setup, validation commands, fixture regeneration, parity workflow, and release checks. |
| `docs/architecture.md` | Mimi architecture facts, implementation boundary, and parity risks. |
| `docs/weights.md` | Official checkpoint details, tensor families, and MLX weight mapping notes. |
| `docs/parity.md` | Current parity status, verification commands, and fixture coverage. |
| `docs/benchmarks.md` | Benchmark entrypoints and current local smoke results. |
| `fixtures/README.md` | Committed fixture manifest, source audio notes, and regeneration command. |

## Non-goals

- No production wrapper around `rustymimi`, PyTorch, `torchaudio`, or upstream
  Moshi code.
- No approximate token parity acceptance for encode paths.
- No hidden token layout conversion. Public APIs use explicit canonical and
  upstream layout helpers.
