# Development Guide

This guide documents the local workflow for changing `mimi-mlx` safely.

## Environment setup

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install reference-only dependencies when you need parity backends or fixture
export:

```bash
python -m pip install -e ".[dev,reference]"
```

The `reference` extra is intentionally separate from the production dependency
set. Production code under `src/mimi_mlx/` should remain MLX-native and should
not import PyTorch, `rustymimi`, `torchaudio`, or upstream Moshi modules.

## Local validation

Run the core checks before committing:

```bash
pytest -q
ruff check .
python -m mimi_mlx.cli --help
```

When local weights are present, also run a parity spot check:

```bash
python scripts/compare_reference.py fixtures/audio/sine_440_025s.wav \
  --weights fixtures/reference/hf --json
```

For CLI changes, verify every command surface still parses:

```bash
python -m mimi_mlx.cli encode --help
python -m mimi_mlx.cli decode --help
python -m mimi_mlx.cli parity --help
python -m mimi_mlx.cli benchmark --help
```

## Reference assets

Download the official Hugging Face assets into the ignored local directory:

```bash
python scripts/download_reference_assets.py
```

Validate the safetensors header:

```bash
python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors
```

The expected checkpoint is documented in `docs/weights.md`. Do not commit the
downloaded `fixtures/reference/hf/` directory.

## Fixture workflow

Committed fixtures live under:

- `fixtures/audio/`
- `fixtures/reference/*.npy`
- `fixtures/reference/manifest.json`

Regenerate reference fixtures after changing fixture audio, the reference
backend, or parity export logic:

```bash
python scripts/export_reference_fixtures.py --weights fixtures/reference/hf
```

Then run:

```bash
pytest -q
```

The tests skip weight-dependent checks when local weights are absent, so run the
full weighted workflow before changing parity-sensitive code.

## Parity workflow

Transformers parity uses the Hugging Face `kyutai/mimi` checkpoint:

```bash
python -m mimi_mlx.cli parity fixtures/audio/sine_440_025s.wav \
  --reference transformers \
  --weights fixtures/reference/hf \
  --json
```

Rust parity uses a separate Moshi tokenizer checkpoint:

```bash
RUSTYMIMI_WEIGHTS="$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    "kyutai/moshika-mlx-bf16",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
))
PY
)"

python -m mimi_mlx.cli parity fixtures/audio/sine_440_025s.wav \
  --reference rustymimi \
  --weights fixtures/reference/hf \
  --reference-weights "$RUSTYMIMI_WEIGHTS" \
  --json
```

Use exact token parity for encode changes. Do not replace token parity with an
approximate metric. Decode waveform comparisons use the tolerance documented in
`docs/parity.md`.

## Benchmark workflow

Run benchmark entrypoints against the committed fixture WAVs:

```bash
python scripts/benchmark_encode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
python scripts/benchmark_decode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
python scripts/benchmark_batching.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
```

If a change intentionally affects performance, update `docs/benchmarks.md` with
the command, hardware context if known, and the new result table.

## Documentation expectations

Update docs in the same change when behavior changes:

| Change | Documentation to check |
| --- | --- |
| CLI arguments or JSON output | `README.md`, `docs/usage.md`, and CLI tests. |
| Public Python API or token metadata | `README.md`, `docs/usage.md`, and relevant tests. |
| Weight mapping or checkpoint assumptions | `docs/weights.md` and `docs/architecture.md`. |
| Parity status, fixtures, or tolerances | `docs/parity.md` and `fixtures/README.md`. |
| Benchmark output or methodology | `docs/benchmarks.md`. |

Keep docs command examples runnable from the repository root.
