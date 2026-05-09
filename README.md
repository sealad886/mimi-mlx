# mimi-mlx

`mimi-mlx` is a correctness-first MLX-native port of Kyutai Mimi for Apple Silicon.

Current status: MLX-native encode/decode is implemented against the Hugging Face
`kyutai/mimi` checkpoint. The committed fixture set has exact encode token parity
against `transformers.MimiModel`; decode matches the upstream waveform within the
documented numerical tolerance.

## Install

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Reference tooling is isolated:

```bash
python -m pip install -e ".[dev,reference]"
```

Download the official Mimi assets with the Hugging Face CLI:

```bash
HF_HUB_DISABLE_XET=1 hf download kyutai/mimi \
  --revision 89091b3e466eb6a9d11e537bf26b144f194978f7 \
  --local-dir fixtures/reference/hf \
  --include config.json \
  --include preprocessor_config.json \
  --include model.safetensors
```

`fixtures/reference/hf/` is intentionally ignored by git.

## Smoke Checks

```bash
pytest -q
ruff check .
python -m mimi_mlx.cli --help
```

Local full parity verification, once weights are present:

```bash
python scripts/export_reference_fixtures.py --weights fixtures/reference/hf
pytest -q
python scripts/compare_reference.py fixtures/audio/sine_440_025s.wav \
  --weights fixtures/reference/hf --json
```

## API

```python
import mlx.core as mx
from mimi_mlx import MimiTokenizer

tokenizer = MimiTokenizer.from_pretrained("fixtures/reference/hf")
audio = mx.zeros((24_000,))
tokens = tokenizer.encode(audio, sample_rate=24_000)
reconstructed = tokenizer.decode(tokens.codes, sample_rate=24_000)
```

`tokens.codes` uses `[batch, frames, codebooks]`. Use
`mimi_mlx.layouts.to_upstream_layout` for upstream `[batch, codebooks, frames]`.

## Non-Goals

- No production wrapper around `rustymimi`, PyTorch, torchaudio, or upstream Moshi code.
- No approximate token parity acceptance.
- Same-length batches are supported directly. Variable-length padded batches are treated as
  full-length unless explicit `lengths` are passed, so downstream callers must pass `lengths`
  for padded inputs.
