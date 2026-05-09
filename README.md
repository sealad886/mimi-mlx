# mimi-mlx

`mimi-mlx` is a correctness-first MLX-native port of Kyutai Mimi for Apple Silicon.

Current status: Stage 0/1 scaffold plus core public contracts. Full encoder, decoder,
official weight loading, exact token parity fixtures, and benchmarks are not complete yet.
Production encode/decode paths intentionally raise `NotImplementedError` until those stages
land with parity evidence.

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

## Smoke Checks

```bash
pytest -q
ruff check .
python -m mimi_mlx.cli --help
```

## Non-Goals

- No production wrapper around `rustymimi`, PyTorch, torchaudio, or upstream Moshi code.
- No approximate token parity acceptance.
- No silent padded batching; variable-length batch encode requires explicit lengths.
