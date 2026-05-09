# Parity Progress

## 2026-05-09

Stage 0/1 scaffold complete for public helper contracts.

- Exact encode token parity: blocked. Full MLX encoder and official weight mapping are not implemented yet.
- Decode waveform parity: blocked. Full MLX decoder and official weight mapping are not implemented yet.
- Batch parity: blocked. HF source notes encoder padding-mask support is incomplete; `mimi_mlx` rejects padded batch encode without explicit lengths until proven safe.
- Current implemented contracts: config loading, token layout conversion, token `.npz`/`.npy` IO, audio shape normalization, MLX linear resampling, toy quantizer unit behavior, CLI command surface.

Verification:

```text
python -m pip install -e ".[dev]"  # pass, local .venv Python 3.12
pytest -q                         # 20 passed
ruff check .                      # all checks passed
python -m mimi_mlx.cli --help     # pass
```

Files changed:

- Package: `src/mimi_mlx/`
- Tests: `tests/`
- Scripts: `scripts/`
- Docs: `README.md`, `docs/`, `fixtures/README.md`
- Packaging: `pyproject.toml`, `LICENSE`

Next blocker: Stage 2 weight-name mapping from HF `model.safetensors` to standalone MLX module names.
