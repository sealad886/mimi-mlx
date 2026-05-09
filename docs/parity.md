# Parity Progress

## 2026-05-09

Stages 0 through 8 are implemented for the current local fixture set.

- Exact encode token parity: passing for all committed fixtures.
- Decode waveform parity: passing for all committed fixtures with `max_abs < 2e-5` and `MSE < 1e-10`.
- Batch parity: passing for same-length batch encode and variable-length encode with explicit `lengths`.
- Padded variable-length inputs without `lengths` remain caller-unsafe and are documented as requiring explicit lengths.
- Current implemented contracts: config loading, token layout conversion, token `.npz`/`.npy` IO, audio shape normalization, MLX linear resampling, official weight loading, MLX encoder/decoder/quantizer, CLI encode/decode/parity, and benchmark entrypoints.

Verification:

```text
.venv/bin/pytest -q                    # 43 passed
.venv/bin/ruff check .                 # all checks passed
.venv/bin/python -m mimi_mlx.cli --help # pass
```

Files changed:

- Package: `src/mimi_mlx/`
- Tests: `tests/`
- Fixtures: `fixtures/audio/*.wav`, `fixtures/reference/*.npy`, `fixtures/reference/manifest.json`
- Scripts: `scripts/`
- Docs: `README.md`, `docs/`, `fixtures/README.md`
- Packaging: `pyproject.toml`, `LICENSE`

Parity fixtures:

- `silence_025s`
- `sine_440_025s`
- `sine_440_100s`
- `impulse_center_025s`
- `noise_low_025s`
- `clipped_025s`
- `odd_length_6017`
- `synthetic_speech_like_050s`
- `real_speech_librispeech_100s`

Root-cause note:

- Initial encode mismatch was caused by treating transformer `q_proj/k_proj/v_proj/o_proj`
  weights as Conv1d projection weights. Restricting transposition to actual Conv1d and
  ConvTranspose1d modules restored exact token parity.
