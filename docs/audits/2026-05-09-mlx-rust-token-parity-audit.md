# MLX/Rust Token Parity Audit

Date: 2026-05-09.

## Scope

Audit target:

- Full `src/mimi_mlx/` MLX encode/decode path.
- Token layout and token IO contracts.
- CLI parity and benchmark entrypoints.
- Rust reference token parity through `rustymimi`.

Out of scope:

- Training.
- Streaming stateful parity.
- Peak MLX allocator measurement, because this repo does not yet have a stable
  per-run memory counter wired into benchmarks.

## Evidence Index

Authoritative external contracts:

- MLX `Conv1d` and `ConvTranspose1d` expect channel-last `NLC` input.
- MLX operations are lazy; proof commands must force evaluation with
  `mx.eval`, NumPy conversion, or scalar extraction.
- Hugging Face Mimi `encode` accepts `[batch, channels, sequence_length]`
  audio and `decode` consumes `[batch, num_quantizers, codes_length]` codes.
- `rustymimi 0.4.1` is the Rust/Python reference package from Kyutai Moshi.

Source links:

- https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv1d.html
- https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html
- https://ml-explore.github.io/mlx/build/html/usage/quick_start.html
- https://huggingface.co/docs/transformers/model_doc/mimi
- https://pypi.org/project/rustymimi/

Repository evidence:

- MLX wrappers keep public/internal audio as `[B,C,T]` and swap to MLX `NLC`
  at convolution boundaries in `src/mimi_mlx/modules.py`.
- Public tokens are `[B,T,K]`; upstream/reference tokens are `[B,K,T]` in
  `src/mimi_mlx/layouts.py`.
- Fixtures cover silence, sine, clipped audio, impulse, noise, odd length,
  synthetic speech-like, and real LibriSpeech-derived speech.

## Baseline

Initial environment defect:

- `.venv` had stale editable metadata pointing at `/Users/andrew/Documents/New project 3`.
- `import mimi_mlx` failed before reinstalling the current repo into `.venv`.
- `.venv/bin/pytest` also pointed at the stale venv path and failed.

Environment remediation:

- Reinstalled current repo with `.venv/bin/python -m pip install -e ".[dev,reference]"`.
- Reinstalled `pytest>=9.0` inside `.venv` to repair the direct console script.

Baseline after environment remediation:

```text
.venv/bin/pytest -q                 # 45 passed before code changes
.venv/bin/python -m ruff check .    # all checks passed
```

## Findings

### F1: Rust token parity was advertised but not executable

Severity: high.

Affected paths:

- `src/mimi_mlx/cli.py`
- `README.md`
- `docs/parity.md`

Evidence:

- CLI accepted `--reference rustymimi`.
- Runtime exited with `rustymimi parity is reference tooling only and is not wired in this CLI yet`.
- `rustymimi.Tokenizer` rejected `fixtures/reference/hf/model.safetensors` because it expects Moshi tokenizer tensor names such as `encoder.model.0.conv.conv.weight_g`.

Root cause:

- The parity command only implemented the Transformers backend and did not
  model the separate Rust tokenizer checkpoint requirement.

Fix:

- Added `--reference-weights` for Rust parity.
- Added `MIMI_RUSTYMIMI_WEIGHTS` fallback.
- Added Rust reference code path using `rustymimi.Tokenizer(..., num_codebooks=32)`.
- Compared Rust `[B,K,T]` output directly against MLX output converted with
  `to_upstream_layout`.

Proof:

```text
.venv/bin/python -m mimi_mlx.cli parity fixtures/audio/sine_440_025s.wav \
  --reference rustymimi \
  --weights fixtures/reference/hf \
  --reference-weights /Users/andrew/.cache/huggingface/hub/models--kyutai--moshika-mlx-bf16/snapshots/03c68ab434ed33ae0716d38b0ee237069477066c/tokenizer-e351c8d8-checkpoint125.safetensors \
  --json
# "ok": true, "codes_shape": [1, 32, 4]
```

Full fixture sweep:

```text
silence_025s (1, 32, 4) ok
sine_440_025s (1, 32, 4) ok
sine_440_100s (1, 32, 13) ok
impulse_center_025s (1, 32, 4) ok
noise_low_025s (1, 32, 4) ok
clipped_025s (1, 32, 4) ok
odd_length_6017 (1, 32, 4) ok
synthetic_speech_like_050s (1, 32, 7) ok
real_speech_librispeech_100s (1, 32, 13) ok
failures 0
```

Status: fixed.

### F2: CLI batching benchmark did not run batching

Severity: medium.

Affected paths:

- `src/mimi_mlx/cli.py`
- `docs/benchmarks.md`

Evidence:

```text
.venv/bin/python -m mimi_mlx.cli benchmark batching \
  --weights fixtures/reference/hf --input-dir fixtures/audio \
  --batch-sizes 1,2 --json
```

The payload was a scalar encode/decode-style summary and omitted per-batch
`results`, meaning `--batch-sizes` was ignored.

Root cause:

- `_benchmark_command` special-cased only `decode`; `batching` fell through to
  single-clip encode logic.

Fix:

- Added real `benchmark batching` handling with parsed positive batch sizes,
  same-rate clip validation, `encode_batch`, and `mx.eval(tokens.codes)`.

Proof:

```text
.venv/bin/pytest tests/test_cli.py -q   # 5 passed
```

Status: fixed.

### F3: Rust parity checkpoint contract was undocumented

Severity: medium.

Affected paths:

- `README.md`
- `docs/architecture.md`
- `docs/parity.md`

Evidence:

- Existing docs described `rustymimi` as a reference dependency but gave no
  command or checkpoint source.
- `rustymimi` cannot load the HF `kyutai/mimi` checkpoint directly.

Fix:

- Documented download/use of Kyutai Moshi
  `tokenizer-e351c8d8-checkpoint125.safetensors`.
- Documented that Rust parity is reference-only and full 32-codebook.

Status: fixed.

## Residual Risks

- Streaming encode/decode parity is not yet covered. Current audit proves
  stateless full-clip token parity.
- Peak memory is not measured. Benchmark docs keep this as a known follow-up.
- The Rust checkpoint is downloaded into the Hugging Face cache for local proof;
  it is not committed to git.

## Final Verification

```text
.venv/bin/pytest -q
# 48 passed in 5.31s

.venv/bin/ruff check .
# All checks passed!

.venv/bin/python -m mimi_mlx.cli parity fixtures/audio/sine_440_025s.wav \
  --reference rustymimi \
  --weights fixtures/reference/hf \
  --reference-weights /Users/andrew/.cache/huggingface/hub/models--kyutai--moshika-mlx-bf16/snapshots/03c68ab434ed33ae0716d38b0ee237069477066c/tokenizer-e351c8d8-checkpoint125.safetensors \
  --json
# "ok": true, "codes_shape": [1, 32, 4]

.venv/bin/python -m mimi_mlx.cli benchmark batching \
  --weights fixtures/reference/hf --input-dir fixtures/audio \
  --batch-sizes 1,2 --json
# returns per-batch results for batch_size 1 and 2
```
