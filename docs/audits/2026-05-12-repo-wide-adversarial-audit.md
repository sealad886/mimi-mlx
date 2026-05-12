# Repo-Wide Adversarial Audit

Date: 2026-05-12.

Scope: correctness-first parity and codebase audit for `mimi-mlx`, with
remediation of accepted in-scope findings.

## Evidence Index

Repository state at start:

- Working directory: `/Users/andrew/zRepos/mimi-mlx`.
- Branch: `main`.
- Initial dirty state: clean.
- Python: `.venv/bin/python` reported Python 3.12.13.
- Package metadata: `.venv/bin/python -m pip show mimi-mlx` reported editable
  install at `/Users/andrew/zRepos/mimi-mlx`, version 0.3.0 before fixes.
- Local HF reference assets: `fixtures/reference/hf/config.json`,
  `fixtures/reference/hf/model.safetensors`, and
  `fixtures/reference/hf/preprocessor_config.json` were present.
- Rust reference checkpoint at baseline: `MIMI_RUSTYMIMI_WEIGHTS` was unset and
  `tokenizer-e351c8d8-checkpoint125.safetensors` was not found in the local
  Hugging Face cache. This was later resolved by downloading the checkpoint
  from `kyutai/moshika-mlx-bf16` revision
  `03c68ab434ed33ae0716d38b0ee237069477066c`.

Authoritative external contracts checked:

- MLX `Conv1d` and `ConvTranspose1d` expect channel-last `NLC` tensors:
  https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv1d.html
  and
  https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.ConvTranspose1d.html.
- MLX is lazy and benchmark/proof paths must force evaluation:
  https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html.
- MLX exposes peak memory counters:
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html
  and
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.reset_peak_memory.html.
- Transformers Mimi uses audio `[batch, channels, sequence_length]` and code
  layout `[batch, num_quantizers, codes_length]`:
  https://huggingface.co/docs/transformers/model_doc/mimi.
- Canonical `kyutai/mimi` config and weights were checked at pinned revision
  `89091b3e466eb6a9d11e537bf26b144f194978f7`.
- PyPA `build` is the canonical frontend for `python -m build`:
  https://build.pypa.io/en/latest/index.html.

## Shard Dependency Graph

```text
Shard 0 baseline
  -> Shards 1,2,3,5,6 first-pass correctness audits
  -> Shard 4 exact reference parity after HF/Rust weight availability check
  -> Shard 7 benchmarks after correctness checks were green
  -> Shard 8 docs, packaging, and release readiness after behavior was known
```

## Baseline Validation

| Command | Outcome |
| --- | --- |
| `pwd` | `/Users/andrew/zRepos/mimi-mlx` |
| `git status --short` | clean at start |
| `.venv/bin/python -m pip show mimi-mlx` | version 0.3.0, editable in this repo |
| `.venv/bin/pytest -q` | `64 passed in 5.40s` |
| `.venv/bin/ruff check .` | `All checks passed!` |
| `.venv/bin/python -m mimi_mlx.cli --help` | succeeded |
| `.venv/bin/python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors` | `350 tensors`, `96151393 parameters`, `20 required tensors present` |

## Finding Ledger

### MIMI-AUD-001: Partial checkpoints loaded with zero-initialized missing parameters

- Severity: high.
- Status: fixed.
- Affected files: `src/mimi_mlx/model.py`, `tests/test_weights.py`.
- Evidence: `MimiModel.load_hf_weights()` assigned tensors present in the
  state dict but did not assert that every model parameter had been assigned.
- Root cause hypothesis: the loader validated unknown tensors but not absent
  expected tensors after object traversal.
- Smallest safe fix: collect expected public `mx.array` parameter names from
  the instantiated MLX model and raise `WeightLoadError` for absent tensors.
- Proof command: `.venv/bin/pytest tests/test_weights.py -q`.
- Residual risk: none known for static full-checkpoint loading.

### MIMI-AUD-002: Canonical remote assets were not pinned to the documented Mimi revision

- Severity: medium.
- Status: fixed.
- Affected files: `src/mimi_mlx/config.py`,
  `scripts/download_reference_assets.py`,
  `scripts/export_reference_fixtures.py`, `tests/test_config.py`,
  `tests/test_export_reference_fixtures.py`, `docs/weights.md`.
- Evidence: `MimiCodecConfig.from_pretrained("kyutai/mimi")` and reference
  fixture export could use moving remote defaults when no revision was passed.
- Root cause hypothesis: local fixtures were revision-specific but remote
  helper defaults did not encode that contract.
- Smallest safe fix: add `DEFAULT_MIMI_REVISION` and apply it only for the
  canonical `kyutai/mimi` repo when the caller did not pass an explicit
  revision.
- Proof command: `.venv/bin/pytest tests/test_config.py tests/test_export_reference_fixtures.py -q`.
- Residual risk: non-canonical model ids remain caller-controlled by design.

### MIMI-AUD-003: Public decode accepted upstream-layout token arrays as canonical

- Severity: high.
- Status: fixed.
- Affected files: `src/mimi_mlx/tokenizer.py`,
  `tests/test_decoder_shapes.py`, `tests/test_layouts.py`.
- Evidence: rank-3 `[B,K,T]` arrays could enter `MimiTokenizer.decode()` and
  then be transposed as if they were public `[B,T,K]` tokens.
- Root cause hypothesis: the public API checked rank but not the canonical
  codebook axis.
- Smallest safe fix: add a canonical token validator requiring the final axis
  to equal the codebook count and explicitly directing upstream-layout callers
  to `from_upstream_layout`.
- Proof command: `.venv/bin/pytest tests/test_decoder_shapes.py tests/test_layouts.py -q`.
- Residual risk: archive validation currently uses the public 32-codebook Mimi
  contract for saved tokens.

### MIMI-AUD-004: Extra or empty codebooks could be silently decoded

- Severity: high.
- Status: fixed.
- Affected files: `src/mimi_mlx/quantizer.py`, `tests/test_quantizer.py`.
- Evidence: RVQ decode loops over available layers and could ignore surplus
  codebooks or fail indirectly for empty codebook inputs.
- Root cause hypothesis: encode paths always generate valid counts, so decode
  bounds were not enforced at the boundary.
- Smallest safe fix: validate rank and codebook-count bounds in residual and
  split residual quantizer decode.
- Proof command: `.venv/bin/pytest tests/test_quantizer.py -q`.
- Residual risk: none known for full and prefix codebook decode.

### MIMI-AUD-005: Variable-length batch `lengths` accepted non-integer values

- Severity: medium.
- Status: fixed.
- Affected files: `src/mimi_mlx/tokenizer.py`, `tests/test_batch_parity.py`.
- Evidence: host conversion followed by `int(length)` could truncate floating
  values and alter the intended audio prefix.
- Root cause hypothesis: shape validation existed but dtype validation did not.
- Smallest safe fix: reject boolean and non-integer `lengths` dtypes before
  grouping samples by explicit prefix length.
- Proof command: `.venv/bin/pytest tests/test_batch_parity.py -q`.
- Residual risk: caller-provided integer values still need to represent real
  sample counts, which is the documented contract.

### MIMI-AUD-006: Rust reference could be pointed at the wrong HF checkpoint file

- Severity: high.
- Status: fixed.
- Affected files: `src/mimi_mlx/cli.py`, `tests/test_cli.py`,
  `tests/test_reference_parity.py`, `docs/parity.md`.
- Evidence: a direct explicit file path such as `fixtures/reference/hf/model.safetensors`
  could pass path existence checks even though `rustymimi` requires a Moshi
  tokenizer checkpoint named `tokenizer-*.safetensors`.
- Root cause hypothesis: directory lookup enforced the tokenizer filename
  pattern but explicit file lookup did not.
- Smallest safe fix: require explicit Rust reference weight files to match
  `tokenizer-*.safetensors`; add an integration test that only runs when
  `MIMI_RUSTYMIMI_WEIGHTS` is available.
- Proof command: `.venv/bin/pytest tests/test_cli.py tests/test_reference_parity.py -q`.
- Additional proof: exact Rust token parity passed for every committed WAV
  fixture after downloading
  `tokenizer-e351c8d8-checkpoint125.safetensors` from `kyutai/moshika-mlx-bf16`.
- Residual risk: none known for stateless full-clip fixture parity.

### MIMI-AUD-007: Text-mode parity failures hid mismatch details

- Severity: medium.
- Status: fixed.
- Affected files: `src/mimi_mlx/cli.py`, `tests/test_cli.py`.
- Evidence: JSON parity output included mismatch detail but text output only
  printed key-value pairs, making first mismatch location less actionable.
- Root cause hypothesis: `_emit()` did not special-case mismatch payloads.
- Smallest safe fix: print mismatch location and values in text mode when
  present.
- Proof command: `.venv/bin/pytest tests/test_cli.py -q`.
- Residual risk: none known.

### MIMI-AUD-008: Benchmarks included first-use lazy materialization and omitted peak memory

- Severity: medium.
- Status: fixed.
- Affected files: `src/mimi_mlx/cli.py`, `scripts/benchmark_batching.py`,
  `tests/test_cli.py`, `docs/benchmarks.md`.
- Evidence: MLX is lazy; encode/decode/batching timing paths needed explicit
  warmup/evaluation boundaries. Docs also treated peak memory as unavailable,
  but current MLX exposes peak memory counters.
- Root cause hypothesis: benchmark code predated use of MLX peak memory APIs
  and did not separate setup from timed decode work.
- Smallest safe fix: add unmeasured warmups, force `mx.eval`, reset peak memory
  before timed loops, report `peak_memory_bytes`, and route the batching script
  through the CLI implementation.
- Proof command: `.venv/bin/pytest tests/test_cli.py tests/test_mlx_optimizations.py -q`.
- Residual risk: peak memory is a process-level MLX allocator counter, not a
  full system RSS measurement.

### MIMI-AUD-009: Fixture export could silently drop the real-speech fixture

- Severity: medium.
- Status: fixed.
- Affected files: `scripts/export_reference_fixtures.py`,
  `tests/test_export_reference_fixtures.py`, `fixtures/README.md`.
- Evidence: a missing ignored LibriSpeech source parquet could produce a
  synthetic-only regenerated fixture set without making the provenance loss
  explicit.
- Root cause hypothesis: source fixture regeneration optimized for local
  convenience over fixture contract strictness.
- Smallest safe fix: fail by default when the speech source is missing and
  require `--allow-missing-speech-source` for a synthetic-only fixture set.
- Proof command: `.venv/bin/pytest tests/test_export_reference_fixtures.py -q`.
- Residual risk: real source parquet remains uncommitted by design.

### MIMI-AUD-010: Packaging readiness was not reproducible from the checked-in dev extra

- Severity: low.
- Status: fixed.
- Affected files: `pyproject.toml`, `.gitignore`, `docs/development.md`.
- Evidence: `python -m build --wheel` initially depended on missing `build`
  tooling in `.venv`; local build artifacts were also not repo-explicit ignores.
- Root cause hypothesis: release checks were documented informally but not
  encoded in the dev dependency set.
- Smallest safe fix: add `build>=1.3` to the `dev` extra, ignore build
  artifacts, document release readiness checks, and bump package version to
  `0.3.1` for the patch remediation.
- Proof command: `.venv/bin/python -m build --wheel`.
- Residual risk: publishing/signing is intentionally outside this audit.

## Blocked Or Residual Items

- Exact Rust token parity is no longer blocked locally. The Moshi tokenizer
  checkpoint was downloaded to:
  `/Users/andrew/.cache/huggingface/hub/models--kyutai--moshika-mlx-bf16/snapshots/03c68ab434ed33ae0716d38b0ee237069477066c/tokenizer-e351c8d8-checkpoint125.safetensors`.

- Benchmark peak memory now reports MLX allocator peak memory, not total
  process resident memory.

## Final Verification Summary

The following commands were run after remediation, from the repo `.venv`:

```text
.venv/bin/python -m pip install -e ".[dev,reference]"
# Successfully installed mimi-mlx-0.3.1

.venv/bin/python -m pip show mimi-mlx
# Version: 0.3.1
# Editable project location: /Users/andrew/zRepos/mimi-mlx

.venv/bin/pytest -q
# 82 passed, 1 skipped in 5.70s

.venv/bin/ruff check .
# All checks passed!

.venv/bin/python -m mimi_mlx.cli --help
# succeeded

.venv/bin/python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors
# 350 tensors, 96151393 parameters, 20 required tensors present

.venv/bin/python scripts/compare_reference.py fixtures/audio/sine_440_025s.wav \
  --weights fixtures/reference/hf --json
# "ok": true, "codes_shape": [1, 32, 4]

env -u MIMI_RUSTYMIMI_WEIGHTS .venv/bin/python -m mimi_mlx.cli parity \
  fixtures/audio/sine_440_025s.wav --reference rustymimi \
  --weights fixtures/reference/hf --json
# blocked as expected:
# rustymimi parity requires --reference-weights pointing at tokenizer-*.safetensors

.venv/bin/python -m mimi_mlx.cli parity fixtures/audio/sine_440_025s.wav \
  --reference rustymimi --weights fixtures/reference/hf \
  --reference-weights /Users/andrew/.cache/huggingface/hub/models--kyutai--moshika-mlx-bf16/snapshots/03c68ab434ed33ae0716d38b0ee237069477066c/tokenizer-e351c8d8-checkpoint125.safetensors \
  --json
# "ok": true, "codes_shape": [1, 32, 4]

for wav in fixtures/audio/*.wav; do
  .venv/bin/python -m mimi_mlx.cli parity "$wav" --reference rustymimi \
    --weights fixtures/reference/hf \
    --reference-weights /Users/andrew/.cache/huggingface/hub/models--kyutai--moshika-mlx-bf16/snapshots/03c68ab434ed33ae0716d38b0ee237069477066c/tokenizer-e351c8d8-checkpoint125.safetensors \
    --json
done
# all 9 committed WAV fixtures returned "ok": true

.venv/bin/python scripts/benchmark_encode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
# "clips": 9, "audio_seconds": 4.000708333333334, "peak_memory_bytes": 514681352

.venv/bin/python scripts/benchmark_decode.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --prefetch-workers 2 --json
# "clips": 9, "audio_seconds": 4.000708333333334, "peak_memory_bytes": 898731388

.venv/bin/python scripts/benchmark_batching.py --weights fixtures/reference/hf \
  --input-dir fixtures/audio --batch-sizes 1,2,4,8 --json
# returned per-batch results including peak_memory_bytes for all requested batch sizes

.venv/bin/python -m build --wheel
# Successfully built mimi_mlx-0.3.1-py3-none-any.whl
```
