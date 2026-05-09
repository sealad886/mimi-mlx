# Stage 0-1 Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the initial importable `mimi_mlx` package with researched architecture notes, config/layout/audio/token IO helpers, CLI help, and focused tests.

**Architecture:** Keep production code MLX-native and standalone. Use upstream Kyutai/Hugging Face only for research, fixture generation, and reference scripts. Expose canonical public token layout as `[batch, frames, codebooks]`, with explicit helpers for upstream `[batch, codebooks, frames]`.

**Tech Stack:** Python 3.12, MLX, NumPy, pytest, ruff, Hugging Face Hub for explicit downloads.

---

### Task 1: Package Metadata And Docs Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `LICENSE`
- Create: `docs/architecture.md`
- Create: `docs/weights.md`
- Create: `docs/parity.md`
- Create: `docs/benchmarks.md`
- Create: `fixtures/README.md`

- [ ] **Step 1: Write package metadata and docs from current research**

Create `pyproject.toml` with Python `>=3.11,<3.14`, runtime deps `mlx`, `numpy`, `soundfile`, `huggingface_hub`, dev deps `pytest`, `ruff`, and reference deps `transformers`, `torch`, `safetensors`, `datasets`, `rustymimi`.

- [ ] **Step 2: Verify metadata can install**

Run: `. .venv/bin/activate && python -m pip install -e ".[dev]"`

Expected: editable install succeeds without PyTorch.

### Task 2: Config Contract

**Files:**
- Create: `src/mimi_mlx/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Test `MimiCodecConfig.from_pretrained` with a local `config.json`, defaults from `kyutai/mimi`, and `hop_length == 1920`.

- [ ] **Step 2: Run test to verify failure**

Run: `. .venv/bin/activate && pytest tests/test_config.py -q`

Expected: FAIL because `mimi_mlx` does not exist yet.

- [ ] **Step 3: Implement minimal config loader**

Support local file, local directory, and Hugging Face repo id through explicit `hf_hub_download`.

- [ ] **Step 4: Run focused test**

Run: `. .venv/bin/activate && pytest tests/test_config.py -q`

Expected: PASS.

### Task 3: Layouts And Token IO

**Files:**
- Create: `src/mimi_mlx/layouts.py`
- Create: `src/mimi_mlx/tokenizer.py`
- Create: `tests/test_layouts.py`

- [ ] **Step 1: Write failing layout tests**

Test canonical `[B,T,K]` to upstream `[B,K,T]`, reverse conversion, invalid layout rejection, `.npz` token round-trip, and `.npy` code round-trip.

- [ ] **Step 2: Run test to verify failure**

Run: `. .venv/bin/activate && pytest tests/test_layouts.py -q`

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement minimal layout and IO helpers**

Use MLX arrays at public boundary, NumPy only for serialization.

- [ ] **Step 4: Run focused test**

Run: `. .venv/bin/activate && pytest tests/test_layouts.py -q`

Expected: PASS.

### Task 4: Audio Helpers

**Files:**
- Create: `src/mimi_mlx/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing audio tests**

Test `[T]`, `[B,T]`, `[B,C,T]` normalization, mono-only rejection, length output, and deterministic 48 kHz to 24 kHz linear resampling shape.

- [ ] **Step 2: Run test to verify failure**

Run: `. .venv/bin/activate && pytest tests/test_audio.py -q`

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement minimal audio helpers**

Return internal audio shape `[B,C,T]`. Keep resampling MLX-native.

- [ ] **Step 4: Run focused test**

Run: `. .venv/bin/activate && pytest tests/test_audio.py -q`

Expected: PASS.

### Task 5: Quantizer Unit

**Files:**
- Create: `src/mimi_mlx/quantizer.py`
- Create: `tests/test_quantizer.py`

- [ ] **Step 1: Write failing quantizer tests**

Test exact nearest centroid, residual ordering, split semantic/acoustic ordering, and code range.

- [ ] **Step 2: Run test to verify failure**

Run: `. .venv/bin/activate && pytest tests/test_quantizer.py -q`

Expected: FAIL because quantizer classes do not exist.

- [ ] **Step 3: Implement MLX-native quantizer**

Use squared-distance equivalent `c2 - x @ embedding.T` like upstream Rust/MLX. No PyTorch or rustymimi imports.

- [ ] **Step 4: Run focused test**

Run: `. .venv/bin/activate && pytest tests/test_quantizer.py -q`

Expected: PASS.

### Task 6: CLI And Placeholders

**Files:**
- Create: `src/mimi_mlx/__init__.py`
- Create: `src/mimi_mlx/cli.py`
- Create: `src/mimi_mlx/model.py`
- Create: `src/mimi_mlx/modules.py`
- Create: `src/mimi_mlx/weights.py`
- Create: `src/mimi_mlx/parity.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_weights.py`
- Create: `tests/test_encoder_shapes.py`
- Create: `tests/test_decoder_shapes.py`
- Create: `tests/test_reference_parity.py`
- Create: `tests/test_batch_parity.py`

- [ ] **Step 1: Write failing tests**

Test CLI help, production package imports without reference deps, missing weights fail clearly, model encode/decode raise `NotImplementedError` until Stage 4/6, and batch without lengths is rejected.

- [ ] **Step 2: Run test to verify failure**

Run: `. .venv/bin/activate && pytest tests/test_cli.py tests/test_weights.py tests/test_encoder_shapes.py tests/test_decoder_shapes.py tests/test_reference_parity.py tests/test_batch_parity.py -q`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement minimal CLI/placeholders**

Expose commands `encode`, `decode`, `parity`, and `benchmark encode/decode/batching` in argparse. Make unimplemented runtime paths fail loudly with non-zero exits.

- [ ] **Step 4: Run focused test**

Run: `. .venv/bin/activate && pytest tests/test_cli.py tests/test_weights.py tests/test_encoder_shapes.py tests/test_decoder_shapes.py tests/test_reference_parity.py tests/test_batch_parity.py -q`

Expected: PASS.

### Task 7: Verification And Commit

**Files:**
- Modify: `docs/parity.md`

- [ ] **Step 1: Run full checks**

Run:

```bash
. .venv/bin/activate
pytest -q
ruff check .
python -m mimi_mlx.cli --help
```

Expected: all PASS.

- [ ] **Step 2: Update progress note**

Record files changed, tests added, tests passing, parity status, and blockers in `docs/parity.md`.

- [ ] **Step 3: Commit**

Run:

```bash
git add pyproject.toml README.md LICENSE src tests docs fixtures
git diff --cached --check
git commit -m "feat: scaffold mimi mlx package"
```

Expected: commit succeeds with only milestone files included.
