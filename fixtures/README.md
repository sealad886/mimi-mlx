# Fixtures

Committed fixtures are small deterministic WAV files plus golden Mimi token outputs from
`transformers.MimiModel` using the official `kyutai/mimi` weights.

See `docs/development.md` for the full fixture regeneration and validation
workflow.

`real_speech_librispeech_100s` is extracted from
`hf-internal-testing/librispeech_asr_dummy`, downloaded into ignored `fixtures/source/`
with the Hugging Face CLI and resampled to 24 kHz before reference token export.

The fixture manifest is `fixtures/reference/manifest.json`. It records:

- upstream implementation and revision,
- official weight SHA256,
- audio path and checksum,
- token path and checksum,
- reconstruction path and checksum,
- upstream layout (`batch_codebook_time`).

Regenerate fixtures after changing the reference implementation or fixture audio:

```bash
python scripts/export_reference_fixtures.py --weights fixtures/reference/hf
```

The official weights are not committed. Download them with:

```bash
python scripts/download_reference_assets.py
```
