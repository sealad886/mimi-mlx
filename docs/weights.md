# Mimi Weight Notes

Research date: 2026-05-09.

See `docs/usage.md` for download commands and `docs/development.md` for the
validation workflow.

## Official Checkpoint

- Repository: `kyutai/mimi`
- Revision: `89091b3e466eb6a9d11e537bf26b144f194978f7`
- File: `model.safetensors`
- Size: 384,649,828 bytes
- SHA-256: `bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e`
- Safetensors metadata: `{"format": "pt"}`
- Tensor count from header: 350
- Parameter dtype: F32
- Parameter count: 96,151,393

Download the checkpoint into the ignored local fixture directory:

```bash
python scripts/download_reference_assets.py
```

Validate the downloaded file:

```bash
python scripts/inspect_weights.py fixtures/reference/hf/model.safetensors
```

## Name Families

HF checkpoint tensor families observed from the safetensors header:

| Family | Example | Shape |
| --- | --- | --- |
| Encoder conv | `encoder.layers.0.conv.weight` | PyTorch `[out, in, kernel]` |
| Decoder conv-transpose | `decoder.layers.2.conv.weight` | PyTorch conv-transpose layout |
| Encoder transformer | `encoder_transformer.layers.0.input_layernorm.weight` | `[512]` |
| Decoder transformer | `decoder_transformer.layers.0.input_layernorm.weight` | `[512]` |
| Semantic quantizer | `quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum` | `[2048, 256]` |
| Acoustic quantizer | `quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum` | `[2048, 256]` |
| Latent downsample | `downsample.conv.weight` | `[512, 512, 4]` |
| Latent upsample | `upsample.conv.weight` | `[512, 1, 4]` |

## MLX Mapping

`MimiModel.load_hf_weights` maps every tensor in `model.safetensors` and validates the
post-conversion MLX shape before assignment. It also verifies that every expected
model tensor was assigned, so partial checkpoints cannot leave parameters at
their zero initialization values.

- Codebook `embed_sum`, `cluster_usage`, and `initialized` map directly to MLX codebook buffers.
- PyTorch conv weights require `[out, in, kernel]` to MLX `[out, kernel, in]`.
- Conv-transpose weights use `[in, out/groups, kernel]` to MLX `[out/groups, kernel, in]`.
- Transformer linear, layer norm, MLP, and layer-scale weights map directly with no transpose.
- Residual block list indices intentionally match HF `.block.1` and `.block.3`.
- Missing, extra, disabled, or shape-mismatched tensors fail with `WeightLoadError`.

## Implemented Validation Slice

`mimi_mlx.weights.validate_hf_mimi_header` validates a required sentinel set across
encoder, decoder, transformer, downsample, upsample, and split-RVQ families.
Runtime remote loading of the canonical `kyutai/mimi` model defaults to revision
`89091b3e466eb6a9d11e537bf26b144f194978f7`; callers can still pass an explicit
`revision=` to opt into another checkpoint.

`scripts/inspect_weights.py` reports JSON or human-readable manifest summaries and exits
non-zero on missing or shape-mismatched required tensors.

Local verification on 2026-05-09:

```text
fixtures/reference/hf/model.safetensors
SHA256 bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e
350 tensors
96,151,393 parameters
```
