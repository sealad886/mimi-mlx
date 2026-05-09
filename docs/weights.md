# Mimi Weight Notes

Research date: 2026-05-09.

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

## Mapping Work Still Required

Stage 2 must map HF/Transformers names to standalone MLX module names and validate every tensor:

- `embed_sum` maps to MLX codebook `embedding_sum`.
- PyTorch conv weights require `[out, in, kernel]` to MLX `[out, kernel, in]`.
- Conv-transpose weights require dedicated transpose handling.
- Transformer layer norm, attention projection, output projection, MLP, and layer-scale names differ between HF, Kyutai PyTorch, and Kyutai MLX modules.
- Missing, extra, or shape-mismatched tensors must fail loudly.

## Implemented Validation Slice

`mimi_mlx.weights.validate_hf_mimi_header` now validates a required sentinel set across
encoder, decoder, transformer, downsample, upsample, and split-RVQ families. This is not
the final full tensor mapping. It is an early guardrail for Stage 2 scripts and tests.

`scripts/inspect_weights.py` reports JSON or human-readable manifest summaries and exits
non-zero on missing or shape-mismatched required tensors.
