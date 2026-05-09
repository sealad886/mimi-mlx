# Mimi Architecture Notes

Research date: 2026-05-09.

Sources inspected:

- Kyutai Moshi repository commit `6d14a61994f38b282ae75bc4e9c3dcc4d35e7183`.
- Hugging Face model repository `kyutai/mimi` commit `89091b3e466eb6a9d11e537bf26b144f194978f7`.
- Hugging Face Transformers Mimi docs and current `modeling_mimi.py`.
- Apple MLX 0.31 documentation for `Conv1d` and array loading.

## Current Facts

- Official HF model id: `kyutai/mimi`.
- HF model file: `model.safetensors`.
- HF model SHA-256: `bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e`.
- HF model has 96,151,393 F32 parameters across 350 tensors.
- HF preprocessor sampling rate: 24,000 Hz.
- Kyutai Moshi Mimi config uses sample rate 24,000 Hz and frame rate 12.5 Hz.
- Frame size is 24,000 / 12.5 = 1,920 samples.
- SEANet ratios are `[8, 6, 5, 4]`; encoder applies them in reverse.
- Encoder frame rate before latent downsampling is 24,000 / (8 * 6 * 5 * 4) = 25 Hz.
- Learned downsample stride is 2, producing 12.5 Hz tokens.
- Default full codec uses 32 codebooks in the standalone Mimi config.
- Moshi language model paths often use only 8 or 16 active codebooks.
- Codebook size is 2,048 and quantizer hidden/codebook dimension is 256.
- Upstream token layout is `[batch, codebook, time]`.
- Public `mimi_mlx` canonical layout is `[batch, time, codebook]`; conversion helpers are explicit.
- Mimi uses causal streamable convolutions, ELU residual blocks, transformer blocks with RoPE, and split residual vector quantization.
- Split RVQ encodes semantic codebooks and acoustic codebooks from the same latent, then concatenates semantic first.
- Current HF Transformers source marks batched padding support as incomplete for the encoder, so padded batch parity remains a direct risk.

## Implementation Boundary

Production `mimi_mlx` encode/decode must be MLX-native and must not import PyTorch, `rustymimi`, `torchaudio`, or upstream Moshi code. Reference tooling may use those dependencies under the `reference` extra.

## Direct Risks To Test

- `[B,K,T]` versus `[B,T,K]` token layout.
- Semantic/acoustic codebook ordering.
- Off-by-one frame counts from causal padding and the learned stride-2 latent downsample.
- Batch padding changing token prefixes.
- Weight transpose mistakes from PyTorch `[out,in,k]` to MLX `[out,k,in]`.
- Conv-transpose/grouped upsample mapping.
- State bleed between independent clips.
