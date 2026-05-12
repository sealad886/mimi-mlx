from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class EuclideanCodebook:
    embedding: mx.array

    def __post_init__(self) -> None:
        if self.embedding.ndim != 2:
            raise ValueError(
                f"Expected codebook embedding [codes, dim], got {self.embedding.shape}"
            )
        self._c2 = mx.sum(mx.square(self.embedding), axis=-1) / 2

    @property
    def size(self) -> int:
        return self.embedding.shape[0]

    def encode(self, vectors: mx.array) -> mx.array:
        if vectors.shape[-1] != self.embedding.shape[-1]:
            raise ValueError(
                f"Vector dim {vectors.shape[-1]} does not match codebook dim "
                f"{self.embedding.shape[-1]}"
            )
        flat = mx.reshape(vectors, (-1, vectors.shape[-1]))
        scores = self._c2[None, :] - flat @ self.embedding.T
        return mx.reshape(mx.argmin(scores, axis=-1), vectors.shape[:-1])

    def decode(self, codes: mx.array) -> mx.array:
        return mx.take(self.embedding, codes, axis=0)


@dataclass
class ResidualVectorQuantization:
    codebooks: list[EuclideanCodebook]

    def __post_init__(self) -> None:
        if not self.codebooks:
            raise ValueError("ResidualVectorQuantization requires at least one codebook")

    @property
    def num_codebooks(self) -> int:
        return len(self.codebooks)

    def encode(self, vectors: mx.array) -> mx.array:
        residual = vectors
        codes = []
        for codebook in self.codebooks:
            code = codebook.encode(residual)
            residual = residual - codebook.decode(code)
            codes.append(code)
        return mx.stack(codes, axis=-1)

    def decode(self, codes: mx.array) -> mx.array:
        if codes.shape[-1] != len(self.codebooks):
            raise ValueError(
                f"Expected {len(self.codebooks)} codebooks, got code shape {codes.shape}"
            )
        quantized = self.codebooks[0].decode(codes[..., 0])
        for index, codebook in enumerate(self.codebooks[1:], start=1):
            quantized = quantized + codebook.decode(codes[..., index])
        return quantized


@dataclass
class SplitResidualVectorQuantizer:
    semantic: ResidualVectorQuantization
    acoustic: ResidualVectorQuantization | None = None

    @property
    def num_codebooks(self) -> int:
        acoustic_count = self.acoustic.num_codebooks if self.acoustic is not None else 0
        return self.semantic.num_codebooks + acoustic_count

    def encode(self, vectors: mx.array) -> mx.array:
        codes = self.semantic.encode(vectors)
        if self.acoustic is not None:
            codes = mx.concatenate([codes, self.acoustic.encode(vectors)], axis=-1)
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        semantic_count = self.semantic.num_codebooks
        quantized = self.semantic.decode(codes[..., :semantic_count])
        if self.acoustic is not None and codes.shape[-1] > semantic_count:
            quantized = quantized + self.acoustic.decode(codes[..., semantic_count:])
        return quantized


class MimiEuclideanCodebook:
    def __init__(self, codebook_size: int, codebook_dim: int, epsilon: float = 1e-5):
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.epsilon = epsilon
        self.initialized = mx.ones((1,), dtype=mx.float32)
        self.cluster_usage = mx.ones((codebook_size,), dtype=mx.float32)
        self.embed_sum = mx.zeros((codebook_size, codebook_dim), dtype=mx.float32)
        self._embed_cache: mx.array | None = None
        self._embed_norm_cache: mx.array | None = None
        self._embed_signature: tuple[int, int] | None = None

    @property
    def embed(self) -> mx.array:
        return self._refresh_embed_cache()[0]

    @property
    def embed_norm(self) -> mx.array:
        return self._refresh_embed_cache()[1]

    def _refresh_embed_cache(self) -> tuple[mx.array, mx.array]:
        signature = (id(self.embed_sum), id(self.cluster_usage))
        if self._embed_signature != signature:
            usage = mx.maximum(self.cluster_usage, self.epsilon)[:, None]
            self._embed_cache = self.embed_sum / usage
            self._embed_norm_cache = mx.sum(mx.square(self._embed_cache), axis=-1) / 2
            self._embed_signature = signature
        return self._embed_cache, self._embed_norm_cache

    def encode(self, vectors: mx.array) -> mx.array:
        shape = vectors.shape[:-1]
        flat = vectors.reshape((-1, vectors.shape[-1]))
        embed = self.embed
        scores = self.embed_norm[None, :] - flat @ embed.T
        return mx.argmin(scores, axis=-1).reshape(shape)

    def decode(self, codes: mx.array) -> mx.array:
        return mx.take(self.embed, codes, axis=0)


class MimiVectorQuantization:
    def __init__(self, codebook_size: int, codebook_dim: int):
        self.codebook = MimiEuclideanCodebook(codebook_size, codebook_dim)

    def encode(self, hidden_states: mx.array) -> mx.array:
        return self.codebook.encode(hidden_states.swapaxes(1, 2))

    def decode(self, codes: mx.array) -> mx.array:
        return self.codebook.decode(codes).swapaxes(1, 2)


class MimiResidualVectorQuantizer:
    def __init__(self, config, num_quantizers: int):
        from .modules import Conv1d

        self.layers = [
            MimiVectorQuantization(config.codebook_size, config.codebook_dim)
            for _ in range(num_quantizers)
        ]
        self.input_proj = None
        self.output_proj = None
        if config.codebook_dim != config.hidden_size:
            self.input_proj = Conv1d(config.hidden_size, config.codebook_dim, 1, bias_enabled=False)
            self.output_proj = Conv1d(
                config.codebook_dim, config.hidden_size, 1, bias_enabled=False
            )

    def encode(self, embeddings: mx.array, num_quantizers: int | None = None) -> mx.array:
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)
        count = len(self.layers) if num_quantizers is None else num_quantizers
        residual = embeddings
        all_indices = []
        for layer in self.layers[:count]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        return mx.stack(all_indices, axis=0)

    def decode(self, codes: mx.array) -> mx.array:
        if codes.ndim != 3:
            raise ValueError(f"Expected codes shape [B,K,T], got {codes.shape}")
        if codes.shape[1] < 1 or codes.shape[1] > len(self.layers):
            raise ValueError(
                f"Expected between 1 and {len(self.layers)} codebooks, got code shape {codes.shape}"
            )
        codes = codes.swapaxes(0, 1)
        quantized = self.layers[0].decode(codes[0])
        for index, layer in enumerate(self.layers[1:], start=1):
            if index >= codes.shape[0]:
                break
            quantized = quantized + layer.decode(codes[index])
        if self.output_proj is not None:
            quantized = self.output_proj(quantized)
        return quantized


class MimiSplitResidualVectorQuantizer:
    def __init__(self, config):
        self.max_num_quantizers = config.num_codebooks
        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.num_acoustic_quantizers = config.num_codebooks - config.num_semantic_quantizers
        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_semantic_quantizers
        )
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config, self.num_acoustic_quantizers
        )

    def encode(self, embeddings: mx.array, num_quantizers: int | None = None) -> mx.array:
        count = self.max_num_quantizers if num_quantizers is None else num_quantizers
        if count > self.max_num_quantizers:
            raise ValueError("Requested more quantizers than model supports")
        if count < self.num_semantic_quantizers:
            raise ValueError("Requested fewer quantizers than semantic quantizers")
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)
        if count > self.num_semantic_quantizers:
            acoustic = self.acoustic_residual_vector_quantizer.encode(
                embeddings,
                num_quantizers=count - self.num_semantic_quantizers,
            )
            codes = mx.concatenate([codes, acoustic], axis=0)
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        if codes.ndim != 3:
            raise ValueError(f"Expected codes shape [B,K,T], got {codes.shape}")
        if (
            codes.shape[1] < self.num_semantic_quantizers
            or codes.shape[1] > self.max_num_quantizers
        ):
            raise ValueError(
                f"Expected between {self.num_semantic_quantizers} and "
                f"{self.max_num_quantizers} codebooks, got code shape {codes.shape}"
            )
        semantic = self.semantic_residual_vector_quantizer.decode(
            codes[:, : self.num_semantic_quantizers]
        )
        if codes.shape[1] <= self.num_semantic_quantizers:
            return semantic
        return semantic + self.acoustic_residual_vector_quantizer.decode(
            codes[:, self.num_semantic_quantizers :]
        )
