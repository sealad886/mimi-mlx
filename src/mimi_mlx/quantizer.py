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
