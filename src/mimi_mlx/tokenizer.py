from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .audio import normalize_audio_shape
from .config import MimiCodecConfig
from .layouts import CANONICAL_LAYOUT, validate_layout


@dataclass(frozen=True)
class MimiTokens:
    codes: mx.array
    lengths: mx.array
    sample_rate: int
    frame_rate: float
    audio_lengths: mx.array | None = None
    layout: str = CANONICAL_LAYOUT

    def __post_init__(self) -> None:
        validate_layout(self.layout)


class MimiTokenizer:
    def __init__(self, config: MimiCodecConfig | None = None):
        self.config = config or MimiCodecConfig.default()

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | Path,
        *,
        config: MimiCodecConfig | None = None,
        revision: str | None = None,
    ) -> MimiTokenizer:
        resolved = config or MimiCodecConfig.from_pretrained(pretrained, revision=revision)
        return cls(config=resolved)

    def reset_state(self) -> None:
        return None

    def encode(self, audio: mx.array, *, sample_rate: int) -> MimiTokens:
        normalize_audio_shape(audio, channels=self.config.channels)
        raise NotImplementedError(
            "MLX Mimi model is not implemented yet; encode lands in Stage 4/5"
        )

    def decode(self, codes: mx.array, *, sample_rate: int) -> mx.array:
        if codes.ndim != 3:
            raise ValueError(f"Expected canonical token shape [B,T,K], got {codes.shape}")
        raise NotImplementedError("MLX Mimi model is not implemented yet; decode lands in Stage 6")

    def encode_batch(
        self,
        audio: mx.array,
        *,
        lengths: mx.array | None = None,
        sample_rate: int,
    ) -> MimiTokens:
        if lengths is None and audio.ndim >= 2 and audio.shape[0] > 1:
            raise ValueError(
                "lengths are required for batch encode until padded-prefix parity is proven"
            )
        return self.encode(audio, sample_rate=sample_rate)

    @staticmethod
    def save_tokens_npz(path: str | Path, tokens: MimiTokens) -> None:
        save_tokens_npz(path, tokens)

    @staticmethod
    def load_tokens_npz(path: str | Path) -> MimiTokens:
        return load_tokens_npz(path)

    @staticmethod
    def save_tokens_npy(path: str | Path, codes: mx.array) -> None:
        save_tokens_npy(path, codes)

    @staticmethod
    def load_tokens_npy(
        path: str | Path,
        *,
        sample_rate: int,
        frame_rate: float,
    ) -> MimiTokens:
        return load_tokens_npy(path, sample_rate=sample_rate, frame_rate=frame_rate)


def save_tokens_npz(path: str | Path, tokens: MimiTokens) -> None:
    arrays: dict[str, np.ndarray | str | int | float] = {
        "codes": np.array(tokens.codes),
        "lengths": np.array(tokens.lengths),
        "sample_rate": tokens.sample_rate,
        "frame_rate": tokens.frame_rate,
        "layout": tokens.layout,
    }
    if tokens.audio_lengths is not None:
        arrays["audio_lengths"] = np.array(tokens.audio_lengths)
    np.savez(path, **arrays)


def load_tokens_npz(path: str | Path) -> MimiTokens:
    with np.load(path, allow_pickle=False) as data:
        audio_lengths = mx.array(data["audio_lengths"]) if "audio_lengths" in data.files else None
        return MimiTokens(
            codes=mx.array(data["codes"]),
            lengths=mx.array(data["lengths"]),
            sample_rate=int(data["sample_rate"]),
            frame_rate=float(data["frame_rate"]),
            audio_lengths=audio_lengths,
            layout=str(data["layout"]),
        )


def save_tokens_npy(path: str | Path, codes: mx.array) -> None:
    np.save(path, np.array(codes))


def load_tokens_npy(path: str | Path, *, sample_rate: int, frame_rate: float) -> MimiTokens:
    codes = mx.array(np.load(path, allow_pickle=False))
    if codes.ndim != 3:
        raise ValueError(f"Expected saved codes with shape [B,T,K], got {codes.shape}")
    lengths = mx.full((codes.shape[0],), codes.shape[1], dtype=mx.int32)
    return MimiTokens(codes=codes, lengths=lengths, sample_rate=sample_rate, frame_rate=frame_rate)
