from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from .audio import normalize_audio_shape
from .config import MimiCodecConfig
from .layouts import CANONICAL_LAYOUT, from_upstream_layout, to_upstream_layout, validate_layout
from .model import MimiModel, MimiModelConfig


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
    def __init__(self, config: MimiCodecConfig | None = None, model: MimiModel | None = None):
        self.config = config or MimiCodecConfig.default()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | Path,
        *,
        config: MimiCodecConfig | None = None,
        revision: str | None = None,
    ) -> MimiTokenizer:
        resolved = config or MimiCodecConfig.from_pretrained(pretrained, revision=revision)
        model_config = MimiModelConfig.from_hf_config(
            _load_hf_config(pretrained, revision=revision)
        )
        model = MimiModel(model_config)
        model.load_hf_weights(_resolve_weights_path(pretrained, revision=revision))
        return cls(config=resolved, model=model)

    def reset_state(self) -> None:
        if self.model is not None:
            self.model.reset_state()

    def encode(self, audio: mx.array, *, sample_rate: int) -> MimiTokens:
        if self.model is None:
            raise NotImplementedError("MLX Mimi model is not loaded")
        normalized, audio_lengths = normalize_audio_shape(audio, channels=self.config.channels)
        if sample_rate != self.config.sample_rate:
            from .audio import resample_linear

            normalized = resample_linear(
                normalized,
                src_rate=sample_rate,
                dst_rate=self.config.sample_rate,
            )
        upstream = self.model.encode(normalized)
        codes = from_upstream_layout(upstream)
        lengths = mx.full((codes.shape[0],), codes.shape[1], dtype=mx.int32)
        return MimiTokens(
            codes=codes,
            lengths=lengths,
            sample_rate=self.config.sample_rate,
            frame_rate=self.config.frame_rate,
            audio_lengths=audio_lengths,
        )

    def decode(self, codes: mx.array, *, sample_rate: int) -> mx.array:
        if self.model is None:
            raise NotImplementedError("MLX Mimi model is not loaded")
        if codes.ndim != 3:
            raise ValueError(f"Expected canonical token shape [B,T,K], got {codes.shape}")
        audio = self.model.decode(to_upstream_layout(codes))
        if sample_rate != self.config.sample_rate:
            from .audio import resample_linear

            audio = resample_linear(audio, src_rate=self.config.sample_rate, dst_rate=sample_rate)
        return audio

    def encode_batch(
        self,
        audio: mx.array,
        *,
        lengths: mx.array | None = None,
        sample_rate: int,
    ) -> MimiTokens:
        if lengths is None:
            return self.encode(audio, sample_rate=sample_rate)

        if self.model is None:
            raise NotImplementedError("MLX Mimi model is not loaded")
        normalized, _ = normalize_audio_shape(audio, channels=self.config.channels)
        if lengths.ndim != 1 or lengths.shape[0] != normalized.shape[0]:
            raise ValueError(f"Expected lengths shape [{normalized.shape[0]}], got {lengths.shape}")

        host_lengths = np.array(lengths)
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(host_lengths.tolist()):
            sample_length = int(length)
            if sample_length < 0 or sample_length > normalized.shape[-1]:
                raise ValueError(
                    f"Invalid audio length {sample_length} for sample {index}; "
                    f"padded length is {normalized.shape[-1]}"
                )
            groups.setdefault(sample_length, []).append(index)

        per_sample: list[mx.array | None] = [None] * normalized.shape[0]
        frame_lengths = [0] * normalized.shape[0]
        for sample_length, indices in groups.items():
            group_audio = mx.take(normalized, mx.array(indices, dtype=mx.int32), axis=0)
            sample_tokens = self.encode(group_audio[:, :, :sample_length], sample_rate=sample_rate)
            for group_index, sample_index in enumerate(indices):
                per_sample[sample_index] = sample_tokens.codes[group_index]
                frame_lengths[sample_index] = sample_tokens.codes.shape[1]

        max_frames = max(frame_lengths, default=0)
        padded = []
        for codes in per_sample:
            if codes is None:
                raise RuntimeError("Internal batching error left a sample unencoded")
            padded.append(mx.pad(codes, [(0, max_frames - codes.shape[0]), (0, 0)]))
        codes = mx.stack(padded, axis=0) if padded else mx.zeros((0, 0, self.config.num_codebooks))
        return MimiTokens(
            codes=codes,
            lengths=mx.array(frame_lengths, dtype=mx.int32),
            sample_rate=self.config.sample_rate,
            frame_rate=self.config.frame_rate,
            audio_lengths=mx.array(host_lengths, dtype=mx.int32),
        )

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
    arrays: dict[str, mx.array] = {
        "codes": tokens.codes,
        "lengths": tokens.lengths,
        "sample_rate": mx.array(tokens.sample_rate, dtype=mx.int32),
        "frame_rate": mx.array(tokens.frame_rate, dtype=mx.float32),
        "layout": _encode_layout(tokens.layout),
    }
    if tokens.audio_lengths is not None:
        arrays["audio_lengths"] = tokens.audio_lengths
    mx.savez(path, **arrays)


def load_tokens_npz(path: str | Path) -> MimiTokens:
    try:
        data = mx.load(path)
    except ValueError:
        return _load_legacy_tokens_npz(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected token archive with named arrays, got {type(data).__name__}")

    audio_lengths = data.get("audio_lengths")
    return MimiTokens(
        codes=data["codes"],
        lengths=data["lengths"],
        sample_rate=int(data["sample_rate"]),
        frame_rate=float(data["frame_rate"]),
        audio_lengths=audio_lengths,
        layout=_decode_layout(data["layout"]),
    )


def _load_legacy_tokens_npz(path: str | Path) -> MimiTokens:
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


def _encode_layout(layout: str) -> mx.array:
    return mx.array(list(layout.encode("utf-8")), dtype=mx.uint8)


def _decode_layout(layout: mx.array) -> str:
    return bytes(layout.tolist()).decode("utf-8")


def save_tokens_npy(path: str | Path, codes: mx.array) -> None:
    mx.save(path, codes)


def load_tokens_npy(path: str | Path, *, sample_rate: int, frame_rate: float) -> MimiTokens:
    codes = mx.load(path)
    if not hasattr(codes, "ndim") or codes.ndim != 3:
        shape = getattr(codes, "shape", None)
        raise ValueError(f"Expected saved codes with shape [B,T,K], got {shape}")
    lengths = mx.full((codes.shape[0],), codes.shape[1], dtype=mx.int32)
    return MimiTokens(codes=codes, lengths=lengths, sample_rate=sample_rate, frame_rate=frame_rate)


def _resolve_weights_path(pretrained: str | Path, *, revision: str | None) -> Path:
    path = Path(pretrained)
    if path.is_dir():
        return path / "model.safetensors"
    if path.is_file():
        return path
    return Path(hf_hub_download(str(pretrained), "model.safetensors", revision=revision))


def _load_hf_config(pretrained: str | Path, *, revision: str | None) -> dict:
    import json

    path = Path(pretrained)
    if path.is_dir():
        config_path = path / "config.json"
    elif path.is_file():
        config_path = path.with_name("config.json")
    else:
        config_path = Path(hf_hub_download(str(pretrained), "config.json", revision=revision))
    return json.loads(config_path.read_text())
