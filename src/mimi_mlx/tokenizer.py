from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from .audio import normalize_audio_shape
from .config import MimiCodecConfig, _resolve_revision
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
        _validate_canonical_codes(
            codes,
            num_codebooks=self.config.num_codebooks,
            context="token codes",
        )
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
        if host_lengths.dtype == np.bool_ or not np.issubdtype(host_lengths.dtype, np.integer):
            raise ValueError("lengths must use an integer dtype with sample counts")
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

    return _tokens_from_archive(data)


def _load_legacy_tokens_npz(path: str | Path) -> MimiTokens:
    with np.load(path, allow_pickle=False) as data:
        return _tokens_from_archive(
            {name: data[name] for name in data.files},
            convert_legacy_numpy=True,
        )


def _tokens_from_archive(data: dict, *, convert_legacy_numpy: bool = False) -> MimiTokens:
    required = ("codes", "lengths", "sample_rate", "frame_rate", "layout")
    for field in required:
        if field not in data:
            raise ValueError(f"Token archive missing required field: {field}")

    codes = mx.array(data["codes"]) if convert_legacy_numpy else data["codes"]
    lengths = mx.array(data["lengths"]) if convert_legacy_numpy else data["lengths"]
    audio_lengths = data.get("audio_lengths")
    if audio_lengths is not None and convert_legacy_numpy:
        audio_lengths = mx.array(audio_lengths)

    _validate_token_archive_shapes(codes, lengths, audio_lengths)
    return MimiTokens(
        codes=codes,
        lengths=lengths,
        sample_rate=int(data["sample_rate"]),
        frame_rate=float(data["frame_rate"]),
        audio_lengths=audio_lengths,
        layout=_decode_layout_value(data["layout"]),
    )


def _validate_token_archive_shapes(
    codes: mx.array,
    lengths: mx.array,
    audio_lengths: mx.array | None,
) -> None:
    _validate_canonical_codes(codes, num_codebooks=32, context="token codes")
    if not hasattr(lengths, "ndim") or lengths.ndim != 1 or lengths.shape[0] != codes.shape[0]:
        shape = getattr(lengths, "shape", None)
        raise ValueError(f"Expected token lengths shape [{codes.shape[0]}], got {shape}")
    if audio_lengths is not None and (
        not hasattr(audio_lengths, "ndim")
        or audio_lengths.ndim != 1
        or audio_lengths.shape[0] != codes.shape[0]
    ):
        shape = getattr(audio_lengths, "shape", None)
        raise ValueError(f"Expected token audio_lengths shape [{codes.shape[0]}], got {shape}")


def _encode_layout(layout: str) -> mx.array:
    return mx.array(list(layout.encode("utf-8")), dtype=mx.uint8)


def _decode_layout(layout: mx.array) -> str:
    return bytes(layout.tolist()).decode("utf-8")


def _decode_layout_value(layout: object) -> str:
    if isinstance(layout, str):
        return layout
    if isinstance(layout, bytes):
        return layout.decode("utf-8")
    if hasattr(layout, "tolist"):
        value = layout.tolist()
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return bytes(value).decode("utf-8")
    raise ValueError(f"Unsupported token layout field type: {type(layout).__name__}")


def save_tokens_npy(path: str | Path, codes: mx.array) -> None:
    mx.save(path, codes)


def load_tokens_npy(path: str | Path, *, sample_rate: int, frame_rate: float) -> MimiTokens:
    codes = mx.load(path)
    _validate_canonical_codes(codes, num_codebooks=32, context="saved canonical codes")
    lengths = mx.full((codes.shape[0],), codes.shape[1], dtype=mx.int32)
    return MimiTokens(codes=codes, lengths=lengths, sample_rate=sample_rate, frame_rate=frame_rate)


def _validate_canonical_codes(codes: mx.array, *, num_codebooks: int, context: str) -> None:
    if not hasattr(codes, "ndim") or codes.ndim != 3:
        shape = getattr(codes, "shape", None)
        raise ValueError(f"Expected {context} shape [B,T,K={num_codebooks}], got {shape}")
    if codes.shape[-1] != num_codebooks:
        if codes.shape[1] == num_codebooks:
            raise ValueError(
                f"Expected {context} shape [B,T,K={num_codebooks}], got {codes.shape}. "
                "Input looks like upstream [B,K,T]; use from_upstream_layout first."
            )
        raise ValueError(f"Expected {context} shape [B,T,K={num_codebooks}], got {codes.shape}")


def _resolve_weights_path(pretrained: str | Path, *, revision: str | None) -> Path:
    path = Path(pretrained)
    if path.is_dir():
        return path / "model.safetensors"
    if path.is_file():
        return path
    return Path(
        hf_hub_download(
            str(pretrained),
            "model.safetensors",
            revision=_resolve_revision(str(pretrained), revision),
        )
    )


def _load_hf_config(pretrained: str | Path, *, revision: str | None) -> dict:
    import json

    path = Path(pretrained)
    if path.is_dir():
        config_path = path / "config.json"
    elif path.is_file():
        config_path = path.with_name("config.json")
    else:
        config_path = Path(
            hf_hub_download(
                str(pretrained),
                "config.json",
                revision=_resolve_revision(str(pretrained), revision),
            )
        )
    return json.loads(config_path.read_text())
