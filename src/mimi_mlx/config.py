from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class MimiCodecConfig:
    sample_rate: int
    mimi_sample_rate: int
    frame_rate: float
    num_codebooks: int
    codebook_size: int
    channels: int
    hop_length: int | None
    model_revision: str | None = None

    @classmethod
    def default(cls) -> MimiCodecConfig:
        return cls(
            sample_rate=24_000,
            mimi_sample_rate=24_000,
            frame_rate=12.5,
            num_codebooks=32,
            codebook_size=2048,
            channels=1,
            hop_length=1920,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | Path,
        *,
        revision: str | None = None,
    ) -> MimiCodecConfig:
        path = _resolve_config_path(pretrained, revision=revision)
        data = json.loads(path.read_text())
        return cls.from_huggingface_config(data, model_revision=revision)

    @classmethod
    def from_huggingface_config(
        cls,
        data: dict[str, Any],
        *,
        model_revision: str | None = None,
    ) -> MimiCodecConfig:
        sample_rate = int(data.get("sampling_rate", 24_000))
        ratios = data.get("upsampling_ratios") or [8, 6, 5, 4]
        hop_length = int(_product([int(ratio) for ratio in ratios]) * int(data.get("compress", 2)))
        frame_rate = float(data.get("frame_rate", sample_rate / hop_length))
        return cls(
            sample_rate=sample_rate,
            mimi_sample_rate=sample_rate,
            frame_rate=frame_rate,
            num_codebooks=int(data.get("num_quantizers", data.get("num_codebooks", 32))),
            codebook_size=int(data.get("codebook_size", 2048)),
            channels=int(data.get("audio_channels", data.get("channels", 1))),
            hop_length=hop_length,
            model_revision=model_revision,
        )


def _resolve_config_path(pretrained: str | Path, *, revision: str | None) -> Path:
    path = Path(pretrained)
    if path.is_dir():
        return path / "config.json"
    if path.is_file():
        return path
    downloaded = hf_hub_download(str(pretrained), "config.json", revision=revision)
    return Path(downloaded)


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
