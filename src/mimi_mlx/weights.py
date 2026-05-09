from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WeightLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetensorsTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]


def inspect_safetensors_header(path: str | Path) -> dict[str, SafetensorsTensorInfo]:
    resolved = Path(path)
    if not resolved.exists():
        raise WeightLoadError(f"Weight file does not exist: {resolved}")
    try:
        with resolved.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise WeightLoadError(f"Could not read safetensors header from {resolved}: {exc}") from exc

    tensors: dict[str, SafetensorsTensorInfo] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = _tensor_info(name, info)
    return tensors


def _tensor_info(name: str, info: dict[str, Any]) -> SafetensorsTensorInfo:
    return SafetensorsTensorInfo(
        name=name,
        dtype=str(info["dtype"]),
        shape=tuple(int(dim) for dim in info["shape"]),
    )
