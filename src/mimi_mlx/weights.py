from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WeightLoadError(RuntimeError):
    pass


HF_MIMI_REQUIRED_TENSORS: dict[str, tuple[int, ...]] = {
    "encoder.layers.0.conv.weight": (64, 1, 7),
    "encoder.layers.0.conv.bias": (64,),
    "encoder.layers.14.conv.weight": (512, 1024, 3),
    "encoder.layers.14.conv.bias": (512,),
    "decoder.layers.0.conv.weight": (1024, 512, 7),
    "decoder.layers.0.conv.bias": (1024,),
    "decoder.layers.14.conv.weight": (1, 64, 3),
    "decoder.layers.14.conv.bias": (1,),
    "encoder_transformer.layers.0.input_layernorm.weight": (512,),
    "decoder_transformer.layers.0.input_layernorm.weight": (512,),
    "downsample.conv.weight": (512, 512, 4),
    "upsample.conv.weight": (512, 1, 4),
    "quantizer.semantic_residual_vector_quantizer.input_proj.weight": (256, 512, 1),
    "quantizer.semantic_residual_vector_quantizer.output_proj.weight": (512, 256, 1),
    "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum": (2048, 256),
    "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage": (2048,),
    "quantizer.acoustic_residual_vector_quantizer.input_proj.weight": (256, 512, 1),
    "quantizer.acoustic_residual_vector_quantizer.output_proj.weight": (512, 256, 1),
    "quantizer.acoustic_residual_vector_quantizer.layers.0.codebook.embed_sum": (2048, 256),
    "quantizer.acoustic_residual_vector_quantizer.layers.30.codebook.embed_sum": (2048, 256),
}


@dataclass(frozen=True)
class SafetensorsTensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class WeightManifest:
    path: Path
    tensor_count: int
    required_count: int
    total_parameters: int


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


def validate_hf_mimi_header(path: str | Path) -> WeightManifest:
    resolved = Path(path)
    tensors = inspect_safetensors_header(resolved)
    missing = sorted(set(HF_MIMI_REQUIRED_TENSORS) - set(tensors))
    if missing:
        preview = ", ".join(missing[:5])
        raise WeightLoadError(f"Missing required Mimi tensors: {preview}")

    for name, expected_shape in HF_MIMI_REQUIRED_TENSORS.items():
        actual_shape = tensors[name].shape
        if actual_shape != expected_shape:
            raise WeightLoadError(
                f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
            )

    total_parameters = sum(_numel(tensor.shape) for tensor in tensors.values())
    return WeightManifest(
        path=resolved,
        tensor_count=len(tensors),
        required_count=len(HF_MIMI_REQUIRED_TENSORS),
        total_parameters=total_parameters,
    )


def _tensor_info(name: str, info: dict[str, Any]) -> SafetensorsTensorInfo:
    return SafetensorsTensorInfo(
        name=name,
        dtype=str(info["dtype"]),
        shape=tuple(int(dim) for dim in info["shape"]),
    )


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
