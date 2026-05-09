from __future__ import annotations

import json
import struct

import pytest

from mimi_mlx.weights import (
    HF_MIMI_REQUIRED_TENSORS,
    WeightLoadError,
    inspect_safetensors_header,
    validate_hf_mimi_header,
)


def test_missing_weight_file_fails_clearly(tmp_path):
    with pytest.raises(WeightLoadError, match="does not exist"):
        inspect_safetensors_header(tmp_path / "missing.safetensors")


def test_inspect_safetensors_header_reads_tensor_metadata(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_header(
        path,
        {
            "encoder.layers.0.conv.weight": {"dtype": "F32", "shape": [64, 1, 7]},
            "__metadata__": {"format": "pt"},
        },
    )

    tensors = inspect_safetensors_header(path)

    assert tensors["encoder.layers.0.conv.weight"].dtype == "F32"
    assert tensors["encoder.layers.0.conv.weight"].shape == (64, 1, 7)


def test_hf_mimi_header_validation_accepts_required_tensors(tmp_path):
    tensors = {
        name: {"dtype": "F32", "shape": list(shape)}
        for name, shape in HF_MIMI_REQUIRED_TENSORS.items()
    }
    path = tmp_path / "model.safetensors"
    _write_header(path, tensors)

    manifest = validate_hf_mimi_header(path)

    assert manifest.tensor_count == len(HF_MIMI_REQUIRED_TENSORS)
    assert manifest.required_count == len(HF_MIMI_REQUIRED_TENSORS)
    expected_parameters = sum(_numel(shape) for shape in HF_MIMI_REQUIRED_TENSORS.values())
    assert manifest.total_parameters == expected_parameters


def test_hf_mimi_header_validation_rejects_missing_required_tensor(tmp_path):
    tensors = {
        name: {"dtype": "F32", "shape": list(shape)}
        for name, shape in HF_MIMI_REQUIRED_TENSORS.items()
    }
    tensors.pop("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum")
    path = tmp_path / "model.safetensors"
    _write_header(path, tensors)

    with pytest.raises(WeightLoadError, match="Missing required Mimi tensors"):
        validate_hf_mimi_header(path)


def test_hf_mimi_header_validation_rejects_shape_mismatch(tmp_path):
    tensors = {
        name: {"dtype": "F32", "shape": list(shape)}
        for name, shape in HF_MIMI_REQUIRED_TENSORS.items()
    }
    tensors["downsample.conv.weight"] = {"dtype": "F32", "shape": [1, 2, 3]}
    path = tmp_path / "model.safetensors"
    _write_header(path, tensors)

    with pytest.raises(WeightLoadError, match="Shape mismatch"):
        validate_hf_mimi_header(path)


def _write_header(path, header):
    payload = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total
