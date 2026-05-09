from __future__ import annotations

import pytest

from mimi_mlx.weights import WeightLoadError, inspect_safetensors_header


def test_missing_weight_file_fails_clearly(tmp_path):
    with pytest.raises(WeightLoadError, match="does not exist"):
        inspect_safetensors_header(tmp_path / "missing.safetensors")
