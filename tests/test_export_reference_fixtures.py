from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_export_script():
    spec = importlib.util.spec_from_file_location(
        "export_reference_fixtures",
        ROOT / "scripts" / "export_reference_fixtures.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_reference_model_load_uses_requested_revision(monkeypatch):
    script = _load_export_script()
    calls = {}

    class FakeMimiModel:
        @classmethod
        def from_pretrained(cls, weights: str, *, revision: str | None = None):
            calls["weights"] = weights
            calls["revision"] = revision
            return types.SimpleNamespace(eval=lambda: "model")

    monkeypatch.setattr(script, "MimiModel", FakeMimiModel)

    model = script.load_reference_model("kyutai/mimi", revision="abc123")

    assert model == "model"
    assert calls == {"weights": "kyutai/mimi", "revision": "abc123"}


def test_local_reference_model_load_does_not_pass_revision(monkeypatch, tmp_path):
    script = _load_export_script()
    calls = {}

    class FakeMimiModel:
        @classmethod
        def from_pretrained(cls, weights: str, *, revision: str | None = None):
            calls["weights"] = weights
            calls["revision"] = revision
            return types.SimpleNamespace(eval=lambda: "model")

    monkeypatch.setattr(script, "MimiModel", FakeMimiModel)

    model = script.load_reference_model(tmp_path, revision="abc123")

    assert model == "model"
    assert calls == {"weights": str(tmp_path), "revision": None}


def test_missing_speech_source_fails_by_default(tmp_path):
    script = _load_export_script()

    missing = tmp_path / "missing.parquet"

    try:
        script.maybe_load_speech_fixture(missing)
    except FileNotFoundError as exc:
        assert "Missing LibriSpeech source parquet" in str(exc)
    else:
        raise AssertionError("missing speech source should fail by default")


def test_missing_speech_source_can_be_explicitly_allowed(tmp_path):
    script = _load_export_script()

    assert script.maybe_load_speech_fixture(
        tmp_path / "missing.parquet",
        allow_missing=True,
    ) is None
