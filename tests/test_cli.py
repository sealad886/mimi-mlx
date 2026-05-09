from __future__ import annotations

import subprocess
import sys


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "mimi_mlx.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "encode" in result.stdout
    assert "decode" in result.stdout
    assert "parity" in result.stdout
    assert "benchmark" in result.stdout
