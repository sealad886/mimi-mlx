from __future__ import annotations

from mimi_mlx.parity import ParityStatus


def test_reference_parity_status_starts_blocked_without_fixtures():
    status = ParityStatus.current()

    assert status.encode == "blocked"
    assert status.decode == "blocked"
    assert "reference fixtures" in status.reason
