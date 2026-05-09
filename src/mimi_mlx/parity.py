from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParityStatus:
    encode: str
    decode: str
    batch: str
    reason: str

    @classmethod
    def current(cls) -> ParityStatus:
        return cls(
            encode="blocked",
            decode="blocked",
            batch="blocked",
            reason="reference fixtures and full MLX model implementation are not present yet",
        )
