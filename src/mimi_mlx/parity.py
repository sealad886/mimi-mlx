from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityStatus:
    encode: str
    decode: str
    batch: str
    reason: str

    @classmethod
    def current(cls) -> ParityStatus:
        return cls(
            encode="exact-fixture-parity",
            decode="waveform-tolerance-parity",
            batch="prefix-parity-tested",
            reason="local fixture parity is enforced by pytest when official weights are present",
        )


@dataclass(frozen=True)
class TokenMismatch:
    batch: int
    frame: int
    codebook: int
    expected: int
    actual: int


def first_token_mismatch(expected: np.ndarray, actual: np.ndarray) -> TokenMismatch | None:
    if expected.shape != actual.shape:
        return TokenMismatch(
            batch=-1,
            frame=-1,
            codebook=-1,
            expected=int(np.prod(expected.shape)),
            actual=int(np.prod(actual.shape)),
        )
    mismatches = np.argwhere(expected != actual)
    if mismatches.size == 0:
        return None
    batch, codebook, frame = (int(v) for v in mismatches[0])
    return TokenMismatch(
        batch=batch,
        frame=frame,
        codebook=codebook,
        expected=int(expected[batch, codebook, frame]),
        actual=int(actual[batch, codebook, frame]),
    )
