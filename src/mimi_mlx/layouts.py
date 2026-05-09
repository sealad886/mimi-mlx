from __future__ import annotations

import mlx.core as mx

CANONICAL_LAYOUT = "batch_time_codebook"
UPSTREAM_LAYOUT = "batch_codebook_time"
SUPPORTED_LAYOUTS = {CANONICAL_LAYOUT, UPSTREAM_LAYOUT}


def validate_layout(layout: str) -> str:
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"Unsupported token layout {layout!r}")
    return layout


def to_upstream_layout(codes: mx.array, *, layout: str = CANONICAL_LAYOUT) -> mx.array:
    validate_layout(layout)
    if codes.ndim != 3:
        raise ValueError(f"Expected 3D token codes, got shape {codes.shape}")
    if layout == UPSTREAM_LAYOUT:
        return codes
    return codes.swapaxes(1, 2)


def from_upstream_layout(codes: mx.array, *, layout: str = UPSTREAM_LAYOUT) -> mx.array:
    validate_layout(layout)
    if codes.ndim != 3:
        raise ValueError(f"Expected 3D token codes, got shape {codes.shape}")
    if layout == CANONICAL_LAYOUT:
        return codes
    return codes.swapaxes(1, 2)
