from __future__ import annotations

import math

import mlx.core as mx


def normalize_audio_shape(audio: mx.array, *, channels: int = 1) -> tuple[mx.array, mx.array]:
    if audio.ndim == 1:
        normalized = audio[None, None, :]
    elif audio.ndim == 2:
        normalized = audio[:, None, :]
    elif audio.ndim == 3:
        normalized = audio
    else:
        raise ValueError(
            "Expected audio shape [samples], [batch, samples], or "
            f"[batch, channels, samples], got {audio.shape}"
        )

    if normalized.shape[1] != channels:
        if channels == 1:
            raise ValueError(f"Expected mono audio with 1 channel, got {normalized.shape[1]}")
        raise ValueError(f"Expected {channels} audio channels, got {normalized.shape[1]}")

    lengths = mx.full((normalized.shape[0],), normalized.shape[-1], dtype=mx.int32)
    return normalized, lengths


def resample_linear(audio: mx.array, *, src_rate: int, dst_rate: int) -> mx.array:
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if src_rate == dst_rate:
        return audio
    if audio.ndim != 3:
        raise ValueError(f"Expected internal audio shape [B,C,T], got {audio.shape}")

    in_length = audio.shape[-1]
    if in_length == 0:
        return audio
    out_length = max(1, int(math.floor(in_length * dst_rate / src_rate)))
    positions = mx.arange(out_length, dtype=mx.float32) * (src_rate / dst_rate)
    left = mx.floor(positions).astype(mx.int32)
    right = mx.minimum(left + 1, in_length - 1)
    frac = positions - left.astype(mx.float32)

    left_values = mx.take(audio, left, axis=-1)
    right_values = mx.take(audio, right, axis=-1)
    return left_values * (1.0 - frac) + right_values * frac
