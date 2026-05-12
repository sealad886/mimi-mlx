from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .modules import (
    Conv1d,
    ConvTranspose1d,
    MimiConv1d,
    MimiConvTranspose1d,
    MimiDecoder,
    MimiEncoder,
    MimiTransformerModel,
)
from .quantizer import MimiSplitResidualVectorQuantizer
from .weights import WeightLoadError


@dataclass(frozen=True)
class MimiModelConfig:
    sampling_rate: int = 24_000
    audio_channels: int = 1
    hidden_size: int = 512
    num_filters: int = 64
    num_residual_layers: int = 1
    upsampling_ratios: tuple[int, ...] = (8, 6, 5, 4)
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    dilation_growth_rate: int = 2
    use_causal_conv: bool = True
    use_conv_shortcut: bool = False
    pad_mode: str = "constant"
    compress: int = 2
    trim_right_ratio: float = 1.0
    codebook_size: int = 2048
    codebook_dim: int = 256
    num_codebooks: int = 32
    num_semantic_quantizers: int = 1
    upsample_groups: int = 512
    num_hidden_layers: int = 8
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 8000
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 250
    attention_dropout: float = 0.0
    layer_scale_initial_scale: float = 0.01
    frame_rate: float = 12.5

    @classmethod
    def from_hf_config(cls, data: dict[str, Any]) -> MimiModelConfig:
        return cls(
            sampling_rate=int(data.get("sampling_rate", 24_000)),
            audio_channels=int(data.get("audio_channels", 1)),
            hidden_size=int(data.get("hidden_size", 512)),
            num_filters=int(data.get("num_filters", 64)),
            num_residual_layers=int(data.get("num_residual_layers", 1)),
            upsampling_ratios=tuple(int(v) for v in data.get("upsampling_ratios", [8, 6, 5, 4])),
            kernel_size=int(data.get("kernel_size", 7)),
            last_kernel_size=int(data.get("last_kernel_size", 3)),
            residual_kernel_size=int(data.get("residual_kernel_size", 3)),
            dilation_growth_rate=int(data.get("dilation_growth_rate", 2)),
            use_causal_conv=bool(data.get("use_causal_conv", True)),
            use_conv_shortcut=bool(data.get("use_conv_shortcut", False)),
            pad_mode=str(data.get("pad_mode", "constant")),
            compress=int(data.get("compress", 2)),
            trim_right_ratio=float(data.get("trim_right_ratio", 1.0)),
            codebook_size=int(data.get("codebook_size", 2048)),
            codebook_dim=int(
                data.get("codebook_dim", data.get("vector_quantization_hidden_dimension", 256))
            ),
            num_codebooks=int(data.get("num_quantizers", data.get("num_codebooks", 32))),
            num_semantic_quantizers=int(data.get("num_semantic_quantizers", 1)),
            upsample_groups=int(data.get("upsample_groups", 512)),
            num_hidden_layers=int(data.get("num_hidden_layers", 8)),
            intermediate_size=int(data.get("intermediate_size", 2048)),
            num_attention_heads=int(data.get("num_attention_heads", 8)),
            num_key_value_heads=int(data.get("num_key_value_heads", 8)),
            head_dim=int(data.get("head_dim", 64)),
            max_position_embeddings=int(data.get("max_position_embeddings", 8000)),
            norm_eps=float(data.get("norm_eps", 1e-5)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            sliding_window=int(data.get("sliding_window", 250)),
            attention_dropout=float(data.get("attention_dropout", 0.0)),
            layer_scale_initial_scale=float(data.get("layer_scale_initial_scale", 0.01)),
            frame_rate=float(data.get("frame_rate", 12.5)),
        )

    @property
    def encodec_frame_rate(self) -> float:
        product = 1
        for ratio in self.upsampling_ratios:
            product *= ratio
        return self.sampling_rate / product


class MimiModel:
    def __init__(self, config: MimiModelConfig | None = None):
        self.config = config or MimiModelConfig()
        self.encoder = MimiEncoder(self.config)
        self.encoder_transformer = MimiTransformerModel(self.config)
        self.downsample = MimiConv1d(
            self.config,
            self.config.hidden_size,
            self.config.hidden_size,
            kernel_size=2 * int(self.config.encodec_frame_rate / self.config.frame_rate),
            stride=2,
            bias=False,
            pad_mode="replicate",
        )
        self.upsample = MimiConvTranspose1d(
            self.config,
            self.config.hidden_size,
            self.config.hidden_size,
            kernel_size=2 * int(self.config.encodec_frame_rate / self.config.frame_rate),
            stride=2,
            bias=False,
            groups=self.config.upsample_groups,
        )
        self.decoder_transformer = MimiTransformerModel(self.config)
        self.decoder = MimiDecoder(self.config)
        self.quantizer = MimiSplitResidualVectorQuantizer(self.config)

    def encode(self, audio: mx.array, *, num_codebooks: int | None = None) -> mx.array:
        embeddings = self.encoder(audio)
        embeddings = self.encoder_transformer(embeddings.swapaxes(1, 2)).swapaxes(1, 2)
        embeddings = self.downsample(embeddings)
        codes = self.quantizer.encode(embeddings, num_quantizers=num_codebooks)
        return codes.swapaxes(0, 1)

    def decode(self, codes: mx.array) -> mx.array:
        embeddings = self.quantizer.decode(codes)
        embeddings = self.upsample(embeddings)
        embeddings = self.decoder_transformer(embeddings.swapaxes(1, 2)).swapaxes(1, 2)
        return self.decoder(embeddings)

    def reset_state(self) -> None:
        return None

    def load_hf_weights(self, path: str | Path) -> None:
        state = mx.load(str(path))
        expected = self._expected_weight_names()
        assigned: set[str] = set()
        missing = []
        for name, value in state.items():
            if name == "__metadata__":
                continue
            if not self._assign_weight(name, value):
                missing.append(name)
            else:
                assigned.add(name)
        if missing:
            preview = ", ".join(missing[:10])
            raise WeightLoadError(f"Could not map {len(missing)} Mimi tensors: {preview}")
        absent = sorted(expected - assigned)
        if absent:
            preview = ", ".join(absent[:10])
            raise WeightLoadError(f"Missing {len(absent)} Mimi tensors: {preview}")

    def _assign_weight(self, name: str, value: mx.array) -> bool:
        target_name = name
        mapped = value
        if target_name.endswith(".weight"):
            module = self._resolve(target_name.removesuffix(".weight"))
            if isinstance(module, ConvTranspose1d):
                if module.groups == 1:
                    mapped = value.transpose(1, 2, 0)
                elif module.groups == module.in_channels == module.out_channels:
                    mapped = value.transpose(0, 2, 1)
                else:
                    return False
            elif isinstance(module, Conv1d):
                mapped = value.swapaxes(-1, -2)
            elif not hasattr(module, "weight"):
                return False
            self._validate_shape(target_name, mapped, module.weight)
            module.weight = mapped
            if hasattr(module, "_refresh_expanded_weight"):
                module._refresh_expanded_weight()
            return True
        target = self._resolve(target_name.rsplit(".", 1)[0])
        leaf = target_name.rsplit(".", 1)[1]
        if not hasattr(target, leaf):
            return False
        self._validate_shape(target_name, mapped, getattr(target, leaf))
        setattr(target, leaf, mapped)
        return True

    @staticmethod
    def _validate_shape(name: str, mapped: mx.array, expected: mx.array | None) -> None:
        if expected is None:
            raise WeightLoadError(f"Unexpected tensor for disabled parameter {name}")
        if tuple(mapped.shape) != tuple(expected.shape):
            raise WeightLoadError(
                f"Shape mismatch for {name}: checkpoint {tuple(mapped.shape)} "
                f"does not match MLX parameter {tuple(expected.shape)}"
            )

    def _resolve(self, path: str):
        current: object = self
        for part in path.split("."):
            if part.isdigit():
                current = current[int(part)]  # type: ignore[index]
            else:
                current = getattr(current, part)
        return current

    def _expected_weight_names(self) -> set[str]:
        names: set[str] = set()
        self._collect_weight_names(self, "", names)
        return names

    def _collect_weight_names(self, value: object, prefix: str, names: set[str]) -> None:
        if isinstance(value, mx.array):
            names.add(prefix)
            return
        if isinstance(value, (str, int, float, bool, type(None))):
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                child = f"{prefix}.{index}" if prefix else str(index)
                self._collect_weight_names(item, child, names)
            return
        if isinstance(value, dict):
            return
        if not hasattr(value, "__dict__"):
            return
        for name, item in vars(value).items():
            if name.startswith("_") or item is None:
                continue
            child = f"{prefix}.{name}" if prefix else name
            self._collect_weight_names(item, child, names)
