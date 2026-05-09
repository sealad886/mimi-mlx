from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


def gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def elu(x: mx.array, alpha: float = 1.0) -> mx.array:
    return mx.maximum(x, 0) + alpha * (mx.exp(mx.minimum(x, 0)) - 1)


def linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
    return ((x - mean) * mx.rsqrt(var + eps)) * weight + bias


def pad1d(x: mx.array, left: int, right: int, mode: str) -> mx.array:
    if left == 0 and right == 0:
        return x
    if mode == "replicate":
        mode = "edge"
    return mx.pad(x, [(0, 0), (0, 0), (left, right)], mode=mode)


def causal_mask(length: int, dtype: mx.Dtype, sliding_window: int | None = None) -> mx.array:
    pos = mx.arange(length)
    mask = pos[None, :] > pos[:, None]
    if sliding_window is not None and sliding_window > 0:
        mask = mx.logical_or(mask, pos[None, :] <= (pos[:, None] - sliding_window))
    return mx.where(mask, mx.array(-1e9, dtype=dtype), mx.array(0.0, dtype=dtype))


def attention_mask(
    length: int,
    dtype: mx.Dtype,
    sliding_window: int | None = None,
) -> str | mx.array:
    if sliding_window is None or sliding_window <= 0 or sliding_window >= length:
        return "causal"
    return causal_mask(length, dtype, sliding_window)


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(
    q: mx.array,
    k: mx.array,
    *,
    base: float,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    dim = q.shape[-1]
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb)[None, None, :, :]
    sin = mx.sin(emb)[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@dataclass
class Conv1d:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    groups: int = 1
    bias_enabled: bool = True

    def __post_init__(self) -> None:
        self.weight = mx.zeros(
            (self.out_channels, self.kernel_size, self.in_channels // self.groups),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((self.out_channels,), dtype=mx.float32) if self.bias_enabled else None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(
            x.swapaxes(1, 2),
            self.weight,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        ).swapaxes(1, 2)
        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y


@dataclass
class ConvTranspose1d:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    groups: int = 1
    bias_enabled: bool = True

    def __post_init__(self) -> None:
        if self.in_channels % self.groups != 0 or self.out_channels % self.groups != 0:
            raise ValueError("ConvTranspose1d channels must be divisible by groups")
        self.weight = mx.zeros(
            (self.out_channels, self.kernel_size, self.in_channels // self.groups),
            dtype=mx.float32,
        )
        self.bias = mx.zeros((self.out_channels,), dtype=mx.float32) if self.bias_enabled else None

    def _refresh_expanded_weight(self) -> None:
        if self.groups != 1 and self.groups != self.in_channels:
            raise ValueError("Grouped ConvTranspose1d only supports depthwise or groups=1")

    def __call__(self, x: mx.array) -> mx.array:
        self._refresh_expanded_weight()
        y = mx.conv_transpose1d(
            x.swapaxes(1, 2),
            self.weight,
            stride=self.stride,
            groups=self.groups,
        ).swapaxes(1, 2)
        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y


class MimiConv1d:
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: str | None = None,
        bias: bool = True,
    ):
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.conv = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias_enabled=bias,
        )
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.stride = stride
        self.padding_total = self.kernel_size - stride
        self.padding_right = self.padding_total // 2
        self.padding_left = self.padding_total - self.padding_right

    def get_output_length(self, input_length: int) -> int:
        nframes = (
            math.ceil((input_length - self.kernel_size + self.padding_total) / self.stride + 1) - 1
        )
        ideal_length = nframes * self.stride + self.kernel_size - self.padding_total
        extra_padding = ideal_length - input_length
        if self.causal:
            padding_left = self.padding_total
            padding_right = extra_padding
        else:
            padding_left = self.padding_left
            padding_right = self.padding_right + extra_padding
        padded = input_length + padding_left + padding_right
        return (padded - self.kernel_size) // self.stride + 1

    def __call__(self, x: mx.array) -> mx.array:
        nframes = (
            math.ceil((x.shape[-1] - self.kernel_size + self.padding_total) / self.stride + 1) - 1
        )
        ideal_length = nframes * self.stride + self.kernel_size - self.padding_total
        extra_padding = max(0, ideal_length - x.shape[-1])
        if self.causal:
            x = pad1d(x, self.padding_total, extra_padding, self.pad_mode)
        else:
            x = pad1d(x, self.padding_left, self.padding_right + extra_padding, self.pad_mode)
        return self.conv(x)


class MimiConvTranspose1d:
    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.conv = ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias_enabled=bias,
        )
        padding_total = kernel_size - stride
        self.padding_right = (
            math.ceil(padding_total * self.trim_right_ratio) if self.causal else padding_total // 2
        )
        self.padding_left = padding_total - self.padding_right

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        end = x.shape[-1] - self.padding_right
        return x[..., self.padding_left : end]


class MimiResnetBlock:
    def __init__(self, config, dim: int, dilations: list[int]):
        hidden = dim // config.compress
        self.block = [
            "elu",
            MimiConv1d(config, dim, hidden, config.residual_kernel_size, dilation=dilations[0]),
            "elu",
            MimiConv1d(config, hidden, dim, 1, dilation=dilations[1]),
        ]
        self.shortcut = None
        if getattr(config, "use_conv_shortcut", False):
            self.shortcut = MimiConv1d(config, dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        for layer in self.block:
            x = elu(x) if layer == "elu" else layer(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return residual + x


class MimiEncoder:
    def __init__(self, config):
        layers: list[object] = [
            MimiConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)
        ]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                layers.append(
                    MimiResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])
                )
            layers.append("elu")
            layers.append(
                MimiConv1d(config, current_scale, current_scale * 2, ratio * 2, stride=ratio)
            )
            scaling *= 2
        layers.append("elu")
        layers.append(
            MimiConv1d(
                config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size
            )
        )
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = elu(x) if layer == "elu" else layer(x)
        return x


class MimiDecoder:
    def __init__(self, config):
        scaling = 2 ** len(config.upsampling_ratios)
        layers: list[object] = [
            MimiConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)
        ]
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            layers.append("elu")
            layers.append(
                MimiConvTranspose1d(
                    config, current_scale, current_scale // 2, ratio * 2, stride=ratio
                )
            )
            for j in range(config.num_residual_layers):
                layers.append(
                    MimiResnetBlock(config, current_scale // 2, [config.dilation_growth_rate**j, 1])
                )
            scaling //= 2
        layers.append("elu")
        layers.append(
            MimiConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)
        )
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = elu(x) if layer == "elu" else layer(x)
        return x


class MimiAttention:
    def __init__(self, config):
        self.config = config
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = LinearWeights(hidden, self.num_heads * self.head_dim)
        self.k_proj = LinearWeights(hidden, self.num_key_value_heads * self.head_dim)
        self.v_proj = LinearWeights(hidden, self.num_key_value_heads * self.head_dim)
        self.o_proj = LinearWeights(self.num_heads * self.head_dim, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = (
            self.k_proj(x)
            .reshape(batch, length, self.num_key_value_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.num_key_value_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        positions = mx.arange(length, dtype=mx.int32)
        q, k = apply_rope(q, k, base=self.config.rope_theta, positions=positions)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=attention_mask(length, q.dtype, self.config.sliding_window),
        )
        out = out.swapaxes(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(out)


@dataclass
class LinearWeights:
    in_features: int
    out_features: int
    bias_enabled: bool = False

    def __post_init__(self) -> None:
        self.weight = mx.zeros((self.out_features, self.in_features), dtype=mx.float32)
        self.bias = mx.zeros((self.out_features,), dtype=mx.float32) if self.bias_enabled else None
        self._weight_t: mx.array | None = None
        self._weight_signature: int | None = None

    def __call__(self, x: mx.array) -> mx.array:
        signature = id(self.weight)
        if self._weight_signature != signature:
            self._weight_t = self.weight.T
            self._weight_signature = signature
        return x @ self._weight_t if self.bias is None else (x @ self._weight_t) + self.bias


class MimiMLP:
    def __init__(self, config):
        self.fc1 = LinearWeights(config.hidden_size, config.intermediate_size)
        self.fc2 = LinearWeights(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(gelu(self.fc1(x)))


class MimiLayerNorm:
    def __init__(self, hidden_size: int):
        self.weight = mx.ones((hidden_size,), dtype=mx.float32)
        self.bias = mx.zeros((hidden_size,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return layer_norm(x, self.weight, self.bias)


class MimiLayerScale:
    def __init__(self, hidden_size: int, initial_scale: float):
        self.scale = mx.full((hidden_size,), initial_scale, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x


class MimiTransformerLayer:
    def __init__(self, config):
        self.self_attn = MimiAttention(config)
        self.mlp = MimiMLP(config)
        self.input_layernorm = MimiLayerNorm(config.hidden_size)
        self.post_attention_layernorm = MimiLayerNorm(config.hidden_size)
        self.self_attn_layer_scale = MimiLayerScale(
            config.hidden_size, config.layer_scale_initial_scale
        )
        self.mlp_layer_scale = MimiLayerScale(config.hidden_size, config.layer_scale_initial_scale)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn_layer_scale(self.self_attn(x))
        residual = x
        x = self.post_attention_layernorm(x)
        return residual + self.mlp_layer_scale(self.mlp(x))


class MimiTransformerModel:
    def __init__(self, config):
        self.layers = [MimiTransformerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x
