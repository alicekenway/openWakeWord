"""Shared temporal CNN and attention blocks for wake-word classifier heads."""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcite1d(nn.Module):
    """Channel recalibration for an input temporal map of shape ``(B, C, T)``."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reduce = nn.Conv1d(channels, hidden, kernel_size=1)
        self.activation = nn.ReLU()
        self.expand = nn.Conv1d(hidden, channels, kernel_size=1)
        self.gate = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x)
        weights = self.reduce(weights)
        weights = self.activation(weights)
        weights = self.expand(weights)
        return x * self.gate(weights)


class TemporalResidualBlock(nn.Module):
    """Length-preserving depthwise temporal residual block for ``(B, C, T)``."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int = 1,
        use_se: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        if expansion < 1:
            raise ValueError("expansion must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        hidden_channels = channels * expansion
        padding = dilation * (kernel_size - 1) // 2
        self.expand = nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False)
        self.expand_norm = nn.BatchNorm1d(hidden_channels)
        self.depthwise = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm1d(hidden_channels)
        self.activation = nn.ReLU()
        self.squeeze_excite = SqueezeExcite1d(hidden_channels) if use_se else nn.Identity()
        self.project = nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False)
        self.project_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.output_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.expand(x)
        y = self.activation(self.expand_norm(y))
        y = self.depthwise(y)
        y = self.activation(self.depthwise_norm(y))
        y = self.squeeze_excite(y)
        y = self.project(y)
        y = self.project_norm(y)
        y = self.dropout(y)
        return self.output_activation(residual + y)


class TransformerTemporalBlock(nn.Module):
    """Pre-normalized temporal Transformer block for tensors shaped ``(B, T, C)``."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if num_heads < 1 or channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if ff_multiplier < 1:
            raise ValueError("ff_multiplier must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, ff_multiplier * channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention_norm(x)
        attention_output, _ = self.attention(y, y, y, need_weights=False)
        x = x + attention_output
        return x + self.feed_forward(self.feed_forward_norm(x))


class TemporalAttentionPooling(nn.Module):
    """Learned weighted temporal pooling from ``(B, T, C)`` to ``(B, C)``."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        hidden = hidden_channels if hidden_channels is not None else max(channels // 2, 16)
        if hidden < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)
