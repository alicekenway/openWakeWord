"""Convolution-attention classifier head for openWakeWord feature embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .blocks import TemporalAttentionPooling, TemporalResidualBlock, TransformerTemporalBlock


class ConvAttentionWakeWordHead(nn.Module):
    """Map fixed ``(B, T, F)`` embeddings to binary logits shaped ``(B, 1)``."""

    def __init__(
        self,
        input_dim: int = 96,
        time_steps: int = 16,
        channels: int = 128,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        dropout: float = 0.05,
        expansion: int = 1,
        local_kernels: Sequence[int] = (3, 3),
        local_dilations: Sequence[int] = (1, 2),
        local_use_se: Sequence[bool] = (False, False),
        classifier_hidden: int = 64,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if time_steps < 1:
            raise ValueError("time_steps must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if classifier_hidden < 1:
            raise ValueError("classifier_hidden must be >= 1")
        if len(local_kernels) != len(local_dilations) or len(local_kernels) != len(local_use_se):
            raise ValueError("local_kernels, local_dilations, and local_use_se must have the same length")
        if not local_kernels:
            raise ValueError("at least one local temporal convolution block is required")

        self.input_dim = input_dim
        self.time_steps = time_steps
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, channels)
        self.local_blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    channels=channels,
                    kernel_size=int(kernel),
                    dilation=int(dilation),
                    expansion=expansion,
                    use_se=bool(se),
                    dropout=dropout,
                )
                for kernel, dilation, se in zip(local_kernels, local_dilations, local_use_se)
            ]
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, time_steps, channels))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.transformer = TransformerTemporalBlock(
            channels=channels,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )
        self.pool = TemporalAttentionPooling(channels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            if x.ndim != 3:
                raise ValueError(f"Expected input with shape (B, T, F), got {tuple(x.shape)}")
            if x.shape[-1] != self.input_dim:
                raise ValueError(f"Expected feature dimension {self.input_dim}, got {x.shape[-1]}")
            if x.shape[1] != self.time_steps:
                raise ValueError(f"Expected {self.time_steps} timesteps, got {x.shape[1]}")
        y = self.input_projection(self.input_norm(x)).transpose(1, 2)
        for block in self.local_blocks:
            y = block(y)
        y = y.transpose(1, 2) + self.position_embedding
        y = self.transformer(y)
        return self.classifier(self.pool(y))
