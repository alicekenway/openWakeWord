"""Pure temporal CNN classifier head for openWakeWord feature embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .blocks import TemporalResidualBlock


class TemporalCNNWakeWordHead(nn.Module):
    """Map ``(B, T, F)`` embeddings to binary logits shaped ``(B, 1)``."""

    def __init__(
        self,
        input_dim: int = 96,
        channels: int = 128,
        expansion: int = 1,
        dropout: float = 0.05,
        kernels: Sequence[int] = (3, 5, 3, 3),
        dilations: Sequence[int] = (1, 1, 2, 4),
        use_se: Sequence[bool] = (False, False, True, True),
        classifier_hidden: int = 64,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if classifier_hidden < 1:
            raise ValueError("classifier_hidden must be >= 1")
        if len(kernels) != len(dilations) or len(kernels) != len(use_se):
            raise ValueError("kernels, dilations, and use_se must have the same length")
        if not kernels:
            raise ValueError("at least one temporal residual block is required")

        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(
                    channels=channels,
                    kernel_size=int(kernel),
                    dilation=int(dilation),
                    expansion=expansion,
                    use_se=bool(se),
                    dropout=dropout,
                )
                for kernel, dilation, se in zip(kernels, dilations, use_se)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            if x.ndim != 3:
                raise ValueError(f"Expected input with shape (B, T, F), got {tuple(x.shape)}")
            if x.shape[-1] != self.input_dim:
                raise ValueError(f"Expected feature dimension {self.input_dim}, got {x.shape[-1]}")
        y = self.input_norm(x).transpose(1, 2)
        y = self.stem(y)
        for block in self.blocks:
            y = block(y)
        mean = torch.mean(y, dim=-1)
        maximum = torch.amax(y, dim=-1)
        return self.classifier(torch.cat((mean, maximum), dim=1))
