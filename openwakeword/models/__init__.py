"""Reusable neural classifier heads for openWakeWord feature embeddings."""

from .attention_wakeword import ConvAttentionWakeWordHead
from .cnn_wakeword import TemporalCNNWakeWordHead

__all__ = [
    "ConvAttentionWakeWordHead",
    "TemporalCNNWakeWordHead",
]
