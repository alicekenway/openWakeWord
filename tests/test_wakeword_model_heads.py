"""Structural and integration tests for the CNN and attention wake-word heads."""

import pytest
import torch

from openwakeword.models import ConvAttentionWakeWordHead, TemporalCNNWakeWordHead
from openwakeword.models.blocks import TemporalResidualBlock
from openwakeword.train import Model as TrainModel


@pytest.mark.parametrize("head", [TemporalCNNWakeWordHead(), ConvAttentionWakeWordHead()])
def test_head_output_shape_is_binary_logits(head):
    logits = head(torch.randn(4, 16, 96))
    assert logits.shape == (4, 1)
    assert torch.isfinite(logits).all()


def test_heads_support_batch_size_one_and_backpropagation():
    for head in (TemporalCNNWakeWordHead(), ConvAttentionWakeWordHead()):
        features = torch.randn(1, 16, 96, requires_grad=True)
        logits = head(features)
        logits.mean().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()


def test_attention_rejects_invalid_feature_and_time_dimensions():
    head = ConvAttentionWakeWordHead(input_dim=96, time_steps=16)
    with pytest.raises(ValueError, match="feature dimension"):
        head(torch.randn(2, 16, 80))
    with pytest.raises(ValueError, match="timesteps"):
        head(torch.randn(2, 12, 96))


def test_temporal_block_requires_odd_kernel_size():
    with pytest.raises(ValueError, match="odd"):
        TemporalResidualBlock(channels=16, kernel_size=4, dilation=1)


@pytest.mark.parametrize(
    ("model_type", "model_kwargs"),
    [
        ("cnn", {"channels": 16, "classifier_hidden": 8}),
        (
            "attention",
            {
                "channels": 16,
                "num_heads": 4,
                "ff_multiplier": 2,
                "classifier_hidden": 8,
            },
        ),
    ],
)
def test_training_wrapper_converts_new_head_logits_to_scores(model_type, model_kwargs):
    trainer = TrainModel(
        n_classes=1,
        input_shape=(16, 96),
        model_type=model_type,
        model_kwargs=model_kwargs,
    )
    scores = trainer.model(torch.randn(2, 16, 96))
    assert scores.shape == (2, 1)
    assert torch.all((scores >= 0) & (scores <= 1))
