"""Unit tests for BirdCLEFModel (Task 9)."""

import torch
import pytest
from birdclef2026.src.model import BirdCLEFModel


BACKBONE = "efficientnet_b0"
NUM_CLASSES = 234
BATCH_SIZE = 2
N_MELS = 128
TIME_FRAMES = 157  # ~5 seconds at hop_length=320, sr=32000


@pytest.fixture
def model_avg():
    """BirdCLEFModel with average pooling (no pretrained weights)."""
    return BirdCLEFModel(backbone_name=BACKBONE, num_classes=NUM_CLASSES, pretrained=False, pool="avg")


@pytest.fixture
def model_attention():
    """BirdCLEFModel with attention pooling (no pretrained weights)."""
    return BirdCLEFModel(backbone_name=BACKBONE, num_classes=NUM_CLASSES, pretrained=False, pool="attention")


@pytest.fixture
def sample_input():
    """A batch of single-channel mel spectrograms."""
    return torch.randn(BATCH_SIZE, 1, N_MELS, TIME_FRAMES)


def test_output_shape_avg_pool(model_avg, sample_input):
    """Output shape should be (B, 234) for avg pooling."""
    model_avg.eval()
    with torch.no_grad():
        logits = model_avg(sample_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"
    )


def test_output_shape_attention_pool(model_attention, sample_input):
    """Output shape should be (B, 234) for attention pooling."""
    model_attention.eval()
    with torch.no_grad():
        logits = model_attention(sample_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"
    )


def test_output_dtype(model_avg, sample_input):
    """Output dtype should be torch.float32."""
    model_avg.eval()
    with torch.no_grad():
        logits = model_avg(sample_input)
    assert logits.dtype == torch.float32, f"Expected float32, got {logits.dtype}"


def test_logits_can_be_outside_unit_interval(model_avg):
    """Logits should be raw (no sigmoid), so values can be outside [0, 1]."""
    model_avg.eval()
    # Use a large input to push logits outside [0, 1]
    x = torch.randn(4, 1, N_MELS, TIME_FRAMES) * 10.0
    with torch.no_grad():
        logits = model_avg(x)
    # After many forward passes with random weights, some logits should be outside [0,1]
    # We just verify no sigmoid was applied by checking the range isn't strictly [0,1]
    has_outside = (logits > 1.0).any() or (logits < 0.0).any()
    assert has_outside, (
        "All logits are in [0, 1] — sigmoid may have been applied in forward(). "
        "Raw logits should be able to exceed this range."
    )


def test_batch_size_one(model_avg):
    """Model should work with batch size 1."""
    model_avg.eval()
    x = torch.randn(1, 1, N_MELS, TIME_FRAMES)
    with torch.no_grad():
        logits = model_avg(x)
    assert logits.shape == (1, NUM_CLASSES)


def test_avg_and_attention_same_output_shape(sample_input):
    """Both pooling modes should produce the same output shape."""
    m_avg = BirdCLEFModel(backbone_name=BACKBONE, num_classes=NUM_CLASSES, pretrained=False, pool="avg")
    m_att = BirdCLEFModel(backbone_name=BACKBONE, num_classes=NUM_CLASSES, pretrained=False, pool="attention")
    m_avg.eval()
    m_att.eval()
    with torch.no_grad():
        out_avg = m_avg(sample_input)
        out_att = m_att(sample_input)
    assert out_avg.shape == out_att.shape
