"""Unit tests for MelSpectrogramExtractor (birdclef2026/src/features.py)."""

import numpy as np
import pytest
import torch

from birdclef2026.src.features import MelSpectrogramExtractor


# Default extractor config matching design defaults
SAMPLE_RATE = 32000
N_MELS = 128
HOP_LENGTH = 320
N_FFT = 1024
SEGMENT_SAMPLES = SAMPLE_RATE * 5  # 5-second segment


@pytest.fixture
def extractor():
    return MelSpectrogramExtractor(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        top_db=80.0,
        mean=0.0,
        std=1.0,
    )


def make_waveform(n_samples: int = SEGMENT_SAMPLES) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(n_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

def test_output_shape_has_correct_n_mels(extractor):
    """Output shape should be (1, n_mels, T) for some T > 0."""
    wav = make_waveform()
    out = extractor(wav)
    assert out.shape[0] == 1
    assert out.shape[1] == N_MELS
    assert out.shape[2] > 0


def test_output_shape_is_3d(extractor):
    """Output tensor must be 3-dimensional."""
    wav = make_waveform()
    out = extractor(wav)
    assert out.dim() == 3


# ---------------------------------------------------------------------------
# Output dtype test
# ---------------------------------------------------------------------------

def test_output_dtype_is_float32(extractor):
    """Output tensor dtype must be torch.float32."""
    wav = make_waveform()
    out = extractor(wav)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# fit_stats tests
# ---------------------------------------------------------------------------

def test_fit_stats_returns_two_floats(extractor):
    """fit_stats() should return a tuple of two Python floats."""
    waveforms = [make_waveform() for _ in range(5)]
    result = extractor.fit_stats(iter(waveforms))
    assert isinstance(result, tuple)
    assert len(result) == 2
    mean, std = result
    assert isinstance(mean, float)
    assert isinstance(std, float)


def test_fit_stats_std_is_positive(extractor):
    """fit_stats() std should be positive for non-constant input."""
    waveforms = [make_waveform(SEGMENT_SAMPLES) for _ in range(3)]
    _, std = extractor.fit_stats(iter(waveforms))
    assert std > 0.0


def test_fit_stats_does_not_mutate_extractor_params(extractor):
    """fit_stats() should not permanently change mean/std on the extractor."""
    original_mean = extractor.mean
    original_std = extractor.std
    waveforms = [make_waveform() for _ in range(3)]
    extractor.fit_stats(iter(waveforms))
    assert extractor.mean == original_mean
    assert extractor.std == original_std


# ---------------------------------------------------------------------------
# Normalization test
# ---------------------------------------------------------------------------

def test_normalization_shifts_output():
    """Using non-zero mean/std should shift the output values."""
    wav = make_waveform()

    ext_default = MelSpectrogramExtractor(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT,
        mean=0.0, std=1.0,
    )
    ext_shifted = MelSpectrogramExtractor(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT,
        mean=10.0, std=2.0,
    )

    out_default = ext_default(wav)
    out_shifted = ext_shifted(wav)

    # They should differ
    assert not torch.allclose(out_default, out_shifted)
    # Round-trip: out_shifted * std + mean ≈ out_default
    recovered = out_shifted * 2.0 + 10.0
    assert torch.allclose(recovered, out_default, atol=1e-4)


# ---------------------------------------------------------------------------
# SpecAugment tests
# ---------------------------------------------------------------------------

from birdclef2026.src.features import SpecAugment, MixupCollator


def test_spec_augment_is_nn_module():
    """SpecAugment must be an nn.Module."""
    import torch.nn as nn
    aug = SpecAugment()
    assert isinstance(aug, nn.Module)


def test_spec_augment_forward_preserves_shape():
    """SpecAugment forward should return a tensor with the same shape as input."""
    aug = SpecAugment(time_mask_param=30, freq_mask_param=20)
    x = torch.randn(1, 128, 500)
    out = aug(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# MixupCollator tests
# ---------------------------------------------------------------------------

def test_mixup_collator_output_shapes_match_input():
    """MixupCollator output shapes should match stacked input shapes."""
    collator = MixupCollator(alpha=0.4)
    batch = [(torch.randn(1, 128, 500), torch.zeros(234)) for _ in range(4)]
    specs, labels = collator(batch)
    assert specs.shape == (4, 1, 128, 500)
    assert labels.shape == (4, 234)


def test_mixup_collator_values_in_valid_range():
    """MixupCollator output should be a convex combination (values in [0, 1] for label inputs in [0, 1])."""
    torch.manual_seed(0)
    np.random.seed(0)
    collator = MixupCollator(alpha=0.4)
    # Use label tensors with values in [0, 1]
    batch = [(torch.randn(1, 128, 100), torch.rand(234)) for _ in range(4)]
    _, labels = collator(batch)
    assert labels.min().item() >= 0.0
    assert labels.max().item() <= 1.0
