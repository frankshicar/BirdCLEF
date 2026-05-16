"""Unit tests for AudioPreprocessor (Task 2)."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdclef2026.src.audio import AudioPreprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ogg(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write a float32 mono waveform to an .ogg file using soundfile."""
    sf.write(path, samples, sample_rate, format="OGG", subtype="VORBIS")


# ---------------------------------------------------------------------------
# normalize() tests
# ---------------------------------------------------------------------------

def test_normalize_scales_to_unit_range():
    pre = AudioPreprocessor()
    waveform = np.array([0.0, 0.5, -2.0, 1.0], dtype=np.float32)
    result = pre.normalize(waveform)
    assert np.max(np.abs(result)) == pytest.approx(1.0)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)


def test_normalize_zero_amplitude_unchanged():
    pre = AudioPreprocessor()
    waveform = np.zeros(100, dtype=np.float32)
    result = pre.normalize(waveform)
    np.testing.assert_array_equal(result, waveform)


def test_normalize_already_unit_range():
    pre = AudioPreprocessor()
    waveform = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    result = pre.normalize(waveform)
    np.testing.assert_array_almost_equal(result, waveform)


def test_normalize_returns_float32():
    pre = AudioPreprocessor()
    waveform = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = pre.normalize(waveform.astype(np.float32))
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# load() tests
# ---------------------------------------------------------------------------

def test_load_returns_mono_float32(tmp_path):
    """load() should return a 1-D float32 numpy array."""
    samples = np.sin(2 * np.pi * 440 * np.arange(32000) / 32000).astype(np.float32)
    ogg_path = str(tmp_path / "test.ogg")
    _write_ogg(ogg_path, samples, 32000)

    pre = AudioPreprocessor(sample_rate=32000)
    result = pre.load(ogg_path)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.dtype == np.float32


def test_load_normalizes_amplitude(tmp_path):
    """load() should return a waveform with max abs value of 1.0."""
    samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(32000) / 32000)).astype(np.float32)
    ogg_path = str(tmp_path / "test.ogg")
    _write_ogg(ogg_path, samples, 32000)

    pre = AudioPreprocessor(sample_rate=32000)
    result = pre.load(ogg_path)

    assert result is not None
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-5)


def test_load_resamples_to_target_rate(tmp_path):
    """load() should resample audio to the configured sample_rate."""
    src_rate = 44100
    duration_sec = 1.0
    n_samples = int(src_rate * duration_sec)
    samples = np.sin(2 * np.pi * 440 * np.arange(n_samples) / src_rate).astype(np.float32)
    ogg_path = str(tmp_path / "test_44k.ogg")
    _write_ogg(ogg_path, samples, src_rate)

    target_rate = 32000
    pre = AudioPreprocessor(sample_rate=target_rate)
    result = pre.load(ogg_path)

    assert result is not None
    expected_len = round(n_samples * target_rate / src_rate)
    # Allow ±1 sample for rounding
    assert abs(len(result) - expected_len) <= 1


def test_load_stereo_converted_to_mono(tmp_path):
    """load() should convert stereo audio to mono."""
    src_rate = 32000
    n_samples = 32000
    stereo = np.stack([
        np.sin(2 * np.pi * 440 * np.arange(n_samples) / src_rate),
        np.sin(2 * np.pi * 880 * np.arange(n_samples) / src_rate),
    ], axis=1).astype(np.float32)
    ogg_path = str(tmp_path / "stereo.ogg")
    sf.write(ogg_path, stereo, src_rate, format="OGG", subtype="VORBIS")

    pre = AudioPreprocessor(sample_rate=src_rate)
    result = pre.load(ogg_path)

    assert result is not None
    assert result.ndim == 1


def test_load_corrupt_file_returns_none(tmp_path):
    """load() should return None and not raise on a corrupt file."""
    corrupt_path = str(tmp_path / "corrupt.ogg")
    Path(corrupt_path).write_bytes(b"this is not a valid ogg file")

    pre = AudioPreprocessor(sample_rate=32000)
    result = pre.load(corrupt_path)

    assert result is None


# ---------------------------------------------------------------------------
# SegmentExtractor tests (Task 3)
# ---------------------------------------------------------------------------

from birdclef2026.src.audio import SegmentExtractor


def test_segment_count_60s_recording():
    """60-second soundscape with 5s segments and 5s hop → 12 segments."""
    sr = 32000
    extractor = SegmentExtractor(segment_duration=5.0, hop_duration=5.0, sample_rate=sr)
    waveform = np.zeros(sr * 60, dtype=np.float32)
    segments = extractor.extract(waveform, "soundscape")
    assert len(segments) == 12


def test_segment_length_is_fixed():
    """Every segment must have exactly segment_samples samples."""
    sr = 32000
    extractor = SegmentExtractor(segment_duration=5.0, hop_duration=5.0, sample_rate=sr)
    waveform = np.random.randn(sr * 60).astype(np.float32)
    for _, seg in extractor.extract(waveform, "sc"):
        assert len(seg) == sr * 5


def test_row_id_format():
    """Row_IDs should follow the {filename}_{end_seconds} pattern."""
    sr = 32000
    extractor = SegmentExtractor(segment_duration=5.0, hop_duration=5.0, sample_rate=sr)
    waveform = np.zeros(sr * 15, dtype=np.float32)
    segments = extractor.extract(waveform, "BC2026_Test_0001")
    row_ids = [rid for rid, _ in segments]
    assert row_ids == ["BC2026_Test_0001_5", "BC2026_Test_0001_10", "BC2026_Test_0001_15"]


def test_short_soundscape_zero_padded():
    """A soundscape shorter than one segment is padded to segment_duration."""
    sr = 32000
    extractor = SegmentExtractor(segment_duration=5.0, hop_duration=5.0, sample_rate=sr)
    # Only 2 seconds of audio
    waveform = np.ones(sr * 2, dtype=np.float32)
    segments = extractor.extract(waveform, "short")
    assert len(segments) == 1
    _, seg = segments[0]
    assert len(seg) == sr * 5
    # The last 3 seconds should be zeros
    assert np.all(seg[sr * 2:] == 0.0)


def test_segment_dtype_is_float32():
    """Segments must be float32 arrays."""
    sr = 32000
    extractor = SegmentExtractor(segment_duration=5.0, hop_duration=5.0, sample_rate=sr)
    waveform = np.random.randn(sr * 10).astype(np.float64)
    for _, seg in extractor.extract(waveform.astype(np.float32), "sc"):
        assert seg.dtype == np.float32


def test_overlapping_segments():
    """With hop < segment_duration, segments should overlap."""
    sr = 32000
    extractor = SegmentExtractor(segment_duration=5.0, hop_duration=2.5, sample_rate=sr)
    waveform = np.zeros(sr * 10, dtype=np.float32)
    segments = extractor.extract(waveform, "sc")
    # starts: 0, 2.5s, 5s, 7.5s → 4 segments
    assert len(segments) == 4
