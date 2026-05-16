"""Audio loading, preprocessing, and segment extraction."""

import logging
from math import ceil

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Load and preprocess .ogg audio files into normalized mono float32 waveforms."""

    def __init__(self, sample_rate: int = 32000, highpass_cutoff: float = 0.0) -> None:
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff  # Hz; 0 = disabled

    def load(self, path: str) -> np.ndarray | None:
        """Load a .ogg file, resample to sample_rate, convert to mono float32,
        and normalize amplitude to [-1.0, 1.0].

        Uses soundfile for I/O and torchaudio.functional.resample() for resampling.

        Args:
            path: Path to the .ogg audio file.

        Returns:
            1-D float32 numpy array, or None if the file is corrupt/unreadable.
        """
        try:
            data, src_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            logger.warning("Could not load audio file '%s': %s", path, exc)
            return None

        # data shape: (samples, channels) — convert to (channels, samples) tensor
        waveform = torch.from_numpy(data.T)  # shape: (channels, samples)

        # Resample if necessary using torchaudio.functional.resample
        if src_rate != self.sample_rate:
            waveform = F.resample(waveform, src_rate, self.sample_rate)

        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply high-pass filter to remove low-frequency noise (wind, hum, etc.)
        if self.highpass_cutoff > 0.0:
            waveform = F.highpass_biquad(waveform, self.sample_rate, self.highpass_cutoff)

        # Convert to float32 numpy array (shape: [samples])
        audio: np.ndarray = waveform.squeeze(0).numpy().astype(np.float32)

        # Skip files that are too short to be useful (< 0.05 seconds)
        if len(audio) < 1600:  # 0.05s at 32kHz
            return None

        return self.normalize(audio)

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform amplitude to [-1.0, 1.0].

        Divides by the maximum absolute value. Returns the waveform unchanged
        if the maximum absolute value is zero (silent audio).

        Args:
            waveform: 1-D float32 numpy array.

        Returns:
            Normalized float32 numpy array.
        """
        max_abs = np.max(np.abs(waveform))
        if max_abs == 0.0:
            return waveform
        return (waveform / max_abs).astype(np.float32)


class SegmentExtractor:
    """Split a waveform into fixed-length segments and assign Row_IDs.

    Row_ID format: ``{filename}_{end_seconds}`` where ``end_seconds`` is an
    integer (e.g. 5, 10, 15, …).  The final segment is zero-padded to
    ``segment_duration`` if it is shorter.
    """

    def __init__(
        self,
        segment_duration: float = 5.0,
        hop_duration: float = 5.0,
        sample_rate: int = 32000,
    ) -> None:
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.sample_rate = sample_rate
        self._segment_samples = int(segment_duration * sample_rate)
        self._hop_samples = int(hop_duration * sample_rate)

    def extract(
        self, waveform: np.ndarray, filename: str
    ) -> list[tuple[str, np.ndarray]]:
        """Split *waveform* into fixed-length segments.

        Args:
            waveform: 1-D float32 numpy array.
            filename: Base filename used to build Row_IDs (without extension).

        Returns:
            List of ``(row_id, segment_waveform)`` tuples.  Each
            ``segment_waveform`` has exactly ``segment_samples`` samples.
            ``row_id`` has the form ``{filename}_{end_seconds}``.
        """
        total_samples = len(waveform)
        segments: list[tuple[str, np.ndarray]] = []

        start = 0
        while start < total_samples:
            end = start + self._segment_samples
            segment = waveform[start:end]

            # Zero-pad the final (possibly short) segment
            if len(segment) < self._segment_samples:
                pad = np.zeros(self._segment_samples - len(segment), dtype=np.float32)
                segment = np.concatenate([segment, pad])

            end_seconds = int((start + self._segment_samples) / self.sample_rate)
            row_id = f"{filename}_{end_seconds}"
            segments.append((row_id, segment.astype(np.float32)))

            start += self._hop_samples

        return segments


class BackgroundNoiseMixer:
    """Mix a foreground bird call with random background noise.

    Simulates the domain gap between clean train_audio and noisy soundscapes.
    Noise is sampled from a directory of .ogg files (e.g. ESC-50 or field recordings).
    If no noise_dir is provided or no files are found, the waveform is returned unchanged.

    Args:
        noise_dir: Directory containing background noise .ogg files.
        sample_rate: Target sample rate.
        snr_db_range: (min_snr, max_snr) in dB. Higher = less noise.
        p: Probability of applying noise augmentation.
    """

    def __init__(
        self,
        noise_dir: str | None = None,
        sample_rate: int = 32000,
        snr_db_range: tuple[float, float] = (5.0, 30.0),
        p: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self.snr_db_range = snr_db_range
        self.p = p
        self._noise_files: list[str] = []

        if noise_dir is not None:
            import glob
            self._noise_files = glob.glob(f"{noise_dir}/**/*.ogg", recursive=True)
            self._noise_files += glob.glob(f"{noise_dir}/**/*.wav", recursive=True)
            logger.info("BackgroundNoiseMixer: found %d noise files", len(self._noise_files))

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Mix waveform with a random noise clip at a random SNR.

        Args:
            waveform: 1-D float32 numpy array (already normalized).

        Returns:
            Mixed waveform as float32 numpy array, re-normalized to [-1, 1].
        """
        if not self._noise_files or np.random.random() > self.p:
            return waveform

        noise_path = self._noise_files[np.random.randint(len(self._noise_files))]
        try:
            noise, src_rate = sf.read(noise_path, dtype="float32", always_2d=True)
            noise = torch.from_numpy(noise.T)
            if src_rate != self.sample_rate:
                noise = F.resample(noise, src_rate, self.sample_rate)
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            noise = noise.squeeze(0).numpy().astype(np.float32)
        except Exception as exc:
            logger.warning("Could not load noise file '%s': %s", noise_path, exc)
            return waveform

        # Tile or crop noise to match waveform length
        target_len = len(waveform)
        if len(noise) < target_len:
            repeats = ceil(target_len / len(noise))
            noise = np.tile(noise, repeats)
        start = np.random.randint(0, len(noise) - target_len + 1)
        noise = noise[start:start + target_len]

        # Mix at random SNR
        snr_db = np.random.uniform(*self.snr_db_range)
        signal_rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
        noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
        target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / noise_rms)

        mixed = waveform + noise
        # Re-normalize
        max_abs = np.max(np.abs(mixed))
        if max_abs > 0:
            mixed = mixed / max_abs
        return mixed.astype(np.float32)
