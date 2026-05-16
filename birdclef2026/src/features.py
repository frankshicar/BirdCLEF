"""Mel spectrogram feature extraction for BirdCLEF 2026."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torchaudio.transforms as T


class _PCEN(torch.nn.Module):
    """Per-Channel Energy Normalization (PCEN) — pure PyTorch implementation.

    Replaces torchaudio.transforms.PCEN which was removed in torchaudio 2.x.

    Formula: PCEN(t,f) = (E / (eps + M)^alpha + delta)^r - delta^r
    where M is an exponential moving average of E along the time axis.

    Args:
        num_bands: Number of mel frequency bands (n_mels).
        hop_length: STFT hop length (used to set default smoothing coefficient).
        alpha: Gain normalization exponent (default 0.98).
        delta: Bias term (default 2.0).
        r: Root compression exponent (default 0.5).
        eps: Stability constant (default 1e-6).
        s: EMA smoothing coefficient (default 0.025).
    """

    def __init__(
        self,
        num_bands: int,
        hop_length: int,
        alpha: float = 0.98,
        delta: float = 2.0,
        r: float = 0.5,
        eps: float = 1e-6,
        s: float = 0.025,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PCEN to a power mel spectrogram.

        Args:
            x: Power spectrogram of shape (n_mels, time_frames) or
               (batch, n_mels, time_frames).

        Returns:
            PCEN-compressed tensor of the same shape.
        """
        # Build EMA background estimate M along time axis
        # x shape: (..., freq, time)
        time_dim = x.dim() - 1
        frames = x.shape[time_dim]

        # Initialise M with the first frame
        m = x.select(time_dim, 0).unsqueeze(time_dim)
        m_frames = [m]
        for t in range(1, frames):
            frame = x.select(time_dim, t).unsqueeze(time_dim)
            m = (1.0 - self.s) * m + self.s * frame
            m_frames.append(m)
        M = torch.cat(m_frames, dim=time_dim)  # same shape as x

        # PCEN compression
        smooth = (self.eps + M) ** (-self.alpha)
        pcen = (x * smooth + self.delta) ** self.r - (self.delta ** self.r)
        return pcen


class MelSpectrogramExtractor:
    """Converts waveform segments to mel spectrogram tensors.

    Pipeline (PCEN mode):  mel filterbank → PCEN
    Pipeline (log-mel mode): mel filterbank → power_to_db → (x - mean) / std
    Output shape: (1, n_mels, time_frames) as torch.float32

    Args:
        use_pcen: If True, use PCEN instead of log-mel + normalization.
                  PCEN dynamically suppresses background noise and enhances
                  transient bird calls. mean/std/top_db are ignored when True.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        hop_length: int,
        n_fft: int,
        top_db: float = 80.0,
        mean: float = 0.0,
        std: float = 1.0,
        f_min: float = 500.0,
        f_max: float | None = 12000.0,
        use_pcen: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.top_db = top_db
        self.mean = mean
        self.std = std
        self.f_min = f_min
        self.f_max = f_max
        self.use_pcen = use_pcen

        self._mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate // 2,
        )

        if use_pcen:
            # torchaudio 2.x removed T.PCEN — use a pure-torch implementation
            self._pcen = _PCEN(num_bands=n_mels, hop_length=hop_length)
            self._to_db = None
        else:
            self._pcen = None
            self._to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        """Convert waveform to mel spectrogram tensor.

        Args:
            waveform: 1-D float32 numpy array.

        Returns:
            Float32 tensor of shape (1, n_mels, time_frames).
        """
        wav = torch.from_numpy(waveform).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        mel = self._mel(wav)  # (1, n_mels, time_frames), power spectrogram

        if self.use_pcen:
            # PCEN expects (freq, time) — squeeze batch/channel, then unsqueeze back
            pcen_out = self._pcen(mel.squeeze(0))  # (n_mels, time_frames)
            return pcen_out.unsqueeze(0).float()   # (1, n_mels, time_frames)
        else:
            mel_db = self._to_db(mel)  # power → dB
            std = self.std if self.std != 0.0 else 1.0
            mel_norm = (mel_db - self.mean) / std
            return mel_norm.float()    # (1, n_mels, time_frames)

    def fit_stats(self, dataset: Iterable[np.ndarray]) -> tuple[float, float]:
        """Compute mean and std over an iterable of waveforms (log-mel mode only).

        When use_pcen=True, PCEN handles normalization internally — this method
        returns (0.0, 1.0) immediately without processing any data.

        Args:
            dataset: Iterable of 1-D float32 numpy waveform arrays.

        Returns:
            (mean, std) tuple of floats.
        """
        if self.use_pcen:
            return 0.0, 1.0

        # Temporarily use identity normalization while computing stats
        saved_mean, saved_std = self.mean, self.std
        self.mean, self.std = 0.0, 1.0

        count = 0
        running_mean = 0.0
        running_m2 = 0.0

        for waveform in dataset:
            tensor = self(waveform)  # (1, n_mels, T)
            values = tensor.numpy().ravel()
            for v in values:
                count += 1
                delta = float(v) - running_mean
                running_mean += delta / count
                delta2 = float(v) - running_mean
                running_m2 += delta * delta2

        self.mean, self.std = saved_mean, saved_std

        if count < 2:
            return float(running_mean), 1.0

        variance = running_m2 / (count - 1)
        return float(running_mean), float(np.sqrt(variance))


import torch.nn as nn


class SpecAugment(nn.Module):
    """Applies time masking and frequency masking to a mel spectrogram.

    Wraps torchaudio.transforms.TimeMasking and FrequencyMasking.
    Requirements: 3.4
    """

    def __init__(self, time_mask_param: int = 30, freq_mask_param: int = 20) -> None:
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time and frequency masking.

        Args:
            x: Float tensor of shape (1, n_mels, time_frames).

        Returns:
            Augmented tensor of the same shape.
        """
        x = self.time_mask(x)
        x = self.freq_mask(x)
        return x


class MixupCollator:
    """Collate function that blends pairs of (spectrogram, label) tensors.

    Uses a Beta-distributed lambda for mixing: lam ~ Beta(alpha, alpha).
    Mixed output: lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2
    Requirements: 3.5
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend pairs of samples using Beta-distributed lambda.

        Args:
            batch: List of (spectrogram, label) tuples.

        Returns:
            (mixed_spectrograms, mixed_labels) stacked tensors.
        """
        specs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        lam = float(np.random.beta(self.alpha, self.alpha))

        # Roll batch by 1 to get pairing partners
        idx = torch.randperm(len(batch))
        mixed_specs = lam * specs + (1.0 - lam) * specs[idx]
        mixed_labels = lam * labels + (1.0 - lam) * labels[idx]

        return mixed_specs, mixed_labels
