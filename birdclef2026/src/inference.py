"""BirdCLEF 2026 Inference Engine — CPU-only soundscape prediction."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from birdclef2026.src.checkpoint_utils import validate_checkpoint

logger = logging.getLogger(__name__)

# Default feature-extraction parameters (overridden by checkpoint config)
# These must match the training defaults in default.yaml / local.yaml
_DEFAULT_CONFIG = {
    "sample_rate": 32000,
    "segment_duration": 5.0,
    "hop_duration": 5.0,
    "n_mels": 160,
    "hop_length": 320,
    "n_fft": 2048,
    "top_db": 80.0,
    "mel_mean": 0.0,
    "mel_std": 1.0,
    "f_min": 50.0,
    "f_max": 15000.0,
    "highpass_cutoff": 50.0,
    "use_pcen": False,
    "use_denoiser": False,
    "denoiser_channels": 64,
    "backbone": "resnet18",
    "pool": "avg",
}


_TTA_VIEWS = 3  # base view plus semantic-preserving time-shifted views


class InferenceEngine:
    """Run CPU-only inference on soundscape recordings.

    Args:
        checkpoint_path: Path to the trained model checkpoint (.pt).
            Used for single-checkpoint inference.
        checkpoint_paths: Optional list of checkpoint paths for ensemble
            inference.  When provided, probabilities are averaged across all
            checkpoints.  ``checkpoint_path`` is still required (it is used as
            the primary checkpoint for config / label_map loading).
        backbone_weights_path: Optional path to pretrained backbone weights.
            If provided, its existence is verified by ``verify_paths()``.
        device: Torch device string (default ``"cpu"``).
        batch_size: Number of segments to process per forward pass.
        tta: If ``True``, average semantic-preserving time-shift views.
            Inference does not use SpecAugment because masking removes
            evidence from short 5-second bioacoustic windows.
        tta_views: Number of augmented views to average when ``tta=True``
            (default 4).
    """

    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_paths: list[str] | None = None,
        backbone_weights_path: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        tta: bool = False,
        tta_views: int = _TTA_VIEWS,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        # Build the full list of checkpoints to use for ensemble.
        # If checkpoint_paths is given, use it; otherwise fall back to the
        # single checkpoint_path so the rest of the code is uniform.
        self.checkpoint_paths: list[str] = (
            checkpoint_paths if checkpoint_paths is not None else [checkpoint_path]
        )
        self.backbone_weights_path = backbone_weights_path
        self.device = device
        self.batch_size = batch_size
        self.tta = tta
        self.tta_views = tta_views

        # One model entry per checkpoint; loaded lazily.
        self._models: list[nn.Module] = []
        # Convenience alias kept for backward-compat with _inject_model helper.
        self._model: nn.Module | None = None
        self._config: dict = {}
        self._label_map: dict[str, int] = {}
        self._species_columns: list[str] = []

    # ------------------------------------------------------------------
    # Path verification
    # ------------------------------------------------------------------

    def verify_paths(self) -> None:
        """Verify that all required paths exist.

        Raises:
            FileNotFoundError: with the missing path in the message.
        """
        for path in [self.checkpoint_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required path does not exist: {path}"
                )
        if self.backbone_weights_path is not None:
            if not os.path.exists(self.backbone_weights_path):
                raise FileNotFoundError(
                    f"Required path does not exist: {self.backbone_weights_path}"
                )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load and prepare all checkpoint models."""
        primary_ckpt = validate_checkpoint(self.checkpoint_path)
        self._config = {**_DEFAULT_CONFIG, **primary_ckpt.get("config", {})}
        self._label_map = primary_ckpt.get("label_map", {})

        if self._label_map:
            self._species_columns = [
                label
                for label, _ in sorted(self._label_map.items(), key=lambda x: x[1])
            ]

        self._models = []
        for ckpt_path in self.checkpoint_paths:
            ckpt = validate_checkpoint(ckpt_path)
            ckpt_config = {**_DEFAULT_CONFIG, **ckpt.get("config", {})}
            model = _build_model_from_config(ckpt_config, ckpt)
            model.eval()
            self._models.append(model)

        self._model = self._models[0] if self._models else None

        self._prepare_processors()

    def _prepare_processors(self) -> None:
        """Build reusable audio/feature processors from the resolved config."""
        from birdclef2026.src.audio import AudioPreprocessor, SegmentExtractor
        from birdclef2026.src.features import MelSpectrogramExtractor

        cfg = {**_DEFAULT_CONFIG, **self._config}
        self._preprocessor = AudioPreprocessor(
            sample_rate=cfg.get("sample_rate", 32000),
            highpass_cutoff=cfg.get("highpass_cutoff", 0.0),
        )
        # Inference always uses non-overlapping 5s windows (hop = segment)
        self._seg_extractor = SegmentExtractor(
            segment_duration=cfg.get("segment_duration", 5.0),
            hop_duration=cfg.get("segment_duration", 5.0),  # no overlap at inference
            sample_rate=cfg.get("sample_rate", 32000),
        )
        self._mel_extractor = MelSpectrogramExtractor(
            sample_rate=cfg.get("sample_rate", 32000),
            n_mels=cfg.get("n_mels", 128),
            hop_length=cfg.get("hop_length", 320),
            n_fft=cfg.get("n_fft", 1024),
            top_db=cfg.get("top_db", 80.0),
            mean=cfg.get("mel_mean", 0.0),
            std=cfg.get("mel_std", 1.0),
            f_min=cfg.get("f_min", 50.0),
            f_max=cfg.get("f_max", 15000.0),
            use_pcen=cfg.get("use_pcen", False),
        )
        self._tta_time_shifts = self._make_tta_time_shifts(self.tta_views) if self.tta else [0]

    # ------------------------------------------------------------------
    # Soundscape prediction
    # ------------------------------------------------------------------

    def predict_soundscape(self, soundscape_path: str) -> dict[str, np.ndarray]:
        """Predict species probabilities for all 5-second segments.

        When multiple checkpoints are configured, probabilities are averaged
        across all checkpoints (ensemble).  When ``tta=True``, each segment is
        rolled along the spectrogram time axis and predictions are averaged.

        Args:
            soundscape_path: Path to a soundscape ``.ogg`` file.

        Returns:
            Mapping ``{row_id: prob_vector}`` where each ``prob_vector`` is a
            float32 array of length 234 with sigmoid-activated probabilities.
        """
        if not self._models:
            self._load_model()
        elif not hasattr(self, "_preprocessor"):
            self._prepare_processors()

        filename = Path(soundscape_path).stem
        waveform = self._preprocessor.load(soundscape_path)
        if waveform is None:
            return {}

        segments = self._seg_extractor.extract(waveform, filename)
        if not segments:
            return {}

        row_ids = [row_id for row_id, _ in segments]
        base_specs = [self._mel_extractor(seg) for _, seg in segments]

        results: dict[str, np.ndarray] = {}

        # ---------------------------------------------------------------
        # Helper: run one model over a list of spec tensors in batches,
        # return (N, num_classes) numpy array of sigmoid probabilities.
        # ---------------------------------------------------------------
        def _run_model(model: nn.Module, specs: list[torch.Tensor]) -> np.ndarray:
            all_probs: list[np.ndarray] = []
            for batch_start in range(0, len(specs), self.batch_size):
                batch_specs = specs[batch_start: batch_start + self.batch_size]
                batch_tensor = torch.stack(batch_specs)  # (B, 1, n_mels, T)
                with torch.no_grad():
                    logits = model(batch_tensor)  # (B, num_classes)
                if torch.any(~torch.isfinite(logits)):
                    logger.warning(
                        "NaN or Inf detected in model output for '%s'; replacing with 0.0.",
                        soundscape_path,
                    )
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                all_probs.append(probs)
            return np.concatenate(all_probs, axis=0)  # (N, num_classes)

        # ---------------------------------------------------------------
        # Ensemble × TTA: accumulate probability arrays, then average.
        # ---------------------------------------------------------------
        # Shape: (N_segments, num_classes)
        accumulated: np.ndarray | None = None
        n_accumulations = 0

        for model in self._models:
            if self.tta:
                tta_acc: np.ndarray | None = None
                for shift_fraction in self._tta_time_shifts:
                    aug_specs = [
                        self._roll_spec_time(spec, shift_fraction)
                        for spec in base_specs
                    ]
                    view_probs = _run_model(model, aug_specs)
                    tta_acc = view_probs if tta_acc is None else tta_acc + view_probs
                assert tta_acc is not None
                model_probs = tta_acc / len(self._tta_time_shifts)
            else:
                model_probs = _run_model(model, base_specs)

            accumulated = model_probs if accumulated is None else accumulated + model_probs
            n_accumulations += 1

        assert accumulated is not None
        final_probs = accumulated / n_accumulations  # element-wise mean

        for row_id, prob_vec in zip(row_ids, final_probs):
            results[row_id] = prob_vec

        return results

    @staticmethod
    def _make_tta_time_shifts(tta_views: int) -> list[float]:
        """Return deterministic fractional time shifts for TTA."""
        if tta_views <= 1:
            return [0.0]
        if tta_views == 2:
            return [0.0, 0.5]
        if tta_views == 3:
            return [0.0, 1.0 / 3.0, 2.0 / 3.0]
        return [i / float(tta_views) for i in range(tta_views)]

    @staticmethod
    def _roll_spec_time(spec: torch.Tensor, shift_fraction: float) -> torch.Tensor:
        """Roll a spectrogram along time without masking or changing values."""
        if shift_fraction == 0.0:
            return spec
        frames = spec.shape[-1]
        shift = int(round(frames * shift_fraction))
        if shift == 0:
            return spec
        return torch.roll(spec, shifts=shift, dims=-1)

    # ------------------------------------------------------------------
    # Full submission run
    # ------------------------------------------------------------------

    def run(
        self,
        soundscape_dir: str,
        sample_submission_path: str,
        output_path: str = "submission.csv",
    ) -> None:
        """Process all soundscapes and write ``submission.csv``.

        Args:
            soundscape_dir: Directory containing ``.ogg`` soundscape files.
            sample_submission_path: Path to ``sample_submission.csv`` which
                defines the required row IDs and column order.
            output_path: Destination path for the generated submission CSV.
        """
        if not self._models:
            self._load_model()

        # Load sample submission to get required row IDs and column order
        sample_df = pd.read_csv(sample_submission_path)
        required_row_ids: list[str] = sample_df["row_id"].tolist()
        species_cols: list[str] = [c for c in sample_df.columns if c != "row_id"]

        # Collect all soundscape files
        soundscape_files = sorted(
            p for p in Path(soundscape_dir).iterdir()
            if p.suffix.lower() == ".ogg"
        )

        # Predict for each soundscape
        all_predictions: dict[str, np.ndarray] = {}
        for sf_path in soundscape_files:
            preds = self.predict_soundscape(str(sf_path))
            all_predictions.update(preds)

        num_classes = len(species_cols)
        zero_vec = np.zeros(num_classes, dtype=np.float32)

        # Build output rows in the order required by sample_submission.csv
        rows = []
        for row_id in required_row_ids:
            prob_vec = all_predictions.get(row_id, zero_vec)
            # Ensure correct length
            if len(prob_vec) != num_classes:
                logger.warning(
                    "Prediction for '%s' has length %d, expected %d; padding with 0.0.",
                    row_id, len(prob_vec), num_classes,
                )
                padded = np.zeros(num_classes, dtype=np.float32)
                padded[: len(prob_vec)] = prob_vec[:num_classes]
                prob_vec = padded
            rows.append([row_id] + prob_vec.tolist())

        out_df = pd.DataFrame(rows, columns=["row_id"] + species_cols)
        out_df.to_csv(output_path, index=False)
        logger.info("Submission written to '%s' (%d rows).", output_path, len(out_df))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model_from_config(config: dict, checkpoint: dict) -> nn.Module:
    """Reconstruct a model from checkpoint config and load its weights.

    Falls back to a simple linear model if the backbone is unavailable
    (e.g. timm not installed), but in practice timm is always present.

    Args:
        config: Resolved config dict from the checkpoint.
        checkpoint: Full checkpoint dict (must contain ``model_state_dict``).

    Returns:
        A ``torch.nn.Module`` with weights loaded, in eval mode.
    """
    num_classes = len(checkpoint.get("label_map", {})) or 234

    try:
        from birdclef2026.src.model import BirdCLEFModel

        backbone = config.get("backbone", "efficientnet_b0")
        pool = config.get("pool", "avg")
        use_denoiser = config.get("use_denoiser", False)
        denoiser_channels = config.get("denoiser_channels", 64)
        model = BirdCLEFModel(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=False,
            pool=pool,
            use_denoiser=use_denoiser,
            denoiser_channels=denoiser_channels,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not build BirdCLEFModel (%s); using fallback.", exc)
        raise

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
