"""Unit tests for InferenceEngine (Task 14).

Uses a tiny synthetic model (not BirdCLEFModel) to avoid loading timm.
"""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torch.nn as nn

from birdclef2026.src.inference import InferenceEngine


# ---------------------------------------------------------------------------
# Tiny synthetic model — avoids timm dependency in tests
# ---------------------------------------------------------------------------

NUM_CLASSES = 234


class _TinyModel(nn.Module):
    """Minimal model: accepts (B, 1, n_mels, T) and outputs (B, NUM_CLASSES) logits.

    Uses adaptive average pooling so it works regardless of the time dimension.
    """

    def __init__(self, n_mels: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T) → pool to (B, 1, 1, 1) → (B, 1) → (B, NUM_CLASSES)
        pooled = self.pool(x)  # (B, 1, 1, 1)
        B = pooled.shape[0]
        return self.fc(pooled.view(B, -1))


class _NaNModel(nn.Module):
    """Model that always returns NaN logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], NUM_CLASSES)
        out[:] = float("nan")
        return out


class _InfModel(nn.Module):
    """Model that always returns Inf logits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], NUM_CLASSES)
        out[:] = float("inf")
        return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_dir: str, model: nn.Module | None = None,
                     n_mels: int = 128) -> str:
    """Save a minimal valid checkpoint and return its path."""
    if model is None:
        model = _TinyModel()

    label_map = {f"species_{i:03d}": i for i in range(NUM_CLASSES)}
    config = {
        "backbone": "efficientnet_b0",
        "pool": "avg",
        "sample_rate": 32000,
        "segment_duration": 5.0,
        "hop_duration": 5.0,
        "n_mels": n_mels,
        "hop_length": 320,
        "n_fft": 1024,
        "top_db": 80.0,
        "mel_mean": 0.0,
        "mel_std": 1.0,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "label_map": label_map,
        "epoch": 1,
        "val_roc_auc": 0.75,
    }
    path = os.path.join(tmp_dir, "checkpoint.pt")
    torch.save(checkpoint, path)
    return path


def _make_soundscape_ogg(tmp_dir: str, duration_sec: float = 10.0,
                          sample_rate: int = 32000,
                          name: str = "test_soundscape") -> str:
    """Write a synthetic mono .ogg soundscape and return its path."""
    n_samples = int(duration_sec * sample_rate)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1
    path = os.path.join(tmp_dir, f"{name}.ogg")
    sf.write(path, audio, sample_rate)
    return path


def _make_sample_submission(tmp_dir: str, row_ids: list[str]) -> str:
    """Write a minimal sample_submission.csv and return its path."""
    species_cols = [f"species_{i:03d}" for i in range(NUM_CLASSES)]
    data = {"row_id": row_ids}
    for col in species_cols:
        data[col] = [0.0] * len(row_ids)
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, "sample_submission.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Helper: patch InferenceEngine._load_model to inject a tiny model
# ---------------------------------------------------------------------------

def _inject_model(engine: InferenceEngine, model: nn.Module,
                  n_mels: int = 128) -> None:
    """Bypass timm by directly setting the engine's internal model and config."""
    engine._model = model
    engine._models = [model]
    engine._config = {
        "sample_rate": 32000,
        "segment_duration": 5.0,
        "hop_duration": 5.0,
        "n_mels": n_mels,
        "hop_length": 320,
        "n_fft": 1024,
        "top_db": 80.0,
        "mel_mean": 0.0,
        "mel_std": 1.0,
        "time_mask_param": 30,
        "freq_mask_param": 20,
    }
    engine._label_map = {f"species_{i:03d}": i for i in range(NUM_CLASSES)}
    engine._species_columns = [f"species_{i:03d}" for i in range(NUM_CLASSES)]


# ===========================================================================
# Test: verify_paths raises FileNotFoundError with missing path in message
# ===========================================================================

def test_verify_paths_raises_for_missing_checkpoint():
    """verify_paths() must raise FileNotFoundError containing the missing path."""
    missing = "/nonexistent/path/checkpoint.pt"
    engine = InferenceEngine(checkpoint_path=missing)
    with pytest.raises(FileNotFoundError) as exc_info:
        engine.verify_paths()
    assert missing in str(exc_info.value), (
        f"Expected missing path '{missing}' in error message, got: {exc_info.value}"
    )


def test_verify_paths_raises_for_missing_backbone_weights():
    """verify_paths() must raise FileNotFoundError for missing backbone_weights_path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(tmp_dir)
        missing_backbone = "/nonexistent/backbone_weights.pt"
        engine = InferenceEngine(
            checkpoint_path=ckpt_path,
            backbone_weights_path=missing_backbone,
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            engine.verify_paths()
        assert missing_backbone in str(exc_info.value), (
            f"Expected missing backbone path in error message, got: {exc_info.value}"
        )


def test_verify_paths_passes_when_all_exist():
    """verify_paths() should not raise when all paths exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = _make_checkpoint(tmp_dir)
        engine = InferenceEngine(checkpoint_path=ckpt_path)
        engine.verify_paths()  # should not raise


def test_verify_paths_message_contains_exact_missing_path():
    """The FileNotFoundError message must contain the exact missing path string."""
    missing = "/some/very/specific/missing_checkpoint.pt"
    engine = InferenceEngine(checkpoint_path=missing)
    with pytest.raises(FileNotFoundError) as exc_info:
        engine.verify_paths()
    assert missing in str(exc_info.value)


# ===========================================================================
# Test: run() produces CSV with all row_ids from sample_submission.csv
# ===========================================================================

def test_run_produces_csv_with_all_row_ids():
    """run() must produce a CSV containing every row_id from sample_submission.csv."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a 10-second soundscape → 2 segments (5s each)
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=10.0,
                                        name="test_soundscape")
        row_ids = [
            "test_soundscape_5",
            "test_soundscape_10",
            "extra_row_id_not_in_soundscape_5",
        ]
        sample_sub_path = _make_sample_submission(tmp_dir, row_ids)
        ckpt_path = _make_checkpoint(tmp_dir)
        output_path = os.path.join(tmp_dir, "submission.csv")

        engine = InferenceEngine(checkpoint_path=ckpt_path, batch_size=4)
        _inject_model(engine, _TinyModel())

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        assert os.path.exists(output_path), "submission.csv should be created"
        out_df = pd.read_csv(output_path)
        assert set(out_df["row_id"].tolist()) == set(row_ids), (
            "Output CSV must contain exactly the row_ids from sample_submission.csv"
        )
        assert len(out_df) == len(row_ids)


def test_run_csv_has_correct_column_count():
    """Output CSV must have row_id + 234 species columns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        row_ids = ["sc1_5"]
        sample_sub_path = _make_sample_submission(tmp_dir, row_ids)
        ckpt_path = _make_checkpoint(tmp_dir)
        output_path = os.path.join(tmp_dir, "submission.csv")

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _TinyModel())

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        out_df = pd.read_csv(output_path)
        # 1 row_id column + NUM_CLASSES species columns
        assert len(out_df.columns) == 1 + NUM_CLASSES


# ===========================================================================
# Test: missing row_ids are filled with 0.0
# ===========================================================================

def test_missing_row_ids_filled_with_zeros():
    """Row IDs in sample_submission.csv not predicted must be filled with 0.0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Soundscape produces row_ids: sc1_5, sc1_10
        _make_soundscape_ogg(tmp_dir, duration_sec=10.0, name="sc1")

        # Include a row_id that will never be predicted
        row_ids = ["sc1_5", "sc1_10", "completely_missing_row_999"]
        sample_sub_path = _make_sample_submission(tmp_dir, row_ids)
        ckpt_path = _make_checkpoint(tmp_dir)
        output_path = os.path.join(tmp_dir, "submission.csv")

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _TinyModel())

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        out_df = pd.read_csv(output_path)
        missing_row = out_df[out_df["row_id"] == "completely_missing_row_999"]
        assert len(missing_row) == 1, "Missing row_id should appear in output"

        species_cols = [c for c in out_df.columns if c != "row_id"]
        values = missing_row[species_cols].values.flatten()
        assert np.all(values == 0.0), (
            "Missing row_id predictions must all be 0.0"
        )


def test_missing_row_ids_all_zeros_not_partial():
    """All 234 values for a missing row_id must be 0.0, not just some."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        row_ids = ["sc1_5", "ghost_row_42"]
        sample_sub_path = _make_sample_submission(tmp_dir, row_ids)
        ckpt_path = _make_checkpoint(tmp_dir)
        output_path = os.path.join(tmp_dir, "submission.csv")

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _TinyModel())

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        out_df = pd.read_csv(output_path)
        ghost = out_df[out_df["row_id"] == "ghost_row_42"]
        species_cols = [c for c in out_df.columns if c != "row_id"]
        assert len(species_cols) == NUM_CLASSES
        values = ghost[species_cols].values.flatten()
        assert np.all(values == 0.0)


# ===========================================================================
# Test: NaN/Inf in model output is replaced with 0.0
# ===========================================================================

def test_nan_in_model_output_replaced_with_zero():
    """NaN values in model output must be replaced with 0.0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        row_ids = ["sc1_5"]
        sample_sub_path = _make_sample_submission(tmp_dir, row_ids)
        ckpt_path = _make_checkpoint(tmp_dir)
        output_path = os.path.join(tmp_dir, "submission.csv")

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _NaNModel())

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        out_df = pd.read_csv(output_path)
        species_cols = [c for c in out_df.columns if c != "row_id"]
        values = out_df[species_cols].values.flatten()
        assert not np.any(np.isnan(values)), "NaN values must not appear in output"
        assert not np.any(np.isinf(values)), "Inf values must not appear in output"
        # All values must be valid probabilities in [0, 1]
        assert np.all(values >= 0.0) and np.all(values <= 1.0), (
            "All output values must be valid probabilities in [0.0, 1.0]"
        )


def test_inf_in_model_output_replaced_with_zero():
    """Inf values in model output must be replaced with 0.0 before sigmoid."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        row_ids = ["sc1_5"]
        sample_sub_path = _make_sample_submission(tmp_dir, row_ids)
        ckpt_path = _make_checkpoint(tmp_dir)
        output_path = os.path.join(tmp_dir, "submission.csv")

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _InfModel())

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        out_df = pd.read_csv(output_path)
        species_cols = [c for c in out_df.columns if c != "row_id"]
        values = out_df[species_cols].values.flatten()
        assert not np.any(np.isinf(values)), "Inf values must not appear in output"
        assert not np.any(np.isnan(values)), "NaN values must not appear in output"


def test_predict_soundscape_nan_replaced():
    """predict_soundscape() must replace NaN logits with 0.0 before sigmoid."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _NaNModel())

        results = engine.predict_soundscape(sf_path)
        assert len(results) > 0
        for row_id, prob_vec in results.items():
            assert not np.any(np.isnan(prob_vec)), (
                f"NaN found in prob_vec for {row_id}"
            )
            assert not np.any(np.isinf(prob_vec)), (
                f"Inf found in prob_vec for {row_id}"
            )


# ===========================================================================
# Test: No .cuda() calls in the implementation
# ===========================================================================

def test_no_cuda_calls_in_inference_source():
    """inference.py must not contain .cuda() calls (CPU-only requirement)."""
    inference_path = Path(__file__).parent.parent / "birdclef2026" / "src" / "inference.py"
    source = inference_path.read_text()

    # Check for .cuda() method calls
    assert ".cuda()" not in source, (
        "inference.py must not contain .cuda() calls (CPU-only environment)"
    )
    # Check for .to("cuda") calls
    assert '.to("cuda")' not in source, (
        'inference.py must not contain .to("cuda") calls'
    )


# ===========================================================================
# Test: predict_soundscape returns correct shape
# ===========================================================================

def test_predict_soundscape_returns_prob_vectors_of_length_234():
    """predict_soundscape() must return prob vectors of length 234."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=10.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _TinyModel())

        results = engine.predict_soundscape(sf_path)
        assert len(results) > 0, "Should return at least one segment"
        for row_id, prob_vec in results.items():
            assert len(prob_vec) == NUM_CLASSES, (
                f"Expected prob_vec of length {NUM_CLASSES}, got {len(prob_vec)}"
            )
            assert prob_vec.dtype == np.float32


def test_predict_soundscape_probabilities_in_range():
    """All probabilities must be in [0.0, 1.0] after sigmoid."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _TinyModel())

        results = engine.predict_soundscape(sf_path)
        for row_id, prob_vec in results.items():
            assert np.all(prob_vec >= 0.0) and np.all(prob_vec <= 1.0), (
                f"Probabilities for {row_id} must be in [0.0, 1.0]"
            )


# ===========================================================================
# Test: batch_size is respected
# ===========================================================================

def test_batch_size_one_produces_same_results_as_batch_size_large():
    """Results should be identical regardless of batch_size."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=15.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        model = _TinyModel()
        model.eval()

        engine1 = InferenceEngine(checkpoint_path=ckpt_path, batch_size=1)
        _inject_model(engine1, model)

        engine2 = InferenceEngine(checkpoint_path=ckpt_path, batch_size=100)
        _inject_model(engine2, model)

        results1 = engine1.predict_soundscape(sf_path)
        results2 = engine2.predict_soundscape(sf_path)

        assert set(results1.keys()) == set(results2.keys())
        for row_id in results1:
            np.testing.assert_allclose(
                results1[row_id], results2[row_id], rtol=1e-5,
                err_msg=f"Results differ for {row_id} between batch_size=1 and batch_size=100"
            )


# ===========================================================================
# Task 15: Ensemble and TTA tests
# ===========================================================================

def _inject_model_ensemble(engine: InferenceEngine, models: list[nn.Module],
                            n_mels: int = 128) -> None:
    """Inject multiple models for ensemble testing."""
    engine._models = models
    engine._model = models[0]
    engine._config = {
        "sample_rate": 32000,
        "segment_duration": 5.0,
        "hop_duration": 5.0,
        "n_mels": n_mels,
        "hop_length": 320,
        "n_fft": 1024,
        "top_db": 80.0,
        "mel_mean": 0.0,
        "mel_std": 1.0,
        "time_mask_param": 30,
        "freq_mask_param": 20,
    }
    engine._label_map = {f"species_{i:03d}": i for i in range(NUM_CLASSES)}
    engine._species_columns = [f"species_{i:03d}" for i in range(NUM_CLASSES)]


class _ConstantModel(nn.Module):
    """Model that always returns a fixed logit value for all classes."""

    def __init__(self, logit_value: float):
        super().__init__()
        self.logit_value = logit_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], NUM_CLASSES), self.logit_value)


def test_ensemble_two_checkpoints_averages_probabilities():
    """Ensemble with 2 checkpoints must produce averaged probabilities."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        # Model A: logit = 2.0  → sigmoid ≈ 0.8808
        # Model B: logit = -2.0 → sigmoid ≈ 0.1192
        # Average ≈ 0.5
        model_a = _ConstantModel(2.0)
        model_b = _ConstantModel(-2.0)

        prob_a = torch.sigmoid(torch.tensor(2.0)).item()
        prob_b = torch.sigmoid(torch.tensor(-2.0)).item()
        expected_avg = (prob_a + prob_b) / 2.0

        engine = InferenceEngine(
            checkpoint_path=ckpt_path,
            checkpoint_paths=[ckpt_path, ckpt_path],  # two paths (models injected below)
        )
        _inject_model_ensemble(engine, [model_a, model_b])

        results = engine.predict_soundscape(sf_path)
        assert len(results) > 0

        for row_id, prob_vec in results.items():
            np.testing.assert_allclose(
                prob_vec,
                np.full(NUM_CLASSES, expected_avg, dtype=np.float32),
                rtol=1e-5,
                err_msg=f"Ensemble average incorrect for {row_id}",
            )


def test_single_checkpoint_still_works():
    """Single checkpoint (original API) must still produce valid probabilities."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        engine = InferenceEngine(checkpoint_path=ckpt_path)
        _inject_model(engine, _TinyModel())

        results = engine.predict_soundscape(sf_path)
        assert len(results) > 0
        for row_id, prob_vec in results.items():
            assert len(prob_vec) == NUM_CLASSES
            assert np.all(prob_vec >= 0.0) and np.all(prob_vec <= 1.0), (
                f"Single-checkpoint probabilities out of [0,1] for {row_id}"
            )


def test_tta_produces_valid_probabilities():
    """TTA with tta=True must produce probabilities in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sf_path = _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")
        ckpt_path = _make_checkpoint(tmp_dir)

        engine = InferenceEngine(checkpoint_path=ckpt_path, tta=True, tta_views=3)
        _inject_model_ensemble(engine, [_TinyModel()])

        results = engine.predict_soundscape(sf_path)
        assert len(results) > 0
        for row_id, prob_vec in results.items():
            assert len(prob_vec) == NUM_CLASSES, (
                f"Expected {NUM_CLASSES} values, got {len(prob_vec)}"
            )
            assert np.all(prob_vec >= 0.0) and np.all(prob_vec <= 1.0), (
                f"TTA probabilities out of [0,1] for {row_id}"
            )
            assert prob_vec.dtype == np.float32


def test_tta_uses_time_rolls_not_specaugment_masks():
    """Inference TTA should preserve spectrogram values and only roll time."""
    spec = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    rolled = InferenceEngine._roll_spec_time(spec, 0.5)

    assert torch.equal(rolled, torch.roll(spec, shifts=2, dims=-1))
    assert torch.equal(torch.sort(rolled.flatten()).values, torch.sort(spec.flatten()).values)


# ===========================================================================
# Task 19.2: submission.csv column order matches sample_submission.csv
# ===========================================================================

def test_run_csv_column_order_matches_sample_submission():
    """Output CSV columns must match sample_submission.csv columns in the same order.

    Requirements: 8.7
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a soundscape that produces one segment
        _make_soundscape_ogg(tmp_dir, duration_sec=5.0, name="sc1")

        # Build a sample_submission with a specific, non-alphabetical column order
        # Use real-looking species codes in a deliberate order
        species_cols_ordered = [
            "zebfin",
            "ashgre1",
            "houspa",
            "osprey",
            "limpki",
            "banana",
        ]
        row_ids = ["sc1_5"]
        data = {"row_id": row_ids}
        for col in species_cols_ordered:
            data[col] = [0.0]
        sample_df = pd.DataFrame(data, columns=["row_id"] + species_cols_ordered)
        sample_sub_path = os.path.join(tmp_dir, "sample_submission.csv")
        sample_df.to_csv(sample_sub_path, index=False)

        # Build a checkpoint whose label_map matches the species columns
        label_map = {sp: i for i, sp in enumerate(species_cols_ordered)}
        n_classes = len(species_cols_ordered)

        class _SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(1, n_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                pooled = self.pool(x)
                return self.fc(pooled.view(x.shape[0], -1))

        model = _SmallModel()
        config = {
            "backbone": "efficientnet_b0",
            "pool": "avg",
            "sample_rate": 32000,
            "segment_duration": 5.0,
            "hop_duration": 5.0,
            "n_mels": 128,
            "hop_length": 320,
            "n_fft": 1024,
            "top_db": 80.0,
            "mel_mean": 0.0,
            "mel_std": 1.0,
        }
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "label_map": label_map,
            "epoch": 1,
            "val_roc_auc": 0.5,
        }
        ckpt_path = os.path.join(tmp_dir, "ckpt_small.pt")
        torch.save(checkpoint, ckpt_path)

        output_path = os.path.join(tmp_dir, "submission.csv")
        engine = InferenceEngine(checkpoint_path=ckpt_path)

        # Inject the small model directly to bypass timm
        engine._model = model
        engine._models = [model]
        engine._config = {
            "sample_rate": 32000,
            "segment_duration": 5.0,
            "hop_duration": 5.0,
            "n_mels": 128,
            "hop_length": 320,
            "n_fft": 1024,
            "top_db": 80.0,
            "mel_mean": 0.0,
            "mel_std": 1.0,
            "time_mask_param": 30,
            "freq_mask_param": 20,
        }
        engine._label_map = label_map
        engine._species_columns = species_cols_ordered

        engine.run(
            soundscape_dir=tmp_dir,
            sample_submission_path=sample_sub_path,
            output_path=output_path,
        )

        assert os.path.exists(output_path), "submission.csv must be created"
        out_df = pd.read_csv(output_path)

        # Column names and order must exactly match sample_submission.csv
        expected_cols = list(sample_df.columns)  # ["row_id", "zebfin", "ashgre1", ...]
        actual_cols = list(out_df.columns)
        assert actual_cols == expected_cols, (
            f"Column order mismatch.\n"
            f"Expected: {expected_cols}\n"
            f"Got:      {actual_cols}"
        )
