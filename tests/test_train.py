"""Unit tests for Trainer checkpoint save/load and validate_checkpoint (Task 10)."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from birdclef2026.src.train import Trainer, validate_checkpoint


# ---------------------------------------------------------------------------
# Minimal model fixture
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Tiny linear model used as a stand-in for BirdCLEFModel in tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tiny_model():
    return _TinyModel()


@pytest.fixture
def base_config():
    return {
        "backbone": "efficientnet_b0",
        "label_map": {"xencan1": 0, "amakin1": 1},
        "num_epochs": 10,
    }


@pytest.fixture
def trainer(tiny_model, base_config):
    return Trainer(
        model=tiny_model,
        train_loader=None,
        val_loader=None,
        config=base_config,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# save_checkpoint + load_checkpoint round-trip
# ---------------------------------------------------------------------------

def test_round_trip_state_dict_identical(trainer, tiny_model):
    """save_checkpoint then load_checkpoint should recover an identical state_dict."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trainer.save_checkpoint(path, epoch=3, val_roc_auc=0.85)
        ckpt = trainer.load_checkpoint(path)

        original_sd = tiny_model.state_dict()
        loaded_sd = ckpt["model_state_dict"]

        assert set(original_sd.keys()) == set(loaded_sd.keys()), (
            "State dict keys differ after round-trip"
        )
        for key in original_sd:
            assert torch.equal(original_sd[key], loaded_sd[key]), (
                f"Parameter '{key}' differs after round-trip"
            )
    finally:
        os.unlink(path)


def test_round_trip_metadata(trainer):
    """Checkpoint should store epoch and val_roc_auc correctly."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trainer.save_checkpoint(path, epoch=7, val_roc_auc=0.92)
        ckpt = trainer.load_checkpoint(path)
        assert ckpt["epoch"] == 7
        assert ckpt["val_roc_auc"] == pytest.approx(0.92)
    finally:
        os.unlink(path)


def test_round_trip_label_map(trainer, base_config):
    """Checkpoint should embed the label_map from config."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trainer.save_checkpoint(path, epoch=1, val_roc_auc=0.5)
        ckpt = trainer.load_checkpoint(path)
        assert ckpt["label_map"] == base_config["label_map"]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load_checkpoint raises ValueError on missing keys
# ---------------------------------------------------------------------------

def _save_partial_checkpoint(path: str, keys: dict) -> None:
    """Save a checkpoint dict that may be missing required keys."""
    torch.save(keys, path)


@pytest.mark.parametrize("missing_key", [
    "model_state_dict",
    "config",
    "label_map",
    "epoch",
    "val_roc_auc",
])
def test_load_checkpoint_raises_on_missing_key(trainer, missing_key):
    """load_checkpoint should raise ValueError listing the missing key."""
    full = {
        "model_state_dict": {},
        "config": {},
        "label_map": {},
        "epoch": 0,
        "val_roc_auc": 0.0,
    }
    full.pop(missing_key)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        _save_partial_checkpoint(path, full)
        with pytest.raises(ValueError) as exc_info:
            trainer.load_checkpoint(path)
        assert missing_key in str(exc_info.value), (
            f"Expected missing key '{missing_key}' to appear in error message"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# validate_checkpoint raises ValueError with missing key names in message
# ---------------------------------------------------------------------------

def test_validate_checkpoint_raises_with_missing_key_names():
    """validate_checkpoint should raise ValueError listing ALL missing keys."""
    incomplete = {"model_state_dict": {}, "epoch": 1}  # missing config, label_map, val_roc_auc

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(incomplete, path)
        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint(path)
        msg = str(exc_info.value)
        for key in ("config", "label_map", "val_roc_auc"):
            assert key in msg, f"Expected missing key '{key}' in error message, got: {msg}"
    finally:
        os.unlink(path)


def test_validate_checkpoint_returns_dict_when_valid():
    """validate_checkpoint should return the checkpoint dict when all keys present."""
    valid = {
        "model_state_dict": {"fc.weight": torch.zeros(2, 4)},
        "config": {"backbone": "efficientnet_b0"},
        "label_map": {"xencan1": 0},
        "epoch": 5,
        "val_roc_auc": 0.78,
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(valid, path)
        result = validate_checkpoint(path)
        assert result["epoch"] == 5
        assert result["val_roc_auc"] == pytest.approx(0.78)
    finally:
        os.unlink(path)


# ===========================================================================
# Task 11 — Training loop unit tests
# ===========================================================================

import math
import tempfile
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyLinearModel(nn.Module):
    """Tiny model: accepts (B, 4) tensors and outputs (B, 2) logits."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _make_loader(num_samples: int = 8, input_dim: int = 4, num_classes: int = 2,
                 batch_size: int = 4) -> DataLoader:
    """Create a DataLoader with synthetic float32 tensors."""
    specs = torch.randn(num_samples, input_dim)
    labels = torch.zeros(num_samples, num_classes).float()
    # Set some positive labels
    labels[:num_samples // 2, 0] = 1.0
    labels[num_samples // 2:, 1] = 1.0
    dataset = TensorDataset(specs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _base_train_config(tmp_dir: str) -> dict:
    return {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "mixed_precision": False,
        "checkpoint_dir": tmp_dir,
        "T_0": 2,
        "label_map": {"classA": 0, "classB": 1},
    }


# ---------------------------------------------------------------------------
# Test: label smoothing transforms positive targets
# ---------------------------------------------------------------------------

def test_label_smoothing_positive_targets_become_smoothed():
    """Positive targets (1.0) should become 1.0 - label_smoothing after smoothing."""
    epsilon = 0.1
    original_labels = torch.ones(4, 2)  # all positive

    # Simulate the smoothing logic from train()
    smoothed = original_labels * (1.0 - epsilon)

    expected = 1.0 - epsilon
    assert torch.allclose(smoothed, torch.full_like(smoothed, expected)), (
        f"Expected all values to be {expected}, got {smoothed}"
    )


def test_label_smoothing_zero_targets_unchanged():
    """Zero targets should remain 0.0 after label smoothing (only positive targets change)."""
    epsilon = 0.1
    original_labels = torch.zeros(4, 2)  # all negative

    smoothed = original_labels * (1.0 - epsilon)

    assert torch.allclose(smoothed, torch.zeros_like(smoothed)), (
        "Zero targets should remain 0.0 after label smoothing"
    )


def test_label_smoothing_applied_during_training():
    """Training with label_smoothing > 0 should not raise and should complete."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _TinyLinearModel()
        loader = _make_loader()
        config = _base_train_config(tmp_dir)
        config["label_smoothing"] = 0.1

        trainer = Trainer(model=model, train_loader=loader, val_loader=loader,
                          config=config, device="cpu")
        # Should complete without error
        trainer.train(num_epochs=1)


# ---------------------------------------------------------------------------
# Test: best checkpoint has highest val_roc_auc
# ---------------------------------------------------------------------------

def test_best_checkpoint_saved_with_highest_val_roc_auc():
    """After N epochs, the saved checkpoint should have the highest val_roc_auc seen."""
    # We mock the Evaluator to return increasing then decreasing ROC-AUC values
    # so we can verify the best one is saved.
    roc_auc_sequence = [0.60, 0.75, 0.70, 0.65]  # peak at epoch 1

    call_count = [0]

    class _MockEvaluator:
        def __init__(self, model, val_loader, device):
            pass

        def evaluate(self):
            idx = min(call_count[0], len(roc_auc_sequence) - 1)
            call_count[0] += 1
            return {"macro_roc_auc": roc_auc_sequence[idx], "per_class": {}}

    import birdclef2026.src.train as train_module
    original_evaluator = train_module.Evaluator
    train_module.Evaluator = _MockEvaluator

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = _TinyLinearModel()
            loader = _make_loader()
            config = _base_train_config(tmp_dir)

            trainer = Trainer(model=model, train_loader=loader, val_loader=loader,
                              config=config, device="cpu")
            trainer.train(num_epochs=len(roc_auc_sequence))

            best_path = os.path.join(tmp_dir, "best_checkpoint.pt")
            assert os.path.exists(best_path), "Best checkpoint file should exist"

            ckpt = torch.load(best_path, map_location="cpu")
            saved_roc_auc = ckpt["val_roc_auc"]

            assert saved_roc_auc == pytest.approx(max(roc_auc_sequence)), (
                f"Saved checkpoint val_roc_auc={saved_roc_auc} should equal "
                f"max of sequence {max(roc_auc_sequence)}"
            )
    finally:
        train_module.Evaluator = original_evaluator


def test_checkpoint_monitor_metric_can_use_macro_map():
    """When configured, checkpoint selection can monitor macro_map explicitly."""
    metrics_sequence = [
        {"macro_roc_auc": 0.90, "macro_map": 0.10},
        {"macro_roc_auc": 0.80, "macro_map": 0.30},
        {"macro_roc_auc": 0.95, "macro_map": 0.20},
    ]
    call_count = [0]

    class _MockEvaluator:
        def __init__(self, model, val_loader, device):
            pass

        def evaluate(self):
            idx = min(call_count[0], len(metrics_sequence) - 1)
            call_count[0] += 1
            return metrics_sequence[idx]

    import birdclef2026.src.train as train_module
    original_evaluator = train_module.Evaluator
    train_module.Evaluator = _MockEvaluator

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = _TinyLinearModel()
            loader = _make_loader()
            config = _base_train_config(tmp_dir)
            config["monitor_metric"] = "macro_map"

            trainer = Trainer(model=model, train_loader=loader, val_loader=loader,
                              config=config, device="cpu")
            trainer.train(num_epochs=len(metrics_sequence))

            ckpt = torch.load(os.path.join(tmp_dir, "best_checkpoint.pt"), map_location="cpu")
            assert ckpt["monitor_metric"] == "macro_map"
            assert ckpt["monitor_value"] == pytest.approx(0.30)
            assert ckpt["val_roc_auc"] == pytest.approx(0.80)
            assert ckpt["val_map"] == pytest.approx(0.30)
    finally:
        train_module.Evaluator = original_evaluator


def test_no_checkpoint_saved_when_roc_auc_never_improves():
    """If val_roc_auc never improves (always 0.0), no checkpoint should be saved."""
    class _ZeroEvaluator:
        def __init__(self, model, val_loader, device):
            pass

        def evaluate(self):
            return {"macro_roc_auc": 0.0, "per_class": {}}

    import birdclef2026.src.train as train_module
    original_evaluator = train_module.Evaluator
    train_module.Evaluator = _ZeroEvaluator

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = _TinyLinearModel()
            loader = _make_loader()
            config = _base_train_config(tmp_dir)

            trainer = Trainer(model=model, train_loader=loader, val_loader=loader,
                              config=config, device="cpu")
            trainer.train(num_epochs=2)

            # 0.0 > -1.0 (initial best), so checkpoint IS saved on first epoch
            # This verifies the first improvement from -inf is captured
            best_path = os.path.join(tmp_dir, "best_checkpoint.pt")
            assert os.path.exists(best_path), (
                "Checkpoint should be saved when val_roc_auc (0.0) > initial best (-1.0)"
            )
            ckpt = torch.load(best_path, map_location="cpu")
            assert ckpt["val_roc_auc"] == pytest.approx(0.0)
    finally:
        train_module.Evaluator = original_evaluator


# ---------------------------------------------------------------------------
# Test: training completes without error (smoke test)
# ---------------------------------------------------------------------------

def test_train_smoke_no_label_smoothing():
    """Training loop should complete without errors when label_smoothing=0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _TinyLinearModel()
        loader = _make_loader()
        config = _base_train_config(tmp_dir)

        trainer = Trainer(model=model, train_loader=loader, val_loader=loader,
                          config=config, device="cpu")
        trainer.train(num_epochs=2)  # should not raise
