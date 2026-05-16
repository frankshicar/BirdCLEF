"""Unit tests for Evaluator (Task 13)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from birdclef2026.src.evaluate import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FixedLogitModel(nn.Module):
    """Model that returns pre-set logits indexed by sample position.

    The DataLoader passes spectrograms whose first channel encodes the sample
    index (set by _make_loader), so the model can look up the correct logit row.
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 1, 1) where x[:, 0, 0, 0] holds the sample indices
        indices = x[:, 0, 0, 0].long()
        return self._logits[indices]


def _make_loader(labels: torch.Tensor, logits: torch.Tensor,
                 batch_size: int = 4) -> DataLoader:
    """Build a DataLoader pairing index-encoded spectrograms with given labels."""
    n = labels.shape[0]
    # Encode sample index in the spectrogram so the model can retrieve the right logit
    specs = torch.zeros(n, 1, 1, 1)
    specs[:, 0, 0, 0] = torch.arange(n, dtype=torch.float32)
    dataset = TensorDataset(specs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_evaluator(labels: torch.Tensor, logits: torch.Tensor,
                    batch_size: int = 4) -> Evaluator:
    """Convenience factory: build Evaluator with a fixed-logit model."""
    model = _FixedLogitModel(logits)
    loader = _make_loader(labels, logits, batch_size=batch_size)
    return Evaluator(model=model, val_loader=loader, device="cpu")


# ---------------------------------------------------------------------------
# Test: evaluate() returns dict with required keys
# ---------------------------------------------------------------------------

def test_evaluate_returns_required_keys():
    """evaluate() must return a dict with 'macro_roc_auc' and 'per_class' keys."""
    n, c = 8, 3
    labels = torch.zeros(n, c)
    labels[:4, 0] = 1.0
    labels[4:, 1] = 1.0
    # class 2 has no positives

    logits = torch.randn(n, c)
    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    assert isinstance(result, dict), "evaluate() should return a dict"
    assert "macro_roc_auc" in result, "Result must contain 'macro_roc_auc'"
    assert "per_class" in result, "Result must contain 'per_class'"


# ---------------------------------------------------------------------------
# Test: classes with no positive labels get None and are excluded from macro
# ---------------------------------------------------------------------------

def test_class_with_no_positives_gets_none():
    """Classes with no positive labels should have None in per_class."""
    n, c = 8, 3
    labels = torch.zeros(n, c)
    labels[:4, 0] = 1.0   # class 0 has positives
    labels[4:, 1] = 1.0   # class 1 has positives
    # class 2: all zeros — no positives

    logits = torch.randn(n, c)
    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    assert result["per_class"]["2"] is None, (
        "Class with no positive labels should have None in per_class"
    )


def test_class_with_no_positives_excluded_from_macro():
    """Macro ROC-AUC should only average over classes with positive labels."""
    n, c = 8, 3
    labels = torch.zeros(n, c)
    labels[:4, 0] = 1.0
    labels[4:, 1] = 1.0
    # class 2: no positives

    logits = torch.randn(n, c)
    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    # Macro should equal mean of class 0 and class 1 scores only
    score_0 = result["per_class"]["0"]
    score_1 = result["per_class"]["1"]
    assert score_0 is not None
    assert score_1 is not None

    expected_macro = (score_0 + score_1) / 2.0
    assert result["macro_roc_auc"] == pytest.approx(expected_macro, abs=1e-6), (
        "Macro ROC-AUC should be mean of valid per-class scores only"
    )


# ---------------------------------------------------------------------------
# Test: macro ROC-AUC equals mean of valid per-class scores
# ---------------------------------------------------------------------------

def test_macro_equals_mean_of_valid_per_class_scores():
    """macro_roc_auc must equal the arithmetic mean of non-None per_class values."""
    n, c = 10, 4
    labels = torch.zeros(n, c)
    labels[:3, 0] = 1.0
    labels[3:6, 1] = 1.0
    labels[6:, 2] = 1.0
    # class 3: no positives

    logits = torch.randn(n, c)
    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    valid_scores = [v for v in result["per_class"].values() if v is not None]
    expected_macro = sum(valid_scores) / len(valid_scores)

    assert result["macro_roc_auc"] == pytest.approx(expected_macro, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: sigmoid is applied (probabilities in [0, 1])
# ---------------------------------------------------------------------------

def test_sigmoid_applied_probabilities_in_range():
    """Evaluator must apply sigmoid so probabilities lie in [0, 1].

    We verify this indirectly: a model that outputs large positive logits for
    class 0 positives and large negative logits for negatives should achieve
    a near-perfect ROC-AUC, which is only possible if sigmoid is applied
    correctly (raw logits would still rank correctly, but we also verify the
    per-class score is in [0, 1]).
    """
    n, c = 10, 2
    labels = torch.zeros(n, c)
    labels[:5, 0] = 1.0
    labels[5:, 1] = 1.0

    # Construct logits that perfectly separate classes
    logits = torch.zeros(n, c)
    logits[:5, 0] = 10.0   # high logit for positives of class 0
    logits[:5, 1] = -10.0
    logits[5:, 0] = -10.0
    logits[5:, 1] = 10.0   # high logit for positives of class 1

    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    for key, score in result["per_class"].items():
        if score is not None:
            assert 0.0 <= score <= 1.0, (
                f"Per-class ROC-AUC for class {key} should be in [0, 1], got {score}"
            )

    # Perfect separation → ROC-AUC should be 1.0
    assert result["macro_roc_auc"] == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_transforms_logits_to_probabilities():
    """Verify sigmoid is applied by checking that large logits yield high scores.

    A model outputting logit=+100 for all samples of a class should still
    produce a valid ROC-AUC (sigmoid(100) ≈ 1.0 for all, but ranking is
    preserved).  Without sigmoid, sklearn would still compute AUC from ranks,
    so we test a more direct property: the evaluator should not crash on
    extreme logits, and scores should remain in [0, 1].
    """
    n, c = 6, 2
    labels = torch.zeros(n, c)
    labels[:3, 0] = 1.0
    labels[3:, 1] = 1.0

    # Extreme logits — sigmoid maps these to ~0 and ~1
    logits = torch.full((n, c), -100.0)
    logits[:3, 0] = 100.0
    logits[3:, 1] = 100.0

    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    assert 0.0 <= result["macro_roc_auc"] <= 1.0
    for score in result["per_class"].values():
        if score is not None:
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test: all classes have no positives → macro_roc_auc = 0.0
# ---------------------------------------------------------------------------

def test_all_classes_no_positives_returns_zero_macro():
    """When no class has positive labels, macro_roc_auc should be 0.0."""
    n, c = 6, 3
    labels = torch.zeros(n, c)  # all zeros
    logits = torch.randn(n, c)

    evaluator = _make_evaluator(labels, logits)
    result = evaluator.evaluate()

    assert result["macro_roc_auc"] == pytest.approx(0.0)
    for score in result["per_class"].values():
        assert score is None
