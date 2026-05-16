"""BirdCLEF 2026 Evaluator — validation metric computation."""

import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


class Evaluator:
    """Computes validation metrics for a BirdCLEF model.

    Applies sigmoid to raw logits, then computes per-class and macro-averaged
    ROC-AUC and mAP using sklearn. Classes with no positive labels in the 
    validation set are excluded from the macro averages and reported as ``None``.

    Args:
        model: the BirdCLEFModel (or any nn.Module) to evaluate
        val_loader: DataLoader yielding (spectrogram, label) batches
        device: torch device string, e.g. ``'cpu'`` or ``'cuda'``
    """

    def __init__(self, model, val_loader, device: str):
        self.model = model
        self.val_loader = val_loader
        self.device = device

    def evaluate(self) -> dict:
        """Run inference on the validation set and compute ROC-AUC and mAP metrics.

        Returns:
            A dict with keys:

            * ``'macro_roc_auc'`` (float): arithmetic mean of per-class ROC-AUC scores
            * ``'macro_map'`` (float): arithmetic mean of per-class mAP scores  
            * ``'per_class'`` (dict): per-class metrics keyed by class index (as string)
              Each value is a dict with 'roc_auc' and 'map' keys, or None for classes
              with no positive labels.
        """
        self.model.eval()
        device = self.device

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                specs, labels = batch
                specs = specs.to(device)
                logits = self.model(specs)
                # Apply sigmoid to convert logits → probabilities (Req 7.2)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Shape: (N, num_classes)
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        num_classes = all_labels.shape[1]
        per_class: dict[str, dict[str, float] | None] = {}
        valid_roc_scores: list[float] = []
        valid_map_scores: list[float] = []

        for cls_idx in range(num_classes):
            key = str(cls_idx)
            y_true = all_labels[:, cls_idx]
            y_score = all_probs[:, cls_idx]

            # Binarize labels for ROC-AUC (threshold at 0.5 to handle soft labels)
            y_true_binary = (y_true >= 0.5).astype(np.int32)
            
            # Exclude classes with no positive labels or only one class present
            n_positive = y_true_binary.sum()
            n_negative = len(y_true_binary) - n_positive
            
            if n_positive == 0 or n_negative == 0:
                per_class[key] = None
                logger.debug("Class %d excluded: n_positive=%d, n_negative=%d", 
                           cls_idx, n_positive, n_negative)
                continue

            roc_score = float(roc_auc_score(y_true_binary, y_score))
            map_score = float(average_precision_score(y_true_binary, y_score))
            
            per_class[key] = {
                'roc_auc': roc_score,
                'map': map_score
            }
            valid_roc_scores.append(roc_score)
            valid_map_scores.append(map_score)

        # Macro averages over valid classes only
        macro_roc_auc = float(np.mean(valid_roc_scores)) if valid_roc_scores else 0.0
        macro_map = float(np.mean(valid_map_scores)) if valid_map_scores else 0.0

        return {
            "macro_roc_auc": macro_roc_auc, 
            "macro_map": macro_map,
            "per_class": per_class
        }
