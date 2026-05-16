"""BirdCLEF 2026 Trainer — checkpoint save/load and training loop."""

import json
import logging
import os

import torch.nn as nn
import torch

from birdclef2026.src.checkpoint_utils import (
    REQUIRED_CHECKPOINT_KEYS,  # re-exported for legacy imports/tests
    validate_checkpoint,
)
from birdclef2026.src.monitoring import TrainingSpectrogramMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluator import — falls back to a placeholder if evaluate.py is absent
# ---------------------------------------------------------------------------
try:
    from birdclef2026.src.evaluate import Evaluator  # noqa: F401
    _EVALUATOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EVALUATOR_AVAILABLE = False

    class Evaluator:  # type: ignore[no-redef]
        """Placeholder Evaluator used when evaluate.py is not yet implemented."""

        def __init__(self, model, val_loader, device: str):
            pass

        def evaluate(self) -> dict:
            return {"macro_roc_auc": 0.0, "per_class": {}}

METRIC_ALIASES = {
    "macro_roc_auc": "macro_roc_auc",
    "roc_auc": "macro_roc_auc",
    "val_roc_auc": "macro_roc_auc",
    "macro_map": "macro_map",
    "map": "macro_map",
    "val_map": "macro_map",
}


class Trainer:
    """Trains a BirdCLEFModel and manages checkpoints.

    Args:
        model: the BirdCLEFModel instance to train
        train_loader: DataLoader for the training set
        val_loader: DataLoader for the validation set
        config: full resolved config dict (must include 'label_map')
        device: torch device string, e.g. 'cpu' or 'cuda'
    """

    def __init__(self, model, train_loader, val_loader, config: dict, device: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize training history
        self.history = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_map": [],
            "learning_rate": []
        }
        
        # Set up history file path
        checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        self.history_path = os.path.join(checkpoint_dir, "training_history.json")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_roc_auc: float,
        val_map: float | None = None,
        monitor_metric: str | None = None,
        monitor_value: float | None = None,
    ) -> None:
        """Save a checkpoint to *path*.

        The checkpoint schema is::

            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "label_map": config.get("label_map", {}),
                "epoch": epoch,
                "val_roc_auc": val_roc_auc,
            }

        Args:
            path: destination file path
            epoch: current epoch number
            val_roc_auc: validation macro ROC-AUC for this checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "label_map": self.config.get("label_map", {}),
            "epoch": epoch,
            "val_roc_auc": val_roc_auc,
        }
        if val_map is not None:
            checkpoint["val_map"] = val_map
        if monitor_metric is not None:
            checkpoint["monitor_metric"] = monitor_metric
        if monitor_value is not None:
            checkpoint["monitor_value"] = monitor_value
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s (epoch=%d, val_roc_auc=%.4f)", path, epoch, val_roc_auc)

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint from *path* and validate required keys.

        Args:
            path: source file path

        Returns:
            The checkpoint dict.

        Raises:
            ValueError: if any of the required keys are absent, listing them.
        """
        return validate_checkpoint(path)

    def save_history(self) -> None:
        """Save training history to JSON file."""
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.debug("Training history saved to %s", self.history_path)

    def _resolve_monitor_metric(self) -> tuple[str, str]:
        """Return the configured metric name and Evaluator key."""
        monitor_metric = self.config.get("monitor_metric", "macro_roc_auc")
        metric_key = METRIC_ALIASES.get(monitor_metric)
        if metric_key is None:
            raise ValueError(
                "monitor_metric must be one of: "
                f"{sorted(METRIC_ALIASES.keys())}; got {monitor_metric!r}"
            )
        return monitor_metric, metric_key

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int) -> None:
        """Main training loop.

        Trains for *num_epochs* epochs, logging training loss, validation
        loss, and validation macro ROC-AUC after each epoch.  Saves a
        checkpoint whenever the validation ROC-AUC improves.

        Supports:
        - Label smoothing (config key ``label_smoothing``)
        - AdamW optimizer (config keys ``learning_rate``, ``weight_decay``)
        - CosineAnnealingWarmRestarts scheduler (config key ``T_0``)
        - Mixed-precision training via GradScaler (config key ``mixed_precision``)
        - Resume from checkpoint (config key ``resume_checkpoint``)

        Args:
            num_epochs: total number of epochs to train.
        """
        device = self.device
        model = self.model.to(device)

        # ---- hyper-parameters from config --------------------------------
        lr = self.config["learning_rate"]
        weight_decay = self.config.get("weight_decay", 1e-4)
        label_smoothing = self.config.get("label_smoothing", 0.0)
        mixed_precision = self.config.get("mixed_precision", False) and str(device).startswith("cuda")
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        T_0 = self.config.get("T_0", 10)
        early_stopping_patience = self.config.get("early_stopping_patience", 0)
        grad_clip_norm = self.config.get("grad_clip_norm", 5.0)
        monitor = TrainingSpectrogramMonitor.from_config(self.config)

        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")

        # ---- loss --------------------------------------------------------
        criterion = nn.BCEWithLogitsLoss()

        # ---- optimizer & scheduler ---------------------------------------
        # Differential learning rate: backbone uses smaller lr to preserve pretrained features
        backbone_lr = self.config.get("backbone_lr", lr)
        head_lr = self.config.get("head_lr", lr)
        
        if hasattr(model, "backbone") and hasattr(model, "classifier"):
            param_groups = [
                {"params": model.backbone.parameters(), "lr": backbone_lr},
            ]

            if hasattr(model, "denoiser") and model.denoiser is not None:
                param_groups.append({"params": model.denoiser.parameters(), "lr": head_lr})

            param_groups.append({"params": model.classifier.parameters(), "lr": head_lr})
        else:
            param_groups = [{"params": model.parameters(), "lr": lr}]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)

        # ---- mixed precision ---------------------------------------------
        scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

        # ---- metric selection -------------------------------------------
        monitor_metric, metric_key = self._resolve_monitor_metric()

        # ---- resume ------------------------------------------------------
        start_epoch = 0
        best_monitor_value = -1.0
        epochs_no_improve = 0

        resume_path = self.config.get("resume_checkpoint")
        if resume_path:
            ckpt = self.load_checkpoint(resume_path)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_monitor_value = ckpt.get(
                "monitor_value",
                ckpt.get("val_roc_auc" if metric_key == "macro_roc_auc" else "val_map", -1.0),
            )
            logger.info(
                "Resumed from checkpoint '%s' (epoch=%d, best_%s=%.4f)",
                resume_path, start_epoch - 1, monitor_metric, best_monitor_value,
            )

        # ---- training loop -----------------------------------------------
        logger.info("Starting training for %d epochs...", num_epochs)
        for epoch in range(start_epoch, start_epoch + num_epochs):
            logger.info("=== Epoch %d/%d ===", epoch + 1, start_epoch + num_epochs)
            monitor.on_epoch_start()
            model.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for batch_idx, (batch_specs, batch_labels) in enumerate(self.train_loader):
                batch_specs = batch_specs.to(device)
                batch_labels = batch_labels.to(device)

                # Apply label smoothing to positive targets
                if label_smoothing > 0.0:
                    batch_labels = batch_labels * (1.0 - label_smoothing)

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        with monitor.capture(model, batch_specs, epoch, batch_idx, "train"):
                            logits = model(batch_specs)
                        loss = criterion(logits, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    with monitor.capture(model, batch_specs, epoch, batch_idx, "train"):
                        logits = model(batch_specs)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

                total_train_loss += loss.item()
                num_train_batches += 1
                
                # Print progress every 200 batches (less frequent)
                if num_train_batches % 200 == 0:
                    logger.debug("  Batch %d/%d, loss=%.4f", 
                              num_train_batches, len(self.train_loader), loss.item())

            scheduler.step(epoch)

            avg_train_loss = total_train_loss / max(num_train_batches, 1)

            # ---- validation ----------------------------------------------
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch_specs, batch_labels in self.val_loader:
                    batch_specs = batch_specs.to(device)
                    batch_labels = batch_labels.to(device)

                    if scaler is not None:
                        with torch.amp.autocast("cuda"):
                            logits = model(batch_specs)
                            val_loss = criterion(logits, batch_labels)
                    else:
                        logits = model(batch_specs)
                        val_loss = criterion(logits, batch_labels)

                    total_val_loss += val_loss.item()
                    num_val_batches += 1

            avg_val_loss = total_val_loss / max(num_val_batches, 1)

            # ---- ROC-AUC & mAP ----------------------------------------------
            evaluator = Evaluator(model, self.val_loader, device)
            metrics = evaluator.evaluate()
            val_roc_auc = metrics.get("macro_roc_auc", 0.0)
            val_map = metrics.get("macro_map", 0.0)
            monitor_value = metrics.get(metric_key, 0.0)

            # ---- Record history ----------------------------------------------
            current_lr = optimizer.param_groups[0]['lr']
            self.history["epochs"].append(epoch)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_roc_auc"].append(val_roc_auc)
            self.history["val_map"].append(val_map)
            self.history["learning_rate"].append(current_lr)
            self.save_history()

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_roc_auc=%.4f | val_mAP=%.4f",
                epoch, avg_train_loss, avg_val_loss, val_roc_auc, val_map,
            )

            # ---- checkpoint (default: macro ROC-AUC, matching competition) -
            if monitor_value > best_monitor_value:
                best_monitor_value = monitor_value
                epochs_no_improve = 0
                self.save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch,
                    val_roc_auc=val_roc_auc,
                    val_map=val_map,
                    monitor_metric=monitor_metric,
                    monitor_value=monitor_value,
                )
                logger.info(
                    "New best checkpoint saved (%s=%.4f, val_roc_auc=%.4f, val_mAP=%.4f)",
                    monitor_metric, monitor_value, val_roc_auc, val_map,
                )
            else:
                epochs_no_improve += 1
                logger.info(
                    "No improvement for %d epoch(s) (best_%s=%.4f)",
                    epochs_no_improve, monitor_metric, best_monitor_value,
                )

            # ---- early stopping ------------------------------------------
            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    epochs_no_improve,
                )
                break
