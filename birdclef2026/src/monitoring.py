"""Training-time spectrogram and activation monitoring utilities."""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    enabled: bool = False
    output_dir: str = "./monitoring"
    epoch_interval: int = 1
    batch_interval: int = 0
    max_batches_per_epoch: int = 1
    max_layers: int = 64
    save_npy: bool = True
    save_pgm: bool = True
    layer_name_regex: str | None = None


class TrainingSpectrogramMonitor:
    """Capture model inputs and layer activations during selected batches.

    The monitor writes two simple artifacts per captured tensor:
    - ``.npy``: raw 2-D numpy array for later analysis.
    - ``.pgm``: dependency-free grayscale heatmap viewable by common tools.

    4-D tensors are reduced from ``(B, C, H, W)`` to ``(H, W)`` using the first
    sample and mean absolute activation across channels.  3-D tensors are
    reduced to ``(C, T)`` or ``(H, W)`` for the first sample.  2-D tensors are
    saved as a single-row heatmap.
    """

    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self._captures_this_epoch = 0
        self._name_pattern = (
            re.compile(config.layer_name_regex) if config.layer_name_regex else None
        )

    @classmethod
    def from_config(cls, config: dict) -> "TrainingSpectrogramMonitor":
        monitor_cfg = config.get("spectrogram_monitor", {})
        return cls(
            MonitorConfig(
                enabled=bool(monitor_cfg.get("enabled", False)),
                output_dir=monitor_cfg.get("output_dir", "./monitoring"),
                epoch_interval=int(monitor_cfg.get("epoch_interval", 1)),
                batch_interval=int(monitor_cfg.get("batch_interval", 0)),
                max_batches_per_epoch=int(monitor_cfg.get("max_batches_per_epoch", 1)),
                max_layers=int(monitor_cfg.get("max_layers", 64)),
                save_npy=bool(monitor_cfg.get("save_npy", True)),
                save_pgm=bool(monitor_cfg.get("save_pgm", True)),
                layer_name_regex=monitor_cfg.get("layer_name_regex"),
            )
        )

    def on_epoch_start(self) -> None:
        self._captures_this_epoch = 0

    def should_capture(self, epoch: int, batch_idx: int) -> bool:
        if not self.config.enabled:
            return False
        if self.config.epoch_interval > 1 and epoch % self.config.epoch_interval != 0:
            return False
        if self._captures_this_epoch >= self.config.max_batches_per_epoch:
            return False
        if self.config.batch_interval <= 0:
            return batch_idx == 0
        return batch_idx % self.config.batch_interval == 0

    @contextmanager
    def capture(
        self,
        model: nn.Module,
        batch_specs: torch.Tensor,
        epoch: int,
        batch_idx: int,
        phase: str,
    ) -> Iterator[None]:
        """Register temporary hooks and save activations after the forward pass."""
        if not self.should_capture(epoch, batch_idx):
            yield
            return

        captures: list[tuple[str, torch.Tensor]] = [
            ("input_mel", batch_specs.detach().cpu())
        ]
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                if len(captures) - 1 >= self.config.max_layers:
                    return
                tensor = self._extract_tensor(output)
                if tensor is not None:
                    captures.append((name, tensor.detach().cpu()))
            return hook

        for name, module in model.named_modules():
            if name == "" or not self._should_hook_module(name, module):
                continue
            handles.append(module.register_forward_hook(make_hook(name)))

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._captures_this_epoch += 1
            self._write_captures(captures, epoch, batch_idx, phase)

    def _should_hook_module(self, name: str, module: nn.Module) -> bool:
        if self._name_pattern is not None and not self._name_pattern.search(name):
            return False
        if any(module.children()):
            return False
        return isinstance(
            module,
            (
                nn.Conv1d,
                nn.Conv2d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.ReLU,
                nn.SiLU,
                nn.GELU,
                nn.AdaptiveAvgPool2d,
                nn.MaxPool2d,
                nn.Linear,
            ),
        )

    @staticmethod
    def _extract_tensor(output) -> torch.Tensor | None:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def _write_captures(
        self,
        captures: list[tuple[str, torch.Tensor]],
        epoch: int,
        batch_idx: int,
        phase: str,
    ) -> None:
        batch_dir = os.path.join(
            self.config.output_dir,
            phase,
            f"epoch_{epoch:04d}",
            f"batch_{batch_idx:05d}",
        )
        os.makedirs(batch_dir, exist_ok=True)

        for order, (name, tensor) in enumerate(captures):
            matrix = self._tensor_to_matrix(tensor)
            if matrix is None:
                continue

            safe_name = self._safe_name(name)
            base = os.path.join(batch_dir, f"{order:03d}_{safe_name}")
            if self.config.save_npy:
                np.save(base + ".npy", matrix)
            if self.config.save_pgm:
                self._write_pgm(base + ".pgm", matrix)

        logger.info(
            "Saved %d training monitor tensors to %s",
            len(captures),
            batch_dir,
        )

    @staticmethod
    def _tensor_to_matrix(tensor: torch.Tensor) -> np.ndarray | None:
        if tensor.numel() == 0:
            return None
        tensor = tensor.float()
        if tensor.dim() == 4:
            matrix = tensor[0].abs().mean(dim=0)
        elif tensor.dim() == 3:
            matrix = tensor[0].abs()
        elif tensor.dim() == 2:
            matrix = tensor[0].abs().unsqueeze(0)
        elif tensor.dim() == 1:
            matrix = tensor.abs().unsqueeze(0)
        else:
            return None
        return matrix.numpy().astype(np.float32)

    @staticmethod
    def _write_pgm(path: str, matrix: np.ndarray) -> None:
        finite = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.0))
        if high <= low:
            high = low + 1e-6
        image = np.clip((finite - low) / (high - low), 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)

        height, width = image.shape
        with open(path, "wb") as f:
            f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
            f.write(image.tobytes())

    @staticmethod
    def _safe_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        return cleaned[:120]
