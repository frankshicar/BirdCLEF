"""
檢查點工具函數
從 train.py 中提取的獨立函數，避免循環依賴
"""

import torch

REQUIRED_CHECKPOINT_KEYS = ["model_state_dict", "config", "label_map", "epoch", "val_roc_auc"]

def validate_checkpoint(path: str) -> dict:
    """Load a checkpoint and validate that all required keys are present.

    Args:
        path: path to the checkpoint file (.pt / .pth)

    Returns:
        The checkpoint dict if all required keys are present.

    Raises:
        ValueError: if any required keys are missing, listing them in the message.
    """
    checkpoint = torch.load(path, map_location="cpu")
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    return checkpoint