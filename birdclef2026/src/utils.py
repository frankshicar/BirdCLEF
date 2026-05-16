"""Utility functions: config loading, seed setup, and run metadata logging."""

import hashlib
import logging
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

# Required keys that must be present in any loaded config
REQUIRED_KEYS = [
    "sample_rate",
    "segment_duration",
    "hop_duration",
    "n_mels",
    "hop_length",
    "n_fft",
    "backbone",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "seed",
]


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file, validate required keys, and return the config dict.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        KeyError: If a required configuration key is missing, with message
                  "Missing required config key: {key}".
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    for key in REQUIRED_KEYS:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    return config


def setup_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed value to apply to all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _hash_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"


def log_run_metadata(config: dict[str, Any], config_path: str | None = None) -> None:
    """Log git commit hash, config file hash, and the full resolved config.

    Args:
        config: The fully resolved configuration dictionary.
        config_path: Optional path to the config file used to compute its hash.
    """
    commit_hash = _get_git_commit_hash()
    config_hash = _hash_file(config_path) if config_path else "unknown"

    logger.info("Git commit hash : %s", commit_hash)
    logger.info("Config file hash: %s", config_hash)
    logger.info("Resolved config :\n%s", yaml.dump(config, default_flow_style=False))
