"""Prepare files for Kaggle submission.

Packages the trained model and birdclef2026 package into a zip where all
files sit at the root level, so that sys.path.insert(0, MODEL_DIR) works
directly in the Kaggle notebook.

Zip structure:
  checkpoint.pt
  birdclef2026/
    __init__.py
    config/
    src/
  dataset-metadata.json
  README.md
"""

import argparse
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Kaggle submission package")
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best_checkpoint.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        default="./kaggle_submission_new",
        help="Output directory for packaged files",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Version suffix for the zip filename (e.g. v9)",
    )
    return parser.parse_args()


def add_dir_to_zip(zf: zipfile.ZipFile, src_dir: Path, arcname_prefix: str) -> None:
    """Recursively add a directory to a zip, skipping pycache/pyc files."""
    for file_path in sorted(src_dir.rglob("*")):
        if file_path.is_dir():
            continue
        # Skip compiled / cache files
        parts = file_path.parts
        if any(p in ("__pycache__", ".pytest_cache") for p in parts):
            continue
        if file_path.suffix in (".pyc", ".pyo"):
            continue
        rel = file_path.relative_to(src_dir)
        arcname = f"{arcname_prefix}/{rel}" if arcname_prefix else str(rel)
        zf.write(file_path, arcname)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    version_suffix = f"-{args.version}" if args.version else ""
    zip_name = f"birdclef-2026-model{version_suffix}.zip"
    zip_path = output_dir / zip_name

    logger.info(f"Building {zip_path} ...")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. checkpoint at root
        logger.info(f"  Adding checkpoint.pt ({checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB)")
        zf.write(checkpoint_path, "checkpoint.pt")

        # 2. birdclef2026 package at root
        logger.info("  Adding birdclef2026/ package ...")
        add_dir_to_zip(zf, Path("birdclef2026"), "birdclef2026")

        # 3. dataset-metadata.json
        metadata = {
            "title": "BirdCLEF 2026 Trained Model",
            "id": "your-username/birdclef-2026-model",
            "licenses": [{"name": "CC0-1.0"}],
        }
        zf.writestr("dataset-metadata.json", json.dumps(metadata, indent=2))

        # 4. README
        readme = (
            "# BirdCLEF 2026 Trained Model\n\n"
            "Contents:\n"
            "- `checkpoint.pt` — trained ResNet-18 checkpoint\n"
            "- `birdclef2026/` — inference package\n\n"
            "## Quick start\n\n"
            "```python\n"
            "import sys\n"
            "MODEL_DIR = '/kaggle/input/datasets/<username>/birdclef-2026-trained-model'\n"
            "sys.path.insert(0, MODEL_DIR)\n"
            "from birdclef2026.src.inference import InferenceEngine\n"
            "```\n"
        )
        zf.writestr("README.md", readme)

    logger.info(f"\n✓ Done — {zip_path.absolute()}")
    logger.info(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Also copy the submission notebook next to the zip
    notebook_src = Path("notebooks/submission.ipynb")
    if notebook_src.exists():
        shutil.copy2(notebook_src, output_dir / "submission.ipynb")
        logger.info(f"  Notebook copied to {output_dir / 'submission.ipynb'}")

    logger.info("\nNext steps:")
    logger.info(f"  1. Upload {zip_path} to Kaggle Datasets")
    logger.info("  2. In the notebook, set MODEL_DIR to the actual Kaggle mount path:")
    logger.info("     /kaggle/input/datasets/<username>/<dataset-slug>")


if __name__ == "__main__":
    main()
