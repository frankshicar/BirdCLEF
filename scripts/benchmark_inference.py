"""Benchmark CPU inference throughput on local soundscape files."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birdclef2026.src.inference import InferenceEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BirdCLEF CPU inference")
    parser.add_argument("--checkpoint", default="./checkpoints/best_checkpoint.pt")
    parser.add_argument("--soundscape-dir", default="./data/test_soundscapes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="0 means all files")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-views", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    soundscape_dir = Path(args.soundscape_dir)
    files = sorted(p for p in soundscape_dir.glob("*.ogg"))
    if args.limit > 0:
        files = files[:args.limit]

    if not files:
        raise SystemExit(f"No .ogg files found in {soundscape_dir}")

    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device="cpu",
        batch_size=args.batch_size,
        tta=args.tta,
        tta_views=args.tta_views,
    )
    engine.verify_paths()

    start = time.perf_counter()
    total_segments = 0
    for path in files:
        preds = engine.predict_soundscape(str(path))
        total_segments += len(preds)
    elapsed = time.perf_counter() - start

    logger.info("Files processed: %d", len(files))
    logger.info("Segments processed: %d", total_segments)
    logger.info("Elapsed seconds: %.2f", elapsed)
    logger.info("Seconds per file: %.3f", elapsed / max(len(files), 1))
    logger.info("Segments per second: %.2f", total_segments / max(elapsed, 1e-9))


if __name__ == "__main__":
    main()
