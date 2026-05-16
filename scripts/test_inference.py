"""Test inference locally using a training audio file.

This script validates that the trained model can successfully run inference
before uploading to Kaggle.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birdclef2026.src.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test inference locally")
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best_checkpoint.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to a test audio file (.ogg). If not provided, uses first training audio.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation",
    )
    return parser.parse_args()


def find_first_audio_file(data_dir: str = "./data/train_audio") -> str | None:
    """Find the first .ogg file in the training audio directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return None
    
    for species_dir in sorted(data_path.iterdir()):
        if species_dir.is_dir():
            for audio_file in sorted(species_dir.iterdir()):
                if audio_file.suffix.lower() == ".ogg":
                    return str(audio_file)
    return None


def main() -> None:
    args = parse_args()

    # Find audio file if not provided
    audio_path = args.audio
    if audio_path is None:
        logger.info("No audio file specified, searching for first training audio...")
        audio_path = find_first_audio_file()
        if audio_path is None:
            logger.error("Could not find any .ogg files in ./data/train_audio/")
            sys.exit(1)
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    logger.info(f"Using audio file: {audio_path}")
    logger.info(f"Using checkpoint: {args.checkpoint}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"TTA enabled: {args.tta}")

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Initialize inference engine
    logger.info("Initializing InferenceEngine...")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        tta=args.tta,
    )

    # Verify paths
    logger.info("Verifying paths...")
    try:
        engine.verify_paths()
        logger.info("✓ All paths verified")
    except FileNotFoundError as e:
        logger.error(f"Path verification failed: {e}")
        sys.exit(1)

    # Run inference on the audio file
    logger.info(f"Running inference on {Path(audio_path).name}...")
    try:
        results = engine.predict_soundscape(audio_path)
        logger.info(f"✓ Inference successful")
        logger.info(f"  Number of segments: {len(results)}")
        
        if results:
            # Show first result
            first_row_id = list(results.keys())[0]
            first_probs = results[first_row_id]
            
            logger.info(f"\nFirst segment: {first_row_id}")
            logger.info(f"  Probability vector shape: {first_probs.shape}")
            logger.info(f"  Probability range: [{first_probs.min():.6f}, {first_probs.max():.6f}]")
            logger.info(f"  Mean probability: {first_probs.mean():.6f}")
            
            # Show top 5 predictions
            top5_indices = np.argsort(first_probs)[-5:][::-1]
            logger.info(f"\n  Top 5 predictions:")
            
            # Load label map from checkpoint to show species names
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            label_map = ckpt.get("label_map", {})
            idx_to_label = {v: k for k, v in label_map.items()}
            
            for rank, idx in enumerate(top5_indices, 1):
                species = idx_to_label.get(idx, f"class_{idx}")
                prob = first_probs[idx]
                logger.info(f"    {rank}. {species:20s} {prob:.6f}")
        
        logger.info("\n✓ Inference test completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Package checkpoint and code for Kaggle")
        logger.info("  2. Upload as Kaggle Datasets")
        logger.info("  3. Run submission.ipynb on Kaggle")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
