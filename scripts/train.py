"""Training entry point for BirdCLEF 2026.

Usage:
    python scripts/train.py --config birdclef2026/config/default.yaml
    python scripts/train.py --config path/to/config.yaml --resume path/to/checkpoint.pt
"""

import argparse
import logging
import os
import sys

# Ensure the project root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from birdclef2026.src.utils import load_config, log_run_metadata, setup_seed
from birdclef2026.src.audio import AudioPreprocessor, BackgroundNoiseMixer
from birdclef2026.src.dataset import DatasetBuilder
from birdclef2026.src.features import MelSpectrogramExtractor, MixupCollator
from birdclef2026.src.model import BirdCLEFModel
from birdclef2026.src.train import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BirdCLEF 2026 model")
    parser.add_argument(
        "--config",
        default="birdclef2026/config/default.yaml",
        help="Path to YAML config file (default: birdclef2026/config/default.yaml)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load config
    config = load_config(args.config)

    # 2. Inject resume path into config if provided
    if args.resume is not None:
        config["resume_checkpoint"] = args.resume

    # 3. Seed all RNGs
    setup_seed(config["seed"])

    # 4. Log run metadata (git hash, config hash, resolved config)
    log_run_metadata(config, config_path=args.config)

    # 5. Build datasets
    data_dir = config.get("data_dir", "/kaggle/input/birdclef-2026")
    taxonomy_path = os.path.join(data_dir, "taxonomy.csv")
    train_csv_path = os.path.join(data_dir, "train.csv")
    audio_dir = os.path.join(data_dir, "train_audio")
    soundscape_dir = os.path.join(data_dir, "train_soundscapes")
    soundscape_labels_path = os.path.join(data_dir, "train_soundscapes_labels.csv")

    # Use soundscape labels only if the file exists
    if not os.path.isfile(soundscape_labels_path):
        soundscape_labels_path = None
    if not os.path.isdir(soundscape_dir):
        soundscape_dir = None

    builder = DatasetBuilder(
        taxonomy_path=taxonomy_path,
        train_csv_path=train_csv_path,
        soundscape_labels_path=soundscape_labels_path,
        audio_dir=audio_dir,
        soundscape_dir=soundscape_dir,
        rating_threshold=config.get("rating_threshold", 0.0),
    )

    # Noise augmentation (optional — set noise_dir in config to enable)
    noise_dir = config.get("noise_dir", None)
    noise_mixer = BackgroundNoiseMixer(
        noise_dir=noise_dir,
        sample_rate=config["sample_rate"],
        snr_db_range=tuple(config.get("noise_snr_db_range", [5.0, 30.0])),
        p=config.get("noise_augment_p", 0.5),
    ) if noise_dir else None

    train_ds, val_ds = builder.build(
        val_fraction=config.get("val_fraction", 0.1),
        seed=config["seed"],
        augment=config.get("use_spec_augment", True),
        noise_mixer=noise_mixer,
        split_strategy=config.get("split_strategy", "soundscape_group"),
        soundscape_group_by=config.get("soundscape_group_by", "filename"),
    )

    logger.info(
        "Dataset split: train=%d val=%d strategy=%s group_by=%s",
        len(train_ds),
        len(val_ds),
        config.get("split_strategy", "soundscape_group"),
        config.get("soundscape_group_by", "filename"),
    )

    # Store label_map in config so it gets embedded in every checkpoint
    config["label_map"] = builder.label_map
    logger.info("label_map contains %d classes", len(builder.label_map))

    # 6. Build mel spectrogram extractor (PCEN or log-mel)
    use_pcen = config.get("use_pcen", False)

    extractor = MelSpectrogramExtractor(
        sample_rate=config["sample_rate"],
        n_mels=config["n_mels"],
        hop_length=config["hop_length"],
        n_fft=config["n_fft"],
        top_db=config.get("top_db", 80.0),
        f_min=config.get("f_min", 50.0),
        f_max=config.get("f_max", 15000.0),
        use_pcen=use_pcen,
    )

    fit_mel_stats = config.get("fit_mel_stats", True)

    if use_pcen:
        logger.info("Using PCEN feature extraction (skipping mel stats fitting)")
        config["mel_mean"] = 0.0
        config["mel_std"] = 1.0
    elif fit_mel_stats:
        logger.info("Fitting mel spectrogram statistics over training set ...")

        preprocessor_for_stats = AudioPreprocessor(
            sample_rate=config["sample_rate"],
            highpass_cutoff=config.get("highpass_cutoff", 0.0),
        )

        def _waveform_iter():
            """Yield raw waveforms from the training dataset for stats fitting."""
            min_samples = config["n_fft"] * 2
            for record in train_ds.samples:
                waveform = preprocessor_for_stats.load(record.audio_path)
                if waveform is not None and len(waveform) >= min_samples:
                    yield waveform

        mel_mean, mel_std = extractor.fit_stats(_waveform_iter())
        config["mel_mean"] = mel_mean
        config["mel_std"] = mel_std
        logger.info("mel_mean=%.4f  mel_std=%.4f", mel_mean, mel_std)

        # Rebuild extractor with fitted stats
        extractor = MelSpectrogramExtractor(
            sample_rate=config["sample_rate"],
            n_mels=config["n_mels"],
            hop_length=config["hop_length"],
            n_fft=config["n_fft"],
            top_db=config.get("top_db", 80.0),
            mean=mel_mean,
            std=mel_std,
            f_min=config.get("f_min", 50.0),
            f_max=config.get("f_max", 15000.0),
            use_pcen=False,
        )
    else:
        logger.info(
            "Using configured mel statistics: mel_mean=%.4f mel_std=%.4f",
            config.get("mel_mean", 0.0),
            config.get("mel_std", 1.0),
        )
        extractor = MelSpectrogramExtractor(
            sample_rate=config["sample_rate"],
            n_mels=config["n_mels"],
            hop_length=config["hop_length"],
            n_fft=config["n_fft"],
            top_db=config.get("top_db", 80.0),
            mean=config.get("mel_mean", 0.0),
            std=config.get("mel_std", 1.0),
            f_min=config.get("f_min", 50.0),
            f_max=config.get("f_max", 15000.0),
            use_pcen=False,
        )

    train_ds.extractor = extractor
    val_ds.extractor = extractor

    # 7. Build data loaders — use WeightedRandomSampler to handle class imbalance
    batch_size = config["batch_size"]
    use_mixup = config.get("use_mixup", False)
    collate_fn = MixupCollator(alpha=config.get("mixup_alpha", 0.4)) if use_mixup else None

    use_weighted_sampler = config.get("use_weighted_sampler", True)
    if use_weighted_sampler:
        source_weights = config.get("source_weights", {})
        sampler = builder.make_weighted_sampler(train_ds, source_weights=source_weights)
        logger.info("Using WeightedRandomSampler for class-balanced training")
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=config.get("num_workers", 2),
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 2),
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=False,
    )

    # 8. Instantiate model
    force_cpu = bool(int(os.environ.get("FORCE_CPU", "0")))
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    logger.info("Using device: %s", device)

    model = BirdCLEFModel(
        backbone_name=config["backbone"],
        num_classes=len(builder.label_map),
        pretrained=config.get("pretrained", False),
        pool=config.get("pool", "avg"),
        use_denoiser=config.get("use_denoiser", False),
        denoiser_channels=config.get("denoiser_channels", 64),
        denoiser_type=config.get("denoiser_type", "residual"),
    )

    # 9. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    trainer.train(num_epochs=config["num_epochs"])

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
