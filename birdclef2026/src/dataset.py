"""Dataset construction for BirdCLEF 2026.

Parses train.csv, taxonomy.csv, and optionally train_soundscapes_labels.csv
to build BirdCLEFDataset instances for training and validation.
"""

from __future__ import annotations

import ast
import csv
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from birdclef2026.src.audio import AudioPreprocessor
from birdclef2026.src.features import MelSpectrogramExtractor

logger = logging.getLogger(__name__)

NUM_CLASSES = 234


@dataclass
class SampleRecord:
    """A single training sample."""

    audio_path: str
    start_sec: float          # 0.0 for train_audio clips
    end_sec: float            # duration for train_audio clips
    label_vector: np.ndarray  # float32[234], multi-hot
    row_id: str | None = None  # set for soundscape-derived samples
    group_id: str | None = None  # used for leakage-safe train/val splitting
    source: str = "train_audio"   # train_audio | soundscape


def _parse_secondary_labels(raw: str) -> list[str]:
    """Parse secondary_labels field from train.csv.

    Handles both Python-list-literal format (e.g. "['compau', 'saffin']")
    and space-separated format (e.g. "compau saffin").
    Returns an empty list for empty/null values.
    """
    raw = raw.strip()
    if not raw or raw in ("[]", "''", '""'):
        return []
    # Try Python list literal first (e.g. "['compau']")
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except (ValueError, SyntaxError):
            pass
    # Fall back to space-separated
    return [s.strip() for s in raw.split() if s.strip()]


def _time_str_to_sec(t: str) -> float:
    """Convert HH:MM:SS time string to seconds."""
    parts = t.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


class DatasetBuilder:
    """Parses CSVs and builds BirdCLEFDataset instances.

    Args:
        taxonomy_path: Path to taxonomy.csv.
        train_csv_path: Path to train.csv.
        soundscape_labels_path: Optional path to train_soundscapes_labels.csv.
        audio_dir: Directory containing train_audio .ogg files.
        soundscape_dir: Optional directory containing train_soundscape .ogg files.
        rating_threshold: Exclude train_audio samples with rating < this value.
    """

    def __init__(
        self,
        taxonomy_path: str,
        train_csv_path: str,
        soundscape_labels_path: str | None,
        audio_dir: str,
        soundscape_dir: str | None,
        rating_threshold: float = 0.0,
    ) -> None:
        self.taxonomy_path = taxonomy_path
        self.train_csv_path = train_csv_path
        self.soundscape_labels_path = soundscape_labels_path
        self.audio_dir = audio_dir
        self.soundscape_dir = soundscape_dir
        self.rating_threshold = rating_threshold

        # Build label_map from taxonomy (frozen after construction)
        self._label_map: dict[str, int] = self._build_label_map()

    def _build_label_map(self) -> dict[str, int]:
        """Build primary_label → class index mapping from taxonomy.csv."""
        label_map: dict[str, int] = {}
        with open(self.taxonomy_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                label = row["primary_label"].strip()
                label_map[label] = idx
        return label_map

    @property
    def label_map(self) -> dict[str, int]:
        """primary_label → class index (0-233), derived from taxonomy.csv."""
        return self._label_map

    def _make_label_vector(
        self, primary_label: str, secondary_labels: list[str]
    ) -> np.ndarray:
        """Build a float32 multi-hot label vector of length NUM_CLASSES.

        Primary label = 1.0, secondary labels = 0.3 (soft label).
        Secondary labels are less certain, so a lower value reduces label noise.
        """
        vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        if primary_label in self._label_map:
            vec[self._label_map[primary_label]] = 1.0
        for lbl in secondary_labels:
            if lbl in self._label_map:
                vec[self._label_map[lbl]] = 0.3
        return vec

    def _load_train_audio_samples(self) -> list[SampleRecord]:
        """Parse train.csv and build SampleRecord list for train_audio files."""
        samples: list[SampleRecord] = []
        with open(self.train_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rating = float(row.get("rating", 0.0))
                if rating < self.rating_threshold:
                    continue

                primary_label = row["primary_label"].strip()
                secondary_labels = _parse_secondary_labels(
                    row.get("secondary_labels", "")
                )
                filename = row["filename"].strip()
                audio_path = os.path.join(self.audio_dir, filename)

                label_vector = self._make_label_vector(primary_label, secondary_labels)

                samples.append(
                    SampleRecord(
                        audio_path=audio_path,
                        start_sec=0.0,
                        end_sec=0.0,  # full clip; AudioPreprocessor loads entire file
                        label_vector=label_vector,
                        row_id=None,
                        group_id=f"audio:{filename}",
                        source="train_audio",
                    )
                )
        return samples

    def _make_soundscape_group_id(self, filename: str, start_sec: float) -> str:
        """Build a grouping key for soundscape validation splits."""
        stem = os.path.splitext(filename)[0]
        group_by = getattr(self, "soundscape_group_by", "filename")

        if group_by == "site":
            parts = stem.split("_")
            site = next((p for p in parts if p.startswith("S") and p[1:].isdigit()), stem)
            return f"soundscape_site:{site}"
        if group_by == "site_date":
            parts = stem.split("_")
            site = next((p for p in parts if p.startswith("S") and p[1:].isdigit()), stem)
            date = next((p for p in parts if len(p) == 8 and p.isdigit()), "unknown")
            return f"soundscape_site_date:{site}:{date}"
        if group_by == "hour":
            hour_block = int(start_sec // 3600)
            return f"soundscape_hour:{stem}:{hour_block}"
        return f"soundscape_file:{stem}"

    def _load_soundscape_samples(self) -> list[SampleRecord]:
        """Parse train_soundscapes_labels.csv and build SampleRecord list.

        Deduplicates rows by (filename, start, end) — the CSV sometimes
        contains duplicate entries for the same time segment.
        Skips segments where none of the label IDs are in the label_map.
        """
        if self.soundscape_labels_path is None or self.soundscape_dir is None:
            return []

        seen: set[tuple[str, float, float]] = set()
        samples: list[SampleRecord] = []

        with open(self.soundscape_labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"].strip()
                start_sec = _time_str_to_sec(row["start"])
                end_sec = _time_str_to_sec(row["end"])

                # Deduplicate by (filename, start, end)
                key = (filename, start_sec, end_sec)
                if key in seen:
                    continue
                seen.add(key)

                # primary_label column contains semicolon-separated species IDs
                raw_labels = row["primary_label"].strip()
                label_ids = [s.strip() for s in raw_labels.split(";") if s.strip()]

                # Keep only IDs that exist in our label_map (skip eBird codes etc.)
                known_ids = [lid for lid in label_ids if lid in self._label_map]
                if not known_ids:
                    continue

                primary_label = known_ids[0]
                secondary_labels = known_ids[1:]

                audio_path = os.path.join(self.soundscape_dir, filename)
                # Skip if the audio file doesn't exist on disk
                if not os.path.isfile(audio_path):
                    continue

                label_vector = self._make_label_vector(primary_label, secondary_labels)

                stem = os.path.splitext(filename)[0]
                end_seconds = int(end_sec)
                row_id = f"{stem}_{end_seconds}"

                samples.append(
                    SampleRecord(
                        audio_path=audio_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label_vector=label_vector,
                        row_id=row_id,
                        group_id=self._make_soundscape_group_id(filename, start_sec),
                        source="soundscape",
                    )
                )

        logger.debug("Loaded %d soundscape segments (after dedup)", len(samples))
        return samples

    def build(
        self,
        val_fraction: float = 0.1,
        seed: int = 42,
        preprocessor: AudioPreprocessor | None = None,
        extractor: MelSpectrogramExtractor | None = None,
        augment: bool = True,
        noise_mixer=None,
        split_strategy: str = "stratified",
        soundscape_group_by: str = "filename",
    ) -> tuple[BirdCLEFDataset, BirdCLEFDataset]:
        """Build train and validation datasets with stratified split.

        Args:
            val_fraction: Fraction of samples to use for validation.
            seed: Random seed for reproducible splits.
            preprocessor: AudioPreprocessor instance (created with defaults if None).
            extractor: MelSpectrogramExtractor instance (created with defaults if None).
            augment: Whether to enable augmentation on the training dataset.

        Returns:
            (train_dataset, val_dataset) tuple.
        """
        if preprocessor is None:
            preprocessor = AudioPreprocessor()
        if extractor is None:
            extractor = MelSpectrogramExtractor(
                sample_rate=32000, n_mels=160, hop_length=320, n_fft=2048,
                f_min=50.0, f_max=15000.0, top_db=80.0,
            )

        self.soundscape_group_by = soundscape_group_by

        all_samples = self._load_train_audio_samples()
        all_samples += self._load_soundscape_samples()

        if split_strategy in {"group", "soundscape_group"}:
            train_samples, val_samples = self._group_split(
                all_samples,
                val_fraction=val_fraction,
                seed=seed,
                soundscape_only=(split_strategy == "soundscape_group"),
            )
        else:
            train_samples, val_samples = self._stratified_split(
                all_samples, val_fraction=val_fraction, seed=seed
            )

        train_ds = BirdCLEFDataset(train_samples, preprocessor, extractor, augment=augment, noise_mixer=noise_mixer)
        val_ds = BirdCLEFDataset(val_samples, preprocessor, extractor, augment=False)
        return train_ds, val_ds

    def make_weighted_sampler(
        self,
        dataset: "BirdCLEFDataset",
        source_weights: dict[str, float] | None = None,
    ) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler that up-samples rare classes.

        Each sample's weight = 1 / (count of its primary class in the dataset).
        This gives rare species the same expected frequency as common ones.
        Optional source_weights can up-weight domains such as soundscape.
        """
        source_weights = source_weights or {}

        # Count samples per primary class
        class_counts: dict[int, int] = {}
        for record in dataset.samples:
            idx = int(np.argmax(record.label_vector))
            class_counts[idx] = class_counts.get(idx, 0) + 1

        weights = []
        for record in dataset.samples:
            idx = int(np.argmax(record.label_vector))
            source_weight = source_weights.get(record.source, 1.0)
            weights.append((1.0 / class_counts[idx]) * source_weight)

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(weights_tensor),
            replacement=True,
        )

    def _stratified_split(        self,
        samples: list[SampleRecord],
        val_fraction: float,
        seed: int,
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Stratified split by primary_label index.

        For each class, puts approximately val_fraction of samples into val.
        Classes with only 1 sample go entirely to train.
        """
        rng = np.random.default_rng(seed)

        # Group samples by primary label (determined by argmax of label_vector)
        from collections import defaultdict
        groups: dict[int, list[SampleRecord]] = defaultdict(list)
        for s in samples:
            primary_idx = int(np.argmax(s.label_vector))
            groups[primary_idx].append(s)

        train_samples: list[SampleRecord] = []
        val_samples: list[SampleRecord] = []

        for idx, group in groups.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            n_val = max(1, round(len(shuffled) * val_fraction)) if len(shuffled) >= 2 else 0
            val_samples.extend(shuffled[:n_val])
            train_samples.extend(shuffled[n_val:])

        return train_samples, val_samples

    def _group_split(
        self,
        samples: list[SampleRecord],
        val_fraction: float,
        seed: int,
        soundscape_only: bool = False,
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Split samples without allowing a group_id to cross train/val.

        When ``soundscape_only`` is true, soundscape records are split by group
        and train_audio records are split with the existing stratified strategy.
        This keeps the validation set close to deployment while preserving
        label coverage from focal recordings.
        """
        if not samples:
            return [], []

        if soundscape_only:
            soundscape = [s for s in samples if s.source == "soundscape"]
            focal = [s for s in samples if s.source != "soundscape"]
            focal_train, focal_val = self._stratified_split(focal, val_fraction, seed)
            sound_train, sound_val = self._split_groups(soundscape, val_fraction, seed)
            return focal_train + sound_train, focal_val + sound_val

        return self._split_groups(samples, val_fraction, seed)

    def _split_groups(
        self,
        samples: list[SampleRecord],
        val_fraction: float,
        seed: int,
    ) -> tuple[list[SampleRecord], list[SampleRecord]]:
        """Group-level random split that keeps all group members together."""
        from collections import defaultdict

        if not samples:
            return [], []

        rng = np.random.default_rng(seed)
        groups: dict[str, list[SampleRecord]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            group_id = sample.group_id or f"sample:{idx}"
            groups[group_id].append(sample)

        group_items = list(groups.items())
        rng.shuffle(group_items)

        target_val = int(round(len(samples) * val_fraction))
        target_val = max(1, target_val) if len(samples) >= 2 and val_fraction > 0 else 0

        val_group_ids: set[str] = set()
        val_count = 0
        for group_id, group_samples in group_items:
            if val_count >= target_val:
                break
            if len(group_items) > 1 and val_count + len(group_samples) >= len(samples):
                continue
            val_group_ids.add(group_id)
            val_count += len(group_samples)

        train_samples: list[SampleRecord] = []
        val_samples: list[SampleRecord] = []
        for group_id, group_samples in group_items:
            if group_id in val_group_ids:
                val_samples.extend(group_samples)
            else:
                train_samples.extend(group_samples)

        return train_samples, val_samples


class BirdCLEFDataset(Dataset):
    """PyTorch Dataset returning (spectrogram_tensor, label_tensor) tuples.

    Args:
        samples: List of SampleRecord instances.
        preprocessor: AudioPreprocessor for loading audio.
        extractor: MelSpectrogramExtractor for computing spectrograms.
        augment: If True, apply SpecAugment during __getitem__.
    """

    def __init__(
        self,
        samples: list[SampleRecord],
        preprocessor: AudioPreprocessor,
        extractor: MelSpectrogramExtractor,
        augment: bool = False,
        noise_mixer=None,
    ) -> None:
        self.samples = samples
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.augment = augment
        self.noise_mixer = noise_mixer

        if augment:
            from birdclef2026.src.features import SpecAugment
            self._spec_augment = SpecAugment()
        else:
            self._spec_augment = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load audio, compute spectrogram, return (spectrogram, label).

        For soundscape-derived samples (start_sec > 0 or end_sec > 0),
        only the relevant segment is extracted from the waveform.

        Returns:
            (spectrogram, label) where spectrogram is float32 (1, n_mels, T)
            and label is float32[234].
        """
        record = self.samples[idx]

        waveform = self.preprocessor.load(record.audio_path)
        if waveform is None:
            # Return zeros on corrupt file
            sr = self.preprocessor.sample_rate
            seg_len = int(5.0 * sr)
            waveform = np.zeros(seg_len, dtype=np.float32)
        else:
            # Extract segment for soundscape samples
            if record.start_sec > 0.0 or record.end_sec > 0.0:
                sr = self.preprocessor.sample_rate
                start_sample = int(record.start_sec * sr)
                end_sample = int(record.end_sec * sr)
                waveform = waveform[start_sample:end_sample]
                if len(waveform) == 0:
                    waveform = np.zeros(int((record.end_sec - record.start_sec) * sr), dtype=np.float32)

        # Ensure fixed-length waveform (5 seconds) for consistent spectrogram shape
        sr = self.preprocessor.sample_rate
        target_len = int(5.0 * sr)
        if len(waveform) < target_len:
            # Pad with zeros
            waveform = np.pad(waveform, (0, target_len - len(waveform)), mode='constant')
        elif len(waveform) > target_len:
            # Random crop during training, center crop during validation
            if self.augment:
                start = np.random.randint(0, len(waveform) - target_len + 1)
            else:
                start = (len(waveform) - target_len) // 2
            waveform = waveform[start:start + target_len]

        # Apply background noise augmentation (training only)
        if self.augment and self.noise_mixer is not None:
            waveform = self.noise_mixer(waveform)

        spectrogram = self.extractor(waveform)  # (1, n_mels, T)

        if self.augment and self._spec_augment is not None:
            spectrogram = self._spec_augment(spectrogram)

        label = torch.from_numpy(record.label_vector)  # float32[234]
        return spectrogram, label
