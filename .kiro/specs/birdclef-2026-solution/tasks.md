# Implementation Plan: BirdCLEF 2026 Solution

## Overview

Implement a BirdCLEF 2026 ML pipeline in Python using PyTorch + torchaudio + timm. The pipeline covers offline GPU training and CPU-only Kaggle submission inference. Tasks follow the module layout in the design document.

## Tasks

- [x] 1. Project scaffold and configuration
  - Create the directory structure: `birdclef2026/src/`, `birdclef2026/config/`, `notebooks/`, `scripts/`, `tests/`
  - Write `config/default.yaml` with all hyperparameters from the configuration schema in the design
  - Write `src/utils.py` with `load_config()`, `setup_seed()`, and `log_run_metadata()`
  - `load_config()` must validate required keys and raise `KeyError("Missing required config key: {key}")` on missing keys
  - `setup_seed()` must seed Python `random`, `numpy`, and `torch` RNGs
  - `log_run_metadata()` must log git commit hash, config file hash, and full resolved config
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.3_

  - [ ]* 1.1 Write property test for config round-trip (Property 23)
    - **Property 23: Config round-trip**
    - **Validates: Requirements 11.1, 11.2**

  - [ ]* 1.2 Write property test for missing config key raises error (Property 24)
    - **Property 24: Missing config key raises error**
    - **Validates: Requirements 11.3**

  - [ ]* 1.3 Write property test for seed determinism (Property 25)
    - **Property 25: Seed determinism**
    - **Validates: Requirements 12.1, 12.2**

- [x] 2. Audio loading and preprocessing (`src/audio.py`)
  - Implement `AudioPreprocessor` with `load()` and `normalize()` methods
  - `load()` reads `.ogg` via torchaudio, resamples to `sample_rate`, converts to mono float32 numpy array, normalizes amplitude, and returns `None` (logging the filename) on corrupt/unreadable files
  - `normalize()` scales waveform to `[-1.0, 1.0]` by dividing by `max(abs(waveform))`; handle zero-amplitude edge case
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.1 Write property test for audio load returns mono float32 (Property 1)
    - **Property 1: Audio load returns mono float32 waveform**
    - **Validates: Requirements 1.2, 1.3**

  - [ ]* 2.2 Write property test for waveform normalization bounds (Property 2)
    - **Property 2: Waveform normalization bounds**
    - **Validates: Requirements 1.5**

  - [ ]* 2.3 Write property test for resampling produces target sample rate (Property 3)
    - **Property 3: Resampling produces target sample rate**
    - **Validates: Requirements 1.1**

  - [ ]* 2.4 Write unit test for corrupt file returns None
    - Test that `AudioPreprocessor.load()` returns `None` and logs on a corrupt file
    - _Requirements: 1.4_

- [x] 3. Segment extraction (`src/audio.py`)
  - Implement `SegmentExtractor` with `extract()` method
  - Split waveform into fixed-length segments using `segment_duration` and `hop_duration`
  - Zero-pad the final segment if shorter than `segment_duration`
  - Assign `row_id` in format `{filename}_{end_seconds}` where `end_seconds` is an integer
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 3.1 Write property test for segment count matches duration (Property 4)
    - **Property 4: Segment count matches duration**
    - **Validates: Requirements 2.1, 2.4**

  - [ ]* 3.2 Write property test for Row_ID format correctness (Property 5)
    - **Property 5: Row_ID format correctness**
    - **Validates: Requirements 2.2**

  - [ ]* 3.3 Write unit test for short soundscape zero-padding
    - Test that a soundscape shorter than one segment is padded to `segment_duration`
    - _Requirements: 2.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Mel spectrogram feature extraction (`src/features.py`)
  - Implement `MelSpectrogramExtractor` using `torchaudio.transforms.MelSpectrogram` and `AmplitudeToDB`
  - Apply mel filterbank → power_to_db → `(x - mean) / std` normalization
  - Output shape must be `(1, n_mels, time_frames)` as `torch.float32`
  - Implement `fit_stats()` to compute mean and std over an iterable of waveforms
  - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [ ]* 5.1 Write property test for mel spectrogram output shape and dtype (Property 6)
    - **Property 6: Mel spectrogram output shape and dtype**
    - **Validates: Requirements 3.1, 3.6**

  - [ ]* 5.2 Write property test for spectrogram normalization statistics (Property 7)
    - **Property 7: Spectrogram normalization statistics**
    - **Validates: Requirements 3.3**

- [x] 6. Data augmentation (`src/features.py`)
  - Implement `SpecAugment` as an `nn.Module` wrapping `torchaudio.transforms.TimeMasking` and `FrequencyMasking`
  - Implement `MixupCollator` as a collate function that blends pairs of `(spectrogram, label)` tensors using a Beta-distributed lambda
  - _Requirements: 3.4, 3.5_

  - [ ]* 6.1 Write property test for mixup produces convex combination (Property 8)
    - **Property 8: Mixup produces convex combination**
    - **Validates: Requirements 3.5**

  - [ ]* 6.2 Write unit test for SpecAugment produces zeros in masked regions
    - Verify that time and frequency masking sets the expected regions to zero
    - _Requirements: 3.4_

- [x] 7. Dataset construction (`src/dataset.py`)
  - Implement `DatasetBuilder` that parses `train.csv`, `taxonomy.csv`, and optionally `train_soundscapes_labels.csv`
  - Build `label_map` (primary_label → int 0–233) from taxonomy order; freeze after loading
  - Assign multi-hot float32 label vectors of length 234 for each sample
  - Filter samples with `rating < rating_threshold`
  - Implement stratified split by `primary_label` using `val_fraction` and `seed`
  - Implement `BirdCLEFDataset` returning `(spectrogram_tensor, label_tensor)` tuples
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 7.1 Write property test for multi-hot label vector correctness (Property 9)
    - **Property 9: Multi-hot label vector correctness**
    - **Validates: Requirements 4.3**

  - [ ]* 7.2 Write property test for rating threshold filtering (Property 10)
    - **Property 10: Rating threshold filtering**
    - **Validates: Requirements 4.4**

  - [ ]* 7.3 Write property test for stratified split coverage (Property 11)
    - **Property 11: Stratified split coverage**
    - **Validates: Requirements 4.5**

  - [ ]* 7.4 Write property test for taxonomy label map consistency (Property 12)
    - **Property 12: Taxonomy label map consistency**
    - **Validates: Requirements 4.6**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Model architecture (`src/model.py`)
  - Implement `BirdCLEFModel` using `timm.create_model(backbone_name, pretrained=False, num_classes=0)`
  - Expand single-channel input to 3 channels by repeating before passing to backbone
  - Apply global average pooling (or attention pooling if `pool="attention"`) over spatial/time dimensions
  - Add `nn.Linear(feature_dim, 234)` classification head; `forward()` returns raw logits with no sigmoid
  - Support loading backbone weights from a local file path via `checkpoint_path`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 9.1 Write property test for model output shape (Property 13)
    - **Property 13: Model output shape**
    - **Validates: Requirements 5.1**

  - [ ]* 9.2 Write unit test for model forward produces no sigmoid
    - Verify that logits can be outside `[0, 1]` (raw logits, not probabilities)
    - _Requirements: 5.6_

- [x] 10. Checkpoint save and load (`src/model.py` + `src/train.py`)
  - Implement checkpoint saving in `Trainer` with the schema: `model_state_dict`, `config`, `label_map`, `epoch`, `val_roc_auc`
  - Implement checkpoint loading in `InferenceEngine.__init__()` with validation of required keys (raise `ValueError` listing missing keys)
  - _Requirements: 9.2, 5.5, 8.1_

  - [ ]* 10.1 Write property test for checkpoint round-trip (Property 14)
    - **Property 14: Checkpoint round-trip**
    - **Validates: Requirements 5.5, 8.1, 9.2**

- [x] 11. Training loop (`src/train.py`)
  - Implement `Trainer` with `train()` method
  - Use `nn.BCEWithLogitsLoss`; apply label smoothing to positive targets: `target = 1.0 - label_smoothing` when `label_smoothing > 0`
  - Use AdamW optimizer and `CosineAnnealingWarmRestarts` scheduler
  - Enable mixed-precision training via `torch.cuda.amp.GradScaler` when `mixed_precision=true`
  - Log training loss, validation loss, and validation macro ROC-AUC after each epoch
  - Save checkpoint when `val_roc_auc` improves; support resume via `config['resume_checkpoint']`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 11.1 Write property test for label smoothing target values (Property 15)
    - **Property 15: Label smoothing target values**
    - **Validates: Requirements 6.2**

  - [ ]* 11.2 Write property test for best checkpoint selection (Property 16)
    - **Property 16: Best checkpoint selection**
    - **Validates: Requirements 6.5**

  - [ ]* 11.3 Write unit test for checkpoint resume produces same metrics
    - Test that resuming from a checkpoint and continuing training yields consistent metrics
    - _Requirements: 6.7_

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Evaluation and metrics (`src/evaluate.py`)
  - Implement `Evaluator` with `evaluate()` method
  - Apply sigmoid to logits before computing ROC-AUC
  - Compute per-class ROC-AUC using `sklearn.metrics.roc_auc_score`; exclude classes with no positive labels (set to `None` in per-class report)
  - Return `{'macro_roc_auc': float, 'per_class': dict[str, float | None]}`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 13.1 Write property test for macro ROC-AUC computation (Property 17)
    - **Property 17: Macro ROC-AUC computation**
    - **Validates: Requirements 7.1, 7.3, 7.4**

- [x] 14. Inference engine (`src/inference.py`)
  - Implement `InferenceEngine` with `verify_paths()`, `predict_soundscape()`, and `run()` methods
  - `verify_paths()` raises `FileNotFoundError` with the missing path in the message
  - `predict_soundscape()` returns `{row_id: prob_vector[234]}` with sigmoid-activated probabilities
  - `run()` processes all soundscapes, fills missing row_ids with `0.0`, writes `submission.csv`
  - Support configurable `batch_size` for CPU memory/throughput balance
  - Replace NaN/Inf in model output with `0.0` and log a warning
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.8, 8.10, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 14.1 Write property test for inference probabilities in valid range (Property 18)
    - **Property 18: Inference probabilities in valid range**
    - **Validates: Requirements 8.4, 8.5**

  - [ ]* 14.2 Write property test for submission completeness (Property 19)
    - **Property 19: Submission completeness**
    - **Validates: Requirements 8.8**

  - [ ]* 14.3 Write property test for missing path raises descriptive error (Property 22)
    - **Property 22: Missing path raises descriptive error**
    - **Validates: Requirements 10.5**

  - [ ]* 14.4 Write unit test for CPU-only inference (no .cuda() calls)
    - Verify that `InferenceEngine` does not invoke `.cuda()` or `.to("cuda")`
    - _Requirements: 10.4_

- [x] 15. Ensemble and TTA support (`src/inference.py`)
  - Extend `InferenceEngine` to accept a list of checkpoint paths; average probabilities across checkpoints
  - Implement TTA: when `tta=True`, generate multiple augmented views per segment and average predictions
  - _Requirements: 8.6, 8.9_

  - [ ]* 15.1 Write property test for ensemble averaging (Property 20)
    - **Property 20: Ensemble averaging**
    - **Validates: Requirements 8.9**

  - [ ]* 15.2 Write property test for TTA averaging (Property 21)
    - **Property 21: TTA averaging**
    - **Validates: Requirements 8.6**

- [x] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Training entry point (`scripts/train.py`)
  - Implement CLI script that wires together: `load_config()` → `setup_seed()` → `log_run_metadata()` → `DatasetBuilder.build()` → `MelSpectrogramExtractor.fit_stats()` → `Trainer.train()`
  - Accept `--config` and `--resume` CLI arguments
  - Produce a self-contained checkpoint including `model_state_dict`, `config`, `label_map`, `epoch`, `val_roc_auc`
  - _Requirements: 9.1, 9.2, 9.5_

- [x] 18. Kaggle submission notebook (`notebooks/submission.ipynb`)
  - Implement `submission.ipynb` that: calls `verify_paths()` at startup, loads checkpoint, runs `InferenceEngine.run()`, writes `submission.csv`
  - Declare all required Kaggle Dataset dependencies in notebook metadata
  - Ensure no GPU-specific calls (`.cuda()`, `.to("cuda")`) appear anywhere in the notebook
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [x] 19. Wire everything together and final validation
  - [x] 19.1 Confirm `label_map` is embedded in every checkpoint and loaded consistently at inference
    - Verify the taxonomy mapping is frozen and identical between training and inference
    - _Requirements: 4.6, 9.2_

  - [x] 19.2 Confirm `submission.csv` column order matches `sample_submission.csv`
    - Write a unit test that checks column names and order against the sample submission schema
    - _Requirements: 8.7_

  - [x] 19.3 Confirm all required config keys are validated at startup for both training and inference runs
    - _Requirements: 11.3_

- [x] 20. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use `hypothesis` with `@settings(max_examples=100)` and are tagged with `# Feature: birdclef-2026-solution, Property N: ...`
- Run tests with `pytest --hypothesis-seed=0` for reproducible CI results
- No GPU is required for any test; all tests run on CPU
- Synthetic `.ogg` fixtures for integration tests are generated at test time using `soundfile`
