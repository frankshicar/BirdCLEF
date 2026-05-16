# Design Document: BirdCLEF 2026 Solution

## Overview

This document describes the technical design for a BirdCLEF 2026 Kaggle competition ML pipeline. The system classifies 234 species (birds, amphibians, mammals, reptiles, insects) from 1-minute soundscape recordings of the Brazilian Pantanal. Predictions are made per 5-second segment and evaluated with macro-averaged ROC-AUC.

The pipeline has two distinct execution environments:
- **Offline training** — runs on GPU hardware outside Kaggle, produces a checkpoint artifact
- **Kaggle submission notebook** — runs CPU-only inside Kaggle, loads the checkpoint, produces `submission.csv`

The core approach: convert audio to mel spectrograms → treat as image classification → use a pretrained CNN/ViT backbone via `timm` → multi-label binary classification with BCEWithLogitsLoss.

### Key Design Decisions

- **Mel spectrogram as image**: Standard and effective for bioacoustic classification; enables reuse of ImageNet-pretrained backbones.
- **`timm` model registry**: Provides a large selection of pretrained backbones (EfficientNet, ConvNeXt, ViT) with a uniform API, and supports loading weights from local files.
- **Offline train / online infer split**: Satisfies the Kaggle CPU-only constraint while allowing full GPU training. The checkpoint bundles everything needed for inference.
- **YAML config**: Single source of truth for all hyperparameters; enables reproducible experiments.
- **PyTorch + torchaudio**: Mature ecosystem, good CPU inference performance, native mixed-precision support.

---

## Architecture

```mermaid
graph TD
    subgraph Offline Training (GPU)
        A[train.csv / taxonomy.csv] --> B[DatasetBuilder]
        C[train_audio/ .ogg files] --> B
        D[train_soundscapes/] --> B
        B --> E[BirdCLEFDataset]
        E --> F[AudioPreprocessor]
        F --> G[MelSpectrogramExtractor]
        G --> H[BirdCLEFModel]
        H --> I[Trainer]
        I --> J[checkpoint.pt]
    end

    subgraph Kaggle Submission Notebook (CPU)
        K[test_soundscapes/] --> L[SegmentExtractor]
        L --> M[AudioPreprocessor]
        M --> N[MelSpectrogramExtractor]
        J --> O[InferenceEngine]
        N --> O
        O --> P[submission.csv]
    end
```

### Module Layout

```
birdclef2026/
├── config/
│   └── default.yaml          # All hyperparameters
├── src/
│   ├── audio.py              # AudioPreprocessor, SegmentExtractor
│   ├── features.py           # MelSpectrogramExtractor
│   ├── dataset.py            # DatasetBuilder, BirdCLEFDataset
│   ├── model.py              # BirdCLEFModel
│   ├── train.py              # Trainer
│   ├── evaluate.py           # Evaluator
│   ├── inference.py          # InferenceEngine
│   └── utils.py              # seed, config loading, logging helpers
├── notebooks/
│   └── submission.ipynb      # Kaggle submission notebook
└── scripts/
    └── train.py              # CLI entry point for offline training
```

---

## Components and Interfaces

### AudioPreprocessor (`src/audio.py`)

Responsible for loading `.ogg` files and returning normalized mono float32 waveforms.

```python
class AudioPreprocessor:
    def __init__(self, sample_rate: int = 32000): ...

    def load(self, path: str) -> np.ndarray:
        """Load .ogg, resample to sample_rate, convert to mono float32,
        normalize to [-1.0, 1.0]. Returns waveform array.
        Logs and returns None on corrupt/unreadable files."""

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize amplitude to [-1.0, 1.0]."""
```

### SegmentExtractor (`src/audio.py`)

Splits a waveform into fixed-length segments and assigns Row_IDs.

```python
class SegmentExtractor:
    def __init__(self, segment_duration: float = 5.0,
                 hop_duration: float = 5.0,
                 sample_rate: int = 32000): ...

    def extract(self, waveform: np.ndarray,
                filename: str) -> list[tuple[str, np.ndarray]]:
        """Returns list of (row_id, segment_waveform) tuples.
        Pads final segment with zeros if shorter than segment_duration.
        row_id format: {filename}_{end_seconds}"""
```

### MelSpectrogramExtractor (`src/features.py`)

Converts waveform segments to mel spectrogram tensors.

```python
class MelSpectrogramExtractor:
    def __init__(self, sample_rate: int, n_mels: int, hop_length: int,
                 n_fft: int, top_db: float = 80.0,
                 mean: float = 0.0, std: float = 1.0): ...

    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        """Returns float32 tensor of shape (1, n_mels, time_frames).
        Applies: mel filterbank → power_to_db → (x - mean) / std"""

    def fit_stats(self, dataset: Iterable[np.ndarray]) -> tuple[float, float]:
        """Compute mean and std over training set for normalization."""
```

Augmentation wrappers (training only):

```python
class SpecAugment(nn.Module):
    """Applies time masking and frequency masking (torchaudio.transforms)."""

class MixupCollator:
    """Collate function that blends pairs of (spectrogram, label) tensors."""
```

### DatasetBuilder (`src/dataset.py`)

Parses CSVs and builds PyTorch datasets.

```python
class DatasetBuilder:
    def __init__(self, taxonomy_path: str, train_csv_path: str,
                 soundscape_labels_path: str | None,
                 audio_dir: str, soundscape_dir: str | None,
                 rating_threshold: float = 0.0): ...

    def build(self, val_fraction: float = 0.1,
              seed: int = 42) -> tuple[BirdCLEFDataset, BirdCLEFDataset]:
        """Returns (train_dataset, val_dataset) with stratified split."""

    @property
    def label_map(self) -> dict[str, int]:
        """primary_label → class index (0-233), derived from taxonomy.csv."""
```

```python
class BirdCLEFDataset(Dataset):
    def __init__(self, samples: list[SampleRecord],
                 preprocessor: AudioPreprocessor,
                 extractor: MelSpectrogramExtractor,
                 augment: bool = False): ...

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (spectrogram, multi_hot_label) where label is float32[234]."""
```

`SampleRecord` is a dataclass holding `(audio_path, start_sec, end_sec, label_vector)`.

### BirdCLEFModel (`src/model.py`)

```python
class BirdCLEFModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 234,
                 pretrained: bool = True,
                 checkpoint_path: str | None = None,
                 pool: str = "avg"): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T) → logits: (B, 234). No sigmoid applied."""
```

Internally:
1. Expand single-channel input to 3 channels (repeat) for ImageNet-pretrained backbones.
2. Extract features via `timm.create_model(backbone_name, pretrained=False, num_classes=0)`.
3. Apply global average pooling (or attention pooling if configured).
4. Linear classification head: `nn.Linear(feature_dim, 234)`.

Checkpoint format (saved by Trainer):
```python
{
    "model_state_dict": ...,
    "config": {...},           # full resolved config dict
    "label_map": {...},        # primary_label → int
    "epoch": int,
    "val_roc_auc": float,
}
```

### Trainer (`src/train.py`)

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 config: dict, device: str): ...

    def train(self, num_epochs: int) -> None:
        """Main training loop. Saves best checkpoint by val ROC-AUC.
        Supports resume via config['resume_checkpoint']."""
```

- Loss: `nn.BCEWithLogitsLoss` with optional label smoothing applied to positive targets.
- Optimizer: AdamW.
- Scheduler: `CosineAnnealingWarmRestarts`.
- Mixed precision: `torch.cuda.amp.GradScaler`.
- Logging: prints epoch summary; optionally writes to a CSV log file.

### Evaluator (`src/evaluate.py`)

```python
class Evaluator:
    def __init__(self, model, val_loader, device: str): ...

    def evaluate(self) -> dict:
        """Returns {'macro_roc_auc': float, 'per_class': dict[str, float]}.
        Excludes species with no positive labels in val set.
        Applies sigmoid before sklearn.metrics.roc_auc_score."""
```

### InferenceEngine (`src/inference.py`)

```python
class InferenceEngine:
    def __init__(self, checkpoint_path: str,
                 backbone_weights_path: str | None = None,
                 device: str = "cpu",
                 batch_size: int = 32,
                 tta: bool = False): ...

    def verify_paths(self) -> None:
        """Raises FileNotFoundError with descriptive message if any
        required path is missing."""

    def predict_soundscape(self, soundscape_path: str) -> dict[str, np.ndarray]:
        """Returns {row_id: prob_vector[234]} for all segments."""

    def run(self, soundscape_dir: str,
            sample_submission_path: str,
            output_path: str = "submission.csv") -> None:
        """Processes all soundscapes, writes submission.csv.
        Fills missing row_ids with 0.0."""
```

Ensemble support: `InferenceEngine` accepts a list of checkpoint paths; probabilities are averaged across checkpoints before writing.

### Config (`src/utils.py` + `config/default.yaml`)

```python
def load_config(path: str) -> dict:
    """Load YAML, validate required keys, raise KeyError with key name if missing."""

def setup_seed(seed: int) -> None:
    """Set random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all."""

def log_run_metadata(config: dict) -> None:
    """Log git commit hash, config file hash, and full resolved config."""
```

---

## Data Models

### SampleRecord

```python
@dataclass
class SampleRecord:
    audio_path: str
    start_sec: float          # 0.0 for train_audio clips
    end_sec: float            # duration for train_audio clips
    label_vector: np.ndarray  # float32[234], multi-hot
    row_id: str | None        # set for soundscape-derived samples
```

### Taxonomy Mapping

Loaded once from `taxonomy.csv` at startup. Stored as:
```python
label_map: dict[str, int]   # "xencan1" → 0, ..., sorted by taxonomy order
index_to_label: list[str]   # inverse lookup
```
The mapping is frozen after loading and embedded in every checkpoint to guarantee consistency between training and inference.

### Checkpoint Schema

```python
{
    "model_state_dict": OrderedDict,   # PyTorch state dict
    "config": {                         # full hyperparameter dict
        "backbone": str,
        "n_mels": int,
        "hop_length": int,
        "n_fft": int,
        "sample_rate": int,
        "segment_duration": float,
        "mel_mean": float,
        "mel_std": float,
        ...
    },
    "label_map": dict[str, int],       # species → class index
    "epoch": int,
    "val_roc_auc": float,
}
```

### Submission CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | str | `{soundscape_filename}_{end_seconds}` |
| `{primary_label}` × 234 | float32 | Probability in [0.0, 1.0] |

Column order matches `sample_submission.csv` exactly.

### Configuration Schema (`config/default.yaml`)

```yaml
# Audio
sample_rate: 32000
segment_duration: 5.0
hop_duration: 5.0           # set < segment_duration for overlapping windows

# Feature extraction
n_mels: 128
hop_length: 320
n_fft: 1024
top_db: 80.0
mel_mean: 0.0               # updated after fit_stats
mel_std: 1.0

# Model
backbone: efficientnet_b0
pool: avg                   # avg | attention

# Training
num_epochs: 30
batch_size: 64
learning_rate: 1e-3
weight_decay: 1e-4
rating_threshold: 3.0
label_smoothing: 0.05
val_fraction: 0.1
seed: 42
mixed_precision: true
resume_checkpoint: null

# Augmentation
use_spec_augment: true
time_mask_param: 30
freq_mask_param: 20
use_mixup: true
mixup_alpha: 0.4

# Inference
inference_batch_size: 32
tta: false
ensemble_checkpoints: []

# Paths (override at runtime)
data_dir: /kaggle/input/birdclef-2026
checkpoint_dir: ./checkpoints
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Audio load returns mono float32 waveform

*For any* `.ogg` audio file (train_audio or soundscape), loading it through `AudioPreprocessor.load()` should return a 1-dimensional NumPy array with dtype `float32`.

**Validates: Requirements 1.2, 1.3**

---

### Property 2: Waveform normalization bounds

*For any* waveform array with non-zero amplitude, after `AudioPreprocessor.normalize()`, all values should lie in the closed interval `[-1.0, 1.0]` and the maximum absolute value should equal `1.0`.

**Validates: Requirements 1.5**

---

### Property 3: Resampling produces target sample rate

*For any* waveform loaded at an arbitrary source sample rate, after resampling, the resulting waveform length should equal `round(original_length * 32000 / source_rate)` (within ±1 sample for rounding).

**Validates: Requirements 1.1**

---

### Property 4: Segment count matches duration

*For any* waveform of duration `D` seconds and a `SegmentExtractor` configured with `segment_duration=S` and `hop_duration=H`, the number of extracted segments should equal `ceil(D / H)`.

**Validates: Requirements 2.1, 2.4**

---

### Property 5: Row_ID format correctness

*For any* waveform and filename passed to `SegmentExtractor.extract()`, every returned `row_id` should match the pattern `{filename}_{end_seconds}` where `end_seconds` is a positive integer multiple of `segment_duration`.

**Validates: Requirements 2.2**

---

### Property 6: Mel spectrogram output shape and dtype

*For any* 5-second waveform segment and `MelSpectrogramExtractor` configured with `n_mels=M`, the output tensor should have shape `(1, M, T)` for some `T > 0` and dtype `torch.float32`.

**Validates: Requirements 3.1, 3.6**

---

### Property 7: Spectrogram normalization statistics

*For any* spectrogram normalized with known `mean` and `std`, the output values should satisfy `(output * std + mean)` approximately recovering the pre-normalization values (round-trip within floating-point tolerance).

**Validates: Requirements 3.3**

---

### Property 8: Mixup produces convex combination

*For any* two spectrogram tensors `s1`, `s2` and label vectors `l1`, `l2`, and any `lambda` in `[0, 1]`, the mixup output should equal `(lambda * s1 + (1 - lambda) * s2, lambda * l1 + (1 - lambda) * l2)` element-wise.

**Validates: Requirements 3.5**

---

### Property 9: Multi-hot label vector correctness

*For any* training sample with a known `primary_label` and `secondary_labels`, the assigned label vector should have length 234, dtype `float32`, value `1.0` at the `primary_label` index, value `1.0` at each `secondary_label` index, and value `0.0` elsewhere.

**Validates: Requirements 4.3**

---

### Property 10: Rating threshold filtering

*For any* dataset built with rating threshold `T`, no sample in the resulting dataset should have a source recording with `rating < T`.

**Validates: Requirements 4.4**

---

### Property 11: Stratified split coverage

*For any* dataset with a set of unique primary labels `L`, after a stratified split, every label in `L` that has at least 2 samples should appear in both the training and validation subsets.

**Validates: Requirements 4.5**

---

### Property 12: Taxonomy label map consistency

*For any* `taxonomy.csv`, the label map loaded from it should be a bijection: each `primary_label` maps to a unique integer in `[0, 233]`, and the same label always maps to the same index regardless of how many times the map is constructed from the same file.

**Validates: Requirements 4.6**

---

### Property 13: Model output shape

*For any* batch of mel spectrogram tensors of shape `(B, 1, n_mels, T)`, the model's `forward()` should return a tensor of shape `(B, 234)`.

**Validates: Requirements 5.1**

---

### Property 14: Checkpoint round-trip

*For any* trained model, saving a checkpoint and loading it back should produce a model whose `state_dict` is identical to the original (all parameter tensors equal element-wise).

**Validates: Requirements 5.5, 8.1, 9.2**

---

### Property 15: Label smoothing target values

*For any* positive label target `1.0` and label smoothing parameter `epsilon`, the smoothed target used in loss computation should equal `1.0 - epsilon`.

**Validates: Requirements 6.2**

---

### Property 16: Best checkpoint selection

*For any* training run over `N` epochs, the checkpoint saved to disk should have a `val_roc_auc` value greater than or equal to the `val_roc_auc` of every individual epoch.

**Validates: Requirements 6.5**

---

### Property 17: Macro ROC-AUC computation

*For any* predictions matrix and label matrix, the macro ROC-AUC computed by the `Evaluator` should equal the arithmetic mean of per-class ROC-AUC scores computed only for columns that have at least one positive label.

**Validates: Requirements 7.1, 7.3, 7.4**

---

### Property 18: Inference probabilities in valid range

*For any* model and input segment batch, the inference engine's output probabilities should all lie in `[0.0, 1.0]` and each row should have exactly 234 values.

**Validates: Requirements 8.4, 8.5**

---

### Property 19: Submission completeness

*For any* `sample_submission.csv` with a set of row IDs `R`, the generated `submission.csv` should contain exactly the rows in `R` (no more, no fewer), with missing predictions filled with `0.0`.

**Validates: Requirements 8.8**

---

### Property 20: Ensemble averaging

*For any* set of `K` model checkpoints and any input segment, the ensemble prediction should equal the element-wise mean of the `K` individual checkpoint predictions.

**Validates: Requirements 8.9**

---

### Property 21: TTA averaging

*For any* segment and `N` augmented views, the TTA prediction should equal the element-wise mean of predictions across all `N` views.

**Validates: Requirements 8.6**

---

### Property 22: Missing path raises descriptive error

*For any* `InferenceEngine` configured with a non-existent checkpoint path, calling `verify_paths()` should raise a `FileNotFoundError` whose message contains the missing path string.

**Validates: Requirements 10.5**

---

### Property 23: Config round-trip

*For any* configuration dictionary written to a YAML file and loaded back via `load_config()`, the loaded dictionary should be equal to the original.

**Validates: Requirements 11.1, 11.2**

---

### Property 24: Missing config key raises error

*For any* configuration file missing a required key `K`, `load_config()` should raise a `KeyError` whose message contains the string `K`.

**Validates: Requirements 11.3**

---

### Property 25: Seed determinism

*For any* seed value `s`, calling `setup_seed(s)` twice in separate processes and then drawing the same sequence of random values from Python, NumPy, and PyTorch should produce identical results both times.

**Validates: Requirements 12.1, 12.2**

---

## Error Handling

| Scenario | Component | Behavior |
|----------|-----------|----------|
| Corrupt / unreadable `.ogg` file | `AudioPreprocessor.load()` | Log filename + exception, return `None`; caller skips the sample |
| Missing required config key | `load_config()` | Raise `KeyError("Missing required config key: {key}")` before any processing |
| Missing Kaggle Dataset mount path | `InferenceEngine.verify_paths()` | Raise `FileNotFoundError` with the missing path; called at notebook startup |
| Soundscape shorter than one segment | `SegmentExtractor.extract()` | Zero-pad the single segment to `segment_duration` seconds |
| Species not in taxonomy at inference | `InferenceEngine` | Log a warning; fill that species column with `0.0` in submission |
| Checkpoint missing required keys | `InferenceEngine.__init__()` | Raise `ValueError` listing the missing keys |
| NaN/Inf in model output | `InferenceEngine` | Replace with `0.0` and log a warning; do not crash |
| Validation set has no positives for a class | `Evaluator.evaluate()` | Silently exclude that class from macro average; include in per-class report as `None` |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required. They are complementary:
- **Unit tests** verify specific examples, integration points, and error conditions.
- **Property-based tests** verify universal correctness across randomly generated inputs.

### Property-Based Testing

**Library**: [`hypothesis`](https://hypothesis.readthedocs.io/) (Python)

Each property from the Correctness Properties section above is implemented as a single Hypothesis test. Tests are configured with `@settings(max_examples=100)` minimum.

Each test is tagged with a comment in the format:
```
# Feature: birdclef-2026-solution, Property {N}: {property_text}
```

Example:
```python
from hypothesis import given, settings, strategies as st
import numpy as np

# Feature: birdclef-2026-solution, Property 2: Waveform normalization bounds
@given(st.arrays(np.float32, shape=st.integers(1, 48000),
                 elements=st.floats(-1e6, 1e6, allow_nan=False)))
@settings(max_examples=100)
def test_normalization_bounds(waveform):
    preprocessor = AudioPreprocessor(sample_rate=32000)
    result = preprocessor.normalize(waveform)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)
    assert np.isclose(np.max(np.abs(result)), 1.0)
```

### Unit Tests

Unit tests focus on:
- **Specific examples**: loading a known `.ogg` file, checking exact segment boundaries
- **Error conditions**: corrupt file handling, missing config keys, missing mount paths
- **Integration**: `DatasetBuilder` → `BirdCLEFDataset` → `MelSpectrogramExtractor` pipeline
- **Submission format**: verifying `submission.csv` column names and row count match `sample_submission.csv`

Avoid writing unit tests that duplicate property tests. Unit tests should cover:
- `test_corrupt_file_returns_none` (Req 1.4)
- `test_short_soundscape_padding` (Req 2.3)
- `test_spec_augment_produces_zeros` (Req 3.4)
- `test_model_no_sigmoid_in_forward` (Req 5.6)
- `test_submission_notebook_cpu_only` (Req 10.4)
- `test_checkpoint_resume_same_metrics` (Req 6.7)

### Test File Layout

```
tests/
├── test_audio.py          # Properties 1-5, unit tests for audio loading
├── test_features.py       # Properties 6-8, unit tests for spectrogram
├── test_dataset.py        # Properties 9-12, unit tests for DatasetBuilder
├── test_model.py          # Properties 13-14, unit tests for model
├── test_train.py          # Properties 15-16, unit tests for Trainer
├── test_evaluate.py       # Property 17, unit tests for Evaluator
├── test_inference.py      # Properties 18-22, unit tests for InferenceEngine
└── test_utils.py          # Properties 23-25, unit tests for config/seed
```

### CI Notes

- Property tests run with `pytest --hypothesis-seed=0` for reproducibility in CI.
- Integration tests that require actual `.ogg` files use a small synthetic fixture generated at test time with `soundfile`.
- No GPU required for any test; all tests run on CPU.
