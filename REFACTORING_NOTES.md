# BirdCLEF+ Refactoring Notes

## What changed

- Checkpoint selection now defaults to `macro_roc_auc`, matching the Kaggle metric.
- Dataset splitting supports leakage-safe group holdouts:
  - `stratified`: legacy per-class random split.
  - `group`: all samples are grouped by `group_id`.
  - `soundscape_group`: focal clips stay stratified while soundscape segments are grouped.
- Soundscape groups can be based on `filename`, `site`, `site_date`, or `hour`.
- Weighted sampling can up-weight domains with `source_weights`, for example soundscape fine-tuning.
- Inference TTA no longer uses SpecAugment masks. If enabled, it uses deterministic time-roll views only.
- Added `scripts/benchmark_inference.py` for CPU runtime checks before ensembling.

## Recommended workflow

1. Train the base model with `monitor_metric: macro_roc_auc`.
2. Use `split_strategy: soundscape_group` for deployment-like validation.
3. Fine-tune from the best checkpoint with lower learning rates and `source_weights.soundscape > 1`.
4. Benchmark CPU inference before adding extra seeds or checkpoints.

## Important defaults

```yaml
monitor_metric: macro_roc_auc
split_strategy: soundscape_group
soundscape_group_by: filename
source_weights:
  train_audio: 1.0
  soundscape: 2.0
tta: false
tta_views: 3
```
