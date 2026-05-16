# 實驗紀錄

## Exp-002 — 資料與訓練策略改進

**日期：** 2026-03-21
**Kaggle Public Score：** 0.784 → 0.801 (+0.017)

### 背景

Exp-001 使用 efficientnet_b0 從頭訓練，val_roc_auc 達 0.9525，但 Kaggle public score 只有 0.784。
落差來自 train audio（乾淨單一鳥叫）與 test soundscape（野外錄音）之間的 domain gap，以及嚴重的長尾分布問題。

### 修改內容

| 項目 | 修改前 | 修改後 | 說明 |
|------|--------|--------|------|
| `pretrained` | `false` | `true` | 使用 ImageNet 預訓練權重，加速收斂並提升特徵品質 |
| `rating_threshold` | `3.0` | `1.0` | 多納入約 14,000 筆資料，稀少種類受益最多 |
| `hop_duration` | `5.0s` | `2.5s` | Inference 時使用 overlapping windows，每個 soundscape 產生更多 segments，提升召回率 |
| Sampler | `shuffle=True` | `WeightedRandomSampler` | 依 class 頻率反比加權，稀少種類被採樣頻率提升，緩解長尾問題 |
| 噪音增強 | 無 | `BackgroundNoiseMixer`（選用） | 混入背景噪音縮小 domain gap，需提供 `noise_dir`（如 ESC-50） |

### 資料集分析摘要

- 總錄音：35,549 筆 / 206 種 / 344.5 小時
- Rating = 0 的錄音佔 36.1%（12,849 筆），threshold 從 3.0 降至 1.0 後多納入約 14,000 筆
- 長尾嚴重：18 個種類 ≤ 5 筆，最少的 3 個種類只有 1 筆；最多的有 499 筆
- 音檔長度中位數 21 秒，7.3% 不足 5 秒（短於一個 segment）

### 結果

| 指標 | Exp-001 | Exp-002 |
|------|---------|---------|
| backbone | efficientnet_b0 (scratch) | efficientnet_b0 (pretrained) |
| val_roc_auc | 0.9525 | 0.9608 |
| Kaggle Public Score | 0.784 | 0.801 |

### 下一步方向

- 提供背景噪音檔案（ESC-50 或從 soundscape 截取）啟用 `BackgroundNoiseMixer`
- 嘗試更大的 backbone（efficientnet_b2 / b3）
- 使用 TTA（`tta: true`）進一步提升 inference 穩定性
- 考慮 focal loss 取代 BCEWithLogitsLoss，對長尾問題更有針對性
