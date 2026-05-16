# BirdCLEF 2026 模型技術報告

**日期：** 2026-03-21（最後更新：2026-04-15）
**Kaggle Public Score：** 0.801（ROC-AUC）
**Checkpoint：** `checkpoints/best_checkpoint.pt`（epoch 28，val_roc_auc 0.9608）

---

## 1. 問題定義

BirdCLEF 2026 是一個多標籤音訊分類任務，目標是從野外錄音（soundscape）中偵測 234 種鳥類及其他生物的叫聲。每段錄音被切成 5 秒片段，模型需輸出每個種類的存在機率。評估指標為 macro ROC-AUC。

---

## 2. 資料集

| 項目 | 數值 |
|------|------|
| 總錄音數 | 35,549 筆 |
| 種類數 | 206 種（234 個分類標籤） |
| 總時長 | 344.5 小時 |
| 主要類別 | Aves（鳥類）162 種，其餘為蛙類、昆蟲、哺乳類 |
| 音檔格式 | .ogg，32kHz，單聲道 |
| 音檔長度中位數 | 21 秒 |

**長尾分布：** 最多的種類有 499 筆，最少的只有 1 筆，18 個種類 ≤ 5 筆。

---

## 3. 資料前處理

### 3.1 音訊載入

- 使用 `soundfile` 讀取 `.ogg` 檔案
- 重採樣至 32,000 Hz（`torchaudio.functional.resample`）
- 轉換為單聲道 float32
- 高通濾波器：50 Hz cutoff（移除風聲、電流雜訊）
- 振幅正規化至 `[-1.0, 1.0]`
- 損壞檔案回傳 `None` 並記錄 warning

### 3.2 片段切割

- 固定長度：5 秒（160,000 samples）
- 訓練時：隨機裁切（random crop）
- 推論時：hop = 2.5 秒的 overlapping windows，確保每個 soundscape 有足夠覆蓋
- 短於 5 秒的片段以 zero-padding 補齊

### 3.3 Mel Spectrogram

| 參數 | 數值 |
|------|------|
| n_mels | 160 |
| n_fft | 2048 |
| hop_length | 320 |
| f_min | 50.0 Hz |
| f_max | 15000.0 Hz |
| top_db | 80.0 |
| 正規化 mean | 0.0 |
| 正規化 std | 1.0 |

pipeline：mel filterbank → AmplitudeToDB → `(x - mean) / std`
輸出 shape：`(1, 160, 500)`，dtype `float32`

---

## 4. 資料增強

### 訓練期間

| 增強方法 | 參數 | 說明 |
|----------|------|------|
| SpecAugment TimeMasking | `time_mask_param=30` | 隨機遮蔽時間軸 |
| SpecAugment FreqMasking | `freq_mask_param=20` | 隨機遮蔽頻率軸 |
| Mixup | `alpha=0.4` | Beta 分布混合兩個樣本及其標籤 |
| BackgroundNoiseMixer | `snr_db=[5, 30]`, `p=0.5` | 混入背景噪音（ESC-50 dataset） |

### 類別不平衡處理

使用 `WeightedRandomSampler`，每個樣本的採樣權重 = `1 / class_count`，讓稀少種類與常見種類有相同的期望採樣頻率。

---

## 5. 模型架構

```
Input: (B, 1, 160, 500)  ← 單通道 mel spectrogram
  ↓
ResidualDenoiser (64 channels)  ← 降噪模組
  ↓
Channel repeat: (B, 3, 160, 500)  ← 複製成 3 通道以符合 ImageNet backbone 輸入
  ↓
EfficientNet-B0 backbone (pretrained=True, num_classes=0)
  ↓ forward_features
Feature map: (B, 1280, H, W)
  ↓
Mean-Max Pooling → concat([GAP, GMP]) → (B, 2560)
  ↓
Linear(2560, 234)
  ↓
Output: (B, 234)  ← raw logits，無 sigmoid
```

| 元件 | 細節 |
|------|------|
| Denoiser | ResidualDenoiser (64 channels) |
| Backbone | EfficientNet-B0（timm） |
| 預訓練 | ImageNet-1k |
| Feature dim | 1280 |
| Pooling | Mean-Max Pooling (concat) → 2560 |
| Head | Linear(2560, 234) |
| 輸出 | Raw logits（推論時套 sigmoid） |

---

## 6. 訓練設定

| 超參數 | 數值 |
|--------|------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts |
| Loss | BCEWithLogitsLoss + label smoothing 0.05（僅正標籤：1.0 → 0.95） |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping | patience=15 |
| Gradient clipping | max_norm=5.0 |
| Mixed precision | fp16（torch.cuda.amp） |
| Rating threshold | 0.0（不過濾） |
| Val fraction | 10% stratified split |
| Seed | 42 |

**註：** 實際訓練在 epoch 28 達到最佳 val_roc_auc 0.9608 後觸發 early stopping。

---

## 7. 推論

- 裝置：CPU only
- Batch size：32
- Hop duration：2.5 秒（overlapping windows）
- 輸出：sigmoid(logits)，機率值 `[0, 1]`
- NaN/Inf 替換為 0.0
- 缺失的 row_id 填 0.0

---

## 8. 實驗結果

| 版本 | Backbone | Pretrained | rating_threshold | Sampler | val_roc_auc | Public Score |
|------|----------|------------|-----------------|---------|-------------|--------------|
| Exp-001 | EfficientNet-B0 | ✗ | 3.0 | shuffle | 0.9525 | 0.784 |
| Exp-002 | EfficientNet-B0 | ✓ | 1.0 | weighted | 0.9608 | 0.801 |

---

## 9. 已知問題與分析

### Val/Public Score 差距
- Validation ROC-AUC: 0.9608
- Public Score: 0.801
- **差距：0.16**

**可能原因：**
1. Train/Val split 可能在 segment 層級而非 recording 層級，造成資料洩漏
2. Validation 使用短片段（5秒），Public test 使用完整 soundscape
3. 模型過度擬合訓練集的音訊特性

### Label Smoothing 實作
當前實作只對正標籤做 smoothing：`y_smooth = y * 0.95`，負標籤保持 0.0。標準 label smoothing 應該讓負標籤也有小機率，可能影響模型校準。

### 種類數說明
- 資料集包含 206 個實際物種
- 模型輸出 234 個分類標籤（28 個額外標籤可能為亞種或變體）

---

## 10. 下一步

**優先級高：**
- 用 soundscape 做 validation，讓 val 指標更接近真實 public score
- 確認 train/val split 在 recording 層級，避免資料洩漏
- 修正 label smoothing 為雙向（正負標籤都 smooth）

**優化方向：**
- 嘗試更大 backbone（EfficientNet-B2/B3）
- 使用 TTA（`tta: true, tta_views: 4`）
- Focal loss 取代 BCE，對長尾更有針對性
- Per-class 分析，找出表現差的種類並針對性增強
