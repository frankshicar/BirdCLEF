# BirdCLEF 2026 模型架構報告

**日期：** 2026-03-25  
**模型版本：** ResNet18 + ResidualDenoiser  
**當前狀態：** 頻率濾波優化 + Enhanced Logging

---

## 🎯 整體架構概覽

```
Audio File (.ogg)
    ↓
AudioPreprocessor (Load + Normalize + Highpass Filter)
    ↓
SegmentExtractor (5s segments with 2.5s hop)
    ↓
MelSpectrogramExtractor (128 mel bins, 50-15000 Hz)
    ↓
[Optional] ResidualDenoiser (4-layer CNN noise reduction)
    ↓
ResNet18 Backbone (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Linear Classifier (234 classes)
    ↓
BCEWithLogitsLoss + Sigmoid → Probabilities
```

---

## 🔊 音訊前處理 (AudioPreprocessor)

### 基本處理
- **採樣率：** 32 kHz
- **格式轉換：** Multi-channel → Mono (平均)
- **正規化：** 振幅正規化到 [-1, 1]

### 頻率濾波 (新增)
- **高通濾波器：** 50 Hz cutoff
  - 移除極低頻噪音（風聲、電流聲）
  - 保留大型鳥類低頻叫聲（貓頭鷹、鴿子）
- **短音訊處理：** 跳過 < 0.05 秒的檔案
  - 從原本 0.1 秒降低，保留更多短鳥叫聲

### 分段處理 (SegmentExtractor)
- **片段長度：** 5 秒固定長度
- **跳躍間隔：** 2.5 秒 (50% overlap)
- **Row ID 格式：** `{filename}_{end_seconds}`
- **零填充：** 不足 5 秒的片段自動補零

---

## 📊 特徵提取 (MelSpectrogramExtractor)

### Mel Spectrogram 參數
- **Mel bins：** 128
- **FFT 大小：** 1024
- **Hop length：** 320 samples
- **頻率範圍：** 50-15000 Hz (優化後)
  - **f_min: 50 Hz** - 保留低頻鳥音
  - **f_max: 15000 Hz** - 保留高頻 harmonics
- **動態範圍：** 80 dB top_db

### 正規化
- **轉換：** Power → dB → Z-score normalization
- **統計量：** 從訓練集自動計算 mel_mean, mel_std
- **輸出形狀：** (1, 128, T) float32

---

## 🔇 降噪模組 (ResidualDenoiser)

### 架構設計
```python
Input: (B, 1, 128, T)
    ↓
Conv2d(1→64, 3×3) + BN + ReLU
    ↓
Conv2d(64→64, 3×3) + BN + ReLU  
    ↓
Conv2d(64→64, 3×3) + BN + ReLU
    ↓
Conv2d(64→1, 3×3) → predicted_noise
    ↓
Output: input - predicted_noise
```

### 降噪原理
- **殘差學習：** 學習預測噪音而非乾淨信號
- **端到端訓練：** 與分類器聯合優化
- **自適應性：** 自動學習什麼是「對分類有害的噪音」

### 目標噪音類型
- 人聲干擾
- 環境噪音（風聲、水聲）
- 多物種混合時的干擾
- 錄音設備噪音

---

## 🏗️ 主幹網路 (ResNet18)

### 架構選擇
- **模型：** ResNet18 (timm 實作)
- **預訓練：** ImageNet weights (`pretrained: true`)
- **輸入適配：** 1-channel → 3-channel (repeat)
- **特徵維度：** 512-d feature vector

### 池化策略
- **方法：** Global Average Pooling
- **替代選項：** Attention Pooling (可配置)

### 分類頭
- **結構：** Linear(512 → 234)
- **輸出：** Raw logits (無 sigmoid)

---

## 🎲 資料增強策略

### 1. SpecAugment (頻譜增強)
- **Time Masking：** 最多遮蔽 30 個時間幀
- **Frequency Masking：** 最多遮蔽 20 個頻率 bin
- **目的：** 提升對時頻變化的魯棒性

### 2. Mixup (樣本混合)
- **實作：** MixupCollator in DataLoader
- **混合係數：** λ ~ Beta(0.4, 0.4)
- **公式：** 
  - `mixed_spec = λ × spec1 + (1-λ) × spec2`
  - `mixed_label = λ × label1 + (1-λ) × label2`

### 3. Background Noise Mixing (可選)
- **SNR 範圍：** 5-30 dB
- **機率：** 50%
- **噪音來源：** 外部噪音檔案庫 (需配置 noise_dir)

---

## ⚖️ 類別不平衡處理

### 1. WeightedRandomSampler
- **目的：** 讓稀少物種被採樣的機率提升
- **權重計算：** 反比於類別頻率
- **效果：** 長尾物種獲得更多訓練機會

### 2. 資料來源多樣化
- **train_audio：** 35,549 個乾淨錄音
- **train_soundscapes：** 野外環境多物種混合 (去重後)
- **評分門檻：** rating_threshold = 0.0 (使用所有資料)

---

## 📈 訓練策略

### 優化器設定
- **優化器：** AdamW
- **學習率：** 0.001
- **權重衰減：** 0.0001
- **調度器：** CosineAnnealingWarmRestarts (T_0=10)

### 訓練技巧
- **混合精度：** `mixed_precision: true` (AMP)
- **標籤平滑：** `label_smoothing: 0.05`
- **Early Stopping：** 5 epochs patience
- **批次大小：** 64

### 驗證策略
- **分割方式：** Stratified split (10% validation)
- **評估指標：** Macro ROC-AUC
- **檢查點：** 保存最佳 val_roc_auc 模型

---

## 🎯 損失函數與評估

### 損失函數
- **主要損失：** `BCEWithLogitsLoss`
- **多標籤分類：** 每個物種獨立二元分類
- **數學形式：** 
  ```
  L = -Σ[y_i × log(σ(z_i)) + (1-y_i) × log(1-σ(z_i))]
  ```
  其中 σ(z) 是 sigmoid 函數

### 評估指標
- **主要指標：** Macro ROC-AUC
  - 每個類別計算 ROC-AUC，再取平均
  - 對長尾物種友善（不受類別不平衡影響）
- **輔助指標：** 
  - Training Loss
  - Validation Loss
  - Per-class ROC-AUC (詳細分析用)

---

## 🔧 當前配置摘要

```yaml
# 模型架構
backbone: resnet18
use_denoiser: true
denoiser_channels: 64

# 音訊處理
sample_rate: 32000
highpass_cutoff: 50.0
f_min: 50.0
f_max: 15000.0

# 訓練設定
batch_size: 64
learning_rate: 0.001
num_epochs: 200
early_stopping_patience: 5

# 增強策略
use_mixup: true
use_spec_augment: true
use_weighted_sampler: true
```

---

## 📊 效能表現

### 歷史成績
- **Baseline (EfficientNet-B0)：** 0.784
- **ResNet18 + 優化：** 0.829
- **頻率濾波調整後：** 待測試

### 最新改進
1. **頻率範圍優化：** 500-12000 Hz → 50-15000 Hz
2. **ResidualDenoiser：** 端到端降噪
3. **Enhanced Logging：** 詳細訓練監控
4. **Soundscape 資料：** 去重後加入訓練

---

## 🚀 下一步優化方向

### 短期目標
1. 解決 Kaggle Notebook timeout 問題
2. 測試新頻率設定的效果
3. 優化 ResidualDenoiser 效率

### 中期目標
1. 實作 U-Net denoiser
2. 加入人聲偵測 (VAD)
3. 多尺度特徵融合

### 長期目標
1. 自監督預訓練
2. 知識蒸餾
3. 模型集成策略

---

**報告生成時間：** 2026-03-25  
**配置檔案：** `birdclef2026/config/local.yaml`  
**模型檔案：** `birdclef2026/src/model.py`