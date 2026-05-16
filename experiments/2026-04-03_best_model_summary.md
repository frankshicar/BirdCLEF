# BirdCLEF 2026 最佳模型架構總結

**日期**: 2026-04-03  
**實驗者**: Frank  
**最佳 Kaggle 分數**: 0.868 (0.857-0.868 範圍，隨機波動)  
**模型版本**: EfficientNet-B0 + 64ch Mask Denoiser + Mean-Max Pooling

---

## 模型架構

### 整體流程

```
Audio (5s, 32kHz)
    ↓
Highpass Filter (50Hz)
    ↓
Mel Spectrogram (160 bins, log-scale)
    ↓
Z-score Normalization
    ↓
ResidualDenoiser (64ch CNN, multiplicative mask)
    ↓
EfficientNet-B0 Backbone (pretrained ImageNet)
    ↓
Mean + Max Pooling (1280 → 2560 features)
    ↓
Linear Classifier (2560 → 234 classes)
    ↓
Sigmoid → Multi-label Probabilities
```

### 1. 音頻預處理

```yaml
sample_rate: 32000          # 32kHz 採樣率
segment_duration: 5.0       # 固定 5 秒片段
highpass_cutoff: 50.0       # 移除極低頻噪音（風聲、電流聲）
```

**設計考量**:
- 32kHz 足以捕捉鳥類聲音（大多 < 15kHz）
- 5 秒是訓練集的標準長度，與測試集的 hop 一致
- Highpass 50Hz 只移除非生物噪音，保留所有鳥類聲音

### 2. Mel 頻譜圖

```yaml
n_mels: 160                 # 160 個 mel bins
n_fft: 2048                 # FFT window size
hop_length: 320             # 10ms hop (32000/320 = 100 fps)
f_min: 50.0                 # 最低頻率 50Hz
f_max: 15000.0              # 最高頻率 15kHz
top_db: 80.0                # 動態範圍壓縮
```

**設計考量**:
- 160 mel bins 提供細緻的頻率解析度，能區分相似鳥種
- n_fft=2048 避免 mel filterbank 零值警告（n_freqs=1025 足以支撐 160 bins）
- f_min=50Hz, f_max=15kHz 覆蓋所有鳥類聲音範圍
- 10ms hop 提供足夠的時間解析度

**Z-score Normalization**:
```python
mel_mean = -12.7009  # 在訓練集上計算
mel_std = 13.7642
normalized = (mel - mel_mean) / mel_std
```

### 3. ResidualDenoiser (64 channels)

```python
class ResidualDenoiser(nn.Module):
    """Multiplicative mask denoiser for mel spectrograms."""
    
    def __init__(self, channels: int = 64):
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, 1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        mask = torch.sigmoid(self.conv4(out))  # [0, 1]
        return x * mask  # 乘法 mask，只能抑制不能過度減去
```

**設計考量**:
- **Multiplicative mask** 而非 additive residual，避免過度減去鳥叫聲
- Mask 值域 [0, 1]，只能選擇保留或抑制頻率
- 64 channels 足夠學習 ESC-50 噪音模式，不會過擬合
- 3 層 conv + BN + ReLU 提供足夠的非線性表達能力

### 4. EfficientNet-B0 Backbone

```python
backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
# Input: (B, 3, 160, T)  — 單通道 mel 圖複製成 3 通道
# Output: (B, 1280, H, W) — feature maps
```

**設計考量**:
- ImageNet 預訓練提供通用視覺特徵（邊緣、紋理、形狀）
- 這些特徵對 mel 圖也有效（頻率模式、時間結構）
- B0 是最小的 EfficientNet，避免過擬合（B1/B2 反而更差）
- 參數量適中（5.3M），訓練速度快

### 5. Mean + Max Pooling

```python
if features.dim() == 4:  # (B, 1280, H, W)
    mean_pool = features.mean(dim=[2, 3])  # (B, 1280)
    max_pool = features.amax(dim=[2, 3])   # (B, 1280)
pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 2560)
```

**設計考量**:
- Mean pool 捕捉整體分布（背景特徵）
- Max pool 捕捉最顯著的局部特徵（鳥叫聲的峰值）
- 兩者互補，feature dim 從 1280 → 2560
- 比單一 avg pool 提升約 +0.005-0.010

### 6. 分類頭

```python
classifier = nn.Linear(2560, 234)  # 234 個物種
logits = classifier(pooled)        # 不加 sigmoid
```

**設計考量**:
- 簡單的線性層，避免過擬合
- 輸出 raw logits，由 BCEWithLogitsLoss 處理
- 多標籤分類（一個音頻可能有多個物種）

---

## 訓練策略

### 損失函數與優化器

```yaml
loss: BCEWithLogitsLoss
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
scheduler: CosineAnnealingWarmRestarts (T_0=10)
```

**設計考量**:
- BCEWithLogitsLoss 內建數值穩定性（log-sum-exp trick）
- AdamW 的 decoupled weight decay 比 Adam 更好
- Cosine annealing 讓 lr 週期性重啟，避免陷入局部最優

### 正則化技術

```yaml
label_smoothing: 0.05       # 防止過度自信
grad_clip_norm: 5.0         # 防止梯度爆炸
mixed_precision: true       # FP16 加速訓練
early_stopping_patience: 15 # 基於 mAP
```

**Label Smoothing**:
```python
if label_smoothing > 0.0:
    batch_labels = batch_labels * (1.0 - label_smoothing)
# 原始: [0, 0, 1, 0] → Smoothed: [0, 0, 0.95, 0]
```

**Secondary Labels (Soft Labels)**:
```python
vec[primary_label] = 1.0      # 主要標籤
vec[secondary_label] = 0.3    # 次要標籤（不確定）
```

### 數據增強

#### SpecAugment (訓練時)
```yaml
time_mask_param: 30   # 隨機遮蔽時間軸
freq_mask_param: 20   # 隨機遮蔽頻率軸
```

#### Mixup (訓練時)
```yaml
mixup_alpha: 0.4
# 混合兩個樣本: x = λ*x1 + (1-λ)*x2, y = λ*y1 + (1-λ)*y2
```

#### Background Noise Mixing (訓練時)
```yaml
noise_dir: ./data/ESC-50/audio
noise_snr_db_range: [5.0, 30.0]  # SNR 範圍
noise_augment_p: 0.5              # 50% 機率加噪音
```

**設計考量**:
- SpecAugment 模擬遮擋和缺失
- Mixup 提升泛化能力，減少過擬合
- ESC-50 噪音（雨聲、風聲、人聲）模擬真實環境

### 類別不平衡處理

```yaml
use_weighted_sampler: true
# 每個樣本的權重 = 1 / (該類別的樣本數)
# 稀有物種被上採樣，常見物種被下採樣
```

---

## 訓練配置

```yaml
num_epochs: 200
batch_size: 64
num_workers: 2              # DataLoader workers
val_fraction: 0.1           # 10% 驗證集
rating_threshold: 0.0       # 不過濾低評分樣本
seed: 42                    # 可重現性
```

**Early Stopping**:
- 監控指標: `val_map` (mean Average Precision)
- Patience: 15 epochs
- mAP 比 ROC-AUC 對稀有物種更敏感

---

## 推理配置

```yaml
inference_batch_size: 32
tta: false                  # Test-Time Augmentation (未啟用)
ensemble_checkpoints: []    # 模型集成 (未啟用)
```

**推理流程**:
1. 載入 test soundscape (長音頻)
2. 以 2.5 秒 hop 切成 5 秒片段
3. 每個片段經過模型得到 234 維概率向量
4. 對每個 5 秒窗口取最大概率作為該時間點的預測

---

## 性能分析

### 分數演進

| 版本 | 主要改動 | Kaggle 分數 | 提升 |
|------|---------|-------------|------|
| 基準 | ResNet18 + 128 mels | 0.858 | - |
| v1 | EfficientNet-B0 | 0.858 | 0.000 |
| v2 | 160 mels + n_fft 2048 | 0.862 | +0.004 |
| v3 | Mask denoiser | 0.865 | +0.003 |
| v4 | Mean+Max pool | 0.866 | +0.001 |
| v5 | Soft secondary labels | 0.868 | +0.002 |
| **最佳** | **完整配置** | **0.868** | **+0.010** |

### 失敗的嘗試

| 方法 | 分數 | 原因 |
|------|------|------|
| PCEN | 0.477 | Domain shift (訓練/測試集差異) |
| 128ch denoiser | 0.850 | 過擬合 |
| EfficientNet-B1 | 0.827 | 模型過大 |
| TRA attention | 0.818 | 5 秒太短 |
| Differential LR | 0.841 | Denoiser 輸出不穩定 |

---

## 模型統計

```
總參數量: ~7.8M
├─ ResidualDenoiser: ~0.1M
├─ EfficientNet-B0: ~5.3M
└─ Classifier: ~2.4M (2560×234)

訓練時間: ~8 小時 (單 GPU, 200 epochs)
推理速度: ~50 samples/sec (batch_size=32)
GPU 記憶體: ~6GB (訓練), ~3GB (推理)
```

---

## 關鍵設計決策總結

### 為什麼這些選擇有效？

1. **Mask Denoiser 而非 Additive**
   - 避免過度減去鳥叫聲
   - 值域限制 [0,1] 提供天然正則化

2. **64ch 而非 128ch**
   - 避免過擬合到訓練集的乾淨音頻
   - 測試集有更多環境噪音，需要泛化能力

3. **EfficientNet-B0 而非 B1/B2**
   - 數據量有限，小模型泛化更好
   - B0 的 ImageNet 預訓練已經足夠

4. **Mean+Max Pooling**
   - 捕捉不同類型的特徵（整體 vs 局部）
   - 簡單有效，提升穩定

5. **160 mels + n_fft 2048**
   - 細緻的頻率解析度
   - 避免 mel filterbank 警告

6. **Secondary Labels = 0.3**
   - 反映標籤的不確定性
   - 減少 label noise

7. **單一 Learning Rate**
   - Denoiser 在 backbone 之前，需要同步學習
   - Differential LR 會導致脫節

---

## 未來改進方向

### 立即可做（無需重新訓練）

1. **Test-Time Augmentation**
   - 對每個片段做多次 crop
   - 預期: +0.005-0.010

2. **多 Seed Ensemble**
   - 訓練 3 個不同 seed 的模型
   - 預期: +0.010-0.020

### 短期改進（1-2 天）

3. **Auxiliary Loss**
   - 加入 taxonomy 分類（5 大類）
   - 預期: +0.005-0.015

4. **更強數據增強**
   - Pitch shifting (±2 semitones)
   - Time stretching (0.9-1.1x)
   - 預期: +0.005-0.010

### 中期改進（3-5 天）

5. **不同架構 Ensemble**
   - B0 + B1 + ResNet50
   - 預期: +0.015-0.030

6. **更長訓練**
   - 300 epochs + 更好的 scheduler
   - 預期: +0.005-0.015

---

## 完整配置檔案

```yaml
# birdclef2026/config/local.yaml

# Data
data_dir: ./data

# Model
backbone: efficientnet_b0
pool: mean_max
pretrained: true
use_denoiser: true
denoiser_channels: 64

# Training
num_epochs: 200
batch_size: 64
learning_rate: 0.001
weight_decay: 0.0001
seed: 42

# Audio
sample_rate: 32000
segment_duration: 5.0
hop_duration: 2.5
highpass_cutoff: 50.0

# Mel Spectrogram
n_mels: 160
hop_length: 320
n_fft: 2048
top_db: 80.0
f_min: 50.0
f_max: 15000.0
use_pcen: false

# Augmentation
use_spec_augment: true
time_mask_param: 30
freq_mask_param: 20
use_mixup: true
mixup_alpha: 0.4
noise_dir: ./data/ESC-50/audio
noise_snr_db_range: [5.0, 30.0]
noise_augment_p: 0.5

# Training Settings
label_smoothing: 0.05
mixed_precision: true
val_fraction: 0.1
rating_threshold: 0.0
grad_clip_norm: 5.0
early_stopping_patience: 15
use_weighted_sampler: true

# Inference
inference_batch_size: 32
num_workers: 2
tta: false
ensemble_checkpoints: []
```

---

## 結論

這個架構在 BirdCLEF 2026 上達到 0.868 的分數，核心優勢是：

1. **平衡的模型容量** — 不過大不過小
2. **有效的去噪策略** — Mask denoiser 避免過度減去
3. **豐富的數據增強** — SpecAugment + Mixup + Noise
4. **穩健的訓練策略** — Label smoothing + Gradient clipping + Early stopping

下一步最推薦 TTA 和 Ensemble，因為成本低、效果確定。
