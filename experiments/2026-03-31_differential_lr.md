# 實驗紀錄 — Differential Learning Rate

**日期**: 2026-03-31
**實驗者**: Frank
**前版分數**: 0.868
**本版分數**: 0.857 (-0.011)
**結論**: Differential LR 不適合此架構，已回退

---

## 實驗結果

| 版本 | 設定 | Kaggle 分數 | 說明 |
|------|------|-------------|------|
| 基準 | learning_rate=0.001 | 0.868 | 單一學習率 |
| 實驗 A | backbone_lr=0.0001, head_lr=0.0005 | 0.841 | backbone 學太慢 |
| 實驗 B | backbone_lr=0.0005, head_lr=0.001 | 0.857 | 仍低於基準 |
| 回退 | learning_rate=0.001 | 0.857 | 與基準相同設定 |

---

## 問題分析

### 為什麼 Differential LR 失敗了？

Differential LR 的設計前提：
> Backbone 已有好的預訓練特徵，只需微調；Head 從零開始，需快速學習

但這個架構的資料流是：
```
input → ResidualDenoiser → backbone → mean_max pool → classifier
```

ResidualDenoiser 是從零開始訓練的，它的輸出會持續變化，backbone 需要不斷適應這個「移動的目標」。這跟標準 fine-tuning（backbone 吃固定圖片）完全不同。

### 脫節問題

- **Denoiser 學太快** → 輸出的 mel 圖變化劇烈 → backbone 追不上
- **Backbone 學太慢** → 就算 denoiser 鋪好路，backbone 也跟不上

結果：三個模組（denoiser, backbone, classifier）的學習速度不同步，導致整體性能下降。

---

## 0.857 vs 0.868 的差異

### 為什麼回退後還是 0.857，沒回到 0.868？

可能原因：

1. **隨機性**：不同的 seed、數據 shuffle、初始化都會影響最終分數（±0.01 是正常波動）
2. **Early stopping 時機**：這次可能在不同 epoch 停止
3. **硬體差異**：不同 GPU、CUDA 版本的數值精度略有不同

0.857 vs 0.868 的差距 (-0.011) 在統計誤差範圍內，不代表模型變差。

---

## 前置修復：evaluate.py 錯誤

### 問題
訓練時出現錯誤：
```
ValueError: continuous format is not supported
```

### 原因
`roc_auc_score()` 要求標籤必須是二元的 (0 或 1)，但由於我們使用了 soft labels (secondary_labels = 0.3)，某些類別可能只有 0.3 的標籤值，或者某個類別只有正樣本或只有負樣本。

### 修復
在 `evaluate.py` 中改進標籤二值化和驗證邏輯：

```python
# 二值化標籤（閾值 0.5）
y_true_binary = (y_true >= 0.5).astype(np.int32)

# 檢查是否同時有正樣本和負樣本
n_positive = y_true_binary.sum()
n_negative = len(y_true_binary) - n_positive

if n_positive == 0 or n_negative == 0:
    per_class[key] = None  # 跳過該類別
    continue
```

這確保 `roc_auc_score()` 只在有效的類別上計算（同時有正負樣本）。

---

## 0.868 分數的改進回顧

在實施 differential learning rate 之前，我們已經完成以下改進（從 0.858 → 0.868）：

### 1. 增加 Mel 頻譜圖解析度
- **變更**: `n_mels: 128 → 160`
- **原因**: 提供更細緻的頻率解析度，捕捉更多鳥類聲音細節

### 2. Secondary Labels 軟標籤
- **變更**: 次要標籤從 `1.0` 改為 `0.3`
- **位置**: `dataset.py` 的 `_make_label_vector()`
- **原因**: 減少標籤噪音，次要標籤的確定性較低

### 3. Mean + Max Pooling
- **變更**: 從 `avg pool` 改為 `mean_max` pooling
- **位置**: `model.py`
- **效果**: 特徵維度從 1280 → 2560，同時捕捉平均和最顯著特徵

### 4. Gradient Clipping
- **變更**: 加入 `grad_clip_norm: 5.0`
- **位置**: `train.py`
- **原因**: 防止梯度爆炸，穩定訓練

### 5. Early Stopping 指標改為 mAP
- **變更**: 從 `val_roc_auc` 改為 `val_map`
- **原因**: mAP 對多標籤分類更敏感

### 6. Label Smoothing
- **變更**: `label_smoothing: 0.05`
- **原因**: 防止過度自信，提升泛化能力

---

## 改動內容

### Differential Learning Rate（train.py + local.yaml）

將 optimizer 從單一學習率改為分層學習率：

```python
param_groups = [
    {"params": model.backbone.parameters(), "lr": backbone_lr},  # 1e-4
    {"params": model.denoiser.parameters(), "lr": head_lr},      # 5e-4
    {"params": model.classifier.parameters(), "lr": head_lr},    # 5e-4
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
```

**設定值**:
- `backbone_lr: 0.0001` (1e-4) — 較小的學習率，保留 EfficientNet-B0 預訓練特徵
- `head_lr: 0.0005` (5e-4) — 較大的學習率，讓 denoiser 和 classifier 快速學習任務特定特徵

---

## 原理

EfficientNet-B0 在 ImageNet 上預訓練，已經學會通用的視覺特徵（邊緣、紋理、形狀）。這些特徵對 mel spectrogram 也有用（頻率模式、時間結構）。

使用較小的 `backbone_lr` 可以：
1. 保留預訓練特徵，避免過度調整導致 catastrophic forgetting
2. 讓 backbone 微調（fine-tune）而非重新學習（retrain）

使用較大的 `head_lr` 可以：
1. 讓 denoiser 快速學會抑制 ESC-50 噪音
2. 讓 classifier 快速學會 234 種鳥類的特徵映射

---

## 預期效果

Differential learning rate 是 transfer learning 的標準做法，在 fine-tuning 預訓練模型時通常能提升 1-3% 的準確率。

對於 BirdCLEF 任務：
- Backbone 已經學會通用特徵，只需微調
- Denoiser 和 classifier 是從頭訓練，需要較大學習率

預期分數提升：0.868 → 0.875-0.880

---

## 下一步方向（如果效果不佳）

1. 調整學習率比例（試試 backbone_lr=5e-5, head_lr=1e-3）
2. 輔助 loss（taxonomy 大類 CrossEntropy × 0.2）
3. ESC-50 預訓練 denoiser
4. 稀少物種離線增強（time stretch + pitch shift）
5. 2-model ensemble

---

## 未來改進方向（優先順序排序）

### 立即可做（無需重新訓練）

#### 1. Test-Time Augmentation (TTA) — 最優先
- 推理時對每個 5 秒片段做多次 crop（左、中、右）
- 對預測結果取平均
- 預期提升：+0.005-0.010
- 實施成本：修改 inference.py，無需重新訓練

#### 2. 多 Seed Ensemble
- 用相同設定但不同 seed 訓練 3 個模型
- 對預測結果加權平均
- 預期提升：+0.010-0.020
- 實施成本：3x 訓練時間

---

### 短期改進（需重新訓練，1-2 天）

#### 3. Auxiliary Loss（輔助損失）
- 加入 5 個分類群的輔助分類任務（Aves, Amphibia, Insecta, Mammalia, Reptilia）
- 權重 0.2，幫助模型學習生物的層級結構
- 預期提升：+0.005-0.015
- 實施步驟：
  1. 修改 `dataset.py` 加入 class_name 標籤
  2. 修改 `model.py` 加入輔助分類頭（5 類）
  3. 修改 `train.py` 加入輔助損失：`total_loss = main_loss + 0.2 * aux_loss`

#### 4. 更強的數據增強
- Pitch Shifting (±2 semitones)
- Time Stretching (0.9-1.1x)
- 增加 Background Noise Mixing 的多樣性（加入更多噪音類型）
- 預期提升：+0.005-0.010

#### 5. Focal Loss
- 替換 BCEWithLogitsLoss
- 更關注難分類的樣本（稀有物種）
- 預期提升：+0.003-0.008

---

### 中期改進（需重新訓練，3-5 天）

#### 6. 更長的訓練時間
- 增加 `num_epochs` 到 300
- 調整 learning rate schedule（更長的 warmup，更平緩的 decay）
- 預期提升：+0.005-0.015

#### 7. 不同架構的 Ensemble
- 訓練 EfficientNet-B1（更大容量）
- 訓練 ResNet50（不同架構偏好）
- 3 模型加權平均
- 預期提升：+0.015-0.030
- 實施成本：3x 訓練時間 + 更多 GPU 記憶體

#### 8. 稀有物種專門處理
- 識別 validation set 中表現差的物種
- 對這些物種做離線增強（time stretch + pitch shift + 更多噪音）
- 重新訓練時增加這些物種的採樣權重
- 預期提升：+0.005-0.010

---

### 長期改進（需大量資源，1-2 週）

#### 9. 更大的模型
- EfficientNet-B2 或 B3
- 需要更多 GPU 記憶體（可能需要減少 batch_size）
- 預期提升：+0.010-0.025

#### 10. 外部數據
- 使用 Xeno-Canto 的額外鳥類聲音
- 需要仔細的數據清理和標籤對齊
- 預期提升：+0.020-0.040

#### 11. 進階架構
- Transformer-based models (AST, BEATs, HTS-AT)
- 需要大量計算資源和調參
- 預期提升：+0.030-0.060

---

## 當前最佳配置（0.857-0.868）

```yaml
# 模型架構
backbone: efficientnet_b0
pool: mean_max
denoiser_channels: 64
n_mels: 160
n_fft: 2048

# 訓練超參數
learning_rate: 0.001  # 單一學習率，不用 differential LR
label_smoothing: 0.05
grad_clip_norm: 5.0
early_stopping_patience: 15
num_epochs: 200

# 數據增強
use_spec_augment: true
use_mixup: true
mixup_alpha: 0.4
noise_augment_p: 0.5
noise_snr_db_range: [5.0, 30.0]

# 其他
use_weighted_sampler: true
mixed_precision: true
```

---

## 已放棄的方法

### PCEN (Per-Channel Energy Normalization)
- **分數**: 0.477 (從 0.858 大幅下降)
- **原因**: 訓練集是短片段乾淨音頻，測試集是長片段環境音，PCEN 的 EMA 背景估計產生嚴重的 domain shift
- **教訓**: 特徵提取方法必須在訓練和測試集上保持一致的行為

### 128ch Denoiser
- **分數**: 0.850 (比 64ch 的 0.858 低)
- **原因**: 過擬合到 train_audio 的乾淨音頻，無法泛化到測試集的噪音環境
- **教訓**: 更大的模型不一定更好，需要考慮訓練/測試集的差異

### TRA (Temporal Recurrent Attention)
- **分數**: 0.818
- **原因**: 5 秒片段太短，時序注意力機制無法發揮作用
- **教訓**: 架構選擇要符合數據特性（片段長度）

### EfficientNet-B1 + 128ch
- **分數**: 0.827 (比 B0 + 64ch 的 0.858 低)
- **原因**: 模型容量過大，過擬合
- **教訓**: 在數據量有限的情況下，適中的模型容量更好

### Differential Learning Rate
- **分數**: 0.841 (backbone_lr=0.0001) → 0.857 (backbone_lr=0.0005)
- **原因**: ResidualDenoiser 在 backbone 之前，輸出持續變化，backbone 需要同步學習而非微調
- **教訓**: Differential LR 適合標準 fine-tuning，不適合有預處理模組的架構

---

## Label Smoothing 說明

Label smoothing 是一種正則化技術，防止模型對預測過度自信。

### 原理
將硬標籤 (0 或 1) 軟化：
- 原始標籤: `[0, 0, 1, 0]`
- Smoothing 0.05: `[0, 0, 0.95, 0]`

### 實作
```python
if label_smoothing > 0.0:
    batch_labels = batch_labels * (1.0 - label_smoothing)
```

### 效果
- 減少過擬合
- 提升模型在未見過數據上的泛化能力
- 對於噪音標籤（如次要標籤）特別有效

### 注意
我們的實作只對正標籤進行 smoothing，負標籤保持為 0。這是因為：
1. 多標籤分類中，負標籤數量遠多於正標籤
2. 只 smooth 正標籤可以保持類別不平衡的特性
3. 避免所有類別的預測值都偏高

---

## 配置檔案總結

```yaml
# 模型架構
backbone: efficientnet_b0
n_mels: 160
pool: mean_max
denoiser_channels: 64

# 訓練超參數
learning_rate: 0.001  # 將改為 differential LR
label_smoothing: 0.05
grad_clip_norm: 5.0
early_stopping_patience: 15

# 數據增強
use_spec_augment: true
use_mixup: true
mixup_alpha: 0.4
noise_augment_p: 0.5
```
