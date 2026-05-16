# 實驗紀錄 — U-Net Denoiser

**日期**: 2026-04-05
**實驗者**: Frank  
**前版分數**: 0.858 (ResidualDenoiser 64ch)  
**本版狀態**: 訓練中  
**假設**: U-Net 的 skip connections 能更好地保留細節，提升降噪效果

---

## 實驗動機

### 為什麼嘗試 U-Net？

ResidualDenoiser (64ch) 已經接近瓶頸：
- 更強的噪音增強沒有提升 (0.858)
- 128ch 反而更差 (0.850)

U-Net 是音頻降噪的標準架構，優勢：
1. **Skip connections** — 保留高頻細節（鳥叫聲的細微特徵）
2. **多尺度特徵** — 同時處理局部和全局噪音
3. **Encoder-Decoder** — 更強的表達能力

### ResidualDenoiser vs U-Net

| 特性 | ResidualDenoiser | U-Net |
|------|------------------|-------|
| 架構 | 3 層 conv | Encoder-Decoder + Skip |
| 參數量 | ~0.1M (64ch) | ~0.5M (32ch base) |
| 感受野 | 小 (7×7) | 大 (多尺度) |
| 細節保留 | 一般 | 好 (skip connections) |
| 訓練難度 | 簡單 | 中等 |

---

## U-Net 架構

```
Input: (B, 1, 160, T)
    ↓
Encoder:
  e1: (B, 32, 160, T)   ← skip connection
  pool → e2: (B, 64, 80, T/2)   ← skip connection
  pool → e3: (B, 128, 40, T/4)  ← skip connection
  pool → bottleneck: (B, 256, 20, T/8)
    ↓
Decoder:
  up → d3: (B, 128, 40, T/4) + e3 → (B, 128, 40, T/4)
  up → d2: (B, 64, 80, T/2) + e2 → (B, 64, 80, T/2)
  up → d1: (B, 32, 160, T) + e1 → (B, 32, 160, T)
    ↓
Output: mask (B, 1, 160, T) in [0, 1]
Final: input * mask
```

### 關鍵設計

1. **Multiplicative mask** (跟 ResidualDenoiser 一樣)
   - 輸出 mask ∈ [0, 1]
   - 只能抑制，不能過度減去

2. **Skip connections**
   - Encoder 的特徵直接傳到 Decoder
   - 保留高頻細節（鳥叫聲的諧波）

3. **多尺度處理**
   - 低層：細節特徵（高頻鳥叫）
   - 高層：全局特徵（背景噪音模式）

4. **參數量控制**
   - base_channels=32 → 總參數 ~0.5M
   - 比 ResidualDenoiser 64ch (~0.1M) 大 5 倍
   - 但比 128ch (~0.2M) 的過擬合風險更可控

---

## 配置變更

```yaml
# 從 ResidualDenoiser 改為 U-Net
denoiser_type: unet        # 新增參數
denoiser_channels: 32      # U-Net 的 base_channels

# 其他設定保持不變
backbone: efficientnet_b0
pool: mean_max
n_mels: 160
learning_rate: 0.001
noise_augment_p: 0.5       # 改回原本的設定
noise_snr_db_range: [5.0, 30.0]
```

---

## 預期效果

### 樂觀情況 (+0.010-0.020)

U-Net 的 skip connections 成功保留鳥叫聲細節：
- 降噪更精準（移除噪音但保留鳥叫）
- 預期分數：0.868-0.878

### 中性情況 (+0.005-0.010)

U-Net 略優於 ResidualDenoiser：
- 預期分數：0.863-0.868

### 悲觀情況 (-0.005 或更差)

U-Net 過擬合或訓練不穩定：
- 參數量太大，過擬合到訓練集
- 預期分數：0.850-0.858

---

## 風險分析

### 風險 1: 過擬合
- U-Net 參數量是 ResidualDenoiser 的 5 倍
- 可能過度擬合 ESC-50 噪音模式
- **緩解**: 使用較小的 base_channels (32 而非 64)

### 風險 2: 訓練不穩定
- Encoder-Decoder 架構更難訓練
- Skip connections 可能導致梯度問題
- **緩解**: 使用 BatchNorm + Gradient clipping

### 風險 3: 推理時間
- U-Net 計算量更大
- 可能超過 90 分鐘 CPU 限制
- **緩解**: 先在本地測試推理速度

---

## 成功指標

### 訓練階段
- Val mAP > 0.95 (當前 ~0.94)
- Val loss 穩定下降
- 沒有明顯過擬合（train/val loss 差距 < 0.1）

### 推理階段
- Kaggle 分數 > 0.865
- 推理時間 < 80 分鐘（留 10 分鐘緩衝）

---

## 下一步方案

### 如果 U-Net 成功（分數 > 0.868）

1. **調整 base_channels**
   - 試試 base_channels=40 或 48
   - 找到最佳的參數量/性能平衡點

2. **加入 TTA**
   - U-Net + TTA 可能達到 0.875+

3. **Ensemble**
   - ResidualDenoiser + U-Net ensemble
   - 兩種架構互補

### 如果 U-Net 持平（分數 0.858-0.868）

1. **回到 ResidualDenoiser**
   - U-Net 沒有明顯優勢
   - 不值得增加複雜度

2. **嘗試其他方向**
   - Auxiliary Loss
   - 頻率分段 Denoiser
   - TTA

### 如果 U-Net 失敗（分數 < 0.858）

1. **立即回退**
   ```yaml
   denoiser_type: residual
   denoiser_channels: 64
   ```

2. **分析失敗原因**
   - 檢查 training_history.json
   - 是否過擬合？
   - 是否訓練不穩定？

3. **考慮更保守的架構**
   - 頻率分段 Denoiser (3×24ch)
   - 或直接做 TTA/Ensemble

---

## 訓練指令

```bash
python scripts/train.py --config birdclef2026/config/local.yaml
```

監控訓練：
```bash
# GPU
watch -n 1 nvidia-smi

# Training history
watch -n 10 'cat checkpoints/training_history.json | jq ".val_map[-5:]"'
```

---

## 實驗結果

**訓練完成後填寫**:

- Kaggle 分數: ___
- 最佳 epoch: ___
- Val mAP: ___
- Val ROC-AUC: ___
- 訓練時間: ___
- 推理時間: ___

**觀察**:
- 
- 
- 

**結論**:
- 
- 

---

## 參數量對比

| Denoiser | 參數量 | 分數 |
|----------|--------|------|
| ResidualDenoiser 64ch | ~0.1M | 0.858 |
| ResidualDenoiser 128ch | ~0.2M | 0.850 |
| U-Net 32ch | ~0.5M | ___ |

---

## 教訓

**待實驗完成後總結**
