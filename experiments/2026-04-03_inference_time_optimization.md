# 推理時間優化策略

**日期**: 2026-04-03  
**當前推理時間**: 30 分鐘  
**時間限制**: 90 分鐘  
**可用空間**: 60 分鐘 (200% 額外容量)

---

## 當前狀態

- **推理時間**: 30 分鐘
- **模型**: EfficientNet-B0 + U-Net/ResidualDenoiser
- **時間利用率**: 33%
- **剩餘空間**: 60 分鐘

這是巨大的優勢！我們可以用這些時間做：
1. Test-Time Augmentation (TTA)
2. 模型 Ensemble
3. 更大的模型

---

## 方案 1: Test-Time Augmentation (TTA) ⭐⭐⭐

**推理時間**: +20-30 分鐘 (總計 50-60 分鐘)  
**預期提升**: +0.005-0.015  
**實施難度**: 極低  
**推薦指數**: ⭐⭐⭐⭐⭐

### 做法

對每個 5 秒片段做多次預測，取平均：

```python
# 方法 1: 多次 crop (最簡單)
def predict_with_tta(model, waveform, n_crops=3):
    """
    對同一個音頻做多次 crop，取平均預測
    """
    predictions = []
    segment_length = int(5.0 * sample_rate)
    
    if len(waveform) > segment_length:
        # 左對齊
        crop1 = waveform[:segment_length]
        predictions.append(model(crop1))
        
        # 中心對齊
        start = (len(waveform) - segment_length) // 2
        crop2 = waveform[start:start + segment_length]
        predictions.append(model(crop2))
        
        # 右對齊
        crop3 = waveform[-segment_length:]
        predictions.append(model(crop3))
    else:
        # 音頻太短，只做一次
        predictions.append(model(waveform))
    
    return torch.stack(predictions).mean(dim=0)

# 方法 2: 加入輕微的數據增強
def predict_with_augmentation(model, waveform):
    """
    對音頻加入輕微變化，取平均預測
    """
    predictions = []
    
    # 原始音頻
    predictions.append(model(waveform))
    
    # 輕微音量調整 (+3dB)
    predictions.append(model(waveform * 1.4))
    
    # 輕微音量調整 (-3dB)
    predictions.append(model(waveform * 0.7))
    
    return torch.stack(predictions).mean(dim=0)
```

### 預期效果

- **3-crop TTA**: +0.005-0.010
- **5-crop TTA**: +0.008-0.015
- **音量增強**: +0.003-0.008

### 時間成本

- 3-crop: +20 分鐘 (總計 50 分鐘)
- 5-crop: +30 分鐘 (總計 60 分鐘)
- 音量增強: +15 分鐘 (總計 45 分鐘)

---

## 方案 2: 2-Model Ensemble ⭐⭐⭐

**推理時間**: +30 分鐘 (總計 60 分鐘)  
**預期提升**: +0.010-0.020  
**實施難度**: 中  
**推薦指數**: ⭐⭐⭐⭐

### 做法

訓練兩個不同的模型，推理時平均預測：

```python
# 模型 1: EfficientNet-B0 + U-Net (當前)
model1 = load_checkpoint("best_checkpoint_unet.pt")

# 模型 2: EfficientNet-B0 + ResidualDenoiser (之前的 0.868)
model2 = load_checkpoint("best_checkpoint_residual.pt")

# Ensemble 推理
pred1 = model1(input)
pred2 = model2(input)
final_pred = (pred1 + pred2) / 2
```

### 模型組合建議

| 組合 | 模型 1 | 模型 2 | 預期提升 |
|------|--------|--------|----------|
| A | U-Net | ResidualDenoiser | +0.010-0.015 |
| B | Seed=42 | Seed=123 | +0.008-0.012 |
| C | B0 | B1 | +0.015-0.025 |

### 時間成本

- 2 模型: +30 分鐘 (總計 60 分鐘)
- 3 模型: +60 分鐘 (總計 90 分鐘，剛好滿)

---

## 方案 3: TTA + Ensemble 組合 ⭐⭐

**推理時間**: 60-90 分鐘 (剛好用滿)  
**預期提升**: +0.015-0.030  
**實施難度**: 中  
**推薦指數**: ⭐⭐⭐⭐⭐

### 做法

```python
# 2 模型 + 3-crop TTA
model1 = load_checkpoint("checkpoint1.pt")
model2 = load_checkpoint("checkpoint2.pt")

predictions = []
for model in [model1, model2]:
    for crop in [left, center, right]:
        predictions.append(model(crop))

final_pred = torch.stack(predictions).mean(dim=0)
# 總共 2×3 = 6 次預測
```

### 時間分配

- 2 模型 × 3 crops = 6 次預測
- 每次預測 ~10 分鐘
- 總計 ~60 分鐘
- 加上載入時間 ~70-80 分鐘

---

## 方案 4: 更大的模型 ⭐

**推理時間**: +30-40 分鐘 (總計 60-70 分鐘)  
**預期提升**: +0.010-0.025  
**實施難度**: 高  
**推薦指數**: ⭐⭐⭐

### 做法

訓練更大的模型：

```yaml
# 選項 A: EfficientNet-B1
backbone: efficientnet_b1  # 比 B0 大 2 倍

# 選項 B: EfficientNet-B2
backbone: efficientnet_b2  # 比 B0 大 3 倍

# 選項 C: ResNet50
backbone: resnet50         # 不同架構
```

### 風險

- B1 之前測試過效果不好 (0.827)
- 更大的模型可能過擬合
- 需要重新訓練（3-5 天）

---

## 推薦方案（按優先順序）

### 🥇 方案 1: 先做 TTA (立即可做)

**步驟**:
1. 修改 `inference.py` 加入 3-crop TTA
2. 重新生成 submission
3. 提交 Kaggle

**預期**:
- 時間: 50 分鐘
- 分數: 0.865-0.875

**優點**:
- 無需重新訓練
- 1 小時內完成
- 效果確定

---

### 🥈 方案 2: TTA 有效後，加入 Ensemble

**步驟**:
1. 用不同 seed 訓練第 2 個模型（3 天）
2. 或者用 ResidualDenoiser + U-Net ensemble
3. 2 模型 + 3-crop TTA

**預期**:
- 時間: 70-80 分鐘
- 分數: 0.875-0.890

**優點**:
- 效果最穩定
- 充分利用時間限制

---

### 🥉 方案 3: 如果還有時間，訓練更大模型

**步驟**:
1. 訓練 EfficientNet-B2 或 ResNet50
2. 3 模型 ensemble + TTA

**預期**:
- 時間: 90 分鐘（用滿）
- 分數: 0.880-0.900

**風險**:
- 需要大量訓練時間
- 可能過擬合

---

## 立即行動計劃

### 今天（1 小時）

**實作 3-crop TTA**:

```python
# 在 inference.py 加入
def predict_soundscape_with_tta(self, audio_path, n_crops=3):
    """Predict with Test-Time Augmentation."""
    # 載入完整音頻
    waveform = load_audio(audio_path)
    
    # 切成 5 秒片段
    segments = split_into_segments(waveform, segment_duration=5.0, hop_duration=2.5)
    
    all_predictions = []
    for segment in segments:
        # 對每個片段做 n_crops 次預測
        crops = make_crops(segment, n_crops=n_crops)
        crop_preds = [self.model(crop) for crop in crops]
        avg_pred = torch.stack(crop_preds).mean(dim=0)
        all_predictions.append(avg_pred)
    
    return all_predictions
```

### 本週（3-5 天）

**訓練第 2 個模型**:
- 選項 A: 不同 seed (seed=123)
- 選項 B: ResidualDenoiser (如果 U-Net 更好)
- 選項 C: 不同架構 (ResNet50)

### 下週

**3 模型 ensemble + TTA**:
- 充分利用 90 分鐘限制
- 目標分數 0.880-0.900

---

## 時間預算表

| 方法 | 單次推理時間 | 總時間 | 剩餘時間 |
|------|-------------|--------|----------|
| 基準 (1 模型) | 30 分鐘 | 30 分鐘 | 60 分鐘 |
| + 3-crop TTA | 10 分鐘/crop | 50 分鐘 | 40 分鐘 |
| + 2nd 模型 | 30 分鐘 | 80 分鐘 | 10 分鐘 |
| + 5-crop TTA | 15 分鐘/crop | 90 分鐘 | 0 分鐘 |

---

## 我的建議

**立即做 TTA**，因為：
1. 1 小時內完成
2. 無需重新訓練
3. 預期 +0.005-0.015
4. 可以立即驗證效果
5. 如果有效，再考慮 ensemble

要我現在幫你實作 TTA 嗎？
