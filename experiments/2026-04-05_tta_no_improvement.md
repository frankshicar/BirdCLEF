# TTA 無效分析 — 0.858

**日期**: 2026-04-05  
**版本**: v15 (ResidualDenoiser + 3-crop TTA)  
**分數**: 0.858  
**基準**: 0.858 (v12, 無 TTA)  
**提升**: 0.000

---

## 結論

**3-crop TTA 對這個任務無效。**

---

## 為什麼 TTA 沒有提升？

### 原因 1: 測試集的音頻已經是固定切片

BirdCLEF 的測試集是這樣的：
```
soundscape.ogg (長音頻)
  ↓ 官方已經切好
row_id: soundscape_0, soundscape_5, soundscape_10, ...
每個 row_id 對應固定的 5 秒片段
```

我們的 TTA 是對這些**已經切好的 5 秒片段**再做 crop：
```
5 秒片段 → 左 crop (前 5s) = 原片段
5 秒片段 → 中 crop (中間 5s) = 原片段
5 秒片段 → 右 crop (後 5s) = 原片段
```

**結果：3 次 crop 其實是同一個片段，取平均沒有意義。**

---

### 原因 2: 我們的 SegmentExtractor 已經是固定切片

看 `inference.py` 的邏輯：
```python
self._seg_extractor = SegmentExtractor(
    segment_duration=5.0,
    hop_duration=5.0,  # 沒有重疊！
)
```

`hop_duration=5.0` 代表每 5 秒切一次，沒有重疊。所以：
- segment 1: 0-5 秒
- segment 2: 5-10 秒
- segment 3: 10-15 秒

每個 segment 剛好 5 秒，TTA 的 3 次 crop 都是同一個片段。

---

### 原因 3: TTA 適用於「有冗餘」的情況

TTA 有效的場景：
```
原始音頻: 10 秒
需要預測: 5 秒片段

TTA:
- 左 crop: 0-5 秒
- 中 crop: 2.5-7.5 秒
- 右 crop: 5-10 秒

3 次 crop 包含不同的內容 → 取平均有意義
```

但我們的情況：
```
原始音頻: 已經切好的 5 秒
需要預測: 5 秒片段

TTA:
- 左 crop: 0-5 秒 = 原片段
- 中 crop: 0-5 秒 = 原片段
- 右 crop: 0-5 秒 = 原片段

3 次 crop 是同一個內容 → 取平均 = 原預測
```

---

## 驗證

讓我檢查 `_make_tta_crops` 的邏輯：

```python
def _make_tta_crops(self, segments):
    for row_id, waveform in segments:
        if len(waveform) <= target_length:  # 5 秒
            # 音頻太短，只做一次
            spec = self._mel_extractor(waveform)
            all_specs.extend([spec, spec, spec])  # 重複 3 次
```

如果 waveform 剛好 5 秒（大部分情況），就會重複 3 次同一個 spec。

即使 waveform > 5 秒，crop 的邏輯也有問題：
```python
# Crop 1: 左對齊
crop1 = waveform[:target_length]  # 0-5s

# Crop 2: 中心對齊
start = (len(waveform) - target_length) // 2
crop2 = waveform[start:start + target_length]  # 中間 5s

# Crop 3: 右對齊
crop3 = waveform[-target_length:]  # 最後 5s
```

但如果 waveform 剛好 5 秒，這 3 個 crop 都是同一個片段。

---

## 教訓

1. **TTA 不適用於固定切片的任務**
   - BirdCLEF 的測試集已經是固定的 5 秒片段
   - 沒有冗餘可以利用

2. **TTA 適用於圖像分類**
   - 圖像可以 crop、翻轉、旋轉
   - 每次變換都產生不同的視角

3. **音頻 TTA 需要不同的策略**
   - 音量調整（±3dB）
   - 輕微的 pitch shifting
   - 但這些可能引入失真

---

## 下一步方向

既然 TTA 無效，我們需要從其他方向提升：

### 方向 1: 模型 Ensemble ⭐⭐⭐⭐⭐

**最推薦**，效果最穩定：

```python
# 訓練 2-3 個不同的模型
model1 = train(seed=42)   # 0.858
model2 = train(seed=123)  # 0.855
model3 = train(seed=456)  # 0.860

# Ensemble
final_pred = (pred1 + pred2 + pred3) / 3
# 預期: 0.868-0.878
```

**優點**:
- 效果確定（+0.010-0.020）
- 降低隨機性
- 不同 seed 的模型有不同的偏好

**成本**:
- 需要訓練 2-3 次（6-9 天）
- 推理時間增加 2-3 倍（60-90 分鐘）

---

### 方向 2: Auxiliary Loss ⭐⭐⭐⭐

加入 taxonomy 分類（5 大類）：

```python
# 主要任務: 234 個物種分類
main_loss = BCEWithLogitsLoss(logits, labels)

# 輔助任務: 5 個大類分類 (Aves, Amphibia, Insecta, Mammalia, Reptilia)
aux_loss = CrossEntropyLoss(aux_logits, class_labels)

# 總損失
total_loss = main_loss + 0.2 * aux_loss
```

**優點**:
- 幫助模型學習層級結構
- 對稀有物種特別有效
- 理論基礎扎實

**成本**:
- 需要修改代碼（1 天）
- 需要重新訓練（3 天）

**預期提升**: +0.005-0.015

---

### 方向 3: 更強的數據增強 ⭐⭐⭐

```python
# Pitch shifting
waveform_shifted = pitch_shift(waveform, n_steps=2)

# Time stretching
waveform_stretched = time_stretch(waveform, rate=1.1)

# 更多噪音類型
# 從 test soundscapes 提取純噪音片段
```

**優點**:
- 提升泛化能力
- 對測試集的噪音更魯棒

**成本**:
- 需要實作增強邏輯（1 天）
- 需要重新訓練（3 天）

**預期提升**: +0.005-0.010

---

### 方向 4: 更大的模型 ⭐⭐

```yaml
backbone: efficientnet_b1  # 或 b2
```

**優點**:
- 更強的表達能力

**缺點**:
- 可能過擬合（B1 之前測試過是 0.827）
- 推理時間增加

**預期提升**: 不確定（可能負提升）

---

### 方向 5: Focal Loss ⭐⭐

替換 BCEWithLogitsLoss：

```python
loss = FocalLoss(alpha=0.25, gamma=2.0)
```

**優點**:
- 更關注難分類樣本
- 對稀有物種有幫助

**成本**:
- 需要實作 Focal Loss（1 小時）
- 需要重新訓練（3 天）

**預期提升**: +0.003-0.008

---

## 我的建議

### 短期（1 週內）

**方向 1: 多 Seed Ensemble**

1. 用當前配置訓練 2 個額外的模型（seed=123, 456）
2. 3 模型 ensemble
3. 預期分數: 0.868-0.878

**理由**:
- 效果最確定
- 不需要改代碼
- 只需要時間

---

### 中期（2 週內）

**方向 2: Auxiliary Loss**

如果 ensemble 有效，再加上 auxiliary loss：
- 3 模型 ensemble + auxiliary loss
- 預期分數: 0.880-0.890

---

### 長期（1 月內）

**外部數據 + Transformer**

如果要衝擊 0.900+：
- 加入 Xeno-Canto 數據
- 或使用 Transformer 架構（AST, BEATs）

---

## 當前狀態總結

| 方法 | 分數 | 提升 | 狀態 |
|------|------|------|------|
| 基準 (ResidualDenoiser) | 0.858 | - | ✓ |
| + 更強噪音增強 | 0.858 | 0.000 | ✗ 無效 |
| + U-Net | 0.847 | -0.011 | ✗ 失敗 |
| + TTA | 0.858 | 0.000 | ✗ 無效 |

**結論**: 當前架構已接近極限，需要 Ensemble 或 Auxiliary Loss 才能進一步提升。

---

## 立即行動

開始訓練第 2 個模型（seed=123）：

```yaml
# birdclef2026/config/local.yaml
seed: 123  # 改成 123

# 其他設定保持不變
backbone: efficientnet_b0
pool: mean_max
denoiser_type: residual
denoiser_channels: 64
```

```bash
python scripts/train.py --config birdclef2026/config/local.yaml
```

訓練完成後，用 2 模型 ensemble 測試效果。
