# 下一步改進計劃

**日期**: 2026-04-03  
**當前最佳分數**: 0.858-0.868 (穩定在 0.86 左右)  
**狀態**: 噪音增強已達瓶頸，需要新方向

---

## 實驗總結

### 已驗證有效的改進
- ✅ EfficientNet-B0 (比 ResNet18 好)
- ✅ 64ch Mask Denoiser (比 128ch 好)
- ✅ Mean+Max Pooling (比單一 avg pool 好)
- ✅ 160 mel bins + n_fft 2048
- ✅ Secondary labels = 0.3 (soft labels)
- ✅ Label smoothing 0.05
- ✅ Gradient clipping 5.0
- ✅ ESC-50 噪音增強

### 已驗證無效或有害的改進
- ❌ PCEN (0.477, domain shift)
- ❌ 128ch denoiser (0.850, 過擬合)
- ❌ EfficientNet-B1 (0.827, 模型過大)
- ❌ TRA attention (0.818, 片段太短)
- ❌ Differential LR (0.841, denoiser 輸出不穩定)
- ⚠️ 更強噪音增強 (0.858, 無明顯效果)

---

## 下一步方向（按優先順序）

### 🥇 優先級 1: 立即可做，成本低，效果確定

#### 1. Test-Time Augmentation (TTA)
**預期提升**: +0.005-0.010  
**實施時間**: 1 小時  
**風險**: 極低

**做法**:
```python
# 推理時對每個 5 秒片段做 3 次 crop
crops = [
    segment[0:5s],      # 左對齊
    segment[center:5s], # 中心對齊
    segment[-5s:]       # 右對齊
]
predictions = [model(crop) for crop in crops]
final_pred = mean(predictions)
```

**優點**:
- 無需重新訓練
- 直接修改 inference.py
- 效果穩定可靠

**缺點**:
- 推理時間增加 3 倍（但仍在 90 分鐘限制內）

---

#### 2. 多 Seed Ensemble (3 個模型)
**預期提升**: +0.010-0.020  
**實施時間**: 3 天（3 次訓練）  
**風險**: 低

**做法**:
```bash
# 訓練 3 個相同設定但不同 seed 的模型
python scripts/train.py --config local.yaml  # seed=42
# 修改 config: seed=123
python scripts/train.py --config local.yaml  # seed=123
# 修改 config: seed=456
python scripts/train.py --config local.yaml  # seed=456

# 推理時平均 3 個模型的預測
```

**優點**:
- 不需要改代碼
- 效果非常穩定
- 降低隨機性影響

**缺點**:
- 需要 3 倍訓練時間
- 推理時間增加 3 倍

---

### 🥈 優先級 2: 短期改進，需重新訓練，效果較確定

#### 3. Auxiliary Loss (Taxonomy 分類)
**預期提升**: +0.005-0.015  
**實施時間**: 1 天  
**風險**: 中

**做法**:
1. 修改 `dataset.py` 加入 class_name 標籤（5 類: Aves, Amphibia, Insecta, Mammalia, Reptilia）
2. 修改 `model.py` 加入輔助分類頭
3. 修改 `train.py` 加入輔助損失: `total_loss = main_loss + 0.2 * aux_loss`

**優點**:
- 幫助模型學習生物的層級結構
- 對稀有物種特別有效
- 理論基礎扎實

**缺點**:
- 需要修改多個檔案
- 需要調整 aux_loss 權重

---

#### 4. 頻率分段 Denoiser
**預期提升**: +0.005-0.015  
**實施時間**: 1 天  
**風險**: 中

**做法**:
```python
class FrequencyAwareDenoiser(nn.Module):
    def __init__(self):
        # 低頻 (0-50 bins): 環境噪音、人聲 → 激進過濾
        self.low_freq = ResidualDenoiser(24)
        # 中頻 (50-120 bins): 鳥類聲音 → 保守過濾
        self.mid_freq = ResidualDenoiser(24)
        # 高頻 (120-160 bins): 高頻鳥叫、昆蟲 → 中等過濾
        self.high_freq = ResidualDenoiser(24)
```

**優點**:
- 符合音頻的物理特性
- 可以針對不同頻段使用不同策略
- 總參數量 3×24ch = 72ch ≈ 64ch

**缺點**:
- 需要手動設定頻率分界點
- 可能需要多次實驗調整

---

### 🥉 優先級 3: 中期改進，成本較高

#### 5. 不同架構 Ensemble
**預期提升**: +0.015-0.030  
**實施時間**: 1 週  
**風險**: 中高

**做法**:
- 訓練 EfficientNet-B0 (已有)
- 訓練 EfficientNet-B1 (更大容量)
- 訓練 ResNet50 (不同架構偏好)
- 推理時加權平均

**優點**:
- 不同架構捕捉不同特徵
- Ensemble 效果通常很好

**缺點**:
- 需要 3 倍訓練時間
- 推理時間增加 3 倍
- B1 之前測試過效果不好 (0.827)

---

#### 6. 更長訓練 + 更好的 Scheduler
**預期提升**: +0.005-0.015  
**實施時間**: 3-5 天  
**風險**: 低

**做法**:
```yaml
num_epochs: 300  # 從 200 增加到 300
# 改用 OneCycleLR 或更長的 warmup
```

**優點**:
- 簡單直接
- 風險低

**缺點**:
- 訓練時間增加 50%
- 可能過擬合

---

### 🏅 優先級 4: 長期改進，需要大量資源

#### 7. 外部數據 (Xeno-Canto)
**預期提升**: +0.020-0.040  
**實施時間**: 2-4 週  
**風險**: 高

**做法**:
- 下載 Xeno-Canto 的鳥類聲音
- 清理和標籤對齊
- 加入訓練集

**優點**:
- 大幅增加數據量
- 提升稀有物種的表現

**缺點**:
- 需要大量數據清理工作
- 標籤質量不確定
- 可能引入噪音標籤

---

#### 8. Transformer 架構 (AST, BEATs)
**預期提升**: +0.030-0.060  
**實施時間**: 2-4 週  
**風險**: 高

**做法**:
- 使用 Audio Spectrogram Transformer (AST)
- 或 BEATs (Self-supervised audio pre-training)

**優點**:
- SOTA 架構
- 可能大幅提升

**缺點**:
- 需要大量 GPU 記憶體
- 訓練時間長
- 推理時間可能超過 90 分鐘限制

---

## 我的建議

### 立即執行（本週）

**方案 A: 保守穩健**
1. TTA (1 小時) → 預期 0.865-0.875
2. 如果有效，再做 3-seed ensemble (3 天) → 預期 0.875-0.885

**方案 B: 積極創新**
1. Auxiliary Loss (1 天) → 預期 0.865-0.880
2. 如果有效，再加 TTA → 預期 0.870-0.890

**方案 C: 架構改進**
1. 頻率分段 Denoiser (1 天) → 預期 0.865-0.880
2. 如果有效，再加 TTA → 預期 0.870-0.890

### 我的推薦順序

1. **TTA** (最優先，1 小時，無風險)
2. **Auxiliary Loss** (次優先，1 天，理論扎實)
3. **3-seed Ensemble** (如果前兩個有效，3 天)
4. **頻率分段 Denoiser** (如果想嘗試新架構，1 天)

### 目標分數

- 短期目標 (1 週): 0.875
- 中期目標 (2 週): 0.885
- 長期目標 (1 月): 0.900+

---

## 實施計劃

### 第 1 天: TTA
- 修改 `inference.py` 加入 TTA
- 重新生成 submission
- 提交 Kaggle

### 第 2-3 天: Auxiliary Loss
- 修改 `dataset.py`, `model.py`, `train.py`
- 訓練新模型
- 提交 Kaggle

### 第 4-6 天: 3-seed Ensemble (如果前面有效)
- 訓練 seed=123, 456 的模型
- Ensemble 推理
- 提交 Kaggle

---

## 決策點

**如果 TTA 提升 < 0.005**:
→ 說明推理端已經優化到極限，需要改進訓練端（Auxiliary Loss 或新架構）

**如果 Auxiliary Loss 提升 < 0.005**:
→ 說明當前特徵已經足夠，需要 Ensemble 或外部數據

**如果所有方法都無效**:
→ 當前架構已達極限，需要考慮 Transformer 或外部數據

---

## 你想先試哪一個？

我建議從 **TTA** 開始，因為：
- 1 小時就能完成
- 無需重新訓練
- 效果確定
- 可以立即驗證是否有效

要我幫你實作 TTA 嗎？
