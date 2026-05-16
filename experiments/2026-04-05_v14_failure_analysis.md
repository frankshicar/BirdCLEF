# v14 失敗分析 — 0.847

**日期**: 2026-04-05  
**版本**: v14 (U-Net + 3-crop TTA)  
**分數**: 0.847  
**基準**: 0.858-0.868  
**下降**: -0.011 ~ -0.021

---

## 問題

v14 的分數 0.847 比基準低，可能的原因：

### 可能性 1: U-Net 訓練不完整 ⭐⭐⭐

**症狀**:
- U-Net 訓練時遇到尺寸不匹配錯誤
- 修復後重新訓練，但可能沒訓練完

**檢查方法**:
```bash
# 查看 training_history.json
cat checkpoints/training_history.json | jq '.val_map[-10:]'

# 查看 checkpoint
python3 -c "import torch; ckpt = torch.load('checkpoints/best_checkpoint.pt', map_location='cpu'); print('epoch:', ckpt.get('epoch')); print('val_roc_auc:', ckpt.get('val_roc_auc'))"
```

**如果是這個問題**:
- U-Net 可能沒收斂
- 或者 U-Net 本身就比 ResidualDenoiser 差

---

### 可能性 2: TTA 實作有 Bug ⭐⭐

**症狀**:
- TTA 的 crop 邏輯有問題
- 或者 reshape 邏輯錯誤

**檢查方法**:
```python
# 測試 TTA 邏輯
segments = [(row_id, waveform) for ...]  # N 個 segments
specs = _make_tta_crops(segments)        # 應該是 3*N 個 specs

# 預測
all_probs = model(specs)                 # (3*N, 234)
reshaped = all_probs.reshape(N, 3, -1)   # (N, 3, 234)
final = reshaped.mean(axis=1)            # (N, 234)
```

**可能的 Bug**:
- `reshape(N, 3, -1)` 的順序不對
- 音頻太短時重複 3 次的邏輯有問題

---

### 可能性 3: Checkpoint 不匹配 ⭐⭐⭐⭐⭐

**症狀**:
- v14 用的 checkpoint 不是最好的
- 或者 checkpoint 是 U-Net 訓練到一半的

**檢查方法**:
```bash
# 查看 v14 zip 包裡的 checkpoint
unzip -l kaggle_submission/birdclef-2026-model-v14-tta.zip | grep checkpoint

# 對比本地最好的 checkpoint
ls -lh checkpoints/best_checkpoint.pt
```

**如果是這個問題**:
- 需要用之前 0.868 的 ResidualDenoiser checkpoint
- 而不是新訓練的 U-Net checkpoint

---

### 可能性 4: TTA 反而有害 ⭐

**症狀**:
- 3-crop TTA 引入了更多噪音
- 或者不同 crop 的預測差異太大

**理論**:
- 如果音頻片段本身就是 5 秒，crop 會重疊
- 重疊的部分可能導致預測不穩定

**檢查方法**:
- 用相同的 checkpoint 但關閉 TTA (`tta=False`)
- 看分數是否回升

---

## 診斷步驟

### 步驟 1: 確認 Checkpoint

```bash
# 查看當前 best_checkpoint.pt 的資訊
python3 -c "
import torch
ckpt = torch.load('checkpoints/best_checkpoint.pt', map_location='cpu')
print('Epoch:', ckpt.get('epoch'))
print('Val ROC-AUC:', ckpt.get('val_roc_auc'))
print('Config keys:', list(ckpt.get('config', {}).keys()))
print('Denoiser type:', ckpt.get('config', {}).get('denoiser_type'))
print('Denoiser channels:', ckpt.get('config', {}).get('denoiser_channels'))
"
```

### 步驟 2: 檢查訓練歷史

```bash
# 查看最後 20 個 epoch 的 val_map
cat checkpoints/training_history.json | jq '.val_map[-20:]'

# 查看最佳 val_map
cat checkpoints/training_history.json | jq '[.val_map[]] | max'
```

### 步驟 3: 測試不同組合

| 測試 | Checkpoint | TTA | 預期分數 |
|------|-----------|-----|----------|
| A | U-Net (新) | False | ? |
| B | U-Net (新) | True | 0.847 (已知) |
| C | ResidualDenoiser (舊 0.868) | False | 0.858-0.868 |
| D | ResidualDenoiser (舊 0.868) | True | ? |

---

## 最可能的原因

根據經驗，我認為是 **可能性 3: Checkpoint 不匹配**

原因：
1. U-Net 訓練時遇到錯誤，修復後重新訓練
2. 新的 checkpoint 可能沒訓練完整
3. 或者 U-Net 本身就比 ResidualDenoiser 差

---

## 立即行動

### 方案 A: 回退到 ResidualDenoiser + TTA

1. 找到之前 0.868 的 checkpoint
2. 確認它用的是 ResidualDenoiser
3. 用這個 checkpoint + TTA 重新打包

```bash
# 找到舊的 checkpoint
ls -lht checkpoints/*.pt

# 或者從 v13 提取
unzip -p kaggle_submission/birdclef-2026-model-v13.zip checkpoint.pt > checkpoints/v13_checkpoint.pt
```

### 方案 B: 關閉 TTA，只用 U-Net

測試 U-Net 本身的效果：
```python
engine = InferenceEngine(
    checkpoint_path="checkpoint.pt",
    tta=False,  # 關閉 TTA
)
```

如果分數回到 0.858+，說明 TTA 有問題。  
如果分數還是 0.847，說明 U-Net 本身有問題。

---

## 建議

**立即做**:
1. 檢查 `checkpoints/best_checkpoint.pt` 的訓練狀態
2. 如果 U-Net 沒訓練完，等它訓練完再測試
3. 如果 U-Net 已經訓練完但效果差，回退到 ResidualDenoiser

**不要做**:
- 不要繼續用 0.847 的 checkpoint
- 不要在不確定問題前繼續實驗

---

## 下一步

### 如果 U-Net 訓練中

等訓練完成，然後：
```bash
# 重新打包
cp checkpoints/best_checkpoint.pt kaggle_submission/v13_temp/checkpoint.pt
cd kaggle_submission/v13_temp && python3 -c "import shutil; shutil.make_archive('../birdclef-2026-model-v15-unet-tta', 'zip', '.')"
```

### 如果 U-Net 已完成但效果差

回退到 ResidualDenoiser + TTA：
```bash
# 從 v13 提取舊 checkpoint
unzip -p kaggle_submission/birdclef-2026-model-v13.zip checkpoint.pt > kaggle_submission/v13_temp/checkpoint.pt

# 確認是 ResidualDenoiser
python3 -c "
import torch
ckpt = torch.load('kaggle_submission/v13_temp/checkpoint.pt', map_location='cpu')
print('Denoiser type:', ckpt.get('config', {}).get('denoiser_type', 'residual'))
"

# 重新打包（已經有 TTA 代碼）
cd kaggle_submission/v13_temp && python3 -c "import shutil; shutil.make_archive('../birdclef-2026-model-v15-residual-tta', 'zip', '.')"
```

---

## 教訓

1. **不要同時改兩個東西**
   - v14 同時改了 denoiser (U-Net) 和 inference (TTA)
   - 無法確定是哪個導致分數下降

2. **先測試單一變更**
   - 應該先測試 U-Net (不加 TTA)
   - 確認有效後再加 TTA

3. **保留已知有效的 checkpoint**
   - 0.868 的 checkpoint 應該備份
   - 不要被新的 checkpoint 覆蓋

---

## 結論

**最可能的原因**: U-Net checkpoint 沒訓練完整或效果本身就差

**立即行動**: 
1. 檢查 U-Net 訓練狀態
2. 如果沒訓練完，等完成
3. 如果已完成但效果差，回退到 ResidualDenoiser + TTA

**目標**: 找到 0.868 的 checkpoint + TTA，預期 0.873-0.883
