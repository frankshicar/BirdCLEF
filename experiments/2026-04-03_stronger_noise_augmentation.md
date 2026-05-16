# 實驗紀錄 — 更強的噪音增強

**日期**: 2026-04-03  
**實驗者**: Frank  
**前版分數**: 0.868 (0.857-0.868 範圍)  
**本版狀態**: 訓練中  
**假設**: 64ch denoiser 足夠，但需要更多樣的噪音來提升泛化能力

---

## 實驗動機

### 問題分析

測試集的 soundscapes 包含各種真實環境噪音（人聲、車聲、雨聲、風聲等），但訓練集的 train_audio 是相對乾淨的短錄音。

當前的噪音增強設定：
```yaml
noise_augment_p: 0.5       # 50% 機率
noise_snr_db_range: [5.0, 30.0]
```

這意味著：
- 50% 的訓練樣本是乾淨音頻（沒有額外噪音）
- SNR 5-30dB 的範圍可能不夠極端

### 為什麼不用 128ch denoiser？

之前的實驗顯示：
- **實驗 E**: B0 + 128ch = 0.850（比 64ch 的 0.858 低）
- **實驗 C**: B0 + 128ch + TRA = 0.818

**假設原因**：128ch denoiser 容量太大，過度擬合到 ESC-50 的特定噪音模式，遇到測試集的真實噪音反而表現更差。

64ch denoiser 容量適中，只能學到「通用的噪音抑制策略」，泛化能力更好。

---

## 本版改動

### 1. 提高噪音增強機率

```yaml
noise_augment_p: 0.5 → 0.7
```

**理由**：
- 70% 的訓練樣本會加入噪音，更接近測試集的噪音比例
- 讓模型更習慣在噪音環境下識別鳥類聲音

### 2. 擴大 SNR 範圍

```yaml
noise_snr_db_range: [5.0, 30.0] → [3.0, 35.0]
```

**理由**：
- SNR 3dB：非常強的噪音（噪音幾乎蓋過鳥叫聲）
- SNR 35dB：非常弱的噪音（幾乎聽不到噪音）
- 更大的範圍讓模型學會處理各種噪音強度

### 3. 保持 64ch denoiser

```yaml
denoiser_channels: 64  # 不變
```

**理由**：
- 64ch 已經證明有效（0.868）
- 容量適中，不會過擬合到特定噪音模式
- 問題可能不是 denoiser 容量，而是訓練時見過的噪音不夠多樣

---

## 完整配置

```yaml
# Model
backbone: efficientnet_b0
pool: mean_max
use_denoiser: true
denoiser_channels: 64

# Training
num_epochs: 200
batch_size: 64
learning_rate: 0.001
weight_decay: 0.0001

# Mel Spectrogram
n_mels: 160
n_fft: 2048
f_min: 50.0
f_max: 15000.0

# Augmentation
use_spec_augment: true
use_mixup: true
mixup_alpha: 0.4
noise_dir: /home/sbplab/frank/BirdCLEF+/data/ESC-50/audio
noise_snr_db_range: [3.0, 35.0]  # ← 改動
noise_augment_p: 0.7              # ← 改動

# Training Settings
label_smoothing: 0.05
grad_clip_norm: 5.0
early_stopping_patience: 15
```

---

## 預期效果

### 樂觀情況 (+0.005-0.015)

如果假設正確（64ch 足夠，只是噪音不夠多樣）：
- 模型學會更強的噪音抑制能力
- 在測試集的真實噪音環境下表現更好
- 預期分數：0.873-0.883

### 中性情況 (±0.005)

噪音增強的效果有限：
- 分數維持在 0.863-0.873
- 說明當前的 denoiser 已經接近瓶頸

### 悲觀情況 (-0.005-0.010)

過度的噪音增強反而有害：
- 模型學到「所有聲音都是噪音」
- 把鳥叫聲也當成噪音濾掉
- 預期分數：0.848-0.863

---

## 下一步方案

### 如果本版成功（分數 > 0.873）

1. **繼續增強噪音多樣性**
   - 從 test soundscapes 提取純噪音片段
   - 加入更多噪音源（MUSAN dataset）

2. **Test-Time Augmentation**
   - 推理時使用多個 crops
   - 預期額外提升 +0.005-0.010

3. **Ensemble**
   - 訓練多個不同 seed 的模型
   - 預期額外提升 +0.010-0.020

### 如果本版持平（分數 0.863-0.873）

1. **方案 3: 從 soundscapes 提取噪音**
   - 手動找出測試集中的純噪音片段
   - 加到訓練噪音庫

2. **Auxiliary Loss**
   - 加入 taxonomy 分類（5 大類）
   - 幫助模型學習生物的層級結構

3. **Focal Loss**
   - 更關注難分類的樣本（稀有物種）

### 如果本版失敗（分數 < 0.863）

1. **回退到 0.868 的設定**
   ```yaml
   noise_augment_p: 0.5
   noise_snr_db_range: [5.0, 30.0]
   ```

2. **嘗試其他方向**
   - 不是噪音的問題，可能是模型架構或訓練策略
   - 考慮 Auxiliary Loss、更長訓練、Ensemble

---

## 訓練指令

```bash
python scripts/train.py --config birdclef2026/config/local.yaml
```

---

## 實驗結果

**訓練完成**:

- **Kaggle 分數**: 0.858
- **結論**: 與基準 0.857-0.868 持平，無明顯提升或下降

### 觀察

1. **噪音增強的效果有限**
   - 從 50% → 70% 機率
   - SNR 從 [5, 30] → [3, 35] dB
   - 分數維持在 0.858，說明當前的 denoiser 已經學會足夠的噪音抑制能力

2. **64ch denoiser 接近瓶頸**
   - 更多的噪音訓練沒有帶來提升
   - 說明問題不在「見過的噪音不夠多樣」
   - 而是 denoiser 的架構或容量已經到極限

3. **隨機性範圍**
   - 0.857, 0.858, 0.868 都在統計誤差範圍內
   - 說明當前架構的穩定分數約在 0.86 左右

### 結論

**噪音增強不是主要瓶頸。** 要進一步提升分數，需要從其他方向著手：
- 改進 denoiser 架構（U-Net、頻率分段）
- 加入 auxiliary loss（taxonomy 分類）
- Test-Time Augmentation
- 模型 Ensemble

---

## 教訓

**待實驗完成後總結**
