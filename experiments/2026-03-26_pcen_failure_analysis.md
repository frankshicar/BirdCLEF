# PCEN 失敗分析與下一步改進

**日期**: 2026-03-26
**實驗者**: Frank
**結果**: Kaggle 分數從 0.858 → 0.477（PCEN 導致泛化失敗）

---

## 失敗原因分析

### 1. Train/Inference Feature Pipeline 不一致（已修復）

`inference.py` 的 `_DEFAULT_CONFIG` 缺少 `use_pcen`、`f_min`、`f_max`、`highpass_cutoff`，
導致推理時用 log-mel + f_min=500 + f_max=12000，而訓練用 PCEN + f_min=50 + f_max=15000。
這是第一個問題，已修復，但修復後分數仍然是 0.477。

### 2. PCEN 在短片段上的根本缺陷（主因）

診斷結果：模型 logits 全部為負數（-11.7 到 -3.1），所有類別機率低於 0.044。
這表示模型對所有類別都沒有信心，退化成全部預測負類的 trivial solution。

**根本原因**：

PCEN 的 EMA 背景估計需要足夠長的音訊才能收斂到穩定的背景估計值。
訓練資料（train_audio）是乾淨的短錄音，PCEN 在這上面的 EMA 幾乎沒有意義，
模型學到的是「PCEN 輸出接近 0 的 feature」對應某個物種。

但 test soundscapes 是長錄音切成的 5 秒片段，每個片段的 PCEN 輸出值域
和 train_audio 的 PCEN 輸出完全不同，造成嚴重的 domain shift。

```
train_audio PCEN:  短錄音，EMA 沒有穩定，輸出值域 [0, ~1]
soundscape PCEN:   長錄音切片，EMA 已收斂，背景被壓制，輸出值域 [0, ~4]
```

### 3. Val AUC 虛高問題

訓練時 val_roc_auc = 0.9536，但 Kaggle 只有 0.477。
Val split 來自 train_audio，跟 test soundscapes 的錄音環境完全不同。
PCEN 在 train_audio 上的 val AUC 無法反映 soundscape 上的真實表現。

---

## 目前狀態

- `use_pcen: false` 已還原到 local.yaml
- 需要用 log-mel 重新訓練，預期回到 0.858 基準

---

## 下一步改進計劃

### 短期（立即執行）

**1. 用 log-mel 重新訓練，確認基準回到 0.858**

```yaml
use_pcen: false
backbone: resnet18
denoiser_channels: 128   # 從 64 提升，增強降噪
noise_dir: /home/sbplab/frank/BirdCLEF+/data/ESC-50/audio
```

ESC-50 noise augmentation 是安全的改動，不影響 feature pipeline 一致性，
預期能提升 soundscape 泛化能力。

**2. 修正 PCEN 實作，讓它在 soundscape 上也能正確運作**

正確做法是在整段 soundscape 上做 PCEN，再切成 5 秒片段，而不是對每個片段獨立做 PCEN：

```python
# 錯誤做法（現在的做法）
for segment in segments:
    pcen_segment = pcen(segment)  # 每段獨立 EMA，沒有上下文

# 正確做法
full_mel = mel_spectrogram(full_waveform)   # 整段音訊的 mel
full_pcen = pcen(full_mel)                  # 整段做 EMA，背景估計正確
segments = split(full_pcen, 5s)             # 再切片
```

這需要修改 `InferenceEngine.predict_soundscape` 的流程。

### 中期（基準穩定後）

**3. Learnable PCEN**

把 PCEN 的 alpha、delta、r、s 設為可學習參數，讓模型自己學習最適合的壓縮方式：

```python
class LearnablePCEN(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_bands, 1), 0.98))
        self.delta = nn.Parameter(torch.full((num_bands, 1), 2.0))
        self.r     = nn.Parameter(torch.full((num_bands, 1), 0.5))
        self.s     = nn.Parameter(torch.full((num_bands, 1), 0.025))
```

這樣 PCEN 參數會跟模型一起訓練，自動適應 train_audio 和 soundscape 的分布差異。

**4. Soundscape-level normalization**

在推理時對整段 soundscape 做全局正規化，再切片，減少片段間的分布差異：

```python
# 推理時
full_mel = mel(full_waveform)
# 用整段的 mean/std 正規化，而不是用訓練集的統計值
full_mel = (full_mel - full_mel.mean()) / (full_mel.std() + 1e-6)
segments = split(full_mel, 5s)
```

### 長期

**5. 2-model ensemble**

ResNet18（快）+ EfficientNet-B0（準），各自用 log-mel 訓練，推理時平均機率。
預期提升 0.01-0.02，推理時間約 60-70 分鐘，在 90 分鐘限制內。

---

## 教訓

PCEN 理論上適合鳥聲辨識，但實作上有兩個陷阱：
1. 必須在完整音訊上做 EMA，不能對切片獨立處理
2. Train/inference 的 feature pipeline 必須完全一致，任何一個參數不同都會導致分數暴跌
