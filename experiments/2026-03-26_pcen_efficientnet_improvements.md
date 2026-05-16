# BirdCLEF 2026 實驗報告 - PCEN + EfficientNet 改進

**日期**: 2026-03-26  
**實驗者**: Frank  
**基準分數**: 0.858 (相較前版本有顯著提升)

## 實驗目標

1. 將 log-mel spectrogram 替換為 PCEN (Per-Channel Energy Normalization)
2. 將 backbone 從 ResNet18 升級到 EfficientNet-B0
3. 啟用背景噪音增強 (ESC-50 dataset)

## 主要修改

### 1. PCEN 特徵提取

**修改檔案**: `birdclef2026/src/features.py`

- 新增 `use_pcen` 參數到 `MelSpectrogramExtractor`
- 實作自定義 `_PCEN` class (因 torchaudio 2.x 移除了 `T.PCEN`)
- PCEN 公式: `(E / (eps + M)^alpha + delta)^r - delta^r`
- 動態背景估計 M 使用指數移動平均

**優勢**:
- 自動壓制持續性背景噪音 (雨聲、風聲)
- 增強短暫的鳥叫聲相對能量
- 比 log-mel 更適合野外錄音環境

### 2. 模型架構升級

**修改檔案**: `birdclef2026/config/local.yaml`

```yaml
# 從 ResNet18 升級到 EfficientNet-B0
backbone: efficientnet_b0  # 原: resnet18

# 啟用 PCEN 特徵提取
use_pcen: true

# 頻率濾波器範圍 (保留低頻鳥叫)
f_min: 50.0    # 保留貓頭鷹、鴿子等低頻叫聲
f_max: 15000.0 # 保留高頻諧波
```

**EfficientNet-B0 優勢**:
- 更好的參數效率 (5.3M vs ResNet18 11.7M)
- 複合縮放策略 (深度+寬度+解析度)
- 在 ImageNet 上更優的準確率/效率比

### 3. 背景噪音增強

**修改檔案**: `birdclef2026/config/local.yaml`

```yaml
# 啟用 ESC-50 噪音資料集
noise_dir: /home/sbplab/frank/BirdCLEF+/data/ESC-50/audio
noise_snr_db_range: [5.0, 30.0]  # SNR 範圍
noise_augment_p: 0.5             # 50% 機率套用
```

**ESC-50 包含**:
- 環境聲音 (雨聲、風聲、海浪)
- 人造噪音 (引擎、工具、人聲)
- 50 類共 2000 個音訊片段

### 4. 訓練流程優化

**修改檔案**: `scripts/train.py`

- PCEN 模式下跳過 mel spectrogram 統計計算
- 自動偵測 `use_pcen` flag 並調整 pipeline
- 修正 `f_min`/`f_max` fallback 值與 config 一致

## 技術實作細節

### PCEN 實作

由於 torchaudio 2.x 移除了 `T.PCEN`，實作純 PyTorch 版本:

```python
class _PCEN(torch.nn.Module):
    def forward(self, x):
        # 沿時間軸建立指數移動平均背景估計
        M = exponential_moving_average(x, smoothing=0.025)
        
        # PCEN 壓縮
        smooth = (eps + M) ** (-alpha)
        pcen = (x * smooth + delta) ** r - (delta ** r)
        return pcen
```

### 特徵提取 Pipeline

**PCEN 模式**:
```
音訊波形 → Mel Spectrogram → PCEN → 模型
```

**Log-mel 模式** (舊版):
```
音訊波形 → Mel Spectrogram → AmplitudeToDB → Z-score 正規化 → 模型
```

## 實驗結果

- **最終分數**: 0.858
- **相較前版本**: 顯著提升
- **訓練穩定性**: 良好，無 NaN 或發散問題

## 效能分析

### PCEN vs Log-mel 比較

| 特徵 | Log-mel | PCEN |
|------|---------|------|
| 背景噪音處理 | 靜態壓縮 | 動態抑制 |
| 短暫信號增強 | 無 | 有 |
| 野外錄音適應性 | 中等 | 優秀 |
| 計算複雜度 | 低 | 中等 |

### EfficientNet-B0 vs ResNet18

| 指標 | ResNet18 | EfficientNet-B0 |
|------|----------|-----------------|
| 參數量 | 11.7M | 5.3M |
| ImageNet Top-1 | 69.8% | 77.1% |
| 推理速度 | 快 | 中等 |
| 記憶體使用 | 中等 | 低 |

## 後續改進方向

### 階段性優化策略

基於當前 0.858 分數，制定三階段優化計劃：

#### 第一階段：模型架構調整 (預期提升至 0.860-0.865)

**當前執行**: ResNet18 + 128 channel denoiser

```yaml
backbone: resnet18        # 從 efficientnet_b0 回退，確保推理速度
denoiser_channels: 128    # 從 64 提升至 128，增強降噪能力
```

**理由分析**:
- **推理時間安全**: ResNet18 比 EfficientNet-B0 快 30-40%，為 90 分鐘 CPU 限制提供緩衝
- **降噪能力提升**: 128 channels 能學習更複雜的噪音模式，特別是多層次的環境噪音
- **風險控制**: 改動最小，容易回滾和調試

#### 第二階段：注意力機制 (預期提升至 0.865-0.875)

**Attention Mechanism 詳解**:

```python
class ChannelAttention(nn.Module):
    """學習哪些頻率通道更重要 - 自動關注鳥叫聲頻段"""
    def forward(self, x):  # (B, C, H, W)
        # Global pooling → FC layers → sigmoid → 重新加權特徵
        weights = self.attention_net(x.mean([2,3]))  # (B, C)
        return x * weights.unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    """學習時頻圖上哪些區域更重要 - 忽略靜音和純噪音片段"""
    def forward(self, x):  # (B, C, H, W)
        attention_map = self.conv(x.mean(1, keepdim=True))  # (B, 1, H, W)
        return x * attention_map
```

**預期效果**:
- 自動學習關注重要頻率範圍 (500Hz-8kHz 鳥叫主頻段)
- 動態忽略無關時間片段 (靜音、風聲、雨聲)
- 典型提升幅度: 1-3% 準確率

**實作策略**:
```python
class AttentionBirdCLEFModel(BirdCLEFModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(self.backbone.num_features)
        
    def forward(self, x):
        if self.use_denoiser:
            x = self.denoiser(x)
        x = x.repeat(1, 3, 1, 1)
        features = self.backbone.forward_features(x)
        features = self.channel_attention(features)  # 新增注意力層
        # ... 後續 pooling + classifier
```

#### 第三階段：模型集成 (預期提升至 0.870-0.880)

**Model Ensemble 策略**:

```python
# 訓練多樣化的模型組合
ensemble_configs = [
    {"backbone": "resnet18", "denoiser_channels": 128, "use_pcen": True},
    {"backbone": "efficientnet_b0", "denoiser_channels": 64, "use_pcen": True},
    {"backbone": "resnet34", "denoiser_channels": 96, "use_pcen": False},
]

# 推理時加權平均
final_pred = 0.4 * pred1 + 0.35 * pred2 + 0.25 * pred3
```

**集成優勢**:
- **誤差互補**: 不同架構捕捉不同特徵模式
- **穩定提升**: 通常是最可靠的 2-5% 提升方法
- **降低方差**: 減少單一模型的偏差和過擬合

**推理時間管理**:
- 2-model ensemble: ~60-70 分鐘 (安全範圍)
- 3-model ensemble: ~80-90 分鐘 (需要優化)

### 進階技術探索

1. **多尺度特徵融合**: 
   - 不同 hop_length (160, 320, 640) 的 spectrogram ensemble
   - 捕捉不同時間解析度的鳥叫特徵

2. **頻域數據增強**:
   - SpecMix: 混合不同物種的頻譜片段
   - FreqOut: 隨機遮蔽特定頻率範圍

3. **自監督預訓練**:
   - 在大量無標籤音訊上預訓練 backbone
   - 學習通用的音訊表示

4. **TTA (Test-Time Augmentation)**:
   - 推理時對同一音訊片段應用多種變換
   - 平均多個預測結果

### 實驗優先級排序

| 階段 | 技術 | 預期提升 | 實作難度 | 推理成本 | 優先級 |
|------|------|----------|----------|----------|--------|
| 1 | ResNet18 + 128ch | +0.002-0.007 | 極低 | 降低 | **最高** |
| 2 | Channel Attention | +0.005-0.010 | 中等 | 微增 | **高** |
| 3 | 2-Model Ensemble | +0.005-0.015 | 低 | 2倍 | **中** |
| 4 | Spatial Attention | +0.003-0.008 | 中等 | 微增 | 中 |
| 5 | 3-Model Ensemble | +0.008-0.020 | 低 | 3倍 | 低 |

### 風險評估與緩解

**主要風險**:
1. **推理時間超限**: 90 分鐘 CPU 限制
2. **過擬合風險**: 模型複雜度增加但訓練資料有限
3. **記憶體限制**: 大模型可能 OOM

**緩解策略**:
1. **漸進式測試**: 每階段都先在小數據集驗證推理時間
2. **Early Stopping**: 嚴格監控 validation loss
3. **混合精度**: 使用 fp16 減少記憶體使用

## 結論

當前 PCEN + EfficientNet-B0 + ESC-50 組合已達到 0.858，為後續優化奠定良好基礎。

**短期目標** (1-2 週): 完成第一階段 ResNet18 + 128ch，目標 0.865  
**中期目標** (2-4 週): 加入 attention mechanism，目標 0.875  
**長期目標** (1-2 月): 完善 ensemble 策略，衝擊 0.880+

這個階段性策略平衡了技術風險、實作複雜度和預期收益，為 BirdCLEF 2026 競賽提供清晰的優化路徑。

## 結論

本次實驗成功將分數提升至 0.858，主要貢獻來自:

1. **PCEN 特徵提取** - 更適合鳥類聲音的動態範圍壓縮
2. **EfficientNet-B0** - 更強的特徵提取能力
3. **ESC-50 噪音增強** - 提升模型對真實環境的泛化能力

這些改進為後續實驗奠定了良好基礎，證明了特徵工程和模型架構選擇對音訊分類任務的重要性。