# 模型修改紀錄 — 加入 CNN 降噪模組

**日期：** 2026-03-24
**修改類型：** 架構改進
**目標：** 在 ResNet18 分類器前加入 ResidualDenoiser，提升對噪音的魯棒性

---

## 修改動機

### 問題分析
1. **Domain Gap：** 訓練資料（乾淨單一鳥叫）vs 測試資料（嘈雜 soundscape）
2. **噪音影響：** 野外錄音包含風聲、水聲、其他動物叫聲等背景噪音
3. **分類性能：** 噪音干擾可能導致特徵提取不準確，影響分類效果

### 解決方案
在 mel spectrogram 上使用輕量級 CNN 進行降噪，然後再進入 ResNet18 分類器。

---

## 技術實現

### 1. ResidualDenoiser 架構

```python
class ResidualDenoiser(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, 1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        predicted_noise = self.conv4(out)
        return residual - predicted_noise  # clean = input - noise
```

### 2. 設計特點

| 特徵 | 說明 |
|------|------|
| **Residual Connection** | `clean = input - predicted_noise`，學習噪音而非乾淨信號 |
| **輕量級** | 4 層 CNN，只增加 75,457 參數（0.7%） |
| **BatchNorm** | 加速收斂，提升穩定性 |
| **End-to-End** | 與分類器一起訓練，針對分類任務優化降噪 |

### 3. 完整架構流程

```
Input: (B, 1, 128, 500) mel spectrogram
  ↓
ResidualDenoiser: 預測並移除噪音
  ↓ clean = input - predicted_noise
Denoised: (B, 1, 128, 500) 降噪後的 mel spectrogram
  ↓
Channel Repeat: (B, 3, 128, 500) 複製成 3 通道
  ↓
ResNet18 Backbone (ImageNet pretrained)
  ↓
Global Average Pooling: (B, 512)
  ↓
Linear Classifier: (B, 234)
  ↓
Output: Raw logits (no sigmoid)
```

---

## 程式碼修改

### 1. 模型檔案 (`birdclef2026/src/model.py`)

**新增：**
- `ResidualDenoiser` 類別
- `BirdCLEFModel` 新增 `use_denoiser` 和 `denoiser_channels` 參數
- Forward pass 中加入可選的降噪步驟

**修改前：**
```python
def forward(self, x):
    x = x.repeat(1, 3, 1, 1)  # 直接複製通道
    features = self.backbone.forward_features(x)
    # ... 後續處理
```

**修改後：**
```python
def forward(self, x):
    if self.use_denoiser:
        x = self.denoiser(x)  # 先降噪
    x = x.repeat(1, 3, 1, 1)  # 再複製通道
    features = self.backbone.forward_features(x)
    # ... 後續處理
```

### 2. 訓練腳本 (`scripts/train.py`)

**新增參數傳遞：**
```python
model = BirdCLEFModel(
    backbone_name=config["backbone"],
    num_classes=len(builder.label_map),
    pretrained=config.get("pretrained", False),
    pool=config.get("pool", "avg"),
    use_denoiser=config.get("use_denoiser", False),      # 新增
    denoiser_channels=config.get("denoiser_channels", 64), # 新增
)
```

### 3. 配置檔案 (`birdclef2026/config/local.yaml`)

**新增設定：**
```yaml
# Model architecture
backbone: resnet18
pool: avg
pretrained: true
use_denoiser: true        # 啟用降噪器
denoiser_channels: 64     # 降噪器通道數
```

---

## 參數統計

| 模型配置 | 總參數數 | 增加參數 | 增加比例 |
|----------|----------|----------|----------|
| ResNet18 only | 11,296,554 | - | - |
| ResNet18 + Denoiser | 11,372,011 | 75,457 | +0.67% |

**Denoiser 參數分解：**
- Conv1: 1→64, 3×3 = 576 + 64 = 640
- Conv2: 64→64, 3×3 = 36,864 + 64 = 36,928  
- Conv3: 64→64, 3×3 = 36,864 + 64 = 36,928
- Conv4: 64→1, 3×3 = 576 + 1 = 577
- BatchNorm: 64×3×2 = 384
- **總計：** 75,457 參數

---

## 預期效果

### 1. 降噪效果
- 移除背景噪音（風聲、水聲、環境音）
- 保留鳥類叫聲的關鍵特徵
- 提升 mel spectrogram 的信噪比

### 2. 分類改進
- 減少噪音對特徵提取的干擾
- 提升模型對 soundscape 的適應性
- 縮小 train/test domain gap

### 3. 訓練策略
- **End-to-End：** denoiser 和 classifier 一起訓練
- **Task-Specific：** 針對鳥類分類任務優化降噪
- **Lightweight：** 參數增加少，訓練效率高

---

## 下一步實驗

### 1. 消融實驗
- [ ] 比較有/無 denoiser 的分類性能
- [ ] 測試不同 `denoiser_channels` (32, 64, 128)
- [ ] 評估降噪效果的可視化

### 2. 進階優化
- [ ] 嘗試 U-Net 風格的 skip connections
- [ ] 加入 attention mechanism 到 denoiser
- [ ] 使用合成噪音進行 pre-training

### 3. 評估指標
- [ ] Kaggle Public Score 提升
- [ ] 降噪前後的 mel spectrogram 對比
- [ ] 不同噪音程度下的魯棒性測試

---

## 檔案清單

**修改的檔案：**
- `birdclef2026/src/model.py` — 新增 ResidualDenoiser 類別
- `scripts/train.py` — 支援 denoiser 參數
- `birdclef2026/config/local.yaml` — 啟用 denoiser 設定

**測試指令：**
```bash
# 訓練帶降噪的模型
python scripts/train.py --config birdclef2026/config/local.yaml

# 測試模型建立
python -c "from birdclef2026.src.model import BirdCLEFModel; print('OK')"
```

---

## 備註

- 這是第一版 ResidualDenoiser，採用最簡單的 residual learning 方式
- 如果效果不佳，可考慮更複雜的架構（U-Net、DenseNet 等）
- 降噪器的設計靈感來自 DnCNN 和 FFDNet 等經典降噪網路
- End-to-end 訓練確保降噪針對分類任務優化，而非通用降噪