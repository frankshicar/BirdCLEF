# Requirements Document

## Introduction

A machine learning pipeline for the BirdCLEF 2026 Kaggle competition. The system identifies which species (birds, amphibians, mammals, reptiles, insects) are calling in one-minute soundscape recordings from the Brazilian Pantanal. For each 5-second segment of each test soundscape, the pipeline predicts the probability of presence for each of 234 species. Submissions are evaluated using macro-averaged ROC-AUC, ignoring classes with no true positive labels.

**Competition Submission Constraints:** Submissions are made via Kaggle Notebooks running on CPU only (GPU submissions are disabled). The notebook must complete within 90 minutes. Internet access is disabled at submission time. Training is performed offline (outside Kaggle on GPU), and the trained checkpoint is uploaded as a Kaggle Dataset. Pre-trained backbone weights must also be bundled as a Kaggle Dataset so they are available locally at inference time.

## Glossary

- **Pipeline**: The end-to-end ML system from raw audio ingestion to submission file generation
- **Soundscape**: A one-minute field recording at 32kHz containing potentially multiple overlapping species calls
- **Segment**: A 5-second window extracted from a soundscape (12 segments per minute-long recording)
- **Primary_Label**: The species identifier used as the column name in submissions (e.g., `xencan1`)
- **Train_Audio**: Short, single-species recordings from XC (Xeno-Canto) or iNat (iNaturalist) collections
- **Train_Soundscape**: Labeled one-minute field recordings with ground truth start/end/species annotations
- **Spectrogram**: A 2D time-frequency representation of an audio signal used as model input
- **Mel_Spectrogram**: A spectrogram with frequency axis mapped to the mel scale, approximating human auditory perception
- **Backbone**: A pretrained CNN or transformer used as the feature extractor in the classifier
- **Macro_ROC_AUC**: The competition metric — mean ROC-AUC across all species that have at least one true positive label
- **Row_ID**: Submission identifier in the format `[soundscape_filename]_[end_time]`, e.g., `BC2026_Test_0001_S05_20250227_010002_20`
- **Rating**: A quality score in train.csv indicating recording quality (higher is better)
- **Secondary_Labels**: Additional species that may be audible in a training recording beyond the primary label
- **Taxonomy**: The 234-species list defined in taxonomy.csv, covering Aves, Amphibia, Mammalia, Insecta, and Reptilia
- **Kaggle_Dataset**: A versioned dataset artifact uploaded to Kaggle and mounted read-only inside a Kaggle Notebook
- **Submission_Notebook**: The Kaggle Notebook that loads a checkpoint, runs CPU inference, and writes submission.csv
- **Offline_Training**: Model training performed outside Kaggle (e.g., on a local GPU or cloud VM) prior to submission

---

## Requirements

### Requirement 1: Audio Ingestion and Preprocessing

**User Story:** As a competition participant, I want to load and preprocess raw audio files uniformly, so that all audio enters the model pipeline in a consistent format regardless of source.

#### Acceptance Criteria

1. THE Pipeline SHALL resample all audio files to 32kHz before any further processing.
2. WHEN a train_audio file is loaded, THE Pipeline SHALL read the `.ogg` file and return a mono float32 waveform array.
3. WHEN a soundscape file is loaded, THE Pipeline SHALL read the `.ogg` file and return a mono float32 waveform array of approximately 60 seconds.
4. IF an audio file is corrupt or unreadable, THEN THE Pipeline SHALL log the filename and skip that file without halting execution.
5. THE Pipeline SHALL normalize waveform amplitude to the range [-1.0, 1.0] before feature extraction.

---

### Requirement 2: Segment Extraction

**User Story:** As a competition participant, I want to split soundscapes into 5-second segments, so that predictions align with the submission format.

#### Acceptance Criteria

1. THE Pipeline SHALL extract non-overlapping 5-second segments from each soundscape, yielding 12 segments per 60-second recording.
2. WHEN extracting segments, THE Pipeline SHALL label each segment with a Row_ID in the format `[soundscape_filename]_[end_time]` where end_time is the segment's end time in seconds (e.g., 5, 10, 15, ..., 60).
3. IF a soundscape is shorter than 60 seconds, THEN THE Pipeline SHALL pad the final segment with zeros to reach 5 seconds.
4. THE Pipeline SHALL support configurable segment duration and hop size to allow experimentation with overlapping windows during training.

---

### Requirement 3: Feature Extraction

**User Story:** As a competition participant, I want to convert audio segments into mel spectrograms, so that the model receives a structured 2D representation suitable for image-based classifiers.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL convert each 5-second audio segment into a mel spectrogram with configurable n_mels, hop_length, and n_fft parameters.
2. THE Feature_Extractor SHALL apply a top-decibel normalization (converting power to dB scale) to each mel spectrogram.
3. THE Feature_Extractor SHALL normalize each spectrogram to zero mean and unit variance using statistics computed from the training set.
4. WHERE data augmentation is enabled, THE Feature_Extractor SHALL apply SpecAugment (time masking and frequency masking) to training spectrograms.
5. WHERE data augmentation is enabled, THE Feature_Extractor SHALL support mixup augmentation by blending two training samples and their labels.
6. THE Feature_Extractor SHALL output spectrograms as float32 tensors of shape (channels, n_mels, time_frames).

---

### Requirement 4: Training Data Construction

**User Story:** As a competition participant, I want to build a unified training dataset from both train_audio and train_soundscapes, so that the model learns from all available labeled data.

#### Acceptance Criteria

1. THE Dataset_Builder SHALL parse `train.csv` and index all train_audio files by primary_label.
2. THE Dataset_Builder SHALL parse `train_soundscapes_labels.csv` and extract labeled segments from train_soundscapes using the start/end columns.
3. THE Dataset_Builder SHALL assign multi-hot label vectors of length 234 to each training sample, setting the primary_label index to 1 and any secondary_labels indices to 1.
4. WHERE a train_audio recording has a rating below a configurable threshold, THE Dataset_Builder SHALL exclude that recording from training.
5. THE Dataset_Builder SHALL support stratified splitting by primary_label to produce training and validation sets.
6. THE Dataset_Builder SHALL load taxonomy.csv and maintain a consistent mapping from primary_label string to integer class index throughout the pipeline.

---

### Requirement 5: Model Architecture

**User Story:** As a competition participant, I want a configurable classifier built on a pretrained backbone, so that I can leverage transfer learning for audio classification.

#### Acceptance Criteria

1. THE Model SHALL accept a mel spectrogram tensor as input and output a vector of 234 logits, one per species in the taxonomy.
2. THE Model SHALL use a pretrained CNN or transformer backbone (e.g., EfficientNet, ConvNeXt, or Vision Transformer) loaded via a model registry such as `timm`.
3. THE Model SHALL apply a global pooling operation (average or attention-based) over the time dimension before the classification head.
4. WHERE multi-scale feature aggregation is configured, THE Model SHALL concatenate features from multiple backbone stages before pooling.
5. THE Model SHALL support loading pretrained weights from a local checkpoint file without requiring internet access.
6. THE Model SHALL output raw logits; sigmoid activation SHALL be applied externally during loss computation and inference.

---

### Requirement 6: Loss Function and Training Loop

**User Story:** As a competition participant, I want a training loop with an appropriate loss function for multi-label classification, so that the model learns to predict species presence probabilities.

#### Acceptance Criteria

1. THE Trainer SHALL use binary cross-entropy with logits loss (BCEWithLogitsLoss) as the primary loss function for multi-label classification.
2. WHERE label smoothing is configured, THE Trainer SHALL apply label smoothing to positive labels during loss computation.
3. THE Trainer SHALL support a configurable learning rate scheduler (e.g., cosine annealing with warm restarts).
4. THE Trainer SHALL log training loss, validation loss, and validation macro ROC-AUC after each epoch.
5. THE Trainer SHALL save the model checkpoint with the highest validation macro ROC-AUC to disk.
6. THE Trainer SHALL support mixed-precision training (float16/bfloat16) to reduce memory usage and training time.
7. WHEN training is interrupted, THE Trainer SHALL support resuming from the last saved checkpoint.

---

### Requirement 7: Validation and Metric Computation

**User Story:** As a competition participant, I want to evaluate model performance using the competition metric during training, so that I can track progress and select the best checkpoint.

#### Acceptance Criteria

1. THE Evaluator SHALL compute macro-averaged ROC-AUC across all species that have at least one positive label in the validation set.
2. THE Evaluator SHALL apply sigmoid activation to model logits before computing ROC-AUC scores.
3. THE Evaluator SHALL report per-class ROC-AUC scores alongside the macro average to identify underperforming species.
4. WHEN a species has no positive labels in the validation set, THE Evaluator SHALL exclude that species from the macro average computation.
5. THE Evaluator SHALL compute the validation metric on full soundscape segments using the same Row_ID format as the submission.

---

### Requirement 8: Inference and Submission Generation

**User Story:** As a competition participant, I want to run CPU-only inference inside a Kaggle Notebook and produce a valid submission file within the 90-minute time limit, so that I can submit predictions to Kaggle.

#### Acceptance Criteria

1. THE Inference_Engine SHALL load a trained model checkpoint from a locally mounted Kaggle_Dataset without downloading any files from the internet.
2. THE Inference_Engine SHALL load pretrained backbone weights from a locally mounted Kaggle_Dataset without downloading any files from the internet.
3. THE Inference_Engine SHALL run inference on CPU and complete processing of all test soundscape segments within 90 minutes on a standard Kaggle CPU notebook environment.
4. THE Inference_Engine SHALL apply sigmoid activation to model logits to produce probabilities in the range [0.0, 1.0].
5. THE Inference_Engine SHALL aggregate predictions for each Row_ID into a single probability vector of length 234.
6. WHERE test-time augmentation is configured, THE Inference_Engine SHALL average predictions across multiple augmented views of each segment.
7. THE Inference_Engine SHALL produce a CSV file named `submission.csv` with a `row_id` column and one column per species primary_label, matching the format of `sample_submission.csv`.
8. THE Inference_Engine SHALL include a row for every Row_ID present in `sample_submission.csv`, filling missing predictions with 0.0.
9. WHEN multiple model checkpoints are provided, THE Inference_Engine SHALL ensemble predictions by averaging probabilities across checkpoints.
10. THE Inference_Engine SHALL process each 5-second segment using a batch size configurable to balance CPU memory usage and throughput within the 90-minute constraint.

---

### Requirement 9: Offline Training Workflow

**User Story:** As a competition participant, I want to train the model offline on GPU hardware outside of Kaggle, so that I can use full GPU acceleration without the submission notebook's CPU-only constraint.

#### Acceptance Criteria

1. THE Offline_Training workflow SHALL execute entirely outside the Kaggle Notebook submission environment, on local GPU hardware or a cloud VM.
2. WHEN Offline_Training completes, THE Trainer SHALL produce a self-contained checkpoint file that includes model weights, architecture configuration, and the species-to-index mapping.
3. THE Offline_Training workflow SHALL document the steps required to upload the trained checkpoint as a Kaggle_Dataset so it can be mounted in the Submission_Notebook.
4. THE Offline_Training workflow SHALL document the steps required to upload pretrained backbone weights as a Kaggle_Dataset so no internet access is needed at submission time.
5. THE Pipeline SHALL maintain a clear separation between training code (runs offline) and inference code (runs in the Submission_Notebook), with no training-only dependencies required at inference time.

---

### Requirement 10: Kaggle Notebook Submission Environment

**User Story:** As a competition participant, I want the submission notebook to satisfy all Kaggle code competition constraints, so that my submission is accepted and runs reliably within the allowed environment.

#### Acceptance Criteria

1. THE Submission_Notebook SHALL run to completion on a Kaggle CPU-only notebook environment within 90 minutes when processing approximately 600 test soundscapes (approximately 7200 five-second segments).
2. THE Submission_Notebook SHALL NOT require internet access at runtime; all model weights, backbone weights, and supporting files SHALL be sourced from locally mounted Kaggle_Datasets.
3. THE Submission_Notebook SHALL write its final output to a file named `submission.csv`.
4. THE Submission_Notebook SHALL NOT invoke GPU-specific operations (e.g., `.cuda()`, `.to("cuda")`) that would fail in a CPU-only environment.
5. WHEN the Submission_Notebook starts, THE Inference_Engine SHALL verify that all required Kaggle_Dataset mount paths exist and raise a descriptive error if any are missing before processing begins.
6. THE Submission_Notebook SHALL declare all required Kaggle_Dataset dependencies (trained checkpoint, backbone weights, competition data) in its notebook metadata so they are automatically mounted at runtime.

---

### Requirement 11: Configuration Management

**User Story:** As a competition participant, I want all pipeline hyperparameters managed in a single configuration file, so that experiments are reproducible and easy to modify.

#### Acceptance Criteria

1. THE Pipeline SHALL read all hyperparameters (sample rate, n_mels, hop_length, n_fft, segment duration, batch size, learning rate, backbone name, augmentation flags, rating threshold) from a single YAML or JSON configuration file.
2. WHEN a configuration file is provided at runtime, THE Pipeline SHALL load it and override default values.
3. IF a required configuration key is missing, THEN THE Pipeline SHALL raise a descriptive error identifying the missing key before execution begins.
4. THE Pipeline SHALL log the full resolved configuration at the start of each training or inference run.

---

### Requirement 12: Reproducibility

**User Story:** As a competition participant, I want deterministic training runs, so that experiments can be reproduced and compared fairly.

#### Acceptance Criteria

1. THE Pipeline SHALL accept a random seed parameter and apply it to Python, NumPy, and PyTorch random number generators at startup.
2. WHEN the same seed, configuration, and data are provided, THE Trainer SHALL produce model checkpoints with identical validation metrics across runs.
3. THE Pipeline SHALL log the git commit hash and configuration file hash at the start of each run to support experiment tracking.

---

---

# 需求文件（繁體中文翻譯）

## 簡介

本系統為 BirdCLEF 2026 Kaggle 競賽的機器學習流程。系統負責辨識巴西潘塔納爾濕地一分鐘聲景錄音中出現的物種（鳥類、兩棲類、哺乳類、爬蟲類、昆蟲）。針對每段測試聲景的每個 5 秒片段，流程需預測 234 個物種各自出現的機率。評分指標為巨集平均 ROC-AUC，忽略無真陽性標籤的類別。

**競賽提交限制：** 提交須透過 Kaggle Notebook 進行，且僅能使用 CPU（GPU 提交已停用）。Notebook 必須在 90 分鐘內完成執行。提交時無法存取網際網路。模型訓練須在離線環境（Kaggle 外部，使用 GPU）完成，訓練完成的檢查點須上傳為 Kaggle Dataset。預訓練骨幹網路權重亦須打包為 Kaggle Dataset，以確保推論時可在本地取得。

## 詞彙表

- **Pipeline（流程）**：從原始音訊輸入到提交檔案生成的端對端機器學習系統
- **Soundscape（聲景）**：以 32kHz 錄製的一分鐘野外錄音，可能包含多個重疊的物種叫聲
- **Segment（片段）**：從聲景中擷取的 5 秒時間窗（每分鐘錄音共 12 個片段）
- **Primary_Label（主要標籤）**：提交檔案中作為欄位名稱的物種識別碼（例如 `xencan1`）
- **Train_Audio（訓練音訊）**：來自 XC（Xeno-Canto）或 iNat（iNaturalist）的單物種短錄音
- **Train_Soundscape（訓練聲景）**：附有起始/結束時間與物種標註的一分鐘野外錄音
- **Spectrogram（頻譜圖）**：音訊訊號的二維時頻表示，作為模型輸入
- **Mel_Spectrogram（梅爾頻譜圖）**：頻率軸映射至梅爾刻度的頻譜圖，近似人類聽覺感知
- **Backbone（骨幹網路）**：用於特徵提取的預訓練 CNN 或 Transformer
- **Macro_ROC_AUC（巨集 ROC-AUC）**：競賽評分指標，為所有至少有一個真陽性標籤的物種之 ROC-AUC 平均值
- **Row_ID（列識別碼）**：提交識別碼，格式為 `[soundscape_filename]_[end_time]`，例如 `BC2026_Test_0001_S05_20250227_010002_20`
- **Rating（評分）**：train.csv 中表示錄音品質的分數（越高越好）
- **Secondary_Labels（次要標籤）**：訓練錄音中除主要標籤外可能聽到的其他物種
- **Taxonomy（分類學）**：taxonomy.csv 中定義的 234 個物種清單，涵蓋鳥綱、兩棲綱、哺乳綱、昆蟲綱及爬蟲綱
- **Kaggle_Dataset（Kaggle 資料集）**：上傳至 Kaggle 並以唯讀方式掛載於 Kaggle Notebook 中的版本化資料集工件
- **Submission_Notebook（提交 Notebook）**：載入檢查點、執行 CPU 推論並寫出 submission.csv 的 Kaggle Notebook
- **Offline_Training（離線訓練）**：在 Kaggle 外部（例如本地 GPU 或雲端 VM）進行的模型訓練

---

## 需求

### 需求 1：音訊輸入與前處理

**使用者故事：** 身為競賽參賽者，我希望能統一載入並前處理原始音訊檔案，使所有音訊無論來源為何，都能以一致的格式進入模型流程。

#### 驗收標準

1. THE Pipeline 應在任何後續處理前，將所有音訊檔案重新取樣至 32kHz。
2. WHEN 載入 train_audio 檔案時，THE Pipeline 應讀取 `.ogg` 檔案並回傳單聲道 float32 波形陣列。
3. WHEN 載入聲景檔案時，THE Pipeline 應讀取 `.ogg` 檔案並回傳約 60 秒的單聲道 float32 波形陣列。
4. IF 音訊檔案損毀或無法讀取，THEN THE Pipeline 應記錄該檔案名稱並跳過，不中斷執行。
5. THE Pipeline 應在特徵提取前，將波形振幅正規化至 [-1.0, 1.0] 範圍。

---

### 需求 2：片段擷取

**使用者故事：** 身為競賽參賽者，我希望將聲景切割為 5 秒片段，使預測結果符合提交格式。

#### 驗收標準

1. THE Pipeline 應從每段聲景中擷取不重疊的 5 秒片段，每段 60 秒錄音共產生 12 個片段。
2. WHEN 擷取片段時，THE Pipeline 應以 `[soundscape_filename]_[end_time]` 格式為每個片段標記 Row_ID，其中 end_time 為片段結束時間（秒），例如 5、10、15、…、60。
3. IF 聲景短於 60 秒，THEN THE Pipeline 應以零值填補最後一個片段至 5 秒。
4. THE Pipeline 應支援可設定的片段時長與跳躍大小，以便在訓練期間實驗重疊窗口。

---

### 需求 3：特徵提取

**使用者故事：** 身為競賽參賽者，我希望將音訊片段轉換為梅爾頻譜圖，使模型接收適合影像分類器的結構化二維表示。

#### 驗收標準

1. THE Feature_Extractor 應將每個 5 秒音訊片段轉換為梅爾頻譜圖，並支援可設定的 n_mels、hop_length 及 n_fft 參數。
2. THE Feature_Extractor 應對每張梅爾頻譜圖套用頂部分貝正規化（將功率轉換為 dB 刻度）。
3. THE Feature_Extractor 應使用從訓練集計算的統計數據，將每張頻譜圖正規化為零均值與單位變異數。
4. WHERE 資料增強已啟用，THE Feature_Extractor 應對訓練頻譜圖套用 SpecAugment（時間遮罩與頻率遮罩）。
5. WHERE 資料增強已啟用，THE Feature_Extractor 應支援 mixup 增強，混合兩個訓練樣本及其標籤。
6. THE Feature_Extractor 應輸出形狀為 (channels, n_mels, time_frames) 的 float32 張量。

---

### 需求 4：訓練資料建構

**使用者故事：** 身為競賽參賽者，我希望從 train_audio 與 train_soundscapes 建立統一的訓練資料集，使模型能從所有可用的標記資料中學習。

#### 驗收標準

1. THE Dataset_Builder 應解析 `train.csv` 並依 primary_label 索引所有 train_audio 檔案。
2. THE Dataset_Builder 應解析 `train_soundscapes_labels.csv`，並使用 start/end 欄位從 train_soundscapes 中擷取標記片段。
3. THE Dataset_Builder 應為每個訓練樣本指定長度為 234 的多熱標籤向量，將 primary_label 索引設為 1，並將所有 secondary_labels 索引設為 1。
4. WHERE train_audio 錄音的評分低於可設定的閾值，THE Dataset_Builder 應將該錄音排除於訓練之外。
5. THE Dataset_Builder 應支援依 primary_label 進行分層切割，以產生訓練集與驗證集。
6. THE Dataset_Builder 應載入 taxonomy.csv，並在整個流程中維護 primary_label 字串至整數類別索引的一致映射。

---

### 需求 5：模型架構

**使用者故事：** 身為競賽參賽者，我希望建立基於預訓練骨幹網路的可設定分類器，以便利用遷移學習進行音訊分類。

#### 驗收標準

1. THE Model 應接受梅爾頻譜圖張量作為輸入，並輸出長度為 234 的 logit 向量，每個物種對應一個值。
2. THE Model 應使用預訓練的 CNN 或 Transformer 骨幹網路（例如 EfficientNet、ConvNeXt 或 Vision Transformer），透過 `timm` 等模型登錄檔載入。
3. THE Model 應在分類頭之前，對時間維度套用全域池化操作（平均或注意力機制）。
4. WHERE 多尺度特徵聚合已設定，THE Model 應在池化前串接來自多個骨幹網路階段的特徵。
5. THE Model 應支援從本地檢查點檔案載入預訓練權重，無需存取網際網路。
6. THE Model 應輸出原始 logit；sigmoid 激活函數應在損失計算與推論時於外部套用。

---

### 需求 6：損失函數與訓練迴圈

**使用者故事：** 身為競賽參賽者，我希望使用適合多標籤分類的損失函數進行訓練，使模型能學習預測物種出現機率。

#### 驗收標準

1. THE Trainer 應使用帶 logit 的二元交叉熵損失（BCEWithLogitsLoss）作為多標籤分類的主要損失函數。
2. WHERE 標籤平滑已設定，THE Trainer 應在損失計算時對正標籤套用標籤平滑。
3. THE Trainer 應支援可設定的學習率排程器（例如帶熱重啟的餘弦退火）。
4. THE Trainer 應在每個 epoch 後記錄訓練損失、驗證損失及驗證巨集 ROC-AUC。
5. THE Trainer 應將驗證巨集 ROC-AUC 最高的模型檢查點儲存至磁碟。
6. THE Trainer 應支援混合精度訓練（float16/bfloat16）以降低記憶體使用量並縮短訓練時間。
7. WHEN 訓練中斷時，THE Trainer 應支援從最後儲存的檢查點繼續訓練。

---

### 需求 7：驗證與指標計算

**使用者故事：** 身為競賽參賽者，我希望在訓練期間使用競賽指標評估模型效能，以便追蹤進度並選擇最佳檢查點。

#### 驗收標準

1. THE Evaluator 應計算驗證集中所有至少有一個正標籤的物種之巨集平均 ROC-AUC。
2. THE Evaluator 應在計算 ROC-AUC 分數前，對模型 logit 套用 sigmoid 激活函數。
3. THE Evaluator 應回報每個類別的 ROC-AUC 分數及巨集平均值，以識別表現不佳的物種。
4. WHEN 某物種在驗證集中無正標籤時，THE Evaluator 應將該物種排除於巨集平均計算之外。
5. THE Evaluator 應使用與提交相同的 Row_ID 格式，在完整聲景片段上計算驗證指標。

---

### 需求 8：推論與提交檔案生成

**使用者故事：** 身為競賽參賽者，我希望在 Kaggle Notebook 中執行僅限 CPU 的推論，並在 90 分鐘時限內產生有效的提交檔案，以便向 Kaggle 提交預測結果。

#### 驗收標準

1. THE Inference_Engine 應從本地掛載的 Kaggle_Dataset 載入訓練完成的模型檢查點，無需從網際網路下載任何檔案。
2. THE Inference_Engine 應從本地掛載的 Kaggle_Dataset 載入預訓練骨幹網路權重，無需從網際網路下載任何檔案。
3. THE Inference_Engine 應在 CPU 上執行推論，並在標準 Kaggle CPU Notebook 環境中，於 90 分鐘內完成所有測試聲景片段的處理。
4. THE Inference_Engine 應對模型 logit 套用 sigmoid 激活函數，產生範圍在 [0.0, 1.0] 的機率值。
5. THE Inference_Engine 應將每個 Row_ID 的預測結果彙整為長度為 234 的單一機率向量。
6. WHERE 測試時增強已設定，THE Inference_Engine 應對每個片段的多個增強視圖取平均預測值。
7. THE Inference_Engine 應產生名為 `submission.csv` 的 CSV 檔案，包含 `row_id` 欄位及每個物種 primary_label 的欄位，格式須符合 `sample_submission.csv`。
8. THE Inference_Engine 應為 `sample_submission.csv` 中的每個 Row_ID 包含一列，缺失預測以 0.0 填補。
9. WHEN 提供多個模型檢查點時，THE Inference_Engine 應透過對各檢查點的機率取平均來進行集成預測。
10. THE Inference_Engine 應以可設定的批次大小處理每個 5 秒片段，以在 90 分鐘限制內平衡 CPU 記憶體使用量與吞吐量。

---

### 需求 9：離線訓練工作流程

**使用者故事：** 身為競賽參賽者，我希望在 Kaggle 外部的 GPU 硬體上離線訓練模型，以便在不受提交 Notebook 僅限 CPU 限制的情況下使用完整的 GPU 加速。

#### 驗收標準

1. THE Offline_Training 工作流程應完全在 Kaggle Notebook 提交環境之外執行，使用本地 GPU 硬體或雲端 VM。
2. WHEN Offline_Training 完成時，THE Trainer 應產生包含模型權重、架構設定及物種至索引映射的自包含檢查點檔案。
3. THE Offline_Training 工作流程應記錄將訓練完成的檢查點上傳為 Kaggle_Dataset 的步驟，以便在 Submission_Notebook 中掛載使用。
4. THE Offline_Training 工作流程應記錄將預訓練骨幹網路權重上傳為 Kaggle_Dataset 的步驟，確保提交時無需存取網際網路。
5. THE Pipeline 應在訓練程式碼（離線執行）與推論程式碼（在 Submission_Notebook 中執行）之間維持清晰的分離，推論時不需要任何僅供訓練使用的相依套件。

---

### 需求 10：Kaggle Notebook 提交環境

**使用者故事：** 身為競賽參賽者，我希望提交 Notebook 能滿足所有 Kaggle 程式碼競賽限制，使我的提交能被接受並在允許的環境中可靠執行。

#### 驗收標準

1. THE Submission_Notebook 在處理約 600 個測試聲景（約 7200 個 5 秒片段）時，應在 Kaggle 僅限 CPU 的 Notebook 環境中於 90 分鐘內完成執行。
2. THE Submission_Notebook 在執行時不應需要網際網路存取；所有模型權重、骨幹網路權重及支援檔案均應來自本地掛載的 Kaggle_Dataset。
3. THE Submission_Notebook 應將最終輸出寫入名為 `submission.csv` 的檔案。
4. THE Submission_Notebook 不應呼叫 GPU 專用操作（例如 `.cuda()`、`.to("cuda")`），以避免在僅限 CPU 的環境中發生錯誤。
5. WHEN Submission_Notebook 啟動時，THE Inference_Engine 應驗證所有必要的 Kaggle_Dataset 掛載路徑是否存在，若有任何路徑缺失，應在開始處理前提出描述性錯誤。
6. THE Submission_Notebook 應在其 Notebook 元資料中宣告所有必要的 Kaggle_Dataset 相依項目（訓練檢查點、骨幹網路權重、競賽資料），以便在執行時自動掛載。

---

### 需求 11：設定管理

**使用者故事：** 身為競賽參賽者，我希望所有流程超參數都在單一設定檔中管理，使實驗具有可重現性且易於修改。

#### 驗收標準

1. THE Pipeline 應從單一 YAML 或 JSON 設定檔讀取所有超參數（取樣率、n_mels、hop_length、n_fft、片段時長、批次大小、學習率、骨幹網路名稱、增強旗標、評分閾值）。
2. WHEN 在執行時提供設定檔時，THE Pipeline 應載入並覆蓋預設值。
3. IF 缺少必要的設定鍵，THEN THE Pipeline 應在執行開始前提出描述性錯誤，指出缺少的鍵。
4. THE Pipeline 應在每次訓練或推論執行開始時記錄完整的已解析設定。

---

### 需求 12：可重現性

**使用者故事：** 身為競賽參賽者，我希望訓練執行具有確定性，使實驗能夠重現並公平比較。

#### 驗收標準

1. THE Pipeline 應接受隨機種子參數，並在啟動時將其套用至 Python、NumPy 及 PyTorch 的隨機數生成器。
2. WHEN 提供相同的種子、設定及資料時，THE Trainer 應在多次執行中產生具有相同驗證指標的模型檢查點。
3. THE Pipeline 應在每次執行開始時記錄 git commit hash 及設定檔 hash，以支援實驗追蹤。
