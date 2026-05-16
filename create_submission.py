#!/usr/bin/env python3
"""
BirdCLEF 2026 Submission Creator
建立 submission.csv 並打包成 zip 檔案
"""

import os
import sys
import zipfile
from pathlib import Path
import logging
import argparse

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_submission(checkpoint_path: str, output_dir: str = "submission"):
    """
    建立 submission 檔案並打包成 zip
    
    Args:
        checkpoint_path: 訓練好的模型檢查點路徑
        output_dir: 輸出目錄名稱
    """
    
    # 檢查檢查點檔案是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"檢查點檔案不存在: {checkpoint_path}")
    
    # 建立輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"建立 submission 目錄: {output_path}")
    
    # 複製必要的程式碼檔案
    code_files = [
        "birdclef2026/__init__.py",
        "birdclef2026/src/__init__.py", 
        "birdclef2026/src/audio.py",
        "birdclef2026/src/features.py",
        "birdclef2026/src/model.py",
        "birdclef2026/src/inference.py",
        "birdclef2026/src/utils.py",
        "birdclef2026/src/checkpoint_utils.py",
        "birdclef2026/config/default.yaml"
    ]
    
    for file_path in code_files:
        src = Path(file_path)
        if src.exists():
            dst = output_path / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # 複製檔案內容
            with open(src, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"複製: {src} -> {dst}")
        else:
            logger.warning(f"檔案不存在，跳過: {src}")
    
    # 複製檢查點檔案 (使用 checkpoint.pt 符合 submission.ipynb)
    checkpoint_dst = output_path / "checkpoint.pt"
    with open(checkpoint_path, 'rb') as src_f:
        with open(checkpoint_dst, 'wb') as dst_f:
            dst_f.write(src_f.read())
    logger.info(f"複製檢查點: {checkpoint_path} -> {checkpoint_dst}")
    
    # 複製 sample_submission.csv
    sample_sub_src = Path("data/sample_submission.csv")
    if sample_sub_src.exists():
        sample_sub_dst = output_path / "sample_submission.csv"
        with open(sample_sub_src, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(sample_sub_dst, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"複製: {sample_sub_src} -> {sample_sub_dst}")
    
    # 建立推論腳本
    inference_script = output_path / "run_inference.py"
    
    script_content = '''#!/usr/bin/env python3
"""
BirdCLEF 2026 推論腳本
用於產生 submission.csv
符合 submission.ipynb 的結構
"""

import os
import sys
from pathlib import Path

def main():
    """執行推論並產生 submission.csv"""
    
    # 設定路徑 (符合 submission.ipynb 的結構)
    MODEL_DIR = str(Path(__file__).parent)  # 當前目錄作為 MODEL_DIR
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pt")  # 使用 checkpoint.pt 而不是 best_checkpoint.pt
    SAMPLE_SUBMISSION = os.path.join(MODEL_DIR, "sample_submission.csv")
    SOUNDSCAPE_DIR = "test_soundscapes"  # 這會在競賽環境中被填充
    OUTPUT_PATH = "submission.csv"
    BATCH_SIZE = 32
    DEVICE = "cpu"
    TTA = False
    TTA_VIEWS = 3
    
    print("開始執行推論...")
    print(f"模型目錄: {MODEL_DIR}")
    print(f"檢查點: {CHECKPOINT_PATH}")
    print(f"樣本提交檔: {SAMPLE_SUBMISSION}")
    print(f"測試音檔目錄: {SOUNDSCAPE_DIR}")
    
    # 清理模組 (符合 submission.ipynb 的邏輯)
    sys.path = [p for p in sys.path if "birdclef" not in p.lower()]
    sys.path.insert(0, MODEL_DIR)
    for mod in list(sys.modules.keys()):
        if "birdclef" in mod:
            del sys.modules[mod]
    
    # 檢查必要檔案
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"檢查點檔案不存在: {CHECKPOINT_PATH}")
    
    if not os.path.exists(SAMPLE_SUBMISSION):
        raise FileNotFoundError(f"樣本提交檔不存在: {SAMPLE_SUBMISSION}")
    
    print("路徑檢查通過")
    
    # 載入推論引擎
    from birdclef2026.src.inference import InferenceEngine
    
    engine = InferenceEngine(
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        tta=TTA,
        tta_views=TTA_VIEWS,
    )
    engine.verify_paths()
    print("推論引擎準備完成")
    
    # 執行推論
    if os.path.exists(SOUNDSCAPE_DIR) and os.listdir(SOUNDSCAPE_DIR):
        print(f"找到測試音檔，開始推論...")
        engine.run(
            soundscape_dir=SOUNDSCAPE_DIR,
            sample_submission_path=SAMPLE_SUBMISSION,
            output_path=OUTPUT_PATH,
        )
        print(f"推論完成，結果儲存至: {OUTPUT_PATH}")
    else:
        print("測試音檔目錄為空或不存在，建立預設 submission...")
        # 建立預設的 submission（全部預測為 0）
        import pandas as pd
        sample_df = pd.read_csv(SAMPLE_SUBMISSION)
        # 將所有預測值設為 0
        for col in sample_df.columns:
            if col != "row_id":
                sample_df[col] = 0.0
        sample_df.to_csv(OUTPUT_PATH, index=False)
        print(f"預設 submission 建立完成: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
'''
    
    with open(inference_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    logger.info(f"建立推論腳本: {inference_script}")
    
    # 建立 requirements.txt
    requirements_path = output_path / "requirements.txt"
    requirements_content = """torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.21.0
pandas>=1.3.0
librosa>=0.8.0
timm>=0.6.0
PyYAML>=5.4.0
"""
    
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    logger.info(f"建立 requirements.txt: {requirements_path}")
    
    # 建立 README.md
    readme_path = output_path / "README.md"
    readme_content = """# BirdCLEF 2026 Submission

這個 submission 包含了訓練好的模型和推論程式碼，符合 submission.ipynb 的結構。

## 檔案結構

- `checkpoint.pt`: 訓練好的模型檢查點
- `run_inference.py`: 推論腳本 (符合 submission.ipynb 邏輯)
- `sample_submission.csv`: 樣本提交格式
- `birdclef2026/`: 模型和推論程式碼
- `requirements.txt`: Python 套件需求

## 使用方法

1. 安裝依賴套件:
   ```bash
   pip install -r requirements.txt
   ```

2. 執行推論:
   ```bash
   python run_inference.py
   ```

3. 結果會儲存在 `submission.csv`

## 與 submission.ipynb 的對應關係

這個 submission 的結構完全符合 `notebooks/submission.ipynb` 的邏輯：

- 使用 `checkpoint.pt` 作為模型檔案名稱
- 包含模組清理邏輯
- 使用相同的參數設定 (BATCH_SIZE=32, DEVICE="cpu", TTA=False)
- 相同的 InferenceEngine 初始化方式

## 注意事項

- 推論使用 CPU 模式以確保相容性
- 如果測試音檔目錄為空，會建立預設的 submission
- 模型會自動處理音檔的前處理和特徵提取
- 包含完整的錯誤處理和路徑驗證
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    logger.info(f"建立 README.md: {readme_path}")
    
    return output_path

def create_zip(submission_dir: str, zip_name: str = "submission.zip"):
    """
    將 submission 目錄打包成 zip 檔案
    
    Args:
        submission_dir: submission 目錄路徑
        zip_name: zip 檔案名稱
    """
    
    submission_path = Path(submission_dir)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission 目錄不存在: {submission_path}")
    
    zip_path = Path(zip_name)
    
    logger.info(f"開始打包 {submission_path} -> {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in submission_path.rglob('*'):
            if file_path.is_file():
                # 計算相對路徑
                arcname = file_path.relative_to(submission_path)
                zipf.write(file_path, arcname)
                logger.info(f"加入檔案: {arcname}")
    
    # 檢查 zip 檔案大小
    zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"打包完成: {zip_path} ({zip_size:.2f} MB)")
    
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="建立 BirdCLEF 2026 submission")
    parser.add_argument(
        "--checkpoint", 
        default="checkpoints/best_checkpoint.pt",
        help="模型檢查點路徑 (預設: checkpoints/best_checkpoint.pt)"
    )
    parser.add_argument(
        "--output-dir",
        default="submission",
        help="輸出目錄名稱 (預設: submission)"
    )
    parser.add_argument(
        "--zip-name",
        default="submission.zip", 
        help="zip 檔案名稱 (預設: submission.zip)"
    )
    
    args = parser.parse_args()
    
    try:
        # 建立 submission 目錄
        submission_dir = create_submission(args.checkpoint, args.output_dir)
        
        # 打包成 zip
        zip_path = create_zip(str(submission_dir), args.zip_name)
        
        logger.info("=" * 50)
        logger.info("Submission 建立完成!")
        logger.info(f"目錄: {submission_dir}")
        logger.info(f"ZIP 檔案: {zip_path}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"建立 submission 時發生錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
