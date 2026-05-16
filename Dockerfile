FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies for soundfile (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torchaudio==2.2.0 \
    timm==0.9.16 \
    numpy \
    pandas \
    scikit-learn \
    soundfile \
    pyyaml

# Copy project source
COPY birdclef2026/ ./birdclef2026/
COPY scripts/ ./scripts/
COPY birdclef2026/config/ ./birdclef2026/config/

ENV PYTHONPATH=/workspace

CMD ["python", "scripts/train.py", "--config", "birdclef2026/config/default.yaml"]
