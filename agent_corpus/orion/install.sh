#!/bin/bash
# Orion — uv + Python 3.8 + CARLA >= 0.9.12
set -e

# Auto-detect CUDA
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
elif [ -d "/usr/local/cuda-11" ]; then
    export CUDA_HOME=/usr/local/cuda-11
fi
export PATH=$CUDA_HOME/bin:$PATH

# Detect CUDA version to match torch variant
CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "11.8")
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_CUDA="cu121"
    echo "[INFO] Detected CUDA $CUDA_VER → using torch+cu121"
else
    TORCH_CUDA="cu118"
    echo "[INFO] Detected CUDA $CUDA_VER → using torch+cu118"
fi

uv pip install --upgrade setuptools wheel
uv pip install "torch==2.4.1+${TORCH_CUDA}" "torchvision==0.19.1+${TORCH_CUDA}" "torchaudio==2.4.1+${TORCH_CUDA}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Install flash-attn build dependencies then flash-attn itself
uv pip install ninja psutil packaging
uv pip install flash-attn --no-build-isolation

uv pip install --no-build-isolation -e "$(dirname "$0")"

# Download weights
cd "$(dirname "$0")"
mkdir -p ckpts

REPO="poleyzdk/Orion"
LOCAL_DIR="ckpts"

echo "[INFO] Downloading HuggingFace repo: $REPO"

if ! command -v hf &> /dev/null; then
    uv pip install --upgrade "huggingface_hub[cli]"
fi

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ -z "$(ls -A "$LOCAL_DIR" 2>/dev/null)" ]; then
    hf download "$REPO" --local-dir "$LOCAL_DIR" --repo-type model
else
    echo "[INFO] Skipping checkpoint download."
fi

echo "[SUCCESS] Repo downloaded to: $LOCAL_DIR"
