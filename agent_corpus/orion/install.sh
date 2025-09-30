#!/bin/bash
set -e  # Exit on error

export CUDA_HOME=/usr/local/cuda-11 # can replace yours

pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install -e "$(dirname "$0")"

# load weights
cd "$(dirname "$0")"
mkdir -p ckpts

REPO="poleyzdk/Orion"
LOCAL_DIR="ckpts"

echo "[INFO] Downloading HuggingFace repo: $REPO"
echo "[INFO] Target local dir: $LOCAL_DIR"

# Ensure hf is installed
if ! command -v hf &> /dev/null; then
    echo "[INFO] Installing huggingface_hub..."
    pip install --upgrade huggingface_hub
fi

# Fix cache permissions
CACHE_DIR="$HOME/.cache/huggingface"
if [ -d "$CACHE_DIR" ]; then
    echo "[INFO] Fixing HuggingFace cache permissions..."
    sudo chown -R "$(whoami)" "$CACHE_DIR"
fi

# Check if --resume-download is supported
if hf download --help 2>&1 | grep -q -- "--resume-download"; then
    hf download "$REPO" --local-dir "$LOCAL_DIR" --repo-type model --resume-download
else
    echo "[WARN] --resume-download not supported in this version, falling back..."
    hf download "$REPO" --local-dir "$LOCAL_DIR" --repo-type model
fi

echo "[SUCCESS] Repo downloaded to: $LOCAL_DIR"
