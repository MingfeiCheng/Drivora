#!/bin/bash
# Updated for uv pip compatibility
set -e  # Exit on error

uv pip install pkgutil-resolve-name==1.3.10
uv pip install numpy==1.23.0
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install ninja psutil packaging
uv pip install flash-attn==2.5.8 --no-build-isolation
uv pip install filterpy
uv pip install transformers
uv pip install huggingface-hub
uv pip install rdp==0.8
uv pip install ujson==5.9.0
uv pip install pytorch-lightning==2.4.0
uv pip install timm==0.9.16
uv pip install peft==0.13.2

uv pip install -e "$(dirname "$0")"

REPO="RenzKa/simlingo"
LOCAL_DIR="$(dirname "$0")"

echo "[INFO] Downloading HuggingFace repo: $REPO"
echo "[INFO] Target local dir: $LOCAL_DIR"

# Ensure hf is installed
if ! command -v hf &> /dev/null; then
    echo "[INFO] Installing huggingface_hub..."
    uv pip install --upgrade huggingface_hub
fi

CKPT_DIR="$(dirname "$0")/simlingo/checkpoints"
if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -d "$CKPT_DIR" ]; then
    hf download "$REPO" --local-dir "$LOCAL_DIR" --repo-type model
else
    echo "[INFO] Skipping checkpoint download."
fi

echo "[SUCCESS] Repo downloaded to: $LOCAL_DIR"
