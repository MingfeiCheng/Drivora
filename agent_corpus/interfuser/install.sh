pip install pkgutil-resolve-name==1.3.10
pip install numpy==1.23.0
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.7.0.post2
pip install filterpy
pip install transformers
pip install huggingface-hub
pip install rdp==0.8
pip install ujson==5.9.0
pip install pytorch-lightning==2.4.0
pip install timm==0.9.16
pip install peft==0.13.2

pip install -e "$(dirname "$0")"

#!/bin/bash
set -e  # Exit on error

REPO="RenzKa/simlingo"
LOCAL_DIR="$(dirname "$0")/checkpoint"

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
