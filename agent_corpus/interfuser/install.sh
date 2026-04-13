#!/bin/bash
# InterFuser — uv + Python 3.8 + CARLA >= 0.9.12
set -e

uv pip install --upgrade setuptools wheel
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install gdown

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FILE_ID="1GKiASmGPbD4FwHkUoVfGRk_lMLrGb2f6"
FILE_NAME="interfuser.pth.tar"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -f "$FILE_NAME" ]; then
    gdown -c --id "$FILE_ID" -O "$FILE_NAME"
else
    echo "[INFO] Skipping checkpoint download."
fi

uv pip install -r requirements.txt

# Install interfuser sub-package using uv (avoids setuptools version conflicts)
cd interfuser
uv pip install -e .
