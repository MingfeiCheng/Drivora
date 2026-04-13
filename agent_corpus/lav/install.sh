#!/bin/bash
# Updated for uv pip and L40S GPU (sm_89) compatibility
set -e

uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
uv pip install einops==0.3.2
uv pip install gdown

cd "$(dirname "$0")"

FILE_ID="1xtG_m_freoR2wzRShd6dOJFA0dq--2iu"
FILE_NAME="weight.zip"


if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -f "$FILE_NAME" ] && [ ! -d "weight" ]; then
    gdown -c --id "$FILE_ID" -O "$FILE_NAME"
    unzip -o "$FILE_NAME" -d .
    find . -type d -name "__MACOSX" -exec rm -rf {} +
    rm -f "$FILE_NAME"
else
    echo "[INFO] Skipping checkpoint download."
fi
