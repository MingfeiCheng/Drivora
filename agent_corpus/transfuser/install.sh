#!/bin/bash
# Transfuser — uv + Python 3.8 + CARLA >= 0.9.12
set -e

# Auto-detect CUDA
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
elif [ -d "/usr/local/cuda-11" ]; then
    export CUDA_HOME=/usr/local/cuda-11
fi

uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
uv pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

uv pip install loguru
uv pip install hydra-core
uv pip install omegaconf
uv pip install natsort
uv pip install scipy
uv pip install tqdm
uv pip install watchdog
uv pip install docker
uv pip install py-trees==0.8.3
uv pip install networkx
uv pip install tabulate
uv pip install shapely
uv pip install timm==0.5.4
uv pip install mmdet==2.28.2
uv pip install mmsegmentation==0.30.0
uv pip install mmengine
uv pip install ujson
uv pip install scikit-image
uv pip install matplotlib
uv pip install "numpy<1.24"

# Download checkpoints
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/model_ckpt"
mkdir -p "$CKPT_DIR"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -d "$CKPT_DIR/models_2022" ]; then
    echo "[INFO] Downloading models to $CKPT_DIR"
    wget -c -O "$CKPT_DIR/models_2022.zip" https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip
    unzip -o "$CKPT_DIR/models_2022.zip" -d "$CKPT_DIR/"
    rm -f "$CKPT_DIR/models_2022.zip"
else
    echo "[INFO] Skipping checkpoint download."
fi
