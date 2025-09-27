#!/bin/bash

# conda create -n tfuse python=3.7 -y 
# export CUDA_HOME=/usr/local/cuda-11.8 # may setup cuda path before install
export CUDA_HOME=/usr/local/cuda-11 # can replace yours

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmcv==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

pip install loguru
pip install hydra-core
pip install omegaconf
pip install natsort
pip install scipy
pip install tqdm
pip install watchdog
pip install docker
pip install py-trees==0.8.3
pip install networkx
pip install tabulate
pip install shapely
pip install timm==0.5.4
pip install mmdet==2.25.0
pip install mmsegmentation==0.25.0
pip install mmengine
pip install ujson
pip install scikit-image==0.16.2
pip install matplotlib==3.1.3
pip install numpy==1.19.5

# === Setup checkpoint directory relative to script ===
set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/model_ckpt"

mkdir -p "$CKPT_DIR"
echo "[INFO] Downloading models to $CKPT_DIR"

wget -c -O "$CKPT_DIR/models_2022.zip" https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip

if ! unzip -tq "$CKPT_DIR/models_2022.zip"; then
    echo "[ERROR] Downloaded file is not a valid zip archive."
    exit 1
fi

unzip -o "$CKPT_DIR/models_2022.zip" -d "$CKPT_DIR/"
rm "$CKPT_DIR/models_2022.zip"