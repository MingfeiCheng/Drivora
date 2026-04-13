#!/bin/bash
# Updated for uv pip and L40S GPU (sm_89) compatibility
set -e

# Auto-detect CUDA
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
elif [ -d "/usr/local/cuda-11" ]; then
    export CUDA_HOME=/usr/local/cuda-11
fi

uv pip install "setuptools>=65.0"
uv pip install networkx==2.5.1
uv pip install scipy==1.6.2
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install openmim
uv pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
uv pip install mmdet==2.28.2
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
uv pip install filterpy==1.4.5
uv pip install munkres==1.1.4
uv pip install rdp==0.8
uv pip install timm==0.4.12
uv pip install ujson==4.2.0
uv pip install scikit-image==0.18.1
uv pip install beartype==0.9.1
uv pip install einops==0.4.0
uv pip install pytorch-lightning==1.5.10
uv pip install safetensors==0.3.1
uv pip install transformers==4.30.2
uv pip install huggingface_hub==0.14.1
uv pip install torchmetrics==0.7.2
uv pip install loguru omegaconf hydra-core py-trees==0.8.3 docker tqdm tabulate shapely

uv pip install -e "$(dirname "$0")"

cd "$(dirname "$0")"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    chmod +x download.sh
    ./download.sh
else
    echo "[INFO] Skipping checkpoint download (SKIP_DOWNLOAD=1)"
fi
