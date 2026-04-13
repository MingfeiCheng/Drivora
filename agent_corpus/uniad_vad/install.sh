#!/bin/bash
# Updated for uv pip compatibility

set -e

# NOTE: the following parts maybe changed according to your system environment
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# Auto-detect CUDA
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
elif [ -d "/usr/local/cuda-11" ]; then
    export CUDA_HOME=/usr/local/cuda-11
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

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
if [ "$TORCH_CUDA" = "cu121" ]; then
    uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
else
    uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
fi

uv pip install ninja packaging

uv pip install --no-build-isolation -e "$(dirname "$0")"


cd "$(dirname "$0")"

repo="rethinklab/Bench2DriveZoo"
repo_dir="Bench2DriveZoo"

git lfs install

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -d "$repo_dir" ]; then
    echo "Start cloning $repo"
    git clone https://huggingface.co/$repo
else
    echo "[INFO] Skipping checkpoint download."
fi
