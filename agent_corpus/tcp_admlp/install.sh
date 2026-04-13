#!/bin/bash
# Updated for uv pip and L40S GPU (sm_89) compatibility
set -e

uv pip install networkx
uv pip install scipy
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install pytorch-lightning==1.9.5

uv pip install -e "$(dirname "$0")"

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
