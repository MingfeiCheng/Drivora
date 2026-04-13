#!/bin/bash
# ROACH agent dependencies — called by install_ads_eval.sh with uv venv active
# Compatible with Python 3.8 + CARLA >= 0.9.12

uv pip install loguru
uv pip install matplotlib
uv pip install hydra-core
uv pip install omegaconf
uv pip install natsort
uv pip install scipy
uv pip install tqdm
uv pip install watchdog
uv pip install msgpack
uv pip install pyzmq
uv pip install msgpack-numpy
uv pip install docker
uv pip install py-trees==0.8.3
uv pip install networkx
uv pip install tabulate
uv pip install shapely
uv pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install h5py
uv pip install "gym==0.17.2"
uv pip install "Pillow==9.5.0"
uv pip install filterpy
uv pip install opencv-python
uv pip install "numpy<1.24"
uv pip install gdown

# Download pre-trained checkpoint
LOG_DIR="$(dirname "$(realpath "$0")")/log"
mkdir -p "$LOG_DIR"
cd "$LOG_DIR" || exit 1

FILE_ID="1dg_nB8OvB9H-wpcSlpUhXNgrXW8_pPIQ"
FILE_NAME="ckpt_11833344.pth"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -f "$FILE_NAME" ]; then
    echo "[INFO] Downloading ROACH checkpoint..."
    gdown --id "$FILE_ID" -O "$FILE_NAME"
else
    echo "[INFO] Skipping checkpoint download."
fi
