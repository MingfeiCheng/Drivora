#!/bin/bash
# Pylot proxy agent + Docker image installation
# Called by install_ads_eval.sh with uv venv active
# Compatible with Python 3.8 + CARLA >= 0.9.12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Host-side proxy dependencies ──
echo "[INFO] Installing Pylot proxy agent dependencies..."

# ZMQ communication
uv pip install pyzmq
uv pip install msgpack
uv pip install msgpack-numpy

# Framework deps (shared with other Drivora agents)
uv pip install loguru
uv pip install "numpy<1.24"
uv pip install opencv-python-headless
uv pip install Pillow
uv pip install py-trees==0.8.3
uv pip install pydantic
uv pip install hydra-core
uv pip install omegaconf
uv pip install networkx
uv pip install shapely
uv pip install docker
uv pip install packaging
uv pip install tqdm
uv pip install matplotlib
uv pip install scipy
uv pip install six
uv pip install setuptools
uv pip install watchdog
uv pip install websocket-client
uv pip install cloudpickle
uv pip install filterpy
uv pip install gdown
uv pip install natsort
uv pip install deap
uv pip install ephem
uv pip install tabulate
uv pip install psutil
uv pip install xmlschema
uv pip install dictor

# ── 2. Build Pylot Docker image ──
PYLOT_IMAGE="drivora/pylot:latest"
if docker image inspect "$PYLOT_IMAGE" &>/dev/null; then
    echo "[INFO] Docker image $PYLOT_IMAGE already exists — skipping build."
else
    echo "[INFO] Building Pylot Docker image ($PYLOT_IMAGE)..."
    echo "[INFO] This may take several minutes (planner compilation + model loading)."
    cd "$SCRIPT_DIR/source_code"
    bash docker/build_drivora.sh
    cd - >/dev/null
fi
