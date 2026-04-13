#!/bin/bash

set -e

if [ $# -lt 3 ]; then
    echo "Usage: bash install_tester.sh <TESTER_NAME> <CARLA_VERSION> <VENV_DIR>"
    echo "Example: bash install_tester.sh random 0.9.10.1 .venv_tester"
    exit 1
fi

TESTER_NAME=$1
CARLA_VERSION=$2
VENV_DIR=$3

# === Check uv is installed ===
if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export UV_LINK_MODE=copy

echo "[INFO] Preparing installation for Tester: ${TESTER_NAME}, CARLA: ${CARLA_VERSION}, venv: ${VENV_DIR}"

# === Determine Python version ===
PYTHON_VERSION="3.8"

# === Setup virtual environment with uv ===
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment with Python ${PYTHON_VERSION} at ${VENV_DIR}..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR" --prompt "drivora-tester"
else
    echo "[INFO] Virtual environment already exists at ${VENV_DIR}. Skipping creation."
fi

echo "[INFO] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# === Install base dependencies ===
uv pip install "setuptools>=65.0"

echo "[INFO] Installing common Python dependencies..."
uv pip install -r requirements.txt

# === Install tester-specific dependencies ===
# Convention: fuzzer/requirements/<tester_name>.txt
TESTER_REQ="fuzzer/requirements/${TESTER_NAME}.txt"
if [ -f "$TESTER_REQ" ]; then
    echo "[INFO] Installing tester-specific dependencies from ${TESTER_REQ}..."
    uv pip install -r "$TESTER_REQ"
else
    echo "[INFO] No tester-specific requirements found at ${TESTER_REQ}. Skipping."
fi

# === Install CARLA Python API (no docker pull, relies on eval environment) ===
if [[ "$CARLA_VERSION" == "0.9.10" || "$CARLA_VERSION" == "0.9.10.1" ]]; then
    CARLA_EGG_REL="pkgs/carla-0.9.10-py3.7-linux-x86_64.egg"
    if [ -f "$CARLA_EGG_REL" ]; then
        CARLA_EGG=$(realpath "$CARLA_EGG_REL")
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "[INFO] Registering $CARLA_EGG in $SITE_PACKAGES/carla.pth"
        echo "$CARLA_EGG" > "${SITE_PACKAGES}/carla.pth"
    else
        echo "[WARN] CARLA egg not found at $CARLA_EGG_REL. Skipping registration."
    fi
elif [[ "$CARLA_VERSION" == "0.9.12" || "$CARLA_VERSION" > "0.9.12" ]]; then
    echo "[INFO] Installing CARLA Python API via pip..."
    uv pip install carla==${CARLA_VERSION}
else
    echo "[WARN] CARLA version ${CARLA_VERSION} not explicitly supported."
fi

echo "[SUCCESS] Installation completed for tester '${TESTER_NAME}' (CARLA ${CARLA_VERSION})."
echo "[INFO] Virtual environment at: ${VENV_DIR}"
