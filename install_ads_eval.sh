#!/bin/bash

set -e  # Exit immediately on error

if [ $# -lt 3 ]; then
    echo "Usage: bash install_ads_eval.sh <ADS_NAME> <CARLA_VERSION> <VENV_DIR>"
    echo "Example: bash install_ads_eval.sh roach 0.9.10.1 .venv_ads"
    exit 1
fi

ADS_NAME=$1
CARLA_VERSION=$2
VENV_DIR=$3

ADS_SCRIPT="agent_corpus/${ADS_NAME}/install.sh"

# === Check ADS-specific script ===
if [ ! -f "$ADS_SCRIPT" ]; then
    echo "[ERROR] ADS installation script not found for '${ADS_NAME}'."
    echo "Expected path: ${ADS_SCRIPT}"
    exit 1
fi

# === Check uv is installed ===
if ! command -v uv >/dev/null 2>&1; then
    echo "[INFO] uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

export UV_LINK_MODE=copy

echo "[INFO] Preparing installation for ADS: ${ADS_NAME}, CARLA: ${CARLA_VERSION}, venv: ${VENV_DIR}"

# === Determine Python version based on CARLA ===
PYTHON_VERSION="3.8"

# === Setup virtual environment with uv ===
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment with Python ${PYTHON_VERSION} at ${VENV_DIR}..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR" --prompt "drivora-ads"
else
    echo "[INFO] Virtual environment already exists at ${VENV_DIR}. Skipping creation."
fi

echo "[INFO] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# === Run ADS-specific installation ===
uv pip install "setuptools>=65.0"

# SKIP_DOWNLOAD=1 to skip large checkpoint downloads
export SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "[INFO] Running ADS-specific script: ${ADS_SCRIPT}"
bash "$ADS_SCRIPT" || echo "[WARN] ADS script had errors, continuing with common deps..."

# === Install common dependencies ===
echo "[INFO] Installing common Python dependencies..."
uv pip install -r requirements.txt

# === Setup CARLA ===
echo "[INFO] Pulling CARLA docker image..."
docker pull carlasim/carla:${CARLA_VERSION}

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
    echo "[WARN] CARLA version ${CARLA_VERSION} not explicitly supported by this script."
fi

echo "[SUCCESS] Installation completed for ADS '${ADS_NAME}' with CARLA '${CARLA_VERSION}'."
echo "[INFO] Virtual environment at: ${VENV_DIR}"
