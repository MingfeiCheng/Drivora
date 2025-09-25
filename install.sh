#!/bin/bash
set -e  # Exit immediately on error

if [ $# -lt 3 ]; then
    echo "Usage: bash install.sh <ADS_NAME> <TESTER_NAME> <CARLA_VERSION>"
    echo "Example: bash install.sh roach avfuzzer 0.9.12"
    exit 1
fi

ADS_NAME=$1
TESTER_NAME=$2
CARLA_VERSION=$3

ADS_SCRIPT="agent_corpus/${ADS_NAME}/install.sh"
TESTER_SCRIPT="fuzzer/open_scenario/${TESTER_NAME}/install.sh"

# === Check ADS-specific script ===
if [ ! -f "$ADS_SCRIPT" ]; then
    echo "[ERROR] ADS installation script not found for '${ADS_NAME}'."
    echo "Expected path: ${ADS_SCRIPT}"
    exit 1
fi

# === Check tester-specific script ===
if [ ! -f "$TESTER_SCRIPT" ]; then
    echo "[ERROR] Tester installation script not found for '${TESTER_NAME}'."
    echo "Expected path: ${TESTER_SCRIPT}"
    exit 1
fi

echo "[INFO] Preparing installation for ADS: ${ADS_NAME}, Tester: ${TESTER_NAME}, CARLA: ${CARLA_VERSION}"

# === Setup project version ===
env_name="drivora-${ADS_NAME}-${TESTER_NAME}"

# === Check Conda or fallback to venv ===
if command -v conda >/dev/null 2>&1; then
    echo "[INFO] Conda detected. Using environment: ${env_name}"

    if conda info --envs | awk '{print $1}' | grep -q "^${env_name}$"; then
        echo "[INFO] Conda environment '${env_name}' already exists. Skipping creation."
    else
        if [[ "$CARLA_VERSION" == "0.9.10" || "$CARLA_VERSION" == "0.9.10.1" ]]; then
            echo "[INFO] Creating conda environment '${env_name}' with Python 3.7..."
            conda create -n "${env_name}" python=3.8 -y
        elif [[ "$CARLA_VERSION" == "0.9.12" || "$CARLA_VERSION" > "0.9.12" ]]; then
            echo "[INFO] Creating conda environment '${env_name}' with Python 3.12..."
            conda create -n "${env_name}" python=3.10 -y
        else
            echo "[WARN] Unknown CARLA version ${CARLA_VERSION}, defaulting to Python 3.8..."
            conda create -n "${env_name}" python=3.8 -y
        fi
    fi

    eval "$(conda shell.bash hook)"
    conda activate "${env_name}"

else
    echo "[WARN] Conda not found. Falling back to Python venv."

    if [ ! -d ".venv/${env_name}" ]; then
        echo "[INFO] Creating virtual environment at .venv/${env_name}..."
        python3 -m venv ".venv/${env_name}"
    fi

    source ".venv/${env_name}/bin/activate"
fi

# === Install common dependencies ===
echo "[INFO] Installing common Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# === Run ADS-specific installation ===
echo "[INFO] Running ADS-specific script: ${ADS_SCRIPT}"
bash "$ADS_SCRIPT"

# === Run tester-specific installation ===
echo "[INFO] Running tester-specific script: ${TESTER_SCRIPT}"
bash "$TESTER_SCRIPT"

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
    pip install carla==${CARLA_VERSION}
else
    echo "[WARN] CARLA version ${CARLA_VERSION} not explicitly supported by this script."
fi

echo "[SUCCESS] Installation completed for ADS '${ADS_NAME}' with tester '${TESTER_NAME}' and CARLA '${CARLA_VERSION}'."
echo "[INFO] Environment '${env_name}' is now active."

# === Keep environment active ===
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[INFO] Relaunching a new shell with environment '${env_name}'..."
    if command -v conda >/dev/null 2>&1; then
        exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${env_name}")
    else
        exec bash --rcfile <(echo "source .venv/${env_name}/bin/activate")
    fi
fi
