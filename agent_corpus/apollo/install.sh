#!/bin/bash
# Apollo proxy agent (host-side) dependency installation.
# Called by install_ads_eval.sh with the uv venv active.
# Package management: uv (per project convention).
#
# Apollo itself runs inside its Docker container (your built Apollo tree); the
# host only needs the bridge/proxy deps + the Apollo protobuf python bindings.
set -e

echo "[INFO] Installing Apollo proxy agent dependencies (uv)..."

# Apollo 7.0 protobuf python bindings (provides apollo_modules.*)
uv pip install "apollo-modules==7.0.2"

# Bridge / proxy deps
uv pip install protobuf
uv pip install docker
uv pip install websocket-client
uv pip install scipy
uv pip install "numpy<1.24"
uv pip install loguru

# Drivora framework deps (shared with other agents)
uv pip install py-trees==0.8.3
uv pip install pydantic
uv pip install hydra-core omegaconf
uv pip install networkx shapely packaging tqdm matplotlib six setuptools deap

echo "[INFO] Apollo proxy deps installed."
echo "[NOTE] You still need, on the Apollo side:"
echo "       1) a built Apollo (build the Apollo source tree once);"
echo "       2) an HD map for your CARLA town under /apollo/modules/map/data/<map_name>;"
echo "       3) sensor calibration matching agent_corpus/apollo sensors;"
echo "       See docs/apollo_integration_plan.md (stages 0 / 0.5)."
