#!/bin/bash
set -e
export UV_LINK_MODE=copy

CARLA_VERSION="0.9.15"
VENV_ROOT=".venvs"
mkdir -p "$VENV_ROOT"

# ADS agents
bash install_ads_eval.sh roach        "$CARLA_VERSION" "$VENV_ROOT/roach" 
bash install_ads_eval.sh interfuser   "$CARLA_VERSION" "$VENV_ROOT/interfuser" 
bash install_ads_eval.sh lav          "$CARLA_VERSION" "$VENV_ROOT/lav" 
bash install_ads_eval.sh orion        "$CARLA_VERSION" "$VENV_ROOT/orion" 
bash install_ads_eval.sh plant        "$CARLA_VERSION" "$VENV_ROOT/plant" 
bash install_ads_eval.sh simlingo     "$CARLA_VERSION" "$VENV_ROOT/simlingo" 
bash install_ads_eval.sh tcp_admlp    "$CARLA_VERSION" "$VENV_ROOT/tcp_admlp" # tcp & admlp
bash install_ads_eval.sh transfuser   "$CARLA_VERSION" "$VENV_ROOT/transfuser" 
bash install_ads_eval.sh uniad_vad    "$CARLA_VERSION" "$VENV_ROOT/uniad_vad" # uniad & vad
bash install_ads_eval.sh pylot    "$CARLA_VERSION" "$VENV_ROOT/pylot"


# Testers
bash install_tester.sh random "$CARLA_VERSION" "$VENV_ROOT/random"
bash install_tester.sh random_multi "$CARLA_VERSION" "$VENV_ROOT/random_multi"
bash install_tester.sh avfuzzer "$CARLA_VERSION" "$VENV_ROOT/avfuzzer"
bash install_tester.sh behavexplore "$CARLA_VERSION" "$VENV_ROOT/behavexplore"
bash install_tester.sh samota "$CARLA_VERSION" "$VENV_ROOT/samota"

echo "[DONE] All environments installed under $VENV_ROOT/"
