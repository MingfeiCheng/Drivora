#!/bin/bash
set -euo pipefail

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=3

# ==== Tester venv (created by install_tester.sh) ====
TESTER_VENV=".venvs/random"
TESTER_PYTHON="${TESTER_VENV}/bin/python"

if [ ! -f "$TESTER_PYTHON" ]; then
    echo "[ERROR] Tester venv not found at ${TESTER_VENV}. Run: bash install_tester.sh random 0.9.15 ${TESTER_VENV}"
    exit 1
fi

# ==== Common Config ====
output_root="results/debug_pylot"
run_tag="pylot_random_debug"
max_sim_time=120.0
open_vis=true
save_agent_internal=false
distribute_num=1

# ==== Agent Config ====
agent_entry_point="agent_corpus.pylot.pylot_proxy_agent:PylotProxyAgent"
agent_config_path="agent_corpus/pylot/config/proxy_config.json"
ads_venv_dir=".venvs/pylot"

# ==== Scenario Config ====
scenario_executor_script="scenario_corpus/openscenario/run_scenario.py"

# ==== Tester / Fuzzer Config ====
tester_type="random"
tester_config_path="fuzzer/configs/debug_pylot.yaml"
time_budget=1
population_size=1

# ==== CARLA Config ====
carla_image="carlasim/carla:0.9.15"
carla_fps=20

# ==== Reminder: Pylot container must be running! ====
echo "========================================="
echo " Drivora Pylot Debug Run"
echo "========================================="
echo ""
echo "IMPORTANT: Ensure the Pylot container is running:"
echo "  docker run --gpus all -p 12667:12667 drivora/pylot:latest"
echo ""
echo "Agent: ${agent_entry_point}"
echo "Config: ${agent_config_path}"
echo "Venv: ${ads_venv_dir}"
echo ""

# ==== Run with tester venv python ====
"$TESTER_PYTHON" start_fuzzer.py \
  fuzzer_dir="fuzzer" \
  output_root="$output_root" \
  run_tag="$run_tag" \
  distribute_num="$distribute_num" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  save_agent_internal="$save_agent_internal" \
  debug=true \
  resume=true \
  carla.image="$carla_image" \
  carla.fps="$carla_fps" \
  tester.type="$tester_type" \
  tester.config_path="$tester_config_path" \
  tester.time_budget="$time_budget" \
  tester.population_size="$population_size" \
  agent.entry_point="$agent_entry_point" \
  agent.config_path="$agent_config_path" \
  agent.ads_venv_dir="$ads_venv_dir" \
  scenario.executor_script="$scenario_executor_script"
