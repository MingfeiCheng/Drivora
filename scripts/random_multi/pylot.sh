#!/bin/bash
set -euo pipefail

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=0

# ==== Tester venv ====
TESTER_VENV=".venvs/random"
TESTER_PYTHON="${TESTER_VENV}/bin/python"

if [ ! -f "$TESTER_PYTHON" ]; then
    echo "[ERROR] Tester venv not found at ${TESTER_VENV}."
    exit 1
fi

# ==== Common Config ====
output_root="results/random_multi_pylot"
run_tag="random_multi_pylot"
max_sim_time=120.0
open_vis=false
save_agent_internal=false
distribute_num=1

# ==== Agent Config ====
agent_entry_point="agent_corpus.pylot.pylot_proxy_agent:PylotProxyAgent"
agent_config_path="agent_corpus/pylot/config/proxy_config.json"
ads_venv_dir=".venvs/pylot"

# ==== Scenario Config ====
scenario_executor_script="scenario_corpus/openscenario/run_scenario.py"

# ==== Tester / Fuzzer Config ====
tester_type="random_multi"
tester_config_path="fuzzer/configs/debug_random_multi.yaml"
time_budget=4
population_size=4

# ==== CARLA Config ====
carla_image="carlasim/carla:0.9.15"
carla_fps=20

# ==== Run ====
"$TESTER_PYTHON" start_fuzzer.py \
  fuzzer_dir="fuzzer" \
  output_root="$output_root" \
  run_tag="$run_tag" \
  distribute_num="$distribute_num" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  save_agent_internal="$save_agent_internal" \
  debug=false \
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
