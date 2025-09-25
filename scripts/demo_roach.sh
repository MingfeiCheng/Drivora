#!/bin/bash
set -euo pipefail

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=2,3   # 限制只使用 GPU0

# ==== Common Config ====
output_root="results"
run_tag="random_roach"
max_sim_time=600.0
open_vis=true

# ==== Agent Config ====
agent_entry_point="agent_corpus.roach.agent:RoachAgent"
agent_config_path="agent_corpus/roach/config/config_agent.yaml"

# ==== Scenario Config ====
scenario_type="open_scenario"
scenario_seed_path="scenario_datasets/open_scenario/Town01/route_50_100/Town01_0001.json"

# ==== Tester Config ====
tester_type="random"
tester_config_path="fuzzer/open_scenario/random/configs/open_scenario.yaml"

# ==== Run (Hydra style overrides) ====
python start_fuzzer.py \
  output_root="$output_root" \
  run_tag="$run_tag" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  tester.type="$tester_type" \
  tester.config_path="$tester_config_path" \
  agent.entry_point="$agent_entry_point" \
  agent.config_path="$agent_config_path" \
  scenario.type="$scenario_type" \
  scenario.seed_path="$scenario_seed_path"
