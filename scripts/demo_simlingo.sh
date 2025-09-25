#!/bin/bash
set -euo pipefail

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=2,3   # 限制只使用 GPU0

# ==== Common Config ====
output_root="results"
run_index=1
max_sim_time=600.0
open_vis=true

# Carla
carla_image="carlasim/carla:0.9.15"

# ==== Agent Config ====
agent_name="simlingo"
agent_entry_point="agent_corpus.simlingo.agent_simlingo:LingoAgent"
agent_config_path="agent_corpus/simlingo/checkpoint/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

# ==== Scenario Config ====
seed_segment="route_100_200"
seed_id="Town01_0001"
scenario_type="open_scenario"
scenario_seed_path="scenario_datasets/open_scenario/0.9.15/${seed_segment}/${seed_id}.json"

# ==== Tester Config ====
tester_type="random"
tester_config_path="fuzzer/open_scenario/random/configs/open_scenario.yaml"

run_tag="${tester_type}_${agent_name}_${seed_segment}_${seed_id}_run${run_index}"

# ==== Run (Hydra style overrides) ====
python start_fuzzer.py \
  output_root="$output_root" \
  run_tag="$run_tag" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  tester.type="$tester_type" \
  tester.config_path="$tester_config_path" \
  agent.entry_point="$agent_entry_point" \
  agent.config_path="'${agent_config_path}'" \
  scenario.type="$scenario_type" \
  scenario.seed_path="$scenario_seed_path" \
  carla.image="'${carla_image}'"
