#!/bin/bash
set -euo pipefail

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=2,3 

# ==== Common Config ====
output_root="results"
run_index=1
max_sim_time=600.0
open_vis=true
distribute_num=2  # Number of distributed execution instances
carla_image="carlasim/carla:0.9.15"

# ==== Agent Config ====
agent_name="vad"
agent_entry_point="agent_corpus.uniad_vad.vad_b2d_agent:VadAgent"
agent_config_path="agent_corpus/uniad_vad/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+agent_corpus/uniad_vad/Bench2DriveZoo/vad_b2d_base.pth"

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
  distribute_num="$distribute_num" \
  run_tag="$run_tag" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  tester.type="$tester_type" \
  tester.config_path="$tester_config_path" \
  agent.entry_point="$agent_entry_point" \
  agent.config_path="$agent_config_path" \
  scenario.type="$scenario_type" \
  scenario.seed_path="$scenario_seed_path" \
  carla.image="'${carla_image}'"
