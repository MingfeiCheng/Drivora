#!/bin/bash
set -euo pipefail

# ============================================================================
# Random fuzzing with Apollo (real LiDAR perception) on Town01.
#
# Single (PARALLEL_NUM=1, default):
#   one Apollo container brought up idempotently by scripts/apollo/start_apollo.sh
#   (reuses apollo_drivora; good for smoke tests / debugging).
#
# Parallel (PARALLEL_NUM=N>1):  multi-seed fuzzing, N CARLA + N Apollo.
#   * CARLA workers <run_tag>_<i> are created by the framework, round-robined over
#     CARLA_GPUS (via CUDA_VISIBLE_DEVICES).
#   * the N Apollo backends are managed by the framework's ApolloBackendManager
#     (Python): lifecycle + health checks + restart-on-failure + per-worker
#     real_config_<i>.json. No bash container loop, no -md (so no host :8888
#     conflict and no useless official map volumes); each Apollo runs Dreamview
#     on its own container IP (HMI works there). Browse one worker with
#     DREAMVIEW_WORKER0=true (needs host :8888 free).
#
# Env knobs:
#   PARALLEL_NUM=1        workers (= distribute_num)
#   CARLA_GPUS=0          CARLA workers round-robin these GPUs
#   APOLLO_GPUS=1         Apollo backends round-robin these GPUs
#   DREAMVIEW_WORKER0=false   publish worker0 Dreamview to host :8888 (parallel)
#   AUTO_START_APOLLO=1   single mode: bring Apollo up (0 = already up)
#   TIME_BUDGET=4         hours; 0.05 ~= smoke test
#   RESUME=true           false = ignore previous run state
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DRIVORA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${DRIVORA_ROOT}"

# ==== Common Config ====
output_root="results"
# Each run gets its OWN containers, named <run_tag>_<i> / apollo_<run_tag>_<i>.
# Concurrent campaigns MUST use distinct RUN_INDEX so they never share containers.
# RESUME=true reuses this run's own containers; RESUME=false recreates them clean.
run_index="${RUN_INDEX:-1}"
max_sim_time=600.0          # Apollo cruises slowly; give the route time to finish
open_vis=true
carla_image="carlasim/carla:0.9.15"

PARALLEL_NUM="${PARALLEL_NUM:-1}"
CARLA_GPUS="${CARLA_GPUS:-0}"
APOLLO_GPUS="${APOLLO_GPUS:-1}"

# ==== Agent Config ====
agent_name="apollo"
agent_entry_point="agent_corpus.apollo.apollo_real_agent:ApolloRealAgent"
agent_config_path="agent_corpus/apollo/config/real_config.json"
agent_venv_dir=".venvs/apollo"

# ==== Tester Config ====
tester_type="random"
tester_config_path="fuzzer/configs/random_apollo.yaml"
time_budget="${TIME_BUDGET:-4}"
resume="${RESUME:-true}"

run_tag="${tester_type}_${agent_name}_town01_run${run_index}"

# ==== Apollo backend ====
# Two ways to run Apollo:
#   single mode  -> PARALLEL_NUM<=1 AND TOPOLOGY!=multi_ads: bring up the one dev
#                   Apollo via start_apollo.sh (reuses apollo_drivora; smoke tests).
#   backend mgr  -> PARALLEL_NUM>1 (multi_seed: N CARLA x 1 Apollo) OR
#                   TOPOLOGY=multi_ads (1 CARLA world x N Apollo, one per ego):
#                   the framework's ApolloBackendManager handles lifecycle/health/
#                   restart + per-worker(/per-ego) real_config_<i>.json.
TOPOLOGY="${TOPOLOGY:-multi_seed}"
backend_arg=()
if [ "${PARALLEL_NUM}" -le 1 ] && [ "${TOPOLOGY}" != "multi_ads" ]; then
  # single mode: bring up the one dev Apollo (idempotent; reuses apollo_drivora)
  if [ "${AUTO_START_APOLLO:-1}" = "1" ]; then
    bash "${DRIVORA_ROOT}/scripts/apollo/start_apollo.sh"
  fi
else
  # backend-manager mode (multi_seed parallel OR multi_ads single-world)
  if [ "${TOPOLOGY}" = "multi_ads" ]; then
    PARALLEL_NUM=1   # multi_ads is ALWAYS a single CARLA world
  fi
  backend_arg=(
    "+agent.backend.entry_point=agent_corpus.apollo.apollo_manager:ApolloBackendManager"
    "+agent.backend.apollo_gpus=[${APOLLO_GPUS}]"
    "+agent.backend.dreamview_worker0=${DREAMVIEW_WORKER0:-false}"
    "+agent.backend.topology=${TOPOLOGY}"
  )
  # Browse EACH backend's Dreamview on localhost:<DV_HOST_PORT + idx> via an
  # in-process TCP relay (no docker host-port publish). e.g. DV_HOST_PORT=8890:
  #   multi_seed PARALLEL_NUM=2 -> worker0 :8890, worker1 :8891
  #   multi_ads  NUM_ADS=2      -> ADS0    :8890, ADS1    :8891
  if [ -n "${DV_HOST_PORT:-}" ]; then
    backend_arg+=("+agent.backend.dreamview_host_port=${DV_HOST_PORT}")
  fi
  # multi_ads (1 CARLA world, N Apollo, one per ego): set TOPOLOGY=multi_ads
  # NUM_ADS=N. The runner auto-forces ego_space.num=N so the scenario spawns N
  # egos (ego k -> ADS k); each Apollo gets its own sensor stream.
  if [ "${TOPOLOGY}" = "multi_ads" ]; then
    backend_arg+=("+agent.backend.num_ads=${NUM_ADS:-2}")
  fi
fi

# CARLA workers use these GPUs (get_available_gpus respects CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES="${CARLA_GPUS}"

# ==== Fuzzer python ====
FUZZER_PY="${FUZZER_PY:-${DRIVORA_ROOT}/.venvs/apollo/bin/python}"

# ==== Run (Hydra style overrides) ====
"$FUZZER_PY" start_fuzzer.py \
  output_root="$output_root" \
  distribute_num="$PARALLEL_NUM" \
  run_tag="$run_tag" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  tester.type="$tester_type" \
  tester.config_path="$tester_config_path" \
  tester.time_budget="$time_budget" \
  resume="$resume" \
  agent.entry_point="$agent_entry_point" \
  agent.config_path="$agent_config_path" \
  agent.ads_venv_dir="$agent_venv_dir" \
  scenario.seed_path="" \
  carla.image="'${carla_image}'" \
  "${backend_arg[@]}"
