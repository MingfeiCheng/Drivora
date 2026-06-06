#!/bin/bash
# ============================================================================
# Demo: ONE scenario, ONE Apollo, with interactive NPC vehicles.
#
# Runs a single fixed Town01 scenario end-to-end through Drivora's single-
# scenario executor (no fuzzer): the ego is driven by Baidu Apollo with REAL
# LiDAR/camera/GNSS perception (no ground-truth), and two NPC vehicles share the
# junction so you can watch Apollo perceive and react to them. Dreamview is
# published so you can watch perception/planning live.
#
# Env knobs:
#   CARLA_GPU=1          GPU for the CARLA server
#   APOLLO_GPU=2         GPU for Apollo (keep it DIFFERENT from CARLA_GPU:
#                        sharing one GPU lets Apollo's perception starve CARLA)
#   MAX_SIM_TIME=180     scenario time budget in seconds
#   AUTO_START_APOLLO=1  bring Apollo up (set 0 if you started it yourself)
#
# Prereqs: Apollo built + agent venv installed
#   (bash install_ads_eval.sh apollo 0.9.15 .venvs/apollo). See
#   agent_corpus/apollo/README.md.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DRIVORA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${DRIVORA_ROOT}"

# ==== Config ====
CARLA_GPU="${CARLA_GPU:-1}"
APOLLO_GPU="${APOLLO_GPU:-2}"
MAX_SIM_TIME="${MAX_SIM_TIME:-180}"

PY="${DRIVORA_ROOT}/.venvs/apollo/bin/python"
SCENARIO="${DRIVORA_ROOT}/agent_corpus/apollo/demo/scenario_demo.json"
REAL_CONFIG="${DRIVORA_ROOT}/agent_corpus/apollo/demo/demo_config.json"
OUT_DIR="${DRIVORA_ROOT}/results/demo_apollo"
CTN_CONFIG="${OUT_DIR}/ctn_config.json"

log() { echo -e "\033[1;35m[demo_apollo]\033[0m $*"; }

[ -x "${PY}" ]       || { echo "[ERROR] agent venv missing: ${PY}  (run: bash install_ads_eval.sh apollo 0.9.15 .venvs/apollo)"; exit 1; }
[ -f "${SCENARIO}" ] || { echo "[ERROR] demo scenario missing: ${SCENARIO}"; exit 1; }
[ -f "${REAL_CONFIG}" ] || { echo "[ERROR] demo config missing: ${REAL_CONFIG}"; exit 1; }
mkdir -p "${OUT_DIR}"

# ==== 1. bring up a single Apollo (idempotent; reuses apollo_drivora) ====
if [ "${AUTO_START_APOLLO:-1}" = "1" ]; then
    log "bringing up Apollo on GPU ${APOLLO_GPU} (Dreamview published) ..."
    APOLLO_GPU="${APOLLO_GPU}" USE_DREAMVIEW=true REAL_CONFIG="${REAL_CONFIG}" \
        bash "${SCRIPT_DIR}/apollo/start_apollo.sh"
else
    log "AUTO_START_APOLLO=0 -> assuming Apollo is already running"
fi

# ==== 2. write the CARLA container config for this demo ====
"${PY}" - "$CTN_CONFIG" "$CARLA_GPU" <<'PYEOF'
import json, sys
path, gpu = sys.argv[1], int(sys.argv[2])
json.dump({
    "idx": 0,
    "container_name": "drivora_apollo_demo",
    "gpu": gpu,
    "random_seed": 42,
    "docker_image": "carlasim/carla:0.9.15",
    "fps": 20,
    "is_sync_mode": True,
}, open(path, "w"), indent=2)
PYEOF

log "scenario : ${SCENARIO##*/}  (ego=Apollo + 2 interactive NPC vehicles)"
log "CARLA GPU ${CARLA_GPU} | Apollo GPU ${APOLLO_GPU} | max_sim_time ${MAX_SIM_TIME}s"
log "Dreamview: http://localhost:8888   (watch perception/planning live)"
log "output   : ${OUT_DIR}"

# ==== 3. run the single scenario ====
set +e
"${PY}" scripts/run_scenario.py \
    --scenario_entry_point "scenario_corpus.openscenario.scenario:OpenScenario" \
    --scenario_config "${SCENARIO}" \
    --ctn_config "${CTN_CONFIG}" \
    --scenario_dir "${OUT_DIR}" \
    --manager_name default \
    --max_sim_time "${MAX_SIM_TIME}" \
    --open_vis
rc=$?
set -e

echo ""
log "exit code: ${rc}"
[ -f "${OUT_DIR}/simulation_status.txt" ] && log "status: $(cat "${OUT_DIR}/simulation_status.txt")"
[ -f "${OUT_DIR}/result.txt" ] && { log "result:"; sed 's/^/   /' "${OUT_DIR}/result.txt"; }
VID="$(find "${OUT_DIR}" -name '*.mp4' 2>/dev/null | head -1)"
[ -n "${VID}" ] && log "video: ${VID}"
log "full logs under: ${OUT_DIR}"
exit ${rc}
