#!/bin/bash
# ============================================================================
# Debug the Apollo <-> CARLA connection by running ONE known-good scenario
# end-to-end, bypassing the fuzzer entirely.
#
# It runs the run25-proven Town01 straight-corridor scenario through Drivora's
# single-scenario executor (scripts/run_scenario.py), which:
#   - starts a CARLA container and connects (carla python API)
#   - spawns the ego with the ApolloRealAgent (real LiDAR perception)
#   - the agent connects to Apollo bridge:9090 + injector:9100, feeds sensors,
#     sends routing, and drives with Apollo's ControlCommand
#   - records a bird/3rd-person video + writes a result
#
# If THIS works, the Apollo<->CARLA path is healthy and any "nothing happens"
# is in the fuzzer/sampling layer. If it fails, the stage where it stops tells
# you whether it's CARLA, the bridge, sensors, or control.
#
# Env:
#   AUTO_START_APOLLO=0   skip the Apollo bring-up (you started it yourself)
#   MAX_SIM_TIME=60       scenario time budget (s); default 120
#   APOLLO_CTN / APOLLO_USER ...  passed through to start_apollo.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DRIVORA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${DRIVORA_ROOT}"

PY="${DRIVORA_ROOT}/.venvs/apollo/bin/python"
SCENARIO="${SCENARIO:-${DRIVORA_ROOT}/agent_corpus/apollo/debug/20260527_215927/scn_real/scenario.json}"
CTN_CONFIG="${DRIVORA_ROOT}/agent_corpus/apollo/debug/scn_debug/ctn_config.json"
OUT_DIR="${DRIVORA_ROOT}/agent_corpus/apollo/debug/scn_debug/run"
MAX_SIM_TIME="${MAX_SIM_TIME:-120}"

log() { echo -e "\033[1;35m[debug_scenario]\033[0m $*"; }

[ -x "${PY}" ]        || { echo "missing ${PY}"; exit 1; }
[ -f "${SCENARIO}" ]  || { echo "missing scenario ${SCENARIO}"; exit 1; }

# ---- 1. Apollo up (idempotent) ----
if [ "${AUTO_START_APOLLO:-1}" = "1" ]; then
    log "ensuring Apollo is up ..."
    bash "${SCRIPT_DIR}/start_apollo.sh"
else
    log "AUTO_START_APOLLO=0 -> assuming Apollo already running"
fi

# ---- 2. show what the agent will connect to ----
log "agent real_config.json:"
"${PY}" -c "import json; print('   ', json.load(open('${DRIVORA_ROOT}/agent_corpus/apollo/config/real_config.json')))"

mkdir -p "${OUT_DIR}"
log "running scenario (max_sim_time=${MAX_SIM_TIME}s) -> ${OUT_DIR}"
log "scenario: ${SCENARIO##*/}   ctn: $(basename "${CTN_CONFIG}")"

# ---- 3. run the single scenario ----
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
[ -f "${OUT_DIR}/result.txt" ]            && { log "result.txt:"; sed 's/^/   /' "${OUT_DIR}/result.txt"; }
VID="$(find "${OUT_DIR}" -name '*.mp4' 2>/dev/null | head -1)"
[ -n "${VID}" ] && log "video: ${VID}"
log "full logs under: ${OUT_DIR}"
exit ${rc}
