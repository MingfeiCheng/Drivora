#!/bin/bash
# ============================================================================
# Idempotent Apollo bring-up for Drivora (real LiDAR perception, Town01).
#
# Brings up everything the ApolloRealAgent needs and is safe to call repeatedly
# (the fuzzer scripts call it on every run):
#   1. Apollo container   (via apollo/docker/scripts/dev_start_ctn.sh; bridge net)
#   2. Dreamview          (bootstrap.sh start; HTTP/ws on :8888, published to host)
#   3. global_flagfile.txt cleaned to a single --map_dir=.../Town01
#   4. cyber_bridge       (:9090, scripts/bridge.sh)
#   5. sensor injector    (:9100, drivora_injector.py inside the container)
#   6. modules            (Dreamview HMI: full real-sensor stack + perception pack)
#   7. patch agent_corpus/apollo/config/real_config.json -> apollo_host = ctn IP
#
# If the container + dreamview + bridge are already healthy it SKIPS the slow
# steps and only re-asserts modules/injector/config.
#
# Env overrides:
#   APOLLO_CTN       container name           (default: apollo_drivora)
#   APOLLO_USER      in-container user account (default: = APOLLO_CTN). Override
#                    when reusing a container whose internal user differs from
#                    its name, e.g. APOLLO_CTN=apollo_dev_myuser APOLLO_USER=myuser
#   APOLLO_ROOT      apollo source tree       (default: <repo>/apollo)
#   APOLLO_MAP       HD map                   (default: Town01)
#   APOLLO_GPU       NVIDIA_VISIBLE_DEVICES   (default: all)
#   USE_DREAMVIEW    pass -md to dev_start_ctn.sh (mount map volumes + publish
#                    :8888 for the Dreamview web UI). true|false (default: true).
#                    Module startup goes through the Dreamview HMI websocket, so
#                    this must be true for the HMI path to work.
# ============================================================================
set -euo pipefail

# ---- paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DRIVORA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
REPO_ROOT="$(cd "${DRIVORA_ROOT}/.." && pwd -P)"

APOLLO_CTN="${APOLLO_CTN:-apollo_drivora}"
# In-container user account for `docker exec --user` (and the USER passed to
# dev_start_ctn.sh, which names both the container AND the account after it).
# Defaults to the container name (matches reference container.py: user == name).
# Override when REUSING a container whose internal user differs from its name,
# e.g. APOLLO_CTN=apollo_dev_myuser APOLLO_USER=myuser.
APOLLO_USER="${APOLLO_USER:-${APOLLO_CTN}}"
APOLLO_ROOT="${APOLLO_ROOT:-${REPO_ROOT}/apollo}"
APOLLO_MAP="${APOLLO_MAP:-Town01}"
# Apollo runs perception on THIS GPU; keep it OFF CARLA's GPU (default 0) so
# CARLA rendering and Apollo's lidar/camera detection don't contend on one GPU
# (that contention jittered the perception cycle -> fusion dropouts). 4x A5000
# here, GPUs 1-3 idle -> put Apollo on GPU 1.
APOLLO_GPU="${APOLLO_GPU:-1}"
USE_DREAMVIEW="${USE_DREAMVIEW:-true}"

DV_PORT=8888
BRIDGE_PORT=9090
INJECTOR_PORT=9100

# REAL_CONFIG is the agent config this Apollo's IP/ports get patched into. Make
# it overridable so a parallel bring-up can write one per worker (real_config_<i>.json).
REAL_CONFIG="${REAL_CONFIG:-${DRIVORA_ROOT}/agent_corpus/apollo/config/real_config.json}"
INJECTOR_SRC="${DRIVORA_ROOT}/agent_corpus/apollo/bridge/incontainer_sensor_injector.py"
HMI_PY="${SCRIPT_DIR}/hmi_start_modules.py"
VENV_PY="${DRIVORA_ROOT}/.venvs/apollo/bin/python"

log()  { echo -e "\033[1;36m[start_apollo]\033[0m $*"; }
warn() { echo -e "\033[1;33m[start_apollo]\033[0m $*"; }
die()  { echo -e "\033[1;31m[start_apollo]\033[0m $*" >&2; exit 1; }

[ -d "${APOLLO_ROOT}" ] || die "APOLLO_ROOT not found: ${APOLLO_ROOT}"
[ -x "${VENV_PY}" ]     || die ".venvs/apollo python not found: ${VENV_PY} (run: uv sync for the apollo venv)"

ctn_exists()  { docker inspect "${APOLLO_CTN}" >/dev/null 2>&1; }
ctn_running() { [ "$(docker inspect -f '{{.State.Running}}' "${APOLLO_CTN}" 2>/dev/null)" = "true" ]; }
ctn_ip()      { docker inspect -f '{{.NetworkSettings.IPAddress}}' "${APOLLO_CTN}" 2>/dev/null; }
port_open()   { timeout 2 bash -c ">/dev/tcp/$1/$2" 2>/dev/null; }   # $1=host $2=port
dv_http_ok()  { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$1:${DV_PORT}" 2>/dev/null)" = "200" ]; }

# ---------------------------------------------------------------- 1. container
if ctn_running; then
    log "container '${APOLLO_CTN}' already running"
elif ctn_exists; then
    log "starting existing container '${APOLLO_CTN}'"
    docker start "${APOLLO_CTN}" >/dev/null
    sleep 2
else
    DEV_OPTS="-y -l"
    if [ "${USE_DREAMVIEW}" = "true" ]; then
        DEV_OPTS="${DEV_OPTS} -md"   # mount map volumes + publish :8888 for Dreamview
    fi
    log "creating container '${APOLLO_CTN}' via dev_start_ctn.sh ${DEV_OPTS} (bridge net) ..."
    ( cd "${APOLLO_ROOT}/docker/scripts" && \
      CURR_DIR="${APOLLO_ROOT}/docker/scripts" \
      APOLLO_ROOT_DIR="${APOLLO_ROOT}" \
      USER="${APOLLO_USER}" \
      DISPLAY="${DISPLAY:-}" \
      bash "${APOLLO_ROOT}/docker/scripts/dev_start_ctn.sh" ${DEV_OPTS} )
    sleep 3
fi
ctn_running || die "container '${APOLLO_CTN}' failed to start"

IP="$(ctn_ip)"
[ -n "${IP}" ] || IP="127.0.0.1"
log "container IP = ${IP}"

# in-container exec helper (runs with apollo env sourced)
dexec()  { docker exec --user "${APOLLO_USER}" "${APOLLO_CTN}" bash -c "source /apollo/scripts/apollo_base.sh >/dev/null 2>&1; $*"; }
dexecd() { docker exec -d --user "${APOLLO_USER}" "${APOLLO_CTN}" bash -c "source /apollo/scripts/apollo_base.sh >/dev/null 2>&1; $*"; }

# ---------------------------------------------- 1b. fix /apollo/data ownership
# /apollo is a SHARED bind-mount; a previous root-running container can leave
# /apollo/data and kv_db.sqlite owned by root, which makes dreamview fail with
# "Permission denied" / "attempt to write a readonly database". Re-own to the
# in-container user (same uid as the host owner -> safe).
log "ensuring /apollo/data is writable by ${APOLLO_USER}"
docker exec --user root "${APOLLO_CTN}" bash -c \
  "mkdir -p /apollo/data/log /apollo/data/core /apollo/data/bag /apollo/records; \
   chown -R ${APOLLO_USER} /apollo/data /apollo/records 2>/dev/null; true" || true

# ------------------------------------------------ 2. clean global_flagfile (Town01)
log "asserting clean global_flagfile.txt (single --map_dir=${APOLLO_MAP})"
FLAGFILE="${APOLLO_ROOT}/modules/common/data/global_flagfile.txt"
if [ -f "${FLAGFILE}" ] && [ "$(grep -c -- '--map_dir' "${FLAGFILE}")" -ne 1 ]; then
    [ -f "${FLAGFILE}.drivora_bak" ] || cp "${FLAGFILE}" "${FLAGFILE}.drivora_bak"
    grep -v -- '--map_dir' "${FLAGFILE}.drivora_bak" 2>/dev/null > "${FLAGFILE}" || true
    printf '\n--map_dir=/apollo/modules/map/data/%s\n' "${APOLLO_MAP}" >> "${FLAGFILE}"
    warn "global_flagfile had duplicate map_dir lines -> rewrote to single ${APOLLO_MAP}"
fi

# ----------------------------------------------------------------- 3. dreamview
# Check ONLY this container's own IP — never 127.0.0.1. With multiple Apollo
# containers, 127.0.0.1:8888 would hit whichever container published the host
# port (apollo_drivora), making us wrongly think THIS container's dreamview is
# up and skip starting it (then HMI would drive the wrong container).
if dv_http_ok "${IP}"; then
    log "dreamview already serving on ${IP}:${DV_PORT}"
else
    # kill any crash-looping dreamview (e.g. started before the perms fix) so the
    # restart picks up the now-writable data dir
    dexec "pkill -f 'modules/dreamview/dreamview' 2>/dev/null; pkill -f dreamview.launch 2>/dev/null; true" || true
    sleep 1
    log "starting dreamview (modules pinned to GPU ${APOLLO_GPU}) ..."
    # dreamview launches the perception modules as children -> they inherit
    # CUDA_VISIBLE_DEVICES, pinning Apollo perception to GPU ${APOLLO_GPU}
    # (separate from CARLA's GPU).
    dexecd "export CUDA_VISIBLE_DEVICES=${APOLLO_GPU} NVIDIA_VISIBLE_DEVICES=${APOLLO_GPU}; /apollo/scripts/bootstrap.sh start"
fi
# wait for dreamview to actually serve HTTP 200 (TCP-accept happens earlier than
# the websocket route being ready)
log "waiting for dreamview HTTP 200 ..."
for _ in $(seq 1 40); do
    if dv_http_ok "${IP}"; then break; fi
    sleep 1.5
done
dv_http_ok "${IP}" || warn "dreamview not serving HTTP 200 on ${IP}:${DV_PORT} yet"

# ------------------------------------------------------------------- 4. bridge
if port_open "${IP}" "${BRIDGE_PORT}"; then
    log "cyber_bridge already up on ${IP}:${BRIDGE_PORT}"
else
    log "starting cyber_bridge ..."
    dexecd "/apollo/scripts/bridge.sh"
    for _ in $(seq 1 20); do port_open "${IP}" "${BRIDGE_PORT}" && break; sleep 1; done
    port_open "${IP}" "${BRIDGE_PORT}" || warn "cyber_bridge not reachable on ${IP}:${BRIDGE_PORT}"
fi

# ----------------------------------------------------------------- 5. injector
log "copying sensor injector into container -> /apollo/drivora_injector.py"
docker cp "${INJECTOR_SRC}" "${APOLLO_CTN}:/apollo/drivora_injector.py" >/dev/null
# One injector PROCESS per sensor (separate python interpreters -> separate GILs)
# so the camera jpeg encode never starves the lidar PointCloud build (that GIL
# contention caused the perception detection flicker).
INJECTOR_PORT_CAM="${INJECTOR_PORT_CAM:-9101}"
INJECTOR_PORT_CAM12="${INJECTOR_PORT_CAM12:-9102}"
start_injector() {  # $1=port $2=sensor $3=logsuffix $4=cam_name(optional)
    if port_open "${IP}" "$1"; then
        log "injector[$3] already up on ${IP}:$1"
    else
        log "starting injector[$3] (:$1, $2 $4) ..."
        dexecd "export NVIDIA_VISIBLE_DEVICES=${APOLLO_GPU} CUDA_VISIBLE_DEVICES=${APOLLO_GPU}; python3 /apollo/drivora_injector.py $1 $2 $4 > /apollo/data/log/drivora_injector_$3.log 2>&1"
        for _ in $(seq 1 20); do port_open "${IP}" "$1" && break; sleep 1; done
        port_open "${IP}" "$1" || warn "injector[$3] not reachable on ${IP}:$1 (see /apollo/data/log/drivora_injector_$3.log)"
    fi
}
start_injector "${INJECTOR_PORT}"       "lidar"  "lidar"  ""
start_injector "${INJECTOR_PORT_CAM}"   "camera" "cam6"   "front_6mm"
start_injector "${INJECTOR_PORT_CAM12}" "camera" "cam12"  "front_12mm"

# ------------------------------------------------------------ 6. modules (HMI)
# HMI must talk to THIS container's dreamview (its own IP), never 127.0.0.1
# (which would be another container's published port).
DV_WS="ws://${IP}:${DV_PORT}/websocket"
log "starting modules via Dreamview HMI (${DV_WS}) ..."
hmi_ok=0
for hmi_try in 1 2; do
    if "${VENV_PY}" "${HMI_PY}" --url "${DV_WS}" --map "${APOLLO_MAP}" \
        --mode "Mkz Standard Debug" --vehicle "Mkz Example"; then
        hmi_ok=1; break
    fi
    warn "HMI module start attempt ${hmi_try} did not bring up all critical modules; retrying ..."
    sleep 3
done
[ "${hmi_ok}" = "1" ] || warn "modules still not all up — check Dreamview (${DV_WS%/websocket} via browser) and /apollo/data/log"

# --------------------------------------------------- 7. patch real_config.json
log "patching ${REAL_CONFIG##*/} -> apollo_host=${IP}"
"${VENV_PY}" - "$REAL_CONFIG" "$IP" "$BRIDGE_PORT" "$INJECTOR_PORT" <<'PY'
import json, sys
path, ip, bport, iport = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(path) as f: cfg = json.load(f)
cfg["apollo_host"] = ip
cfg["bridge_port"] = bport
cfg["injector_port"] = iport
with open(path, "w") as f: json.dump(cfg, f, indent=2)
print(f"  apollo_host={ip} bridge_port={bport} injector_port={iport}")
PY

log "Apollo ready."
log "  container : ${APOLLO_CTN}  (user ${APOLLO_USER}, ip ${IP})"
log "  dreamview : http://localhost:${DV_PORT}  (browser)"
log "  bridge    : ${IP}:${BRIDGE_PORT}   injector: ${IP}:${INJECTOR_PORT}"
