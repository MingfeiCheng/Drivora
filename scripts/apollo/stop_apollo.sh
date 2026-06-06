#!/bin/bash
# Stop the Drivora Apollo container (and its modules/bridge/injector with it).
#
#   scripts/apollo/stop_apollo.sh          # stop container (keeps it for reuse)
#   scripts/apollo/stop_apollo.sh --rm     # stop and remove the container
#
# Env: APOLLO_CTN (default: apollo_drivora)
set -euo pipefail

APOLLO_CTN="${APOLLO_CTN:-apollo_drivora}"
RM=0
[ "${1:-}" = "--rm" ] && RM=1

log() { echo -e "\033[1;36m[stop_apollo]\033[0m $*"; }

if docker inspect "${APOLLO_CTN}" >/dev/null 2>&1; then
    if [ "$(docker inspect -f '{{.State.Running}}' "${APOLLO_CTN}" 2>/dev/null)" = "true" ]; then
        log "stopping container ${APOLLO_CTN}"
        docker stop "${APOLLO_CTN}" >/dev/null
    else
        log "container ${APOLLO_CTN} already stopped"
    fi
    if [ "${RM}" -eq 1 ]; then
        log "removing container ${APOLLO_CTN}"
        docker rm "${APOLLO_CTN}" >/dev/null
    fi
else
    log "no container named ${APOLLO_CTN}"
fi
