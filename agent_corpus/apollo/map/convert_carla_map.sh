#!/bin/bash
# Convert a CARLA OpenDRIVE (.xodr) town into a complete Apollo HD map and
# install it under the Apollo tree. Drivora-owned tool — uses only the imap_box
# converter (pip) + Apollo's own map tools (run in a disposable container).
# Does NOT use anything external except the Apollo source tree itself.
#
# Usage:
#   bash convert_carla_map.sh <Town> <path/to/Town.xodr> [apollo_root]
# Example:
#   bash convert_carla_map.sh Town01 /home/$USER/carlaCache/Carla/Maps/OpenDrive/Town01.xodr
#
# Produces in agent_corpus/apollo/map/<Town>/ and apollo/modules/map/data/<Town>/:
#   base_map.{txt,bin}  routing_map.{txt,bin}  sim_map.{txt,bin}  default_end_way_point.txt
set -e

TOWN="${1:?usage: convert_carla_map.sh <Town> <xodr> [apollo_root]}"
XODR="${2:?missing xodr path}"
APOLLO_ROOT="${3:-}"
APOLLO_IMG="apolloauto/apollo:dev-x86_64-18.04-20210914_1336"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVORA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# default apollo source location: an "apollo" tree beside the Drivora repo
# (override via arg 3 or the APOLLO_ROOT env var)
APOLLO_ROOT="${APOLLO_ROOT:-$DRIVORA_ROOT/../apollo}"
OUT="$SCRIPT_DIR/$TOWN"
IMAP_VENV="$DRIVORA_ROOT/.venvs/imap"
mkdir -p "$OUT"

# 0) ensure imap_box converter venv
if [ ! -x "$IMAP_VENV/bin/imap" ]; then
    echo "[INFO] creating imap converter venv (uv)..."
    uv venv "$IMAP_VENV" --python 3.8
    VIRTUAL_ENV="$IMAP_VENV" uv pip install imap_box
fi

# 1) sanitize the CARLA geoReference (drop egm96 geoid grid → pyproj-friendly)
SAN="$OUT/$TOWN.sanitized.xodr"
"$IMAP_VENV/bin/python" - "$XODR" "$SAN" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
for bad in ("+geoidgrids=egm96_15.gtx ", "+geoidgrids=egm96_15.gtx",
            "+vunits=m ", "+geoid_crs=WGS84 "):
    s = s.replace(bad, "")
open(dst, "w", encoding="utf-8").write(s)
print("sanitized xodr ->", dst)
PY

# 2) OpenDRIVE -> Apollo base_map (txt + bin)
echo "[INFO] converting $TOWN with imap_box..."
"$IMAP_VENV/bin/imap" -f -i "$SAN" -o "$OUT/base_map.txt"

# 3) place base_map into apollo tree, generate routing_map + sim_map via Apollo tools
APMAP="$APOLLO_ROOT/modules/map/data/$TOWN"
mkdir -p "$APMAP"
cp "$OUT/base_map.bin" "$OUT/base_map.txt" "$APMAP/"

echo "[INFO] generating routing_map + sim_map in a disposable Apollo container..."
docker run --rm -v "$APOLLO_ROOT":/apollo --entrypoint bash "$APOLLO_IMG" -lc "
  set -e
  source /apollo/cyber/setup.bash 2>/dev/null || true
  export GLOG_log_dir=/tmp GLOG_logtostderr=0
  M=/apollo/modules/map/data/$TOWN
  /apollo/bazel-bin/modules/routing/topo_creator/topo_creator --map_dir=\$M
  /apollo/bazel-bin/modules/map/tools/sim_map_generator --map_dir=\$M --output_dir=\$M
  touch \$M/default_end_way_point.txt
  chown -R $(id -u):$(id -g) \$M
" 2>&1 | grep -v ttyname | tail -5

# 4) sync full map back to the Drivora-owned copy
cp "$APMAP"/routing_map.* "$APMAP"/sim_map.* "$APMAP"/default_end_way_point.txt "$OUT/"

echo "[DONE] $TOWN map ready:"
echo "  apollo tree : $APMAP"
echo "  drivora copy: $OUT"
echo "Select it at runtime with --map_dir=$APMAP (do NOT edit global_flagfile.txt)."
