#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"  # source_code/

echo "========================================"
echo " Drivora Pylot Docker Build"
echo " Context: $SOURCE_DIR"
echo "========================================"

# ── Stage 1: Build base pylot image (ERDOS + Pylot + models) ──
# Skip if base image already exists or was pulled
if ! docker image inspect erdosproject/pylot:latest &>/dev/null; then
    echo ""
    echo "[1/2] Building base image: erdosproject/pylot ..."
    docker build \
        -t erdosproject/pylot:latest \
        -f "$SCRIPT_DIR/Dockerfile" \
        "$SOURCE_DIR"
else
    echo "[1/2] Base image erdosproject/pylot:latest already exists — skipping."
fi

# ── Stage 2: Overlay with Drivora server ──
echo ""
echo "[2/2] Building server image: drivora/pylot ..."
docker build \
    -t drivora/pylot:latest \
    --build-arg BASE_IMAGE=erdosproject/pylot:latest \
    -f "$SCRIPT_DIR/Dockerfile.drivora" \
    "$SOURCE_DIR"

echo ""
echo "========================================"
echo " Build complete!"
echo "   drivora/pylot:latest"
echo "========================================"
echo ""
echo "Run:"
echo "  docker run --gpus all -p 12667:12667 drivora/pylot:latest"
