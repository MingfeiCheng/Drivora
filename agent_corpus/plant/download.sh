#!/usr/bin/env bash

if [ -d "checkpoints" ] && [ -n "$(ls -A checkpoints 2>/dev/null)" ]; then
    echo "[INFO] Checkpoints already exist, skipping download."
    exit 0
fi

echo "[INFO] Downloading PlanT checkpoints..."
wget -c https://s3.eu-central-1.amazonaws.com/avg-projects/plant/checkpoints.zip
unzip -q checkpoints.zip
rm -f checkpoints.zip
