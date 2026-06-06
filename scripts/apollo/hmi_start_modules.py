#!/usr/bin/env python
"""Start Apollo modules through the Dreamview HMI websocket.

Run from the HOST (uses .venvs/apollo which has websocket-client). Talks to
Dreamview at ws://<host>:<port>/websocket. Sets mode/map/vehicle, then issues
START_MODULE for the full real-sensor stack, and finally polls HMIStatus to
report which modules came up.

Usage:
    python hmi_start_modules.py --url ws://127.0.0.1:8888/websocket \
        --mode "Mkz Standard Debug" --map Town01 --vehicle "Mkz Example"
"""
import argparse
import json
import time
import sys

import websocket  # from websocket-client

# Full real-sensor stack (skip Canbus/Recorder/Camera/Radar/Velodyne/GPS/etc —
# no hardware; Perception uses the full pack dag_streaming_perception.dag).
MODULES = [
    "Transform", "Localization", "Routing", "Perception", "Traffic Light",
    "Prediction", "Planning", "Control", "Storytelling", "Guardian",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8888/websocket")
    ap.add_argument("--mode", default="Mkz Standard Debug")
    ap.add_argument("--map", default="Town01")
    ap.add_argument("--vehicle", default="Mkz Example")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="seconds to wait for modules to report running")
    args = ap.parse_args()

    # Dreamview may accept TCP before its /websocket route is ready right after
    # container/dreamview startup -> retry the handshake.
    ws = None
    last_err = None
    for attempt in range(20):
        try:
            ws = websocket.create_connection(args.url, timeout=15)
            break
        except Exception as e:  # ConnectionReset/refused while dreamview warms up
            last_err = e
            print(f"[hmi] dreamview not ready (attempt {attempt + 1}/20): {e}")
            time.sleep(3.0)
    if ws is None:
        print(f"[hmi] ERROR: could not connect to {args.url}: {last_err}")
        return 1

    def send(action, value=None, pause=0.6):
        msg = {"type": "HMIAction", "action": action}
        if value is not None:
            msg["value"] = value
        ws.send(json.dumps(msg))
        time.sleep(pause)

    def current_modules():
        """Read the latest HMIStatus module on/off map (empty dict if none)."""
        ws.settimeout(2.0)
        latest = {}
        end = time.time() + 4
        while time.time() < end:
            try:
                raw = ws.recv()
            except Exception:
                break
            if isinstance(raw, str):
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                if d.get("type") == "HMIStatus":
                    latest = d.get("data", {}).get("modules", {})
                    break
        return latest

    print(f"[hmi] mode={args.mode!r} map={args.map!r} vehicle={args.vehicle!r}")
    send("CHANGE_MODE", args.mode, pause=1.5)
    send("CHANGE_MAP", args.map, pause=2.0)
    send("CHANGE_VEHICLE", args.vehicle, pause=1.5)
    time.sleep(2.0)

    # Only START modules that are currently OFF. Re-issuing START_MODULE for an
    # already-running module can launch a SECOND instance -> duplicate cyber node
    # (node_manager: "duplicated node[RecognitionComponent]") -> the perception
    # process is terminated and dumps a multi-GB core, repeatedly. This guard is
    # what makes the bring-up safe to call idempotently on every run.
    running = current_modules()
    for mod in MODULES:
        if running.get(mod, False):
            print(f"[hmi] {mod} already running, skip")
            continue
        print(f"[hmi] START_MODULE {mod}")
        send("START_MODULE", mod, pause=0.8)

    # poll HMIStatus for module readiness
    deadline = time.time() + args.wait
    ws.settimeout(2.0)
    last = {}
    while time.time() < deadline:
        try:
            raw = ws.recv()
        except Exception:
            continue
        if not isinstance(raw, str):
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        if d.get("type") != "HMIStatus":
            continue
        modules = d.get("data", {}).get("modules", {})
        last = modules
        on = [m for m, v in modules.items() if v]
        off = [m for m in MODULES if not modules.get(m, False)]
        print(f"[hmi] ON={sorted(on)}")
        if not off:
            print("[hmi] all requested modules running")
            ws.close()
            return 0
        time.sleep(1.0)

    off = [m for m in MODULES if not last.get(m, False)]
    print(f"[hmi] WARNING: modules still off after {args.wait}s: {off}")
    ws.close()
    # Critical modules that must be running for the agent to drive. (Traffic
    # Light has no camera input here, so don't gate on it.)
    critical = [m for m in ("Transform", "Localization", "Routing",
                            "Perception", "Prediction", "Planning", "Control")
                if not last.get(m, False)]
    if critical:
        print(f"[hmi] ERROR: critical modules not running: {critical}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
