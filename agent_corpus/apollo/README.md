# Apollo agent for Drivora (CARLA + real-sensor perception)

Runs **Baidu Apollo** as an ADS under test inside Drivora's CARLA framework.
CARLA camera / LiDAR / IMU / GNSS are forwarded to Apollo over CyberBridge;
Apollo runs its **own perception and localization** (no ground-truth injection),
and its `ControlCommand` is converted back to `carla.VehicleControl`.

The CyberBridge layer under `bridge/` is an independent, self-contained rewrite;
only the Apollo source tree is shared, bind-mounted by the Apollo container at
runtime.

## Layout

```
apollo/
├── apollo_real_agent.py    # ApolloRealAgent — real-sensor agent (entry point)
├── apollo_manager.py       # Apollo container lifecycle / health / restart
├── transform.py            # CARLA ↔ Apollo coordinate / sensor conversions
├── config/real_config.json # agent config template (host / ports / sensors)
├── demo/                   # self-contained single-scenario demo
│   ├── scenario_demo.json  #   Town01 scenario: ego + 2 interactive NPC vehicles
│   └── demo_config.json    #   agent config used by the demo
├── map/Town01/             # Apollo HD map for the town under test
└── bridge/                 # self-contained CyberBridge layer
    ├── cyber_bridge.py  container.py  dreamview.py  messenger.py
    ├── publishers/         # camera, lidar, imu, gnss, chassis, routing, control_pad
    └── subscribers/        # control
```

## Prerequisites

1. **Build Apollo once** (heavy, one-time) and have the Apollo source tree
   available on the host (bind-mounted into the container).
2. **HD map** for your CARLA town installed under the Apollo map data dir
   (`map/Town01/` here mirrors what the container loads).
3. **Install agent deps**: `bash install_ads_eval.sh apollo 0.9.15 .venvs/apollo`
   (uv-based; see [docs/installation.md](../../docs/installation.md)).

## Quick start — demo

Run **one scenario** with **one Apollo** and interactive NPC vehicles. The ego is
driven by Apollo with real LiDAR/camera/GNSS perception; two NPC vehicles share
the junction so you can watch Apollo perceive and react. Dreamview is published
so you can watch perception/planning live.

```bash
CARLA_GPU=1 APOLLO_GPU=2 MAX_SIM_TIME=180 bash scripts/demo_apollo.sh
```

- `CARLA_GPU` / `APOLLO_GPU` — keep CARLA and Apollo on **different** GPUs
  (sharing one GPU lets Apollo's perception saturate it and starve CARLA's tick).
- `MAX_SIM_TIME` — scenario time budget in seconds.
- Dreamview: `http://localhost:8888`.
- Output (logs, video, result) is written to `results/demo_apollo/`.
- Already started Apollo yourself? add `AUTO_START_APOLLO=0`.

## Status

The single-scenario demo above is the supported entry point today. Larger-scale
use — parallel multi-seed fuzzing (N CARLA × 1 Apollo) and multi-ADS in one world
(1 CARLA × N Apollo) — is **under active development** and not yet documented for
external use. Sensor calibration, map setup, and control tuning are
environment-specific and kept in internal notes.

## Get involved

This Apollo integration is actively evolving and we'd love collaborators. If you
want to build on it, compare ADSs, or contribute, please get in touch:

- 💬 **Discord** — join our community (ask for an invite via email below)
- ✉️ **Email** — [snowbirds.mf@gmail.com](mailto:snowbirds.mf@gmail.com)
- 💚 **WeChat** — reach out by email for the contact

Feedback, issues, and PRs are all welcome.
