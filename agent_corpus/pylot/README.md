# Pylot — Container-Based ADS Agent

Pylot is a modular ADS built on the [ERDOS](https://github.com/erdos-project/erdos) dataflow framework. Due to its complex dependencies (Rust-compiled ERDOS, TensorFlow 2.5, C++ planners), it runs inside a Docker container. The host-side proxy communicates via ZMQ.

## Architecture

```
Drivora Subprocess                       Docker Container (drivora/pylot)
┌─────────────────────────┐             ┌──────────────────────────────┐
│ PylotProxyAgent          │ ZMQ REQ/REP│ PylotServer                   │
│                          │            │                               │
│ setup():                 │            │ __init__():                    │
│   auto docker run ───────┼──────────► │   parse pylot FLAGS            │
│   wait for ready (ping)  │            │   build ERDOS dataflow         │
│   get_sensors ───────────┼──────────► │   load TF models (~60s)        │
│                          │            │   ZMQ REP bind                  │
│ run_step():              │            │                               │
│   JPEG encode cameras    │            │ handle_tick():                  │
│   pack LiDAR/IMU/GNSS ──┼──────────► │   decode JPEG → CameraFrame    │
│                          │            │   FasterRCNN detection          │
│                          │            │   SORT tracking                 │
│                          │            │   Linear prediction             │
│   control ◄──────────────┼─────────── │   Hybrid A* planning            │
│   → apply_control()      │            │   PID control → return          │
│                          │            │                               │
│ destroy():               │            │ handle_destroy():              │
│   send destroy ──────────┼──────────► │   rebuild ERDOS pipeline        │
│   (container stays alive)│            │   (ready for next scenario)     │
└─────────────────────────┘             └──────────────────────────────┘
        │                                        │
        │ CARLA API (sensors, apply_control)      │ No CARLA connection
        ▼                                        │ Pure computation node
     CARLA Container
```

## Components

| File | Location | Role |
|------|----------|------|
| `pylot_proxy_agent.py` | Host (agent venv) | ZMQ client, auto-manages Docker lifecycle |
| `source_code/pylot_server.py` | Container | ZMQ server wrapping ERDOS pipeline |
| `source_code/pylot/` | Container | Pylot core (perception, planning, control) |
| `source_code/docker/Dockerfile.drivora` | Build | Overlay on `erdosproject/pylot` base |
| `config/proxy_config.json` | Host | Container host/port/image/GPU config |
| `source_code/configs/drivora.conf` | Container | Pylot flagfile (detection, tracking, planning) |

## Installation

```bash
# Installs proxy venv + builds Docker image
bash install_ads_eval.sh pylot 0.9.15 .venvs/pylot
```

Or manually:
```bash
# 1. Host-side proxy venv
uv venv .venvs/pylot --python 3.8
source .venvs/pylot/bin/activate
uv pip install pyzmq msgpack msgpack-numpy loguru opencv-python-headless carla==0.9.15 ...

# 2. Docker image
cd agent_corpus/pylot/source_code
bash docker/build_drivora.sh
```

## Configuration

`config/proxy_config.json`:
```json
{
    "container_host": "localhost",
    "base_port": 12667,
    "jpeg_quality": 90,
    "docker_image": "drivora/pylot:latest",
    "container_name_prefix": "drivora-pylot",
    "gpu": 0,
    "pylot_config": "configs/drivora.conf",
    "auto_stop_container": false
}
```

## Multi-Ego Support

Each ego vehicle automatically gets its own container:

| Ego ID | Container Name | Port |
|--------|---------------|------|
| `ego_0` | `drivora-pylot-ego_0` | 12667 |
| `ego_1` | `drivora-pylot-ego_1` | 12668 |
| `ego_2` | `drivora-pylot-ego_2` | 12669 |

Derived from `self.id` (set by `setup_env()` before `setup()`).

## ZMQ Protocol

| Command | Direction | Payload |
|---------|-----------|---------|
| `ping` | proxy → server | `{}` |
| `get_sensors` | proxy → server | `{}` → returns sensor specs |
| `init` | proxy → server | `{opendrive, route, vehicle_id}` |
| `tick` | proxy → server | `{timestamp, sensors: {cameras, lidar, imu, gnss, speed}}` |
| `destroy` | proxy → server | `{}` → resets ERDOS pipeline |
| `shutdown` | proxy → server | `{}` → exits server process |

Camera data is JPEG-encoded (~150KB per 1080p frame). All payloads use msgpack + msgpack-numpy.

## Docker Image

The image is built in two stages:

1. **Base** (`erdosproject/pylot:latest`): ERDOS + Pylot + TF + models (~33GB, pulled from Docker Hub)
2. **Overlay** (`drivora/pylot:latest`): ZMQ deps + pylot_server.py + drivora.conf + planner recompilation

```bash
# Rebuild overlay only (fast, ~30s)
cd agent_corpus/pylot/source_code
docker build --no-cache -t drivora/pylot:latest -f docker/Dockerfile.drivora .
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `cv2 GStreamerPipeline` error | opencv-python-headless conflicts with base image | Rebuild with `--no-cache` |
| `No module named 'agents'` | CARLA PythonAPI not in PYTHONPATH | Already fixed in Dockerfile.drivora |
| `perfect_localization` flag missing | Not in pylot/flags.py | Defined in pylot_server.py |
| `Control read timed out` | First tick race condition | Normal for first 1-2 frames |
| `TimestampError` on 2nd scenario | ERDOS requires increasing timestamps | `destroy` rebuilds pipeline |
| Container takes 60s to start | TF model loading | Proxy polls with ping, waits up to 120s |
