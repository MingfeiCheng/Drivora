# Installation Guide

## Prerequisites

- **Python** >= 3.8
- **Docker** with NVIDIA Container Toolkit (`nvidia-docker`)
- **CUDA** 11.x or 12.x (with `nvcc` available)
- **Git LFS** (for downloading large model checkpoints)
- **GPU**: NVIDIA GPU with >= 16GB VRAM (tested on A5000, L40S)

> Drivora uses [uv](https://github.com/astral-sh/uv) for Python environment management. It will be installed automatically if not present.

## Quick Install (All Agents)

```bash
git clone https://github.com/MingfeiCheng/Drivora.git
cd Drivora
bash install_all.sh
```

This installs all ADS agents + the tester with CARLA 0.9.15. Environments are created under `.venvs/`:

```
.venvs/
├── roach/          # ROACH agent
├── interfuser/     # InterFuser agent
├── lav/            # LAV agent
├── orion/          # Orion agent
├── plant/          # PlanT agent
├── pylot/          # Pylot proxy agent (+ auto-builds Docker image)
├── simlingo/       # Simlingo agent
├── tcp_admlp/      # TCP & ADMLP agents (shared)
├── transfuser/     # TransFuser agent
├── uniad_vad/      # UniAD & VAD agents (shared)
└── random/         # Tester (fuzzer) venv
```

### Skip Checkpoint Downloads

Model checkpoints can be large (1-10GB each). To install dependencies only:

```bash
SKIP_DOWNLOAD=1 bash install_all.sh
```

Download checkpoints later by re-running individual install scripts:

```bash
bash install_ads_eval.sh roach 0.9.15 .venvs/roach
```

## Install Individual Components

### ADS Agent

```bash
bash install_ads_eval.sh <agent_name> <carla_version> <venv_path>
```

Examples:
```bash
bash install_ads_eval.sh roach 0.9.15 .venvs/roach
bash install_ads_eval.sh transfuser 0.9.15 .venvs/transfuser
bash install_ads_eval.sh pylot 0.9.15 .venvs/pylot    # also builds Docker image
```

### Tester (Fuzzer)

```bash
bash install_tester.sh <tester_name> <carla_version> <venv_path>
```

Example:
```bash
bash install_tester.sh random 0.9.15 .venvs/random
```

Tester-specific requirements are in `fuzzer/requirements/<tester_name>.txt`. All fuzzer dependencies (including BehAVExplor and SAMOTA extras like `scikit-learn`, `hdbscan`) are included in the base `requirements.txt` since the tester venv imports all fuzzers at startup.

## Pylot (Container-Based Agent)

Pylot requires a Docker image in addition to the host-side venv:

```bash
# Automatic (install.sh builds image if not present)
bash install_ads_eval.sh pylot 0.9.15 .venvs/pylot

# Or build manually
cd agent_corpus/pylot/source_code
bash docker/build_drivora.sh
```

The Docker image `drivora/pylot:latest` contains:
- ERDOS dataflow framework (Rust + Python)
- TensorFlow 2.5 + FasterRCNN detection models
- Hybrid A*, RRT*, Frenet planners (C++ compiled)
- ZMQ server for host communication

At runtime, the `PylotProxyAgent` auto-starts/stops the container. Multi-ego scenarios get separate containers with unique ports.

## Supported CARLA Versions

| CARLA Version | Python | Install Method |
|--------------|--------|---------------|
| 0.9.10 / 0.9.10.1 | 3.8 | Egg registration (`pkgs/carla-0.9.10-py3.7-linux-x86_64.egg`) |
| >= 0.9.12 | 3.8 | `uv pip install carla==<version>` |
| 0.9.15 (recommended) | 3.8 | `uv pip install carla==0.9.15` |

## CUDA Compatibility

Agents that compile CUDA C++ extensions (Orion, UniAD/VAD) automatically detect the system CUDA version:

- **CUDA 12.x**: Uses `torch+cu121`
- **CUDA 11.x**: Uses `torch+cu118`

All other agents use `torch+cu118` which is compatible with both CUDA 11 and 12 at runtime.

## Troubleshooting

### `libtiff.so.5: cannot open shared object file`
System has `libtiff.so.6` but CARLA 0.9.10 egg needs `.so.5`:
```bash
sudo ln -sf /usr/lib/x86_64-linux-gnu/libtiff.so.6 /usr/lib/x86_64-linux-gnu/libtiff.so.5
```

### `No module named 'carla'`
The CARLA Python API was not installed. Re-run the install script or manually:
```bash
source .venvs/<agent>/bin/activate
uv pip install carla==0.9.15
```

### `No module named 'joblib'` / `'hdbscan'` / `'sklearn'`
The tester venv is missing dependencies needed by BehAVExplor or SAMOTA. Re-install:
```bash
bash install_tester.sh random 0.9.15 .venvs/random
```

### `CUDA version mismatch` during build
The system CUDA version doesn't match the torch CUDA variant. For agents with CUDA extensions (Orion, UniAD), the install script auto-detects CUDA. Ensure `nvcc --version` is available.

### Pylot: `cv2 GStreamerPipeline` error
The Docker image has a conflicting opencv version. Rebuild with `--no-cache`:
```bash
cd agent_corpus/pylot/source_code
docker build --no-cache -t drivora/pylot:latest -f docker/Dockerfile.drivora .
```

### Hydra `=` parsing error
Checkpoint paths containing `=` (e.g., `epoch=013.ckpt`) need escaping in demo scripts:
```bash
agent_config_path='path/to/epoch\=013.ckpt/model.pt'
```

### `Time budget exhausted — nothing to do`
The fuzzer has already run for the configured `time_budget`. Clear the checkpoint:
```bash
rm results/<run_tag>/tmp/time_counter.txt
```
