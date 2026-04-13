<p align="center">
  <img src="assets/drivora_logo.png" alt="Drivora Logo" width="600" style="margin-bottom: -20px;" />
</p>

<br />
<div align="center">
  <h1 align="center">Drivora</h1>
  <p align="center">
    <b>A Unified and Extensible Infrastructure for Autonomous Driving Testing</b>
  </p>
</div>

<p align="center">
  <img src="assets/demo1.gif" width="22%"/>&nbsp;&nbsp;&nbsp;
  <img src="assets/demo2.gif" width="22%"/>&nbsp;&nbsp;&nbsp;
  <img src="assets/demo3.gif" width="22%"/>&nbsp;&nbsp;&nbsp;
  <img src="assets/demo4.gif" width="22%"/>
</p>

---

## Overview

**Drivora** is a research-oriented infrastructure for **search-based testing of Autonomous Driving Systems (ADSs)**.  
It supports:

- Diverse **state-of-the-art ADS architectures** (end-to-end, vision-language, module-based, containerized)
- A variety of **advanced ADS testing techniques** (Random, AVFuzzer, BehAVExplor, SAMOTA, and more)
- **Distributed and parallel execution** for large-scale testing
- **Multi-agent and multi-vehicle** testing settings

Drivora enables **unified, extensible, and automated testing** of ADS safety and reliability across complex driving scenarios.

If you find **Drivora** useful, please consider giving it a star on GitHub!

<p align="center">
  <img src="assets/design.png" alt="Drivora Design" width="700" style="margin-bottom: -20px;" />
</p>


## Features

- **Fuzzing/Testing** — Built-in scenario fuzzing and adversarial scenario generation
- **ADS-Agnostic Integration** — Isolated venv-based interfaces for any ADS; Docker container support for complex agents (e.g., Pylot)
- **Distributed & Parallel Execution** — Scale across multiple CARLA containers
- **Multi-Agent Testing** — Multi-vehicle evaluation with coordinated behaviors
- **8 Runtime Safety Criteria** — Collision, stuck, offroad, overspeed, red light, stop sign, wrong lane, route completion
- **Agent Internal Visualization** — Controllable saving of each agent's debug output


## Prerequisites

- Python >= 3.8
- [uv](https://github.com/astral-sh/uv) — fast Python package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [CARLA](https://carla.org/) >= 0.9.12
- [Docker](https://www.docker.com/) with NVIDIA Container Toolkit
- CUDA 11.x or 12.x
- Git LFS


## Quick Start

### Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/MingfeiCheng/Drivora.git
cd Drivora

# Install all agents + tester (CARLA 0.9.15)
bash install_all.sh

# Or install individually
bash install_ads_eval.sh roach 0.9.15 .venvs/roach
bash install_tester.sh random 0.9.15 .venvs/random

# Skip large checkpoint downloads
SKIP_DOWNLOAD=1 bash install_all.sh
```

### Run

```bash
bash scripts/demo_roach.sh
```

Results are saved to `results/debug_roach/roach_random_debug/`.

See [docs/installation.md](docs/installation.md) for detailed installation instructions and [docs/usage.md](docs/usage.md) for usage guide.


## Directory Structure

```
Drivora/
├── agent_corpus/           # ADS agents under test
│   ├── roach/              # Each agent has install.sh + agent code
│   ├── interfuser/
│   ├── pylot/              # Container-based agent (Docker + ZMQ)
│   │   ├── pylot_proxy_agent.py   # Host-side proxy
│   │   ├── source_code/           # Pylot ERDOS pipeline + Dockerfile
│   │   └── config/                # Proxy config
│   └── ...
├── fuzzer/                 # Fuzzing framework
│   ├── runner_base.py      # Base fuzzer with all reusable infrastructure
│   ├── runner_random.py    # Random fuzzer
│   ├── runner_avfuzzer.py  # AVFuzzer (GA-based)
│   ├── runner_behavexplor.py # BehAVExplor (coverage-guided)
│   ├── runner_samota.py    # SAMOTA (surrogate-assisted)
│   ├── runner_random_multi.py # Multi-ego random fuzzer
│   ├── oracle/             # Safety evaluation
│   ├── feedback/           # Fitness scoring (per-fuzzer)
│   ├── mutator/            # Scenario sampling/mutation
│   ├── misc/               # Surrogate models, coverage models
│   ├── configs/            # Fuzzer pipeline configs
│   └── requirements/       # Per-fuzzer Python requirements
├── scenario_corpus/        # Scenario definitions (OpenScenario)
├── scenario_elements/      # Behavior tree nodes, criteria, triggers
├── scenario_runner/        # CARLA scenario execution engine
├── registry/               # Dynamic module discovery & registration
├── scripts/                # Demo + per-fuzzer per-agent run scripts
├── docs/                   # Documentation
├── config.yaml             # Main Hydra configuration
├── install_all.sh          # Install all agents + testers
├── install_ads_eval.sh     # Install a single ADS agent
├── install_tester.sh       # Install a tester
└── start_fuzzer.py         # Entry point
```


## ADS Corpus

**11 ADS agents** supported, covering end-to-end, vision-language, module-based, and container-based systems:

| Agent | Type | Entry Point | Repository |
|-------|------|-------------|------------|
| Roach | End-to-End | `agent_corpus.roach.agent:RoachAgent` | [carla-roach](https://github.com/zhejz/carla-roach) |
| InterFuser | End-to-End | `agent_corpus.interfuser.interfuser_agent:InterfuserAgent` | [InterFuser](https://github.com/opendilab/InterFuser) |
| LAV | End-to-End | `agent_corpus.lav.lav_agent:LAVAgent` | [LAV](https://github.com/dotchen/LAV) |
| TransFuser | End-to-End | `agent_corpus.transfuser.agent:HybridAgent` | [TransFuser](https://github.com/autonomousvision/transfuser) |
| PlanT | End-to-End | `agent_corpus.plant.PlanT_agent:PlanTPerceptionAgent` | [PlanT](https://github.com/autonomousvision/plant) |
| TCP | End-to-End | `agent_corpus.tcp_admlp.tcp_b2d_agent:TCPAgent` | [TCP](https://github.com/OpenDriveLab/TCP) |
| ADMLP | End-to-End | `agent_corpus.tcp_admlp.admlp_b2d_agent:ADMLPAgent` | [AD-MLP](https://github.com/E2E-AD/AD-MLP) |
| UniAD | End-to-End | `agent_corpus.uniad_vad.uniad_b2d_agent:UniadAgent` | [UniAD](https://github.com/OpenDriveLab/UniAD) |
| VAD | End-to-End | `agent_corpus.uniad_vad.vad_b2d_agent:VadAgent` | [VAD](https://github.com/hustvl/VAD) |
| Simlingo | Vision-Language | `agent_corpus.simlingo.agent_simlingo:LingoAgent` | [Simlingo](https://github.com/RenzKa/simlingo) |
| Orion | Vision-Language | `agent_corpus.orion.orion_b2d_agent:OrionAgent` | [Orion](https://github.com/xiaomi-mlab/Orion) |
| Pylot | Module-Based (Container) | `agent_corpus.pylot.pylot_proxy_agent:PylotProxyAgent` | [Pylot](https://github.com/erdos-project/pylot) |


## Testing Tools

| Tool | Type | Description |
|------|------|------------|
| Random | Baseline | Random scenario sampling with collision-proximity feedback |
| AVFuzzer | GA-based | Genetic algorithm with crossover, mutation, restart, and local iterative search (LIS) |
| BehAVExplor | Coverage-guided | KMeans behavior coverage model + energy-based seed selection |
| SAMOTA | Surrogate-assisted | Ensemble surrogate models (RBF + Kriging + PR) with global/local search |
| Random Multi | Multi-ego | Multi-ego testing with ego-to-ego collision distance feedback |

> We provide prototype implementations following the core methodology of each paper. These are not guaranteed to be fully identical to the original implementations.

### Running Different Fuzzers

```bash
# Per-agent scripts for each fuzzer
bash scripts/random/roach.sh
bash scripts/avfuzzer/roach.sh
bash scripts/behavexplor/roach.sh
bash scripts/samota/roach.sh
bash scripts/random_multi/roach.sh

# Or use Hydra overrides directly
python start_fuzzer.py tester.type="avfuzzer" tester.config_path="fuzzer/configs/debug_avfuzzer.yaml" ...
```


## Pylot (Container-Based Agent)

Pylot uses a Docker container for its ERDOS dataflow pipeline. See [agent_corpus/pylot/README.md](agent_corpus/pylot/README.md) for architecture details.

```bash
# Install proxy + build Docker image
bash install_ads_eval.sh pylot 0.9.15 .venvs/pylot

# Run (container auto-starts)
bash scripts/random/pylot.sh
```


## Scenario Definition

Scenarios use the **OpenScenario** format with actionable parameters:

<p align="center">
  <img src="assets/OpenScenario.png" alt="Scenario Design" width="700" />
</p>

Each scenario defines: ego vehicle routes, NPC vehicles, AI walkers, static obstacles, traffic light behavior, and weather conditions.


## Documentation

- [Installation Guide](docs/installation.md) — Detailed setup instructions and troubleshooting
- [Usage Guide](docs/usage.md) — Running demos, parameters, results structure
- [Architecture](docs/architecture.md) — System design, data flow, module overview
- [Extending Drivora](docs/extending.md) — Adding new agents, fuzzers, and feedback calculators


## Citation

If you use **Drivora** in your work, please cite:

```bibtex
@article{cheng2026drivora,
  title     = {Drivora: A Unified and Extensible Infrastructure for Search-based Autonomous Driving Testing},
  author    = {Cheng, Mingfei and Briand, Lionel and Zhou, Yuan},
  journal   = {arXiv preprint arXiv:2601.05685},
  year      = {2026}
}
```

## Contributing

Contributions are welcome! Please open an issue first for discussion.

1. Fork this repository
2. Create a new branch
3. Commit and push your changes
4. Open a Pull Request


## Acknowledgements

- All open-source Autonomous Driving Systems
- [CARLA Simulator](https://carla.org/)
- [CARLA ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)


## Contact & License

For inquiries, please contact **Mingfei Cheng** at [snowbirds.mf@gmail.com](mailto:snowbirds.mf@gmail.com).

This project is licensed under the [MIT License](LICENSE).
