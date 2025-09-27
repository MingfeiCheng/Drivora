<p align="center">
  <img src="assets/drivora_logo.png" alt="Drivora Logo" width="600" style="margin-bottom: -20px;" />
</p>

<br />
<div align="center">
  <h1 align="center">Drivora</h1>
  <p align="center">
    <b>A Unified and Scalable Infrastructure for Autonomous Driving Testing</b>
  </p>
</div>

---

## 🧭 Overview

**Drivora** is a research-oriented infrastructure for **search-based testing of Autonomous Driving Systems (ADSs)**.  
It is designed to support:

- 🚗 Diverse **state-of-the-art ADS architectures**  
- 🧪 A variety of **advanced ADS testing techniques**  
- ⚡ **Distributed and parallel execution** for large-scale testing  
- 👥 **Multi-agent and multi-vehicle** testing settings  

Drivora enables **scalable, automated, and reproducible evaluation** of ADS safety and reliability across complex driving scenarios.  
Its modular design allows researchers to **prototype and extend new testing methods** without dealing with low-level deployment details.

---

## 🚀 Features

- 🔬 **Fuzzing**  
  Built-in support for diverse scenario fuzzing and adversarial input generation.

- 🧩 **ADS-Agnostic Integration**  
  Containerized interfaces for black-box and white-box ADSs.

- ⚡ **Distributed & Parallel Execution**  
  Scale up testing across multiple scenario execution instances.

- 👥 **Multi-Agent Testing**  
  Supports multi-vehicle evaluation with coordinated or independent ADS behaviors.

---

## 📦 Getting Started

### Prerequisites
- [Docker](https://www.docker.com/)  
- [Anaconda](https://www.anaconda.com/) (recommended)

### Clone the Repository
```bash
git clone https://github.com/MingfeiCheng/Drivora.git
cd Drivora
```

---

## 📂 Directory Structure

```
Carla/
├── agent_corpus/       # ADSs under test
├── fuzzer/             # Fuzzing tools and logic
├── pkgs/               # Environment packages
├── registry/           # Dynamic component loading
├── scenario_corpus/    # Scenario templates / DSLs
├── scenario_elements/  # Low-level scenario behavior nodes
├── scenario_runner/    # Scenario execution components
├── seed_generator/     # Seed scenario generation
├── tools/              # Helper scripts
├── scripts/            # Demo usage scripts
├── config.yaml         # Main configuration
├── install.sh          # Quick install script
└── start_fuzzer.py     # Entrypoint for launching tests
```

---

## ⚙️ Installation

Different ADSs and testing techniques often depend on heterogeneous libraries, which may cause dependency conflicts.  
We provide a quick script for installation. For example, to test **Roach** under **Random** testing with CARLA `0.9.10.1`:

```bash
bash install.sh roach random 0.9.10.1
```

- **First parameter** → ADS under test (e.g., `roach`)  
- **Second parameter** → Testing method (e.g., `random`)  
- **Third parameter** → Compatible CARLA version (check official repo of each ADS for supported versions)

---

## 🚦 Usage (Quick Demo)

### Step 1: Generate Seed Scenarios
```bash
python -m seed_generator.open_scenario \
  --num 10 --town Town01 \
  --min_length 50 --max_length 200 \
  --out_dir scenario_datasets \
  --image carlasim/carla:0.9.10.1
```

This generates 10 initial seeds under `scenario_datasets`, e.g.:

```
scenario_datasets/open_scenario/0.9.10.1/route_100_200/Town01_0001.json
scenario_datasets/open_scenario/0.9.10.1/route_100_200/Town01_0002.json
...
```

---

### Step 2: Run Testing

You can configure testing for any seed scenario and ADS by editing demo scripts in `scripts/`.  
Example: run **Random testing** for **Roach** with an initial seed:

```bash
bash scripts/demo_roach.sh
```

Inside `scripts/demo_roach.sh`:

```bash
#!/bin/bash
set -euo pipefail

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=2,3 

# ==== Common Config ====
output_root="results"
run_index=1
max_sim_time=600.0
open_vis=true
distribute_num=2  # Number of distributed execution instances

# ==== Agent Config ====
agent_name="roach"
agent_entry_point="agent_corpus.roach.agent:RoachAgent"
agent_config_path="agent_corpus/roach/config/config_agent.yaml"

# ==== Scenario Config ====
seed_segment="route_100_200"
seed_id="Town01_0001"
scenario_type="open_scenario"
scenario_seed_path="scenario_datasets/open_scenario/0.9.10.1/${seed_segment}/${seed_id}.json"

# ==== Tester Config ====
tester_type="random"
tester_config_path="fuzzer/open_scenario/random/configs/open_scenario.yaml"

run_tag="${tester_type}_${agent_name}_${seed_segment}_${seed_id}_run${run_index}"

# ==== Run (Hydra style overrides) ====
python start_fuzzer.py \
  output_root="$output_root" \
  distribute_num="$distribute_num" \
  run_tag="$run_tag" \
  max_sim_time="$max_sim_time" \
  open_vis="$open_vis" \
  tester.type="$tester_type" \
  tester.config_path="$tester_config_path" \
  agent.entry_point="$agent_entry_point" \
  agent.config_path="$agent_config_path" \
  scenario.type="$scenario_type" \
  scenario.seed_path="$scenario_seed_path"
```

---

## 🧱 ADS Corpus

Currently, **12 ADSs** are supported, covering **module-based**, **end-to-end**, and **vision-language-based** ADSs:

| ADS Agent  | ADS Type              | Original Repo                                                                                      | Entry Point                                   | Config Path                                                   |
|------------|-----------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------|---------------------------------------------------------------|
| Pylot      | Module-based          | [erdos-project/pylot](https://github.com/erdos-project/pylot)  |           release soon     |
| LAV      | End-to-End            | [LAV](https://github.com/dotchen/LAV) | agent_corpus.lav.lav_agent:LAVAgent | agent_corpus/lav/config_v2.yaml |
| Roach      | End-to-End            | [carla-roach](https://github.com/zhejz/carla-roach) | agent_corpus.roach.agent:RoachAgent           | agent_corpus/roach/config/config_agent.yaml                   |
| InterFuser | End-to-End  | [InterFuser](https://github.com/opendilab/InterFuser)   |  agent_corpus.interfuser.interfuser_agent:InterfuserAgent |  agent_corpus/interfuser/interfuser_config.py |
| TransFuser | End-to-End            | [transfuser](https://github.com/autonomousvision/transfuser)   |     agent_corpus.transfuser.agent:HybridAgent  | agent_corpus/transfuser/model_ckpt/models_2022/transfuser |
| PlanT      | End-to-End            | [plant](https://github.com/autonomousvision/plant) | agent_corpus.plant.PlanT_agent:PlanTPerceptionAgent  |  agent_corpus/plant/carla_agent_files/config/experiments/PlanTSubmission.yaml |
| TCP        | End-to-End            | [TCP](https://github.com/OpenDriveLab/TCP), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/tree/main) |       agent_corpus.tcp_admlp.tcp_b2d_agent:TCPAgent   |  agent_corpus/tcp_admlp/Bench2DriveZoo/tcp_b2d.ckpt   |
| ADMLP      | End-to-End            | [ADMLP](https://github.com/E2E-AD/AD-MLP), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/tree/main) |     agent_corpus.tcp_admlp.admlp_b2d_agent:ADMLPAgent       |  agent_corpus/tcp_admlp/Bench2DriveZoo/admlp_b2d.ckpt  |
| Uniad      | End-to-End            | [UniAD](https://github.com/OpenDriveLab/UniAD), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/tree/main) |     agent_corpus.uniad_vad.uniad_b2d_agent:UniadAgent     |  agent_corpus/uniad_vad/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py+agent_corpus/uniad_vad/Bench2DriveZoo/uniad_base_b2d.pth |
| VAD        | End-to-End            | [VAD](https://github.com/hustvl/VAD), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive/tree/main) |    agent_corpus.uniad_vad.vad_b2d_agent:VadAgent      | agent_corpus/uniad_vad/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+agent_corpus/uniad_vad/Bench2DriveZoo/vad_b2d_base.pth |
| Simlingo   | Vision-Language-based | [simlingo](https://github.com/RenzKa/simlingo)                                                     | agent_corpus.simlingo.agent_simlingo:LingoAgent | agent_corpus/simlingo/checkpoint/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt |
| Orion      | Vision-Language-based | [Orion](https://github.com/xiaomi-mlab/Orion)     |        agent_corpus.orion.orion_b2d_agent:OrionAgent   |  will release soon |

📌 See the [Agent Integration Guide](agents/atomic/README.md) for integrating your own ADS.

## 🔬 Fuzzing Tools

Drivora incorporates multiple fuzzers, each with different scenario definitions, mutation strategies, feedback, and oracles.

✅ **Currently Supported Tools**
- [Random](fuzzer/open_scenario/random/README.md)  
- [AVFuzzer](fuzzer/open_scenario/avfuzzer/README.md)  
- [Behavexplor](fuzzer/open_scenario/behavexplor/README.md)  
- [SAMOTA](fuzzer/open_scenario/samota/README.md)  
- ... 🔄 more coming soon!

---

## 🧩 Extension

To develop your own **search-based testing methods**, please refer to the provided examples and associated papers.

---

## ✅ TODO

- [ ] Provide more detailed **documentation and tutorials**  
- [ ] Release more testing methods
- [ ] Release more ADSs
 
## 📬 Contact

We welcome issues, suggestions, and collaboration opportunities.  

**Mingfei Cheng**  
📧 [snowbirds.mf@gmail.com](mailto:snowbirds.mf@gmail.com)

---

## 🤝 Contributing

Contributions of all kinds are welcome!  
1. Fork this repository  
2. Create a new branch  
3. Commit and push your changes  
4. Open a Pull Request  

---

## 📖 Citation

If you use Drivora in your work, please cite:

```bibtex
@article{cheng2024drivetester,
  title     = {Drivetester: A unified platform for simulation-based autonomous driving testing},
  author    = {Cheng, Mingfei and Zhou, Yuan and Xie, Xiaofei},
  journal   = {arXiv preprint arXiv:2412.12656},
  year      = {2024}
}

@article{cheng2025stclocker,
  title     = {STCLocker: Deadlock Avoidance Testing for Autonomous Driving Systems},
  author    = {Cheng, Mingfei and Wang, Renzhi and Xie, Xiaofei and Zhou, Yuan and Ma, Lei},
  journal   = {arXiv preprint arXiv:2506.23995},
  year      = {2025}
}
```

---

## Acknowledgements

We would like to acknowledge the following open-source projects and communities that our work builds upon:

- All open-source Autonomous Driving Systems (ADSs)
- [CARLA ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)

## 📝 License

This project is licensed under the [MIT License](LICENSE).  
© 2024 Mingfei Cheng
