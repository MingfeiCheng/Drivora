# Drivora - CARLA

## 🛠️ Installation

Please navigate to the `Drivora/Carla` directory:

```bash
cd Drivora/Carla
```

Then run the installation script:

```bash
bash install.sh
```

This script will:

- Create a conda environment named `drivora-carla-${CARLA_VERSION}`
- Install all dependencies listed in `requirements.txt`
- Pull the required CARLA Docker image from `carlasim/carla:${CARLA_VERSION}`

You can check the current CARLA version used in the `VERSION` file.

---

## 🚀 Usage

### 📁 Directory Structure: `drivora/Carla`

The `drivora/Carla` directory is organized as follows:

```
Carla/
├── agents/             # Contains the ADSs under test
├── examples/           # Example usage of Drivora for testing ADSs
├── fuzzer/             # Fuzzing techniques and components
├── pkgs/               # Required environment packages
├── registry/           # Registry classes for dynamic component loading
├── scenario_runner/    # Scenario execution engine and scenario templates
├── tools/              # Helper scripts (e.g., logging, module loading)
├── config.yaml         # Main configuration file
└── main.py             # Entrypoint for launching tests
```

Drivora consists of two main components you need to work with:

1. **Autonomous Driving Systems**
2. **Fuzzing tools**


### 🧱 Building ADS

Drivora integrates several state-of-the-art Autonomous Driving Systems (ADSs), organized under the following directory structure:

```
Carla/
├── agents/
│   ├── ctn_operator/   # The interface between the fuzzer engine and ADS agents
│   ├── images/         # Contains ADS implementations under test (e.g., Pylot, Roach)
│   └── local_agents/   # Rule-based agents used to control NPCs
...
```

Each ADS agent’s source code, Docker build logic, and integration scripts are located in `Carla/agents/images`.

> 📌 To unify usage across different ADSs, Drivora provides a standardized interface template located at `Carla/agents/images/template`. This interface wraps each ADS implementation with a consistent entrypoint and communication layer. See [template guide](agents/images/template/README.md) for details on using and adapting the interface.

#### ⚙️ Building an ADS

To use a built-in ADS agent in Drivora, please refer to the detailed usage guide provided for each supported agent.

✅ **Currently Supported ADSs**

| ADS Agent     | Drivora Integration Guide                                | Original Repository                                        |
|---------------|----------------------------------------------------------|------------------------------------------------------------|
| Pylot         | [Usage](agents/images/pylot/README_Drivora.md)           | [erdos-project/pylot](https://github.com/erdos-project/pylot) |
| Roach         | [Usage](agents/images/roach/README_Drivora.md)           | [carla-roach](https://github.com/zhejz/carla-roach)          |
| InterFuser    | [Usage](agents/images/interfuser/README_Drivora.md)      | [InterFuser](https://github.com/opendilab/InterFuser)        |
| TransFuser    | [Usage](agents/images/transfuser/README_Drivora.md)      | [transfuser](https://github.com/autonomousvision/transfuser) |
| PlanT         | [Usage](agents/images/plant/README_Drivora.md)           | [plant](https://github.com/autonomousvision/plant)           |

> 📌 Each usage guide explains how to build the ADS and run the ADS within the Drivora infrastructure.

<!-- 1. Copy `agents/images/template` to a new folder `agents/images/<your_agent_name>/`
2. Develop your agent logics under `agents/images/<your_agent_name>/`
3. Write a `build_base.sh` script and Dockerfile to install dependencies for the agent required packages -->

<!-- > 📌 See the [Agent Template Guide](agents/images/template/README.md) for a full walkthrough on integration. -->


<!-- Step 1. **Enter the corresponding agent folder**

```bash
cd Carla/agents/images/pylot  # or roach, etc.
```

Step 2. **Build the base Docker image**: Each ADS has a `build_base.sh` script to build its runtime environment locally.

```bash
bash build_base.sh
```

   Alternatively, you can pull the prebuilt image from Docker Hub:

```bash
docker pull mingfeicheng/drivora:pylot_base_0.9.10.1
```

> 🔍 You can check the image tag directly in the `build_base.sh` script.

3. **Build the custom Docker image**  
   This image layers your modified ADS code on top of the base image to support fast iteration.

```bash
bash build_custom.sh  # (if available)
``` -->


### 🔬 Building Fuzzing Tools

Drivora incorporates multiple ADS fuzzing tools, each with its own scenario definition, mutation strategy, feedback mechanism, and oracle design. Because of these differences, please refer to the specific usage guide for the tool you intend to use.


✅ **Currently Supported Fuzzing Tools**

- [Random](fuzzer/random/README.md)
- [AVFuzzer](fuzzer/avfuzzer/README.md)
- [BehAVExplor](fuzzer/behavexplor/README.md)
- [SAMOTA](fuzzer/samota/README.md)
<!-- - DoppelTest *(in progress)*
- ADFuzz *(in progress)*
- LawBreaker *(in progress)* -->
- ... > 🔄 More ADS fuzzing tools are continuously being added. Stay tuned!



### Quick Start
To better understand the piepline of usage, we provide a simple usage of Roach + Random, you can work with the tutorial step by step: 

---

## 🧩 Extension

### Extension 1: Adding Your Own ADS

Drivora is designed for extensibility. To integrate your own ADS agent, please see the [Agent Template Guide](agents/images/template/README.md) for a full walkthrough on integration.

### Extension 2: Developing Your Own Fuzzing Tools

From my experience, designing a unified and easily extensible framework for ADS fuzzing is a non-trivial task. I recommend reading the source code of existing fuzzing tools included in Drivora, and using them as a foundation to build your own.

Fortunately, Drivora provides a flexible and extensible **scenario runner**, which allows you to define custom scenarios using `py_trees` and execute them within CARLA. You can easily build your own fuzzing logic by extending this component.

> 📘 For details, refer to the [Scenario Runner Guide](scenario_runner/README.md).
