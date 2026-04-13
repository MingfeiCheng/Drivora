# Extending Drivora

## Adding a New ADS Agent

### Standard Agent (venv-based)

#### 1. Create agent directory

```
agent_corpus/my_agent/
├── __init__.py
├── my_agent.py          # Agent implementation
├── install.sh           # Dependencies (called by install_ads_eval.sh)
└── config/
    └── config.yaml      # Agent config
```

#### 2. Implement the agent

```python
# my_agent.py
from agent_corpus.atomic.base_agent import AutonomousAgent

class MyAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        # Load model, config, etc.
        pass

    def sensors(self):
        return [
            {'type': 'sensor.camera.rgb', 'x': 1.3, 'y': 0.0, 'z': 2.3,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 960, 'height': 480, 'fov': 100, 'id': 'rgb_front'},
            {'type': 'sensor.other.imu', 'x': 0, 'y': 0, 'z': 0,
             'roll': 0, 'pitch': 0, 'yaw': 0, 'id': 'imu'},
            {'type': 'sensor.other.gnss', 'x': 0, 'y': 0, 'z': 0, 'id': 'gps'},
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
        ]

    def run_step(self, input_data, timestamp):
        control = carla.VehicleControl()
        log_data = {}
        return control, log_data
```

#### 3. Create install script

```bash
#!/bin/bash
# install.sh — called by install_ads_eval.sh with uv venv active
uv pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
uv pip install my-other-deps

if [ "${SKIP_DOWNLOAD:-0}" != "1" ] && [ ! -f "checkpoint.pth" ]; then
    wget https://example.com/checkpoint.pth
fi
```

#### 4. Install and run

```bash
bash install_ads_eval.sh my_agent 0.9.15 .venvs/my_agent
bash scripts/demo_my_agent.sh
```

### Container-Based Agent (Docker)

For agents with complex dependencies (ERDOS, old TF, Rust, etc.), use the container pattern:

#### 1. Create proxy + server

```
agent_corpus/my_agent/
├── my_proxy_agent.py       # Host-side: inherits AutonomousAgent, ZMQ client
├── source_code/
│   ├── my_server.py        # Container-side: ZMQ server wrapping the pipeline
│   └── docker/
│       └── Dockerfile      # Build instructions
├── config/
│   └── proxy_config.json   # Host/port/image config
└── install.sh              # Installs proxy deps + builds Docker image
```

#### 2. Proxy agent

```python
class MyProxyAgent(AutonomousAgent):
    def setup(self, config_path):
        # Auto-start container, connect ZMQ, get sensor specs
        pass

    def sensors(self):
        return self._sensor_specs  # fetched from container

    def run_step(self, input_data, timestamp):
        # Pack sensors → ZMQ send → receive control
        pass
```

See `agent_corpus/pylot/` for a complete reference implementation with multi-ego support.


## Adding a New Fuzzer

### 1. Create the runner

```python
# fuzzer/runner_my_fuzzer.py
from registry import FUZZER_REGISTRY
from fuzzer.runner_base import Fuzzer, FuzzSeed
from .mutator.random_sample import RandomSampler
from .feedback.my_feedback import MyFeedbackCalculator
from .oracle.general_oracle import ScenarioOracle
from .scenario_space import ScenarioODDSpace

@FUZZER_REGISTRY.register("fuzzer.my_fuzzer")
class MyFuzzer(Fuzzer):
    def __init__(self, fuzzer_config, agent_config, scenario_config):
        super().__init__(fuzzer_config, agent_config, scenario_config)

        self.scenario_space = ScenarioODDSpace(self.pipeline_config['scenario_space'])
        self.sampler = RandomSampler(self.scenario_space, self.mutator_config)
        self.feedback = MyFeedbackCalculator(self.feedback_config)
        self.oracle = ScenarioOracle(self.oracle_config)

        self.setup_deap()

    def setup_deap(self):
        # Register DEAP fitness + operators
        super().setup_deap()

    def assign_feedback_to_ind(self, ind, feedback_result):
        ind.fitness.values = (feedback_result['score'],)
        return ind

    def _run(self, start_time):
        while not self.termination_check(start_time):
            self.global_search_step += 1
            # Your fuzzing logic: sample/mutate → evaluate → update
            batch = [[ind] for ind in my_individuals]
            batch = self.toolbox.evaluate(batch)  # uses base class pipeline
            self.save_checkpoint()
```

### 2. Add feedback calculator

```python
# fuzzer/feedback/my_feedback.py
class MyFeedbackCalculator:
    def __init__(self, config):
        self.config = config

    def get_default_feedback(self):
        return {"score": 1.0, "single_score": 1.0, "mutliple_scores": [1.0]}

    def evaluate(self, observation_data, oracle_result):
        # observation_data: list of per-frame dicts with egos/NPCs/bboxes
        # oracle_result: dict with criteria_summary, runtime_results
        return {"score": 0.5, "single_score": 0.5, "mutliple_scores": [0.5]}
```

### 3. Add config

```yaml
# fuzzer/configs/my_fuzzer.yaml
feedback:
  my_param: 10.0

oracle:
  collision_recheck: true

mutator: {}

scenario_space:
  map_region_space:
    town: "Town02"
    region_x_min: 40.0
    region_x_max: 125.0
    region_y_min: 200.0
    region_y_max: 295.0
  ego_space:
    route_length: [50.0, 150.0]
  npc_vehicle_space:
    num: [1, 3]
```

### 4. Add requirements

```
# fuzzer/requirements/my_fuzzer.txt
my-special-dependency>=1.0
```

Also add any imports to `requirements.txt` (base) since the tester venv imports all fuzzers at startup via registry.

### 5. Add scripts

```bash
# scripts/my_fuzzer/roach.sh
tester_type="my_fuzzer"
tester_config_path="fuzzer/configs/my_fuzzer.yaml"
```

### 6. Install and run

```bash
bash install_tester.sh my_fuzzer 0.9.15 .venvs/random
bash scripts/my_fuzzer/roach.sh
```

## Adding a New Mutation Operator

Place in `fuzzer/mutator/`:

```python
# fuzzer/mutator/my_mutator.py
class MyMutator:
    def __init__(self, config):
        self.config = config

    def perturb(self, scenario: ScenarioConfig) -> ScenarioConfig:
        """Mutate scenario in-place and return."""
        scenario = copy.deepcopy(scenario)
        # Modify NPC speeds, triggers, weather, etc.
        return scenario
```

## Adding a New Oracle

Place in `fuzzer/oracle/`:

```python
# fuzzer/oracle/my_oracle.py
class MyOracle:
    def __init__(self, config):
        self.config = config

    def evaluate(self, scenario_observation, runtime_results):
        return {
            "expected": False,    # True if safety violation
            "ignored": False,
            "runtime_results": runtime_results,
            "criteria_summary": {},
            "offline_results": {},
        }
```
