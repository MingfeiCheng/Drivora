# Architecture

## System Overview

```
start_fuzzer.py                        # Entry point (Hydra config)
  ├── discover_modules()               # Auto-register all fuzzers/managers
  ├── Fuzzer (runner_base)             # Base class with all infrastructure
  │     ├── RandomSampler              # Initial scenario sampling
  │     ├── ScenarioOracle             # Safety evaluation (8 criteria)
  │     ├── FeedbackCalculator         # Fitness scoring (per-fuzzer)
  │     └── _run()                     # Main loop (overridden by each fuzzer)
  │           └── execute_population() # Launch subprocess(es)
  │                 └── run_scenario.py
  │                       └── ScenarioManager
  │                             ├── OpenScenario (behavior tree)
  │                             ├── Agent (ADS under test)
  │                             └── Criteria (runtime monitors)
  └── save_checkpoint()                # Persist state for resume
```

## Two-Process Architecture

Drivora uses a **two-process architecture** for robustness:

1. **Fuzzer process** (tester venv): Runs the fuzzing loop, scenario sampling, and evaluation. Uses `.venvs/random/bin/python`.

2. **Scenario subprocess** (ADS venv): Executes each scenario in an isolated subprocess. Uses `.venvs/<agent>/bin/python`. This ensures:
   - CARLA crashes (segfaults) don't kill the fuzzer
   - Different Python/torch versions per agent
   - Clean isolation between fuzzer and ADS dependencies

For container-based agents (e.g., Pylot), there is a third component:

3. **Agent container** (Docker): Runs heavy inference (ERDOS + TF). The proxy agent in the subprocess communicates via ZMQ.

## Fuzzer Architecture

All fuzzers inherit from `Fuzzer` (runner_base.py) and share:
- Scenario execution pipeline (`execute_population`)
- Oracle + feedback evaluation
- Checkpoint save/load with extensible hooks
- Container lifecycle management
- DEAP toolbox integration

```
Fuzzer (base)
  ├── RandomFuzzer          # Random sampling each generation
  ├── AVFuzzer              # GA: crossover + mutation + roulette/top2 + restart + LIS
  ├── BehAVExplor           # Coverage-guided: KMeans behavior model + energy selection
  ├── SAMOTAFuzzer          # Surrogate-assisted: ensemble models + global/local search
  └── RandomMultiFuzzer     # Multi-ego: same ADS on multiple vehicles
```

### RandomFuzzer
- **Mutation**: re-sample entirely from `RandomSampler`
- **Selection**: random
- **Feedback**: collision proximity (single objective)

### AVFuzzer
- **Crossover**: swap one NPC between two parent scenarios
- **Mutation**: perturb one NPC's speed or trigger time
- **Selection**: roulette wheel (fitness-proportional)
- **Restart**: if no progress for N generations, re-initialize population
- **LIS**: local iterative search around global best with boosted mutation rate
- **Feedback**: composite (collision + stuck + destination)
- **IDs**: `global_gen_N_ind_I` / `local_gen_N_lis_L_ind_I`

### BehAVExplor
- **Phase 1**: Random sampling to build initial corpus
- **Phase 2**: Coverage-guided fuzzing with energy-based seed selection
- **Coverage**: KMeans clustering on ego behavior time series (velocity, acceleration, yaw, control)
- **Mutation**: energy-driven (high energy → small perturbation, low → large resample)
- **Corpus update**: add seed if new coverage or better safety score
- **Feedback**: safety score + diversity score (coverage distance)

### SAMOTA
- **Phase 1**: Random sampling to build database
- **Phase 2**: Train ensemble surrogate models (RBF + Kriging + Polynomial Regression) per objective
- **Global Search**: Surrogate-guided NSGA-II to find promising candidate vectors
- **Local Search**: HDBSCAN clustering + per-cluster RBF + local GA
- **Candidate → Scenario**: find nearest scenario in database, apply AVFuzzer-style perturbation
- **Objectives**: 6 binary objectives (DfC, DfV, DfP, DfM, DT, Dest) with continuous values for surrogate training
- **Feedback**: per-type collision distances + lane/traffic/destination checks

### RandomMultiFuzzer
- Same as RandomFuzzer but `ego_space.num` can be > 1
- **Feedback**: per-ego metrics + ego-to-ego minimum distance

## Key Modules

### `fuzzer/runner_base.py` — Fuzzer Base Class
Provides all reusable infrastructure:
- DEAP toolbox setup
- Checkpoint save/load with extensible hooks (`_get_checkpoint_data`, `_restore_checkpoint_data`)
- Scenario execution via subprocess (`execute_population`)
- Oracle + feedback evaluation pipeline (`execute_evaluate`)
- Container lifecycle management
- Stall detection (kill stuck subprocesses)
- Time budget management with early termination

### `fuzzer/mutator/` — Mutation Operators
- `random_sample.py` — `RandomSampler`: samples scenarios from `ScenarioODDSpace`. Supports multi-ADS (list of entry_points). Auto-generates map cache from town name.
- `avfuzzer_mutator.py` — `AVFuzzerMutator`: perturbation-based mutation (speeds, triggers, weather, traffic lights)
- `behavior_mutator.py` — `ScenarioMutator`: small (perturb) / large (resample) mutation for BehAVExplor

### `fuzzer/feedback/` — Feedback Calculators
| Calculator | Used By | Objectives |
|---|---|---|
| `RandomFeedbackCalculator` | Random, RandomMulti | Collision proximity (single) |
| `AVFuzzerFeedbackCalculator` | AVFuzzer | Collision + stuck + destination (composite) |
| `BehaviorFeedbackCalculator` | BehAVExplor | Safety score + diversity score |
| `SAMOTAFeedbackCalculator` | SAMOTA | 6 objectives: DfC, DfV, DfP, DfM, DT, Dest |
| `MultiADSFeedbackCalculator` | RandomMulti | Per-ego + ego-ego distance |

### `fuzzer/misc/` — Algorithm-specific models
- `behavior_model.py` — KMeans coverage model for BehAVExplor
- `surrogate_models.py` — RBF, Kriging, Polynomial Regression, Ensemble for SAMOTA
- `samota_utils.py` — NSGA-II helpers, archive management, global/local search

### `fuzzer/oracle/general_oracle.py` — Scenario Oracle
Evaluates all 8 safety criteria per actor:
- Runtime criteria from `scenario_elements/criteria/`
- Offline collision recheck via 2D bounding-box intersection
- Returns `expected=True` if any safety violation occurred

### `scenario_runner/scenario_manager.py` — Scenario Execution
- Loads CARLA world, spawns actors, sets up agent
- Runs simulation tick loop
- Collects observation data and criteria results
- **Always saves observation + video** (even on error/timeout — moved to `stop()`)
- Handles CARLA connection loss gracefully

### `agent_corpus/atomic/base_agent.py` — Agent Base Class
All agents inherit from `AutonomousAgent`:
- `setup_env()`: Binds vehicle, sets up `internal_save_dir`
- `setup()`: Agent-specific initialization
- `sensors()`: Define required sensors
- `run_step()`: Generate control output per tick
- `set_global_plan()`: Receive route from scenario

### `agent_corpus/pylot/` — Container-Based Agent
- `pylot_proxy_agent.py`: Host-side ZMQ client, auto-manages Docker container lifecycle
- `source_code/pylot_server.py`: Container-side ZMQ server wrapping ERDOS pipeline
- Multi-ego support: per-ego container name + port auto-derived from ego ID

## Data Flow

```
Fuzzer                    Subprocess                    Container (Pylot only)
  │                          │                              │
  ├─ sample scenario ──────> │                              │
  │  (via CARLA container)   │                              │
  │                          │                              │
  ├─ write scenario.json ──> │                              │
  ├─ write ctn_config.json > │                              │
  │                          │                              │
  ├─ launch subprocess ────> ├─ load scenario               │
  │  (ads venv python)       ├─ connect to CARLA            │
  │                          ├─ spawn actors + agent         │
  │                          │   └─ PylotProxy.setup() ───> ├─ docker run (auto)
  │                          │      (ZMQ connect)            ├─ ERDOS pipeline init
  │                          ├─ run simulation loop          │
  │                          │   └─ run_step():              │
  │                          │      pack sensors ──ZMQ────> ├─ FasterRCNN detect
  │                          │      control <──ZMQ────────  ├─ Track → Predict
  │                          │                              ├─ Plan → PID
  │                          ├─ write result.json            │
  │                          ├─ write observation.jsonl.gz   │
  │                          └─ write simulation_status.txt  │
  │                          │                              │
  ├─ read results <──────────┘                              │
  ├─ oracle.evaluate()                                      │
  ├─ feedback.evaluate()                                    │
  └─ save checkpoint                                        │
```
