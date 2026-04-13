# Architecture

## Data Flow

Drivora uses a **two-process architecture**: the fuzzer runs in a main process, and each scenario is executed in an isolated subprocess with the ADS agent's own Python environment.

```
                            CARLA Container
                           ┌──────────────┐
                           │  CARLA Server │
                           │  (Docker)     │
                           └──────┬───────┘
                                  │ TCP
                ┌─────────────────┼──────────────────────┐
                │                 │                       │
     ┌──────────┴──────────┐     │     ┌─────────────────┴─────────────┐
     │  Fuzzer Process      │     │     │  Scenario Subprocess          │
     │  (.venvs/random)     │     │     │  (.venvs/<agent>)             │
     │                      │     │     │                               │
     │  1. Sample scenario ─┼─────┘     │  4. ScenarioManager           │
     │     (connect CARLA,  │           │     ├─ load world             │
     │      sample routes,  │           │     ├─ spawn ego + NPCs       │
     │      cleanup)        │           │     ├─ setup Agent            │
     │                      │           │     ├─ run tick loop:         │
     │  2. Write files: ────┼──────────►│     │   Agent.run_step()      │
     │     scenario.json    │           │     │   Criteria.check()      │
     │     ctn_config.json  │           │     │   record observation    │
     │                      │           │     └─ stop():                │
     │  3. Launch subprocess┼──────────►│        save result.json       │
     │     (Popen, ads venv)│           │        save observation.gz    │
     │                      │           │        save video             │
     │  5. Read results: ◄──┼───────────┤                               │
     │     result.json      │           └───────────────────────────────┘
     │     observation.gz   │
     │                      │
     │  6. Oracle.evaluate()│
     │  7. Feedback.evaluate│
     │  8. Update population│
     │  9. Save checkpoint  │
     └─────────────────────┘
```

**Why two processes?**
- CARLA segfaults don't kill the fuzzer
- Each agent can have its own Python/torch version
- Clean isolation between fuzzer and ADS dependencies
- Stall detection: fuzzer kills stuck subprocesses automatically

## Fuzzer Loop (Main Process)

Each fuzzer inherits from `Fuzzer` (runner_base.py) and implements `_run()`:

```
_run(start_time):
  while not termination_check():
    ┌─────────────────────────────────────────────────┐
    │ 1. Generate scenarios                            │
    │    (sample / mutate / surrogate-guided search)   │
    │                                                  │
    │ 2. execute_evaluate(batch)     ← base class      │
    │    ├─ execute_population()     ← launches Popen  │
    │    │    writes scenario.json + ctn_config.json   │
    │    │    waits for result.json + timeout/stall    │
    │    ├─ oracle.evaluate()        ← 8 safety checks │
    │    ├─ feedback.evaluate()      ← fitness score   │
    │    └─ assign_feedback_to_ind() ← DEAP fitness    │
    │                                                  │
    │ 3. Update population / corpus / archive          │
    │ 4. save_checkpoint()                             │
    └─────────────────────────────────────────────────┘
```

All fuzzers share steps 2-4 via the base class. Only step 1 (generation strategy) differs:

| Fuzzer | Generation Strategy |
|--------|-------------------|
| Random | Re-sample from `ScenarioODDSpace` each generation |
| AVFuzzer | GA: crossover (swap NPC) + mutation (perturb speed/trigger) + roulette selection + restart + LIS |
| BehAVExplor | Energy-based seed selection + small/large mutation + KMeans coverage model |
| SAMOTA | Ensemble surrogate (RBF+PR+Kriging) → NSGA-II global search + HDBSCAN local search → nearest-scenario mutation |
| RandomMulti | Same as Random but with `ego_space.num > 1` |

## Scenario Execution (Subprocess)

Each scenario runs in a subprocess via `ScenarioManager`:

```
ScenarioManager.run():
  1. Load CARLA world (or reuse if same town)
  2. Build scenario tree (OpenScenario behavior tree):
     ├─ EgoVehicle behaviors (route following)
     ├─ NPC vehicle behaviors (waypoint following)
     ├─ Walker behaviors (AI navigation)
     ├─ Static obstacles
     ├─ Traffic light control
     └─ Weather setup
  3. Setup ADS agent:
     ├─ load_entry_point("agent_corpus.roach.agent:RoachAgent")
     ├─ agent.set_global_plan(route)
     ├─ agent.setup_env(ego_id, vehicle, ctn_operator, config)
     └─ AgentWrapper.setup_sensors()
  4. Tick loop:
     for each CARLA tick:
       ├─ agent.run_step(sensor_data, timestamp) → control
       ├─ ego.apply_control(control)
       ├─ scenario_tree.tick() (NPC behaviors + criteria)
       └─ record observation frame
  5. stop():
     ├─ cleanup sensors, agents, actors
     ├─ analyze_scenario() → result.json
     ├─ save observation.jsonl.gz (always, even on error)
     └─ save video
```

## Scenario Configuration

Each scenario is defined by `ScenarioConfig` (Pydantic model):

```
ScenarioConfig
├── ego_vehicles: List[EgoConfig]
│   ├── route: List[Waypoint]          # x, y, z, pitch, yaw, roll, speed
│   ├── entry_point: str               # "agent_corpus.roach.agent:RoachAgent"
│   ├── config_path: str               # agent config file
│   └── trigger_time: float            # when to start
├── npc_vehicles: List[WaypointVehicleConfig]
│   ├── route: List[Waypoint]          # NPC route with speeds
│   └── trigger_time: float
├── npc_walkers: List[AIWalkerConfig]
├── npc_statics: List[StaticObstacleConfig]
├── weather: WeatherConfig             # 10 parameters
├── traffic_light: TrafficLightBehaviorConfig
└── map_region: MapConfig              # town + region bounds
```

The fuzzer mutates these parameters. Map cache (driving waypoints, routes) is shared per town: `cache/<town>_map_cache.json`.

## Safety Evaluation Pipeline

```
Scenario results
       │
       ▼
  ScenarioOracle.evaluate()
  ├─ Runtime criteria (from scenario_elements/criteria/):
  │   collision, stuck, offroad, overspeed,
  │   running_stop, running_red_light, wrong_lane, reach_destination
  ├─ Offline collision recheck (2D bbox intersection)
  └─ → oracle_result: { expected: bool, criteria_summary: {...} }
       │
       ▼
  FeedbackCalculator.evaluate()
  ├─ Extracts continuous metrics from observation + oracle_result
  ├─ Computes fitness score(s)
  └─ → feedback_result: { score, details, ... }
```

Each fuzzer uses its own feedback calculator:

| Feedback | Objectives | Used By |
|----------|-----------|---------|
| `RandomFeedbackCalculator` | Collision proximity (single) | Random, RandomMulti |
| `AVFuzzerFeedbackCalculator` | Collision + stuck + destination (composite) | AVFuzzer |
| `BehaviorFeedbackCalculator` | Safety score + diversity score (coverage) | BehAVExplor |
| `SAMOTAFeedbackCalculator` | 6 per-type objectives: DfC, DfV, DfP, DfM, DT, Dest | SAMOTA |
| `MultiADSFeedbackCalculator` | Per-ego + ego-ego distance | RandomMulti |

## Checkpoint & Resume

All fuzzers save state after each generation via `save_checkpoint()`:

```
results/<run_tag>/
├── tmp/
│   ├── checkpoint.pkl          # full fuzzer state (extensible per fuzzer)
│   ├── time_counter.txt        # accumulated wall-clock time
│   └── ensembles/              # SAMOTA: cached surrogate models
├── results/
│   └── <scenario_id>/         # per-scenario outputs
├── overview.json               # summary of all seeds
└── logbook.json                # per-generation metrics
```

Base class provides hooks: `_get_checkpoint_data()` / `_restore_checkpoint_data()`. Each fuzzer extends these to save its own state (population, archive, coverage model, surrogates, etc.).

## Key Source Files

| File | Role |
|------|------|
| `start_fuzzer.py` | Entry point (Hydra config + registry discovery) |
| `fuzzer/runner_base.py` | Base class: execution, evaluation, checkpoint, container management |
| `fuzzer/runner_*.py` | Per-fuzzer implementations |
| `fuzzer/mutator/random_sample.py` | Scenario sampling from `ScenarioODDSpace` |
| `fuzzer/mutator/avfuzzer_mutator.py` | Perturbation mutation (speed, trigger, weather) |
| `fuzzer/mutator/behavior_mutator.py` | Small/large mutation for BehAVExplor |
| `fuzzer/feedback/*.py` | Per-fuzzer fitness calculators |
| `fuzzer/oracle/general_oracle.py` | 8-criteria safety oracle + collision recheck |
| `fuzzer/misc/surrogate_models.py` | RBF, Kriging, PR, Ensemble (SAMOTA) |
| `fuzzer/misc/behavior_model.py` | KMeans coverage model (BehAVExplor) |
| `fuzzer/misc/samota_utils.py` | NSGA-II, archive, global/local search (SAMOTA) |
| `scenario_runner/scenario_manager.py` | CARLA scenario execution + observation recording |
| `scenario_runner/ctn_operator.py` | Docker container lifecycle for CARLA |
| `scenario_corpus/openscenario/` | Scenario tree + config models |
| `scenario_elements/criteria/` | Runtime safety monitors (8 types) |
| `agent_corpus/atomic/base_agent.py` | Agent base class (`AutonomousAgent`) |
