# Usage Guide

## Running a Demo

Each agent has a demo script (random fuzzer) in `scripts/`:

```bash
bash scripts/demo_roach.sh
bash scripts/demo_transfuser.sh
bash scripts/demo_pylot.sh
# ... etc
```

## Running with Different Fuzzers

Per-fuzzer per-agent scripts are in `scripts/<fuzzer>/`:

```bash
# Random fuzzer
bash scripts/random/roach.sh

# AVFuzzer (GA-based)
bash scripts/avfuzzer/roach.sh

# BehAVExplor (coverage-guided)
bash scripts/behavexplor/roach.sh

# SAMOTA (surrogate-assisted)
bash scripts/samota/roach.sh

# Multi-ego random
bash scripts/random_multi/roach.sh
```

Replace `roach` with any agent: `interfuser`, `lav`, `transfuser`, `plant`, `tcp`, `admlp`, `uniad`, `vad`, `simlingo`, `orion`, `pylot`.

## Script Structure

```bash
#!/bin/bash

# ==== GPU Config ====
export CUDA_VISIBLE_DEVICES=0

# ==== Tester venv ====
TESTER_VENV=".venvs/random"

# ==== Common Config ====
output_root="results/avfuzzer_roach"
run_tag="avfuzzer_roach"
max_sim_time=120.0
distribute_num=1

# ==== Agent Config ====
agent_entry_point="agent_corpus.roach.agent:RoachAgent"
agent_config_path="agent_corpus/roach/config/config_agent.yaml"
ads_venv_dir=".venvs/roach"

# ==== Tester Config ====
tester_type="avfuzzer"
tester_config_path="fuzzer/configs/debug_avfuzzer.yaml"
time_budget=4
population_size=4
```

## Key Parameters

| Parameter | Description | Default |
|-----------|------------|---------|
| `distribute_num` | Number of parallel CARLA containers | 1 |
| `max_sim_time` | Max simulation time per scenario (seconds) | 120.0 |
| `time_budget` | Total fuzzing time budget (hours) | 4 |
| `population_size` | Scenarios sampled per generation | 4 |
| `resume` | Resume from last checkpoint | true |
| `open_vis` | Save bird-eye-view video | false |
| `save_agent_internal` | Save agent's internal visualization data | false |
| `debug` | Enable debug logging | false |

## Fuzzer-Specific Parameters

### AVFuzzer (`fuzzer/configs/debug_avfuzzer.yaml`)
| Parameter | Description | Default |
|-----------|------------|---------|
| `mutation_prob` | Per-individual mutation probability | 0.6 |
| `crossover_prob` | Crossover probability | 0.6 |
| `selection` | Selection method (`roulette` or `top2`) | `roulette` |
| `stagnation_gens` | Restart after N gens without improvement | 5 |
| `lis_interval` | Local search trigger interval | 3 |
| `lis_generations` | LIS inner GA generations | 5 |

### BehAVExplor (`fuzzer/configs/debug_behavexplor.yaml`)
| Parameter | Description | Default |
|-----------|------------|---------|
| `initial_corpus_size` | Build corpus of this size before fuzzing | 4 |
| `batch_size` | Scenarios per generation | 1 |
| `coverage_cluster_num` | KMeans clusters for behavior model | 50 |
| `coverage_threshold` | New coverage distance threshold | 0.4 |

### SAMOTA (`fuzzer/configs/debug_samota.yaml`)
| Parameter | Description | Default |
|-----------|------------|---------|
| `initial_db_size` | Initial database size (random sampling) | 6 |
| `gs_pop_size` | Population size for surrogate-guided NSGA-II | 6 |
| `gs_generations` | GA generations in Global Search | 100 |
| `thresholds` | Per-objective archive thresholds (6 values) | [0.1, 0.0, ...] |

## Resume & Checkpointing

By default, `resume=true`. The fuzzer saves checkpoints after each generation:

```
results/<run_tag>/
├── tmp/
│   ├── checkpoint.pkl     # fuzzer state
│   └── time_counter.txt   # accumulated time
├── overview.json          # results summary
└── logbook.json           # per-generation metrics
```

To start fresh, either:
- Set `resume=false` in the demo script
- Delete the time counter: `rm results/<run_tag>/tmp/time_counter.txt`
- Delete everything: `rm -rf results/<run_tag>/`

## Results Structure

Each scenario execution produces:

```
results/<run_tag>/results/global_gen_5_ind_2/
├── scenario.json           # scenario configuration
├── ctn_config.json         # CARLA container config
├── eval.log                # subprocess stdout+stderr log
├── simulation_status.txt   # SUCCESS or FAILED
├── result.json             # criteria results (8 types)
├── result.txt              # human-readable results table
├── observation.jsonl.gz    # per-frame observation data (always saved, even on error)
└── agent/
    ├── internal/           # agent's internal visualization (if save_agent_internal=true)
    └── video/              # bird-eye-view video (if open_vis=true)
```

## Evaluation Criteria

Each scenario evaluates 8 runtime criteria:

| Criterion | Description |
|-----------|------------|
| `collision` | Collision with vehicles, pedestrians, or static objects |
| `stuck` | Vehicle blocked below min speed for too long |
| `reach_destination` | Route completion percentage and distance to goal |
| `offroad` | Sidewalk invasion or leaving driving lanes |
| `overspeed` | Exceeding speed limit |
| `running_stop` | Running a stop sign |
| `running_red_light` | Running a red traffic light |
| `wrong_lane` | Driving in wrong direction lane |

## Multi-GPU / Parallel Execution

Set `distribute_num` to use multiple CARLA containers in parallel:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
distribute_num=4
```

Each container is assigned to a GPU in round-robin fashion.

## Multi-Ego Testing

Use `random_multi` fuzzer with `ego_space.num > 1` in the scenario space config:

```yaml
# In fuzzer config
scenario_space:
  ego_space:
    num: [2, 2]    # exactly 2 ego vehicles (same ADS)
```

All egos run the same agent. Feedback includes ego-to-ego collision distance.
