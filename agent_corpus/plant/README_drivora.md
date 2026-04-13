# PlanT — Drivora Integration

## Install

```bash
bash install_ads_eval.sh plant 0.9.15 .venvs/plant
```

## Run

```bash
# Random fuzzer
bash scripts/random/plant.sh

# AVFuzzer
bash scripts/avfuzzer/plant.sh

# BehAVExplor
bash scripts/behavexplor/plant.sh

# SAMOTA
bash scripts/samota/plant.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
