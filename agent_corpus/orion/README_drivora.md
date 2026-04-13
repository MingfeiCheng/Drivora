# Orion — Drivora Integration

## Install

```bash
bash install_ads_eval.sh orion 0.9.15 .venvs/orion
```

## Run

```bash
# Random fuzzer
bash scripts/random/orion.sh

# AVFuzzer
bash scripts/avfuzzer/orion.sh

# BehAVExplor
bash scripts/behavexplor/orion.sh

# SAMOTA
bash scripts/samota/orion.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
