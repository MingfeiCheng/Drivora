# Roach — Drivora Integration

## Install

```bash
bash install_ads_eval.sh roach 0.9.15 .venvs/roach
```

## Run

```bash
# Random fuzzer
bash scripts/random/roach.sh

# AVFuzzer
bash scripts/avfuzzer/roach.sh

# BehAVExplor
bash scripts/behavexplor/roach.sh

# SAMOTA
bash scripts/samota/roach.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
