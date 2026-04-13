# LAV — Drivora Integration

## Install

```bash
bash install_ads_eval.sh lav 0.9.15 .venvs/lav
```

## Run

```bash
# Random fuzzer
bash scripts/random/lav.sh

# AVFuzzer
bash scripts/avfuzzer/lav.sh

# BehAVExplor
bash scripts/behavexplor/lav.sh

# SAMOTA
bash scripts/samota/lav.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
