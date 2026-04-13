# Simlingo — Drivora Integration

## Install

```bash
bash install_ads_eval.sh simlingo 0.9.15 .venvs/simlingo
```

## Run

```bash
# Random fuzzer
bash scripts/random/simlingo.sh

# AVFuzzer
bash scripts/avfuzzer/simlingo.sh

# BehAVExplor
bash scripts/behavexplor/simlingo.sh

# SAMOTA
bash scripts/samota/simlingo.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
