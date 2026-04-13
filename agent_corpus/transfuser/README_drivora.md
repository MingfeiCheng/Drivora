# TransFuser — Drivora Integration

## Install

```bash
bash install_ads_eval.sh transfuser 0.9.15 .venvs/transfuser
```

## Run

```bash
# Random fuzzer
bash scripts/random/transfuser.sh

# AVFuzzer
bash scripts/avfuzzer/transfuser.sh

# BehAVExplor
bash scripts/behavexplor/transfuser.sh

# SAMOTA
bash scripts/samota/transfuser.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
