# InterFuser — Drivora Integration

## Install

```bash
bash install_ads_eval.sh interfuser 0.9.15 .venvs/interfuser
```

## Run

```bash
# Random fuzzer
bash scripts/random/interfuser.sh

# AVFuzzer
bash scripts/avfuzzer/interfuser.sh

# BehAVExplor
bash scripts/behavexplor/interfuser.sh

# SAMOTA
bash scripts/samota/interfuser.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
