# UniAD / VAD — Drivora Integration

## Install

```bash
bash install_ads_eval.sh uniad_vad 0.9.15 .venvs/uniad_vad
```

## Run

```bash
# Random fuzzer
bash scripts/random/uniad_vad.sh

# AVFuzzer
bash scripts/avfuzzer/uniad_vad.sh

# BehAVExplor
bash scripts/behavexplor/uniad_vad.sh

# SAMOTA
bash scripts/samota/uniad_vad.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
