# TCP / ADMLP — Drivora Integration

## Install

```bash
bash install_ads_eval.sh tcp_admlp 0.9.15 .venvs/tcp_admlp
```

## Run

```bash
# Random fuzzer
bash scripts/random/tcp_admlp.sh

# AVFuzzer
bash scripts/avfuzzer/tcp_admlp.sh

# BehAVExplor
bash scripts/behavexplor/tcp_admlp.sh

# SAMOTA
bash scripts/samota/tcp_admlp.sh
```

## Configuration

- Agent entry point and config are defined in the run scripts under `scripts/`
- Fuzzer configs are in `fuzzer/configs/`
