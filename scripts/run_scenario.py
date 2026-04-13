#!/usr/bin/env python
"""
CLI entry point for running a single scenario in a subprocess.
Called by the fuzzer's execute_instance() via:

    python scripts/run_scenario.py \
        --scenario_entry_point scenario_corpus.openscenario.scenario:OpenScenario \
        --scenario_config /path/to/scenario.json \
        --ctn_config /path/to/ctn_config.json \
        --scenario_dir /path/to/output/ \
        --manager_name default \
        --max_sim_time 120 \
        [--debug] [--pytree_debug] [--open_vis]
"""

import os
import sys
import json
import argparse

# Ensure project root (Drivora/) is on sys.path
# This file lives at: Drivora/scripts/run_scenario.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from loguru import logger
from registry.utils import discover_modules
from scenario_corpus.openscenario.config import ScenarioConfig
from scenario_corpus.openscenario.scenario_executor import run_scenario


def main():
    # Discover and register all modules (ScenarioManager, etc.)
    discover_modules(PROJECT_ROOT, fuzzer_dir=None)

    parser = argparse.ArgumentParser(description="Run a single CARLA scenario")
    parser.add_argument("--scenario_entry_point", type=str, required=True)
    parser.add_argument("--scenario_config", type=str, required=True, help="Path to scenario JSON")
    parser.add_argument("--ctn_config", type=str, required=True, help="Path to container config JSON")
    parser.add_argument("--scenario_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--manager_name", type=str, default="default")
    parser.add_argument("--max_sim_time", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pytree_debug", action="store_true")
    parser.add_argument("--open_vis", action="store_true")
    args = parser.parse_args()

    # Load configs from JSON files
    with open(args.scenario_config, 'r') as f:
        scenario_data = json.load(f)
    scenario_config = ScenarioConfig.model_validate(scenario_data)

    with open(args.ctn_config, 'r') as f:
        ctn_config = json.load(f)

    logger.info(f"Running scenario {scenario_config.id} ...")

    status = run_scenario(
        scenario_entry_point=args.scenario_entry_point,
        scenario_config=scenario_config,
        ctn_config=ctn_config,
        scenario_dir=args.scenario_dir,
        manager_name=args.manager_name,
        max_sim_time=args.max_sim_time,
        debug=args.debug,
        pytree_debug=args.pytree_debug,
        open_vis=args.open_vis,
    )

    # Write status file for the fuzzer to read
    status_file = os.path.join(args.scenario_dir, 'simulation_status.txt')
    with open(status_file, 'w') as f:
        f.write('SUCCESS' if status else 'FAILED')

    sys.exit(0 if status else 1)


if __name__ == '__main__':
    main()
