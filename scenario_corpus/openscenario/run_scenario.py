#!/usr/bin/env python
"""
CLI entry point for running a single scenario in a subprocess.
Called by the fuzzer's execute_instance().
"""

import os
import sys
import json
import argparse

# Ensure project root (Drivora/) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from loguru import logger
from scenario_corpus.openscenario.config import ScenarioConfig
from scenario_corpus.openscenario.scenario_executor import run_scenario


def main():
    parser = argparse.ArgumentParser(description="Run a single CARLA scenario")
    parser.add_argument("--scenario_entry_point", type=str, required=True)
    parser.add_argument("--scenario_config", type=str, required=True)
    parser.add_argument("--ctn_config", type=str, required=True)
    parser.add_argument("--scenario_dir", type=str, required=True)
    parser.add_argument("--manager_name", type=str, default="default")
    parser.add_argument("--max_sim_time", type=float, default=120.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pytree_debug", action="store_true")
    parser.add_argument("--open_vis", action="store_true")
    parser.add_argument("--save_agent_internal", action="store_true")
    args = parser.parse_args()

    with open(args.scenario_config, 'r') as f:
        scenario_config = ScenarioConfig.model_validate(json.load(f))

    with open(args.ctn_config, 'r') as f:
        ctn_config = json.load(f)

    logger.info(f"Running scenario {scenario_config.id} ...")

    status = False
    try:
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
            save_agent_internal=args.save_agent_internal,
        )
    except Exception as e:
        logger.error(f"Scenario execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        status_file = os.path.join(args.scenario_dir, 'simulation_status.txt')
        with open(status_file, 'w') as f:
            f.write('SUCCESS' if status else 'FAILED')

    sys.exit(0 if status else 1)


if __name__ == '__main__':
    main()
