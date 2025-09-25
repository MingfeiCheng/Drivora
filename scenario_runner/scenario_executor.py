import os
import time
import json

from loguru import logger

from registry import MANAGER_REGISTRY

from .ctn_manager import CtnConfig, CtnSimOperator
from .scenario_manager import ScenarioConfigType

def run_scenario(
    scenario_entry_point: str,
    scenario_config: ScenarioConfigType,
    ctn_config: dict,
    scenario_dir: str = None,
    manager_name: str = "default",
    max_sim_time: float = 300, # seconds
    debug: bool = False,
    pytree_debug: bool = False,
    open_vis: bool = True
):
    """
    TODO: update this structure
    Execute the scenario based on the specification
    |_scenario
        |- scenario_idx
            |- scenario.json
            |- records_simulator.pkl
            |- simulation_result.json
            |- records_apollo (folder)
            |- debug (folder)
    """
    if scenario_entry_point is None:
        raise RuntimeError('Please specify a valid scenario type')
    
    ctn_config = CtnConfig.from_dict(ctn_config)
    
    ctn_operator = CtnSimOperator(
        idx=ctn_config.idx,
        container_name=ctn_config.container_name,
        gpu=ctn_config.gpu,
        random_seed=ctn_config.random_seed,
        docker_image=ctn_config.docker_image,
        fps=ctn_config.fps,
        is_sync_mode=ctn_config.is_sync_mode
    )
    ctn_operator.start() 
    
    if not os.path.exists(scenario_dir):
        os.makedirs(scenario_dir)
        logger.info(f'--> Create scenario folder: {scenario_dir}')

    # 1. save scenario file
    with open(os.path.join(scenario_dir, 'scenario.json'), "w") as f:
        json.dump(scenario_config.model_dump(), f, indent=4)

    # 2. build scenario
    scenario_manager_class = MANAGER_REGISTRY.get(f"scenario_manager.{manager_name}")
    scenario_manager = scenario_manager_class(
        scenario_entry_point=scenario_entry_point,
        scenario_config=scenario_config,
        ctn_operator=ctn_operator,
        scenario_dir=scenario_dir,
        debug=debug,
        pytree_debug=pytree_debug,
        require_vis=open_vis,
        max_sim_time=max_sim_time,
    )

    # 4. run all components
    m_start_time = time.time()
    run_status = scenario_manager.run()
    m_end_time = time.time()
    simulation_spend_time = m_end_time - m_start_time
    logger.info('--> [Simulation Time] Simulation Spend Time (seconds): [=]{}[=]', simulation_spend_time)
    
    return run_status # TRUE or FALSE