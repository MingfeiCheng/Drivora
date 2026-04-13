import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import hydra
import multiprocessing as mp

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scenario_runner.config import GlobalConfig # this is a global config for the system
from registry import FUZZER_REGISTRY
from registry.utils import discover_modules

def get_version():
    """
    Get the version of the CARLA simulator.
    """
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    if not os.path.exists(version_file):
        raise FileNotFoundError(f"Version file not found: {version_file}")
    
    with open(version_file, 'r') as f:
        version = f.read().strip()
    
    return version

@hydra.main(config_path='.', config_name='config', version_base=None)
def main(cfg: DictConfig):
    
    agent_config = cfg.get('agent', None)
    if agent_config is None:
        raise ValueError("Please provide the agent config.")
    
    scenario_config = cfg.get('scenario', None)
    if scenario_config is None:
        raise ValueError("Please provide the scenario config.")
    
    fuzzer_config = cfg.get('tester', None)
    if fuzzer_config is None:
        raise ValueError("Please provide the fuzzer config.")
    
    fuzzer_type = fuzzer_config.get('type', None)
    fuzzer_dir = cfg.get('fuzzer_dir', 'fuzzer')

    discover_modules(os.path.dirname(os.path.abspath(__file__)), fuzzer_dir)


    GlobalConfig.debug = cfg.debug
    GlobalConfig.pytree_debug = cfg.pytree_debug  # Enable py_trees debug mode
    GlobalConfig.open_vis = cfg.open_vis  # Open visualization
    GlobalConfig.save_agent_internal = cfg.get('save_agent_internal', False)
    GlobalConfig.parallel_num = cfg.get('distribute_num', 1)  # number of parallel scenarios to run, only for fuzzing
    # setup logger
    level = "DEBUG" if GlobalConfig.debug else "INFO"
    logger.configure(handlers=[{"sink": sys.stderr, "level": level}])
    
    GlobalConfig.resume = cfg.resume
    GlobalConfig.run_tag = cfg.run_tag
    GlobalConfig.max_sim_time = cfg.max_sim_time  # seconds, default is 120 seconds -> the scenario max running time
    
    GlobalConfig.output_root = os.path.join(cfg.output_root, GlobalConfig.run_tag)  # also used in fuzzing
    
    # we use docker here
    GlobalConfig.carla_image = cfg.carla.image
    GlobalConfig.carla_fps = cfg.carla.fps
    GlobalConfig.carla_is_sync = cfg.carla.is_sync
    GlobalConfig.carla_random_seed = cfg.carla.random_seed
    
    GlobalConfig.print_config()
    
    output_root = GlobalConfig.output_root
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    logger_file = os.path.join(output_root, 'run.log')
    _ = logger.add(logger_file, level=level, mode="a")  # Use mode="a" for append
    # save configs
    OmegaConf.save(config=cfg, f=os.path.join(output_root, 'config.yaml'))
    
    # load testing config    
    logger.info(f'Fuzzer type: {fuzzer_type}')
    fuzzer_class = FUZZER_REGISTRY.get(f"fuzzer.{fuzzer_type}")
    logger.info(f'Load fuzzer class from: {fuzzer_class}')
    
    # set global ADS venv & eval scripts
    GlobalConfig.ads_venv_dir = agent_config.ads_venv_dir
    GlobalConfig.scenario_executor_script = scenario_config.executor_script

    fuzzer_instance = fuzzer_class(
        fuzzer_config,
        agent_config,
        scenario_config
    )
    try:
        fuzzer_instance.run()
    except KeyboardInterrupt:
        fuzzer_instance.close()
    finally:
        fuzzer_instance.close()

if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    
    main()
    logger.info('[:D] -> Fuzzing DONE!')
    sys.exit(0)