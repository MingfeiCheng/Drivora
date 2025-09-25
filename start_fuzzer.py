import os
import sys
import hydra

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scenario_runner.config import GlobalConfig # this is a global config for the system
from registry import FUZZER_REGISTRY
from tools.module_loader import discover_modules

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
    
    discover_modules(os.path.dirname(os.path.abspath(__file__)))
    
    agent_config = cfg.get('agent', None)
    if agent_config is None:
        raise ValueError("Please provide the agent config.")
    
    scenario_config = cfg.get('scenario', None)
    if scenario_config is None:
        raise ValueError("Please provide the scenario config.")
    
    fuzzer_config = cfg.get('tester', None)
    if fuzzer_config is None:
        raise ValueError("Please provide the fuzzer config.")
    
    scenario_type = scenario_config.get('type', None)
    fuzzer_type = fuzzer_config.get('type', None)

    GlobalConfig.debug = cfg.debug
    GlobalConfig.pytree_debug = cfg.pytree_debug  # Enable py_trees debug mode
    GlobalConfig.open_vis = cfg.open_vis  # Open visualization
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
    
    GlobalConfig.print()
    
    output_root = GlobalConfig.output_root
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    logger_file = os.path.join(output_root, 'run.log')
    _ = logger.add(logger_file, level=level, mode="a")  # Use mode="a" for append
    # save configs
    OmegaConf.save(config=cfg, f=os.path.join(output_root, 'config.yaml'))
    
    # load testing config    
    logger.info(f'Fuzzer type: {scenario_type}.{fuzzer_type}')
    fuzzer_class = FUZZER_REGISTRY.get(f"fuzzer.{scenario_type}.{fuzzer_type}")
    logger.info(f'Load fuzzer class from: {fuzzer_class}')

    fuzzer_instance = fuzzer_class(
        fuzzer_config,
        agent_config,
        scenario_config
    )
    fuzzer_instance.run()
    fuzzer_instance.close()

if __name__ == '__main__':
    main()
    logger.info('[:D] -> Fuzzing DONE!')
    sys.exit(0)