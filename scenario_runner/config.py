from __future__ import annotations

import os

class GlobalConfig:
    
    # some tag
    run_tag: str = "default_run"

    # config
    debug: bool = False
    pytree_debug: bool = False  # Enable py_trees debug mode
    open_vis: bool = False  # Open visualization
    resume: bool = True  # if resume the previous run    
    parallel_num: int = 1  # number of parallel scenarios to run, only for fuzzing
    
    # scenario settings
    max_sim_time: float = 120.0  # seconds, default is 300 seconds
    output_root: str = ""  # also used in fuzzing
    
    # carla settings
    carla_image: str = "carlasim/carla:0.9.15"
    carla_fps: int = 20
    carla_is_sync: bool = True
    carla_random_seed: int = 42  # random seed for carla server
    
    @staticmethod
    def get_scenario_folder(scenario_id: str):
        scenario_folder = os.path.join(GlobalConfig.output_root, f'scenario/{scenario_id}')
        if not os.path.exists(scenario_folder):
            os.makedirs(scenario_folder)
        return scenario_folder
    
    @staticmethod
    def print():
        print("Global Config [Carla]:")
        print(f"  run_tag: {GlobalConfig.run_tag}")
        print(f"  debug: {GlobalConfig.debug}")
        print(f"  pytree_debug: {GlobalConfig.pytree_debug}")
        print(f"  open_vis: {GlobalConfig.open_vis}")
        print(f"  resume: {GlobalConfig.resume}")
        print(f"  max_sim_time: {GlobalConfig.max_sim_time}")
        print(f"  output_root: {GlobalConfig.output_root}")
        print(f"  carla_image: {GlobalConfig.carla_image}")
        print(f"  carla_fps: {GlobalConfig.carla_fps}")
        print(f"  carla_is_sync: {GlobalConfig.carla_is_sync}")
        print(f"  carla_random_seed: {GlobalConfig.carla_random_seed}")
        print(f"  distribute_num: {GlobalConfig.parallel_num}")