import multiprocessing

from dataclasses import dataclass
from loguru import logger
from typing import List, Optional

from tools.env_tools import get_available_gpus
from scenario_runner.config import GlobalConfig
from .ctn_operator import CtnSimOperator

def create_ctn_manager(run_tag, carla_image, carla_fps, random_seed, is_sync,
                       worker_agent_configs: Optional[List[List[str]]] = None) -> 'CtnManager':
    # worker_agent_configs: optional PER-WORKER list of PER-EGO agent config paths,
    # i.e. worker_agent_configs[i] = [cfg_ego0, cfg_ego1, ...] for CARLA worker i.
    #   * multi-seed  (N CARLA x 1 ADS): [[ads0], [ads1], ...] one ego per worker
    #   * multi-ADS   (1 CARLA x N ADS): [[ads0, ads1, ..., adsN]] N egos, one sim
    # None -> every ego uses the scenario's default config_path (other agents).
    # Detect available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.warning("[WARN] No GPU detected, containers will run on CPU (gpu=None).")
        raise RuntimeWarning("No GPU detected, containers will run on CPU (gpu=None).")

    parallel_num = GlobalConfig.parallel_num
    num_gpus = len(available_gpus)
    logger.info(f"Setting up {parallel_num} containers for parallel execution with AVAILABLE {num_gpus} GPUs: {available_gpus}")

    operators = []
    for i in range(parallel_num):
        # If there is only one GPU, assign all containers to it
        if num_gpus == 1:
            gpu_id = available_gpus[0]
        else:
            # With multiple GPUs, assign containers in a round-robin fashion
            gpu_id = available_gpus[i % num_gpus]

        ego_cfgs = None
        if worker_agent_configs and i < len(worker_agent_configs):
            ego_cfgs = list(worker_agent_configs[i])

        op_cfg = CtnConfig(
            idx=i,
            container_name=f"{run_tag}_{i}",
            gpu=gpu_id,
            random_seed=random_seed,  # Ensure different seeds for each container
            docker_image=carla_image,
            fps=carla_fps,
            is_sync_mode=is_sync,
            agent_config_paths=ego_cfgs,
        )
        logger.info(f"Container {op_cfg.container_name} started on GPU {gpu_id}.")
        operators.append(op_cfg)

    return CtnManager(operators)


@dataclass
class CtnConfig:
    idx: int
    container_name: str
    gpu: int
    random_seed: int = 42
    docker_image: str = 'carlasim/carla:0.9.10.1'
    fps: int = 25
    is_sync_mode: bool = True
    # per-ego agent config paths for this worker (ego k -> agent_config_paths[k]);
    # multi-seed = [single_ads], multi-ADS = [ads0, ads1, ...]. None = scenario default.
    agent_config_paths: Optional[List[str]] = None

    def to_dict(self):
        return {
            "idx": self.idx,
            "container_name": self.container_name,
            "gpu": self.gpu,
            "random_seed": self.random_seed,
            "docker_image": self.docker_image,
            "fps": self.fps,
            "is_sync_mode": self.is_sync_mode,
            "agent_config_paths": self.agent_config_paths,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            idx=data.get("idx", 0),
            container_name=data.get("container_name", "default_ctn"),
            gpu=data.get("gpu", 0),
            random_seed=data.get("random_seed", 42),
            docker_image=data.get("docker_image", 'carlasim/carla:0.9.10.1'),
            fps=data.get("fps", 25),
            is_sync_mode=data.get("is_sync_mode", True),
            agent_config_paths=data.get("agent_config_paths", None),
        )

class CtnManager:

    def __init__(self, operators: List[CtnConfig]):
        """
        operators: list of CtnConfig
        """
        self._all_ops = operators  # 保存所有 operator 引用
        
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()
        for op in self._all_ops:
            self.queue.put(op)

    @property
    def capacity(self):
        return len(self._all_ops)

    def acquire(self, block=True, timeout=None):
        op_config = self.queue.get(block=block, timeout=timeout)
        return op_config

    def release(self, operator):
        self.queue.put(operator)

    def shutdown(self):
        self.manager.shutdown()

if __name__ == "__main__":
    configs = [
        CtnConfig(idx=i, container_name=f"test_ctn_{i}", gpu=0)
        for i in range(3)
    ]
    cm = CtnManager(configs)
    cm.shutdown()
