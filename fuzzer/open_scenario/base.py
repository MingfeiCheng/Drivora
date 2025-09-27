import os
import json
import queue
import multiprocessing as mp

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from typing import Dict
from dataclasses import dataclass

from scenario_corpus.open_scenario.config import ScenarioConfig
from scenario_runner.config import GlobalConfig
from scenario_runner.ctn_manager import create_ctn_manager


@dataclass
class FuzzSeed:
    """Data container for fuzzing seeds."""
    id: str
    scenario: ScenarioConfig
    oracle_result: Dict
    feedback_result: Dict
    is_expected: bool

    def set_id(self, new_id: str):
        self.id = new_id
        self.scenario.id = new_id

    @classmethod
    def load_from_scenario_file(cls, config_path: str):
        """Load the initial seed from the given path"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        scenario = ScenarioConfig.model_validate(data)
        return cls(
            id="init_seed",
            scenario=scenario,
            oracle_result={},
            feedback_result={},
            is_expected=False,
        )

    @classmethod
    def load_from_dict(cls, data: dict):
        scenario = ScenarioConfig.model_validate(data['scenario'])
        return cls(
            id=data.get("id", "seed"),
            scenario=scenario,
            oracle_result=data.get("oracle_result", {}),
            feedback_result=data.get("feedback_result", {}),
            is_expected=data.get("is_expected", False),
        )

    def to_dict(self):
        return {
            "id": self.id,
            "scenario": self.scenario.model_dump(),
            "oracle_result": self.oracle_result,
            "feedback_result": self.feedback_result,
            "is_expected": self.is_expected,
        }

class Fuzzer(object):
    
    SCENARIO_ENTRY = "scenario_corpus.open_scenario.scenario:OpenScenario"
    MANAGER_NAME = "default"

    def __init__(
            self,
            fuzzer_config: DictConfig,
            agent_config: DictConfig,
            scenario_config: DictConfig
    ):
        self.fuzzer_config = fuzzer_config
        self.agent_config = agent_config
        self.scenario_config = scenario_config
        
        self.resume = GlobalConfig.resume
        self.output_root = GlobalConfig.output_root
        
        # result dir
        self.result_folder = os.path.join(self.output_root, 'results')
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
            
        # tmp dir
        self.tmp_dir = os.path.join(self.output_root, 'tmp') # save some temporary files
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.checkpoint_path = os.path.join(self.tmp_dir, 'checkpoint.pkl')
        
        # save cfgs
        fuzzer_config_path = os.path.join(self.output_root, 'fuzzer_config.yaml')
        OmegaConf.save(config=fuzzer_config, f=fuzzer_config_path)
        logger.info('Fuzzer config saved to {}', fuzzer_config_path)
        
        # basic fuzzer config
        self.time_budget = fuzzer_config.get('time_budget', 1.0)  # in hours        
        # basic scenario config
        self.seed_path = self.scenario_config.get('seed_path', None)
        if self.seed_path is None or not os.path.isfile(self.seed_path):
            logger.error(f"Please provide a valid seed scenario path: {self.seed_path}")
            raise ValueError("Invalid seed scenario path.")
        
        # basic agent config
        self.agent_entry_point = self.agent_config.get('entry_point', None)
        if self.agent_entry_point is None:
            logger.error(f"Please provide a valid agent entry point: {self.agent_entry_point}")
            raise ValueError("Invalid agent entry point.")
        self.agent_config_path = self.agent_config.get('config_path', {}) # can be empty
        if self.agent_config_path is None:
            logger.error(f"Please provide a valid agent config path: {self.agent_config_path}")
            raise ValueError("Invalid agent config path.")
        
        # check time counter & load previous time, before execution, we check if we have already finished the testing
        self.time_counter_file = os.path.join(self.tmp_dir, 'time_counter.txt')
        self.time_counter = 0.0
        if os.path.exists(self.time_counter_file):
            with open(self.time_counter_file, 'r') as f:
                line = f.readline()
                if line:
                    self.time_counter = float(line.rstrip())
                    
        if self.termination_check(datetime.now()):
            logger.info(f"Already tested for {self.time_budget} hours, skip.")
            return
        
        # Load detailed configs
        fuzzer_config_path = fuzzer_config.get('config_path', None)
        if fuzzer_config_path is None or not os.path.isfile(fuzzer_config_path):
            logger.error(f"Please provide a valid fuzzer config path: {fuzzer_config_path}")
            raise ValueError("Invalid fuzzer config path.")
        
        self.pipeline_config = OmegaConf.load(fuzzer_config_path)
        
        # 1. create container manager
        self.ctn_manager = create_ctn_manager(
            run_tag=GlobalConfig.run_tag,
            carla_image=GlobalConfig.carla_image,
            carla_fps=GlobalConfig.carla_fps,
            random_seed=GlobalConfig.carla_random_seed,
            is_sync=GlobalConfig.carla_is_sync
        )
        
        # 2. mutation 
        self.mutator_config = self.pipeline_config.get('mutator', {})

        # 3. create the feedback calculator
        # NOTE: fuzzer-specific feedback calculator, if you want to use a different feedback calculator
        self.feedback_config = self.pipeline_config.get('feedback', {})
        
        # 4. create oracle (here is the offline oracle, online shoud defined criteria in feedback calculator)
        self.oracle_config = self.pipeline_config.get('oracle', {})
        
        # 5. setup toolbox
        self.toolbox = None
        self.setup_deap()
        
    def setup_deap(self):
        raise NotImplementedError("Method setup_deap not implemented")

    def run(
            self
    ):
        raise NotImplementedError("Method run not implemented")
    
    def close(
            self
    ):
        """
        Close the fuzzer, such as closing the database connection, etc.
        """
        pass


    def termination_check(self, start_time) -> bool:
        curr_time = datetime.now()
        t_delta = (curr_time - start_time).total_seconds()
        total_time = t_delta + self.time_counter
        # update total time
        with open(self.time_counter_file, 'w') as f:
            f.write(str(total_time))
            f.write('\n')
            
        if (self.time_budget is not None) and total_time / 3600.0 > self.time_budget:
            return True
        
        if self.time_budget is None:
            logger.info(f"Note that you set [Infinite] testing budget.")
            
        return False
    
    @staticmethod
    def execution_instance(
        output_dir,
        ctn_config,
        scenario_entry_point,
        manager_name,
        seed_dict, 
        max_sim_time,
        debug,
        pytree_debug,
        open_vis,
    ):
        """
        Static evaluation function for multiprocessing.
        Args:
            seed_dict (dict): Serialized FuzzSeed (dict form).
            global_cfg: GlobalConfig (must be serializable / simple namespace).
            oracle_cfg (dict): Oracle config.
            feedback_cfg (dict): Feedback config.
        Returns:
            tuple: (fitness_score, updated_seed_dict)
        """
        import os, json, shutil
        
        from loguru import logger
        from scenario_runner.scenario_executor import run_scenario

        # reconstruct objects
        seed = FuzzSeed.load_from_dict(seed_dict)
        
        logger.info(f"Evaluating seed {seed.id} in process {os.getpid()} ...")
        logger.info(f"ctn_config: {ctn_config}")
        
        # prepare scenario directory
        scenario_dir = os.path.join(output_dir, f"{seed.id}")
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)
        os.makedirs(scenario_dir)

        # save seed to file (so run_scenario can access)
        seed_path = os.path.join(scenario_dir, "seed.json")
        with open(seed_path, "w") as f:
            json.dump(seed.to_dict(), f, indent=4)
            
        scenario_path = os.path.join(scenario_dir, "scenario.json")
        with open(scenario_path, "w") as f:
            json.dump(seed.scenario.model_dump(), f, indent=4)

        # define evaluation task
        run_status = run_scenario(
            scenario_entry_point=scenario_entry_point,
            scenario_config=seed.scenario,
            ctn_config=ctn_config,
            scenario_dir=scenario_dir,
            manager_name=manager_name,
            max_sim_time=max_sim_time,
            debug=debug,
            pytree_debug=pytree_debug,
            open_vis=open_vis,
        )
        
        if not run_status:
            logger.error(f"Scenario {seed.id} run failed.")
            return (float("inf"),)

        # overwrite seed file
        with open(seed_path, "w") as f:
            json.dump(seed.to_dict(), f, indent=4)
            
        return scenario_dir

    @staticmethod
    def worker(task_queue, result_queue, ctn_manager, fuzzer_execution_instance):
        """Worker process that pulls tasks from the queue and executes them"""
        while True:
            try:
                ind_index, partial_task = task_queue.get(timeout=2)  
            except queue.Empty:
                break

            try:
                # Acquire container
                ctn_cfg = ctn_manager.acquire()
                gpu_id = getattr(ctn_cfg, "gpu", 0)

                if gpu_id is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
                # Merge ctn_cfg into the task
                full_task = list(partial_task)
                full_task[1] = ctn_cfg.to_dict()   # index 1 is ctn_config
                scenario_dir = fuzzer_execution_instance(*full_task)

                result_queue.put((ind_index, scenario_dir))

            except Exception as e:
                logger.error(f"Worker error: {e}")
            finally:
                ctn_manager.release(ctn_cfg)

    def execute_population(self, individuals):
        results = []
        task_queue = mp.Manager().Queue()
        result_queue = mp.Manager().Queue()

        # Prepare tasks
        for ind_index, ind in enumerate(individuals):
            task = (
                self.result_folder,
                {},  # placeholder, can be replaced with ctn_cfg.to_dict() in worker if needed
                self.SCENARIO_ENTRY,
                self.MANAGER_NAME,
                ind.to_dict(),
                GlobalConfig.max_sim_time,
                GlobalConfig.debug,
                GlobalConfig.pytree_debug,
                GlobalConfig.open_vis,
            )
            task_queue.put((ind_index, task))

        # Start workers
        num_workers = min(GlobalConfig.parallel_num, len(individuals))
        workers = []
        for _ in range(num_workers):
            p = mp.Process(
                target=self.worker,
                args=(task_queue, result_queue, self.ctn_manager, self.execution_instance),  # pass instance method
            )
            p.start()
            workers.append(p)

        # Wait for workers to finish
        for p in workers:
            p.join()

        # Collect results
        while not result_queue.empty():
            results.append(result_queue.get())

        return results
