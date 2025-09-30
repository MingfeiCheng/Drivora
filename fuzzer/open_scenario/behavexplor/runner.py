import os
import copy
import json
import pickle
import random
import multiprocessing
import numpy as np

from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from deap import base, tools
from typing import List, Dict, Any
from dataclasses import dataclass, field, fields, replace
from registry import FUZZER_REGISTRY

from scenario_runner.config import GlobalConfig
from scenario_runner.ctn_manager import CtnSimOperator

from fuzzer.open_scenario.base import Fuzzer, FuzzSeed
from tools.recorder_tool import load_observation, load_runtime_result, visualize_trajectories

from .mutator import ScenarioMutator
from .feedback import FeedbackCalculator
from .oracle import ScenarioOracle

@dataclass
class BehSeed(FuzzSeed):
    is_new: bool = field(default=False)          # whether it has new coverage
    parent_index: int = field(default=-1)        # index in all_seeds
    select_num: int = field(default=0)
    fail_num: int = field(default=0)
    safety_score: float = field(default=1.0)     # [0.0, 1.0], lower is better
    diversity_score: float = field(default=0.0)  # [0.0, 1.0], higher is better
    energy: float = field(default=1.0)           # [0.0, 1.0], higher is better
    scenario_dir: str = field(default="")        # directory where the scenario is stored

    @classmethod
    def load_from_scenario_file(cls, config_path: str) -> "BehSeed":
        base = super().load_from_scenario_file(config_path)
        # upgrade FuzzSeed instance into BehSeed by replacing/adding fields
        return replace(base)

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> "BehSeed":
        base = super().load_from_dict(data)
        # collect extra fields defined in BehSeed but not in FuzzSeed
        extra = {
            f.name: data.get(f.name, getattr(cls, f.name))
            for f in fields(cls)
            if f.name not in {f.name for f in fields(FuzzSeed)}
        }
        return replace(base, **extra)

    def to_dict(self) -> dict:
        data = super().to_dict()
        # automatically include subclass-specific fields
        for f in fields(self):
            if f.name not in data:
                data[f.name] = getattr(self, f.name)
        return data

@FUZZER_REGISTRY.register("fuzzer.open_scenario.behavexplor")
class BehAVExplor(Fuzzer):
    
    def __init__(
        self, 
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig
    ):
        super(BehAVExplor, self).__init__(
            fuzzer_config,
            agent_config,
            scenario_config
        )
        
        # 2. create mutator
        # NOTE: fuzzer-specific mutator, if you want to use a different mutator, you can change it here
        self.mutator = ScenarioMutator(self.mutator_config)

        # 3. create the feedback calculator
        # NOTE: fuzzer-specific feedback calculator, if you want to use a different feedback calculator
        self.feedback = FeedbackCalculator(self.feedback_config) # 
        
        # 4. create oracle (here is the offline oracle, online shoud defined criteria in feedback calculator)
        self.oracle = ScenarioOracle(self.oracle_config)
        
        ###### The following should be in checkpoint ########
        # 5. pipeline config
        self.initial_corpus_size = self.pipeline_config.get('initial_corpus_size', 4)
        self.batch_size = GlobalConfig.parallel_num # we use the parallel number as batch size
        
        # 6. internal parameters & checkpoint information
        # internal parameters used in fuzzer
        # we need (1) all seeds, (2) corpus index and 
        self.generation_step = 0
        self.best_safety = 1.0 # lower is better
        self.best_diversity = 0.0 # larger is better
        self.all_seeds: List[BehSeed] = [] 
        self.seed_corpus: List[int] = [] # keeps existing seeds of index in all_seeds
        self.F_corpus: List[int] = [] # keeps valid fault seeds of index in all_seeds
        self.new_corpus: List[int] = [] # keeps valid new coverage seeds of index in all_seeds
        
        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "overall", "safety", "diversity"]
        
        # 7. load initial seed
        self.initial_seed = BehSeed.load_from_scenario_file(self.seed_path)
        
        # 8. load checkpoint path
        if self.resume:
            # NOTE: save iteration into checkpoint, if not resume start from 0
            self.load_checkpoint()
        else:
            self.time_counter = 0.0
            logger.info("Start from scratch, time counter reset to 0.")    

    def setup_deap(self):
        
        self.toolbox = base.Toolbox()
        
        self.toolbox.register("initialize_individual", self.initialize_individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mutate", self.mutation)
        self.toolbox.register("select", self.select) # random
        self.toolbox.register("update_corpus", self.update_corpus)
        
        # setup parallel pool
        self.parallel_num = GlobalConfig.parallel_num
        if self.parallel_num > 1:
            self.pool = multiprocessing.Pool(processes=self.parallel_num)
            self.toolbox.register("map", self.pool.map)
            logger.info(f"Parallel evaluation with {self.parallel_num} processes.")
        else:
            self.toolbox.register("map", map)
            self.pool = None
            logger.info("Sequential evaluation (parallel_num=1).")
    
    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.initial_corpus_size = checkpoint_data['initial_corpus_size']
            self.generation_step = checkpoint_data['generation_step']
            self.F_corpus = checkpoint_data['F_corpus']
            self.seed_corpus = checkpoint_data['seed_corpus']
            self.new_corpus = checkpoint_data['new_corpus']
            self.best_safety = checkpoint_data['best_safety']
            self.best_diversity = checkpoint_data['best_diversity']
            
            self.all_seeds = [
                BehSeed.load_from_dict(seed_dict) for seed_dict in checkpoint_data['all_seeds']
            ]
            self.initial_seed = BehSeed.load_from_dict(checkpoint_data['initial_seed'])
            
            self.feedback.load_checkpoint(checkpoint_data['feedback_model'])
            
            # load logbook
            logbook_file = os.path.join(self.output_root, "logbook.json")
            if os.path.exists(logbook_file):
                with open(logbook_file, 'r') as f:
                    log_data = json.load(f)
                    for entry in log_data:
                        self.logbook.record(**entry)
                        
            logger.info('Load checkpoint from {}', self.checkpoint_path)
        else:
            logger.warning('Checkpoint file not found, start from scratch.')
            
    def save_checkpoint(self):
        """
        Save checkpoint
        """
        checkpoint_data = {
            "best_safety": self.best_safety,
            "best_diversity": self.best_diversity,
            "initial_corpus_size": self.initial_corpus_size,
            "generation_step": self.generation_step,
            "F_corpus": self.F_corpus,
            "seed_corpus": self.seed_corpus,
            "new_corpus": self.new_corpus,
            "all_seeds": [seed.to_dict() for seed in self.all_seeds],
            "initial_seed": self.initial_seed.to_dict(),
            "feedback_model": self.feedback.save_checkpoint(os.path.dirname(self.checkpoint_path))
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)            
        logger.info('Save checkpoint to {}', self.checkpoint_path)
        
        # save a result overview
        overview_res = {
            'summary': {
                'total_iterations': self.generation_step,
                'F_size': len(self.F_corpus),
                'best_safety': self.best_safety,
                'best_diversity': self.best_diversity,
                'new_coverage_size': len(self.new_corpus),
                'time_budget_hours': self.time_budget,
                'time_used_hours': self.used_time / 3600.0,
                'seed_corpus_size': len(self.seed_corpus),
                'total_seeds': len(self.all_seeds),
                'F_corpus': self.F_corpus,
            },
            'details': {
            }
        }
        
        for seed_brief in self.all_seeds:
            overview_item_detail = {
                'scenario_id': seed_brief['id'],
                'is_expected': seed_brief['is_expected'],
                'oracle_result': seed_brief['oracle_result'],
                'feedback_result': seed_brief['feedback_result'],
            }
            overview_res['details'][seed_brief['id']] = overview_item_detail
            
        overview_res_file = os.path.join(self.output_root, 'overview.json')
        with open(overview_res_file, 'w') as f:
            json.dump(overview_res, f, indent=4)
            
        # save lookbook
        with open(os.path.join(self.output_root, "logbook.json"), 'w') as f:
            json.dump(self.logbook, f, indent=2, default=str)
    
    def initialize_individual(self, ind: BehSeed) -> BehSeed:
        # initialize the individual by mutating the initial seed
        logger.info("Initialize individual ...")
        
        op_seed = copy.deepcopy(ind)
        
        ctn_config = self.ctn_manager.acquire()
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
        
        op_seed = self.mutator.generate(
            source_seed=op_seed,
            ctn_operator=ctn_operator,
            ego_entry_point=self.agent_entry_point,
            ego_config_path=self.agent_config_path
        )       
        self.ctn_manager.release(ctn_config) 
        return op_seed
    
    def update_corpus(self, seed: BehSeed):
        
        # update to dataset
        seed.energy = 1.0 # as new seed
        self.all_seeds.append(copy.deepcopy(seed))
        parent_index = seed.parent_index
        
        if parent_index <= 0:
            logger.info("Add seed {} to corpus (no parent).", seed.id)
            self.seed_corpus.append(len(self.all_seeds) - 1)
            return
        
        # find violations
        if seed.is_expected == True:
            logger.info("Add seed {} to corpus (expected).", seed.id)
            self.F_corpus.append(len(self.all_seeds) - 1)
            self.all_seeds[parent_index].fail_num += 1
                        
        # update for safe seeds
        parent_safety_score = self.all_seeds[parent_index].safety_score
        child_safety_score = seed.safety_score
        
        parent_energy = self.all_seeds[parent_index].energy
        parent_select_num = self.all_seeds[parent_index].select_num
        parent_fail_num = self.all_seeds[parent_index].fail_num
        
        delta_safety = parent_safety_score - child_safety_score
        
        if parent_select_num == 0:
            delta_fail = 0.0
        else:
            delta_fail = parent_fail_num / float(parent_select_num + 1e-5)
        
        delta_select = - 0.1
        new_energy = parent_energy + 0.3 * np.tanh(delta_safety) + 0.5 * delta_fail + delta_select
        new_energy = float(np.clip(new_energy, 0.0, 1.0))
        self.all_seeds[parent_index].energy = new_energy
        
        # update corpus
        if not seed.is_expected:
            if child_safety_score < parent_safety_score:
                logger.info("Add seed {} to corpus (better safety).", seed.id)
                self.seed_corpus.append(len(self.all_seeds) - 1)
            elif seed.is_new == True:
                logger.info("Add seed {} to corpus (new coverage).", seed.id)
                self.seed_corpus.append(len(self.all_seeds) - 1)
                self.new_corpus.append(len(self.all_seeds) - 1)
            else:
                logger.info("Do not add seed {} to corpus (not better).", seed.id)
        
    def select(self) -> BehSeed:
        
        corpus_energy = [self.all_seeds[i].energy for i in self.seed_corpus]
                
        select_probabilities = corpus_energy
        select_probabilities = np.array(select_probabilities) + 1e-5
        select_probabilities /= (select_probabilities.sum())
        source_seed_corpus_index = np.random.choice(list(np.arange(0, len(self.seed_corpus))), p=select_probabilities)
        source_seed_index = self.seed_corpus[source_seed_corpus_index]
        
        selected_seed = self.all_seeds[source_seed_index]
        selected_seed.select_num += 1
        self.all_seeds[source_seed_index] = selected_seed
        
        op_seed = copy.deepcopy(selected_seed)
        op_seed.parent_index = source_seed_index
        return op_seed
    
    def mutation(self, source_seed: BehSeed) -> BehSeed:
        # this should be in the logical scenario space
        logger.info("Start mutation ...")
        
        op_seed = copy.deepcopy(source_seed)
        
        ctn_config = self.ctn_manager.acquire()
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
        
        source_seed_energy = op_seed.energy
        
        if random.random() < source_seed_energy:
            mutation_stage = 'small'
        else:
            mutation_stage = 'large'
        
        op_seed = self.mutator.step(
            source_seed=op_seed,
            ctn_operator=ctn_operator,
            mutation_stage=mutation_stage
        )       
        self.ctn_manager.release(ctn_config) 
        return op_seed

    def evaluate(self, individuals: List[BehSeed]) -> List[BehSeed]:
        """
        Evaluate a list of individuals using multiprocessing + container pool.
        Args:
            individuals (list): List of individuals to evaluate.
        Returns:
            list of (individual, fitness_score)
        """
        # Run execution in parallel (container + scenario simulation)
        exec_results = self.execute_population(individuals)

        for ind_index, scenario_dir in exec_results:
            # Evaluate oracle and feedback on the produced scenario
            
            visualize_trajectories(scenario_dir)
            
            scenario_observation = load_observation(scenario_dir)
            runtime_oracle_results = load_runtime_result(scenario_dir)
            
            oracle_result = self.oracle.evaluate(scenario_observation, runtime_oracle_results)
            
            # some info can reused in feedback
            feedback_result = self.feedback.evaluate(
                scenario_observation, 
                oracle_result
            )

            ind = individuals[ind_index]
            ind.oracle_result = oracle_result            
            ind.feedback_result = feedback_result
            
            # update some attributes
            ind.is_expected = oracle_result['expected']
            ind.safety_score = feedback_result['safety_score']
            ind.diversity_score = feedback_result['diversity_score']
            ind.is_new = feedback_result['is_new']
            ind.scenario_dir = scenario_dir
            
            individuals[ind_index] = ind

        return individuals

    def run(self):
        
        # minimize is better
        logger.info('===== Start Fuzzer (BehAVExplor) =====')
        start_time = datetime.now()
        
        while len(self.seed_corpus) < self.initial_corpus_size:
            
            if self.termination_check(start_time):
                break
            
            self.generation_step += 1
            logger.info(f'==================== Run initial Iter {self.generation_step} ====================')
            
            pop = []
            for i in range(self.batch_size):
                
                ind_id = f'gen_{self.generation_step}_ind_{i}'
                ind = copy.deepcopy(self.initial_seed)
                ind.set_id(ind_id)
                ind = self.toolbox.initialize_individual(ind)
                pop.append(ind)
                
            pop = self.toolbox.evaluate(pop)
            for ind in pop:
                self.toolbox.update_corpus(ind)
                
                # update best safety and diversity
                if ind.safety_score < self.best_safety:
                    self.best_safety = ind.safety_score
                if ind.diversity_score > self.best_diversity:
                    self.best_diversity = ind.diversity_score
        
        # initialize the coverage model
        if self.feedback.is_initialized == False:
            logger.info("Initialize coverage model ...")
            corpus_seeds = [self.all_seeds[i] for i in self.seed_corpus]
            scenario_dir_list = [seed.scenario_dir for seed in corpus_seeds]
            # load all observations
            scenario_observation_list = []
            for scenario_dir in scenario_dir_list:
                scenario_observation = load_observation(scenario_dir)
                scenario_observation_list.append(scenario_observation)
            # initialize coverage model
            self.feedback.initialize_coverage_model(
                scenario_observation_list
            )
        
        self.save_checkpoint()
         
        # start fuzzing
        logger.info(f'Initial corpus size reached: {len(self.seed_corpus)} seeds.')
        logger.info(f'Start fuzzing ...')

        while True:
            
            if self.termination_check(start_time):
                break
            
            self.generation_step += 1
            logger.info(f'==================== Run Iter {self.generation_step} ====================')
            
            pop = []
            for i in range(self.batch_size):
                
                ind_id = f'gen_{self.generation_step}_ind_{i}'
                ind = self.toolbox.select()
                ind.set_id(ind_id)                
                ind = self.toolbox.mutate(ind)
                
                pop.append(ind)
                                
            # evaluation
            pop = self.toolbox.evaluate(pop)
            
            for ind in pop:
                self.toolbox.update_corpus(ind)
                
                # update best safety and diversity
                if ind.safety_score < self.best_safety:
                    self.best_safety = ind.safety_score
                if ind.diversity_score > self.best_diversity:
                    self.best_diversity = ind.diversity_score
            
            self.save_checkpoint()
            
