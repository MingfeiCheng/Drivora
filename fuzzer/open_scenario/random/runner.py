import os
import copy
import json
import pickle
import random
import multiprocessing

from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from deap import base, creator, tools

from registry import FUZZER_REGISTRY

from scenario_runner.config import GlobalConfig
from scenario_runner.ctn_manager import CtnSimOperator

from fuzzer.open_scenario.base import Fuzzer, FuzzSeed
from tools.recorder_tool import load_observation, load_runtime_result, visualize_trajectories

from .mutator import ScenarioMutator
from .feedback import FeedbackCalculator
from .oracle import ScenarioOracle

@FUZZER_REGISTRY.register("fuzzer.open_scenario.random")
class RandomFuzzer(Fuzzer):
    
    def __init__(
        self, 
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig
    ):
        super(RandomFuzzer, self).__init__(
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
        self.population_size = self.pipeline_config.get('population_size', 1)
        self.mutation_prob = self.pipeline_config.get('mutation_prob', 0.5)
        
        # 6. internal parameters & checkpoint information
        # internal parameters used in fuzzer
        self.generation_step = 0
        self.seed_recorder = []
        self.F_corpus = [] # save all expected corpus
        # used for saving
        self.pop = []
        self.best_score = float("inf") # min is better
        
        # 7. load initial seed
        self.initial_seed = FuzzSeed.load_from_scenario_file(self.seed_path)
        
        # 8. load checkpoint path
        if self.resume:
            # NOTE: save iteration into checkpoint, if not resume start from 0
            self.load_checkpoint()
        else:
            self.time_counter = 0.0
            logger.info("Start from scratch, time counter reset to 0.")    

    def setup_deap(self):
        self.toolbox = base.Toolbox()
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", FuzzSeed, fitness=creator.FitnessMin)
            
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mutate", self.mutation)
        self.toolbox.register("select", tools.selRandom) # random
        
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
            
            self.population_size = checkpoint_data['population_size']
            self.generation_step = checkpoint_data['generation_step']
            self.F_corpus = checkpoint_data['F_corpus']
            _pop = [
                FuzzSeed.load_from_dict(seed_dict) for seed_dict in checkpoint_data['pop']
            ]
            self.pop = [
                creator.Individual(
                    id=seed.id,
                    scenario=seed.scenario,
                    oracle_result=seed.oracle_result,
                    feedback_result=seed.feedback_result,
                    is_expected=seed.is_expected,
                ) for seed in _pop
            ]
            self.initial_seed = FuzzSeed.load_from_dict(checkpoint_data['initial_seed'])
            
            logger.info('Load checkpoint from {}', self.checkpoint_path)
        else:
            logger.warning('Checkpoint file not found, start from scratch.')
            
    def save_checkpoint(self):
        """
        Save checkpoint
        """
        checkpoint_data = {
            "population_size": self.population_size,
            "generation_step": self.generation_step,
            "F_corpus": self.F_corpus,
            "seed_recorder": self.seed_recorder,
            "pop": [seed.to_dict() for seed in self.pop], # TODO: check this
            "initial_seed": self.initial_seed.to_dict(),
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)            
        logger.info('Save checkpoint to {}', self.checkpoint_path)
        
        # save a result overview
        overview_res = {
            'summary': {
                'total_iterations': self.generation_step,
                'F_size': len(self.F_corpus),
                'time_budget_hours': self.time_budget,
                'time_used_hours': (self.time_counter / 3600.0 if self.time_budget is not None else None),
                'best_score': self.best_score
            },
            'details': {
            }
        }
        
        for seed_brief in self.seed_recorder:
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
    
    def mutation(self, source_seed: FuzzSeed) -> FuzzSeed:
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
        
        op_seed = self.mutator.generate(
            source_seed=op_seed,
            ctn_operator=ctn_operator,
            ego_entry_point=self.agent_entry_point,
            ego_config_path=self.agent_config_path
        )       
        self.ctn_manager.release(ctn_config) 
        return (op_seed,)

    def evaluate(self, individuals: list):
        """
        Evaluate a list of individuals using multiprocessing + container pool.
        Args:
            individuals (list): List of individuals to evaluate.
        Returns:
            list of (individual, fitness_score)
        """
        # Run execution in parallel (container + scenario simulation)
        exec_results = self.execute_population(individuals)
        results = []

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
            ind.is_expected = oracle_result['expected']
            
            ind.feedback_result = feedback_result

            # Example: use feedback score as fitness
            score = feedback_result.get("score", float("inf"))
            ind.fitness.values = (score, )
            
            # add breif
            self.seed_recorder.append({
                'id': ind.id,
                'is_expected': ind.is_expected,
                'oracle_result': ind.oracle_result,
                'feedback_result': ind.feedback_result,
            })
            
            individuals[ind_index] = ind

        return individuals

    
    def run(self):
        start_time = datetime.now()

        # ========== Initialize logbook ==========
        logbook = tools.Logbook()
        logbook.header = ["gen", "fitness", "best_so_far"]
        best_fitness_so_far = float("inf")

        # ========== Initialize population ==========
        if len(self.pop) != self.population_size:
            self.pop = []
            for i in range(self.population_size):
                logger.info(f"Initialize individual {i} in GA population.")

                ind_id = f'gen_{self.generation_step}_ind_{i}'

                ind = creator.Individual(
                    id=ind_id,
                    scenario=copy.deepcopy(self.initial_seed.scenario),
                    oracle_result={},
                    feedback_result={},
                    is_expected=False,
                )
                ind.set_id(ind_id)

                # mutation
                ind, = self.toolbox.mutate(ind)
                del ind.fitness.values
                self.pop.append(ind)

        # evaluate the initial population
        self.pop = self.evaluate(self.pop)
        self.save_checkpoint()

        # ========== GA loop ==========
        while True:
            
            if self.termination_check(start_time):
                break

            self.generation_step += 1
            logger.info(f"=== Generation {self.generation_step} ===")

            # selection + clone
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = list(map(copy.deepcopy, offspring))

            # mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    mut_id = f'gen_{self.generation_step}_ind_{i}'
                    offspring[i].set_id(mut_id)
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    if hasattr(offspring[i].fitness, "values"):
                        del offspring[i].fitness.values

            # evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                invalid_ind = self.evaluate(invalid_ind)

            # update population
            self.pop[:] = offspring

            # ========== Metrics ==========
            best = tools.selBest(self.pop, 1)[0]
            logger.debug(f"Best individual: {best.id} with fitness {best.fitness.values}")
            best_val = best.fitness.values[0]
            best_fitness_so_far = min(best_fitness_so_far, best_val)

            logbook.record(
                gen=self.generation_step,
                fitness=best_val,
                best_so_far=best_fitness_so_far
            )

            logger.info(
                f"[Gen {self.generation_step}] "
                f"Best of generation = {best_val:.4f}, "
                f"Best so far = {best_fitness_so_far:.4f}"
            )

            # save results
            best_dir = os.path.join(self.output_root, "best_seeds", f"iter_{self.generation_step}")
            os.makedirs(best_dir, exist_ok=True)
            with open(os.path.join(best_dir, "logbook.json"), 'w') as f:
                json.dump(logbook, f, indent=2, default=str)

            logger.info(f"[Global Iter {self.generation_step}] Saved best seed to {best_dir}")
            self.save_checkpoint()
