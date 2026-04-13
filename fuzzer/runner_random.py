import os
import copy
import time
import random

from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from deap import base, creator, tools

from registry import FUZZER_REGISTRY
from scenario_runner.ctn_manager import CtnSimOperator

from fuzzer.runner_base import Fuzzer, FuzzSeed

from .mutator.random_sample import RandomSampler
from .feedback.random_feedback import RandomFeedbackCalculator
from .oracle.general_oracle import ScenarioOracle
from .scenario_space import ScenarioODDSpace


@FUZZER_REGISTRY.register("fuzzer.random")
class RandomFuzzer(Fuzzer):
    """Random scenario fuzzer — samples new scenarios each generation,
    evaluates them, and keeps the best via simple GA selection."""

    def __init__(
        self,
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig,
    ):
        super().__init__(fuzzer_config, agent_config, scenario_config)

        # extra directories
        for d in ("feedback_cache", "sample_cache", "fuzz_cache"):
            os.makedirs(os.path.join(self.output_root, d), exist_ok=True)

        # 1. scenario space
        self.scenario_space = ScenarioODDSpace(self.pipeline_config['scenario_space'])

        # 2. mutator
        self.mutator = RandomSampler(self.scenario_space, self.mutator_config)

        # 3. feedback
        self.feedback = RandomFeedbackCalculator(self.feedback_config)

        # 4. oracle
        self.oracle = ScenarioOracle(self.oracle_config)

        # GA parameters
        self.population_size = fuzzer_config.get('population_size', 5)
        self.mutation_prob = fuzzer_config.get('mutation_prob', 0.8)
        self.best_score = float('inf')

        # 5. DEAP setup
        self.setup_deap()

        if not self.resume:
            self.time_counter = 0.0
            logger.info("Start from scratch, time counter reset to 0.")

    # ──────────────────────────────────────────────────────────────
    # DEAP
    # ──────────────────────────────────────────────────────────────

    def setup_deap(self):
        """Register DEAP types and operators."""
        # Fitness: minimize (lower collision distance = more dangerous)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", FuzzSeed, fitness=creator.FitnessMin)

        super().setup_deap()
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selRandom)

    def assign_feedback_to_ind(self, ind, feedback_result):
        """Single-objective: collision proximity score."""
        ind.fitness.values = (feedback_result['single_score'],)
        return ind

    # ──────────────────────────────────────────────────────────────
    # Mutation: random re-sample
    # ──────────────────────────────────────────────────────────────

    def _mutate(self, source_seed: FuzzSeed) -> tuple:
        """Mutate by re-sampling a new scenario from the scenario space."""
        seed = self.random_sample_seed()
        seed.set_id(source_seed.id)  # keep the same id slot
        return (seed,)

    def random_sample_seed(self, max_retries=3) -> FuzzSeed:
        """Sample a new scenario. Retries with container restart on failure."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Sampling new scenario (attempt {attempt}/{max_retries}) ...")
                ctn_config = self.ctn_manager.acquire()
                ctn_operator = CtnSimOperator(
                    idx=ctn_config.idx,
                    container_name=ctn_config.container_name,
                    gpu=ctn_config.gpu,
                    random_seed=ctn_config.random_seed,
                    docker_image=ctn_config.docker_image,
                    fps=ctn_config.fps,
                    is_sync_mode=ctn_config.is_sync_mode,
                )

                sampled_scenario = self.mutator.sample(
                    ctn_operator=ctn_operator,
                    ego_entry_point=self.agent_entry_point,
                    ego_config_path=self.agent_config_path,
                )

                # Disconnect from CARLA before subprocess uses the same container
                ctn_operator.cleanup()
                self.ctn_manager.release(ctn_config)

                if sampled_scenario is not None:
                    return FuzzSeed(id="unnamed", scenario=sampled_scenario)

                logger.warning(f"Sampling returned None, retrying...")

            except Exception as e:
                logger.error(f"Sampling failed (attempt {attempt}): {e}")
                try:
                    self.ctn_manager.release(ctn_config)
                except Exception:
                    pass

            # Restart containers before next attempt
            if attempt < max_retries:
                self.restart_containers()

        raise RuntimeError(f"Failed to sample scenario after {max_retries} attempts")

    # ──────────────────────────────────────────────────────────────
    # Checkpoint extensions
    # ──────────────────────────────────────────────────────────────

    def _get_checkpoint_data(self) -> dict:
        data = super()._get_checkpoint_data()
        data['population_size'] = self.population_size
        data['best_score'] = self.best_score
        return data

    def _restore_checkpoint_data(self, data: dict):
        super()._restore_checkpoint_data(data)
        self.population_size = data.get('population_size', self.population_size)
        self.best_score = data.get('best_score', float('inf'))

    # ──────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────

    def close(self):
        super().close()
        if hasattr(self, 'feedback') and self.feedback is not None:
            del self.feedback

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def _run(self, start_time):
        """Random fuzzing loop: sample batch → evaluate → record → repeat."""
        container_restart_interval = 300.0  # restart containers every 5 minutes
        last_restart_time = time.time()

        while not self.termination_check(start_time):
            self.global_search_step += 1
            logger.info(f"=== Generation {self.global_search_step} ===")

            # Periodic container restart to keep CARLA healthy
            if time.time() - last_restart_time > container_restart_interval:
                self.restart_containers()
                last_restart_time = time.time()

            # Sample a fresh batch each generation
            batch = []
            for i in range(self.population_size):
                ind_id = f"gen_{self.global_search_step}_ind_{i}"
                try:
                    seed = self.random_sample_seed()
                except RuntimeError as e:
                    logger.error(f"Skipping gen {self.global_search_step} ind {i}: {e}")
                    continue
                ind = creator.Individual(**seed.to_deap_args())
                ind.set_id(ind_id)
                batch.append([ind])

            if not batch:
                logger.warning("No scenarios sampled this generation, skipping.")
                continue

            # Evaluate
            batch = self.toolbox.evaluate(batch)

            # Track best
            for ind_wrapper in batch:
                ind = ind_wrapper[0]
                if ind.fitness.valid:
                    val = ind.fitness.values[0]
                    if val < self.best_score:
                        self.best_score = val

            self.record_logbook(self.global_search_step, batch)

            logger.info(
                f"[Gen {self.global_search_step}] "
                f"Best so far = {self.best_score:.4f}, "
                f"F_corpus size = {len(self.F_corpus)}"
            )

            self.save_checkpoint()

        logger.info("Time budget exhausted. Fuzzing complete.")
