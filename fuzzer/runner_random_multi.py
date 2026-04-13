"""
RandomMultiFuzzer — Random scenario fuzzer with multi-ego support.

Tests the same ADS agent controlling multiple ego vehicles simultaneously.
The number of ego vehicles is controlled by scenario_space.ego_space.num.

Differences from RandomFuzzer:
  - ego_space.num can be > 1 (e.g., [2, 3] for 2-3 ego vehicles)
  - Feedback: per-ego metrics + ego-to-ego min distance
  - Same entry_point/config_path applied to all egos
"""

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
from .feedback.multi_feedback import MultiADSFeedbackCalculator
from .oracle.general_oracle import ScenarioOracle
from .scenario_space import ScenarioODDSpace


@FUZZER_REGISTRY.register("fuzzer.random_multi")
class RandomMultiFuzzer(Fuzzer):
    """Random scenario fuzzer for multi-ego (same ADS) testing."""

    def __init__(
        self,
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig,
    ):
        super().__init__(fuzzer_config, agent_config, scenario_config)

        for d in ("feedback_cache", "sample_cache", "fuzz_cache"):
            os.makedirs(os.path.join(self.output_root, d), exist_ok=True)

        # 1. Scenario space (ego_space.num controls how many ego vehicles)
        self.scenario_space = ScenarioODDSpace(self.pipeline_config['scenario_space'])

        # 2. Mutator
        self.mutator = RandomSampler(self.scenario_space, self.mutator_config)

        # 3. Feedback (multi-ego: per-ego + ego-ego min distance)
        self.feedback = MultiADSFeedbackCalculator(self.feedback_config)

        # 4. Oracle
        self.oracle = ScenarioOracle(self.oracle_config)

        # GA parameters
        self.population_size = fuzzer_config.get('population_size', 5)
        self.mutation_prob = fuzzer_config.get('mutation_prob', 0.8)
        self.best_score = float('inf')

        self.setup_deap()

        if not self.resume:
            self.time_counter = 0.0
            logger.info("Start from scratch, time counter reset to 0.")

    # ──────────────────────────────────────────────────────────────
    # DEAP
    # ──────────────────────────────────────────────────────────────

    def setup_deap(self):
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", FuzzSeed, fitness=creator.FitnessMin)

        super().setup_deap()
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selRandom)

    def assign_feedback_to_ind(self, ind, feedback_result):
        ind.fitness.values = (feedback_result['score'],)
        return ind

    # ──────────────────────────────────────────────────────────────
    # Sampling
    # ──────────────────────────────────────────────────────────────

    def _mutate(self, source_seed: FuzzSeed) -> tuple:
        seed = self.random_sample_seed()
        seed.set_id(source_seed.id)
        return (seed,)

    def random_sample_seed(self, max_retries=3) -> FuzzSeed:
        """Sample a multi-ego scenario. Same ADS for all egos."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Sampling multi-ego scenario (attempt {attempt}/{max_retries}) ...")
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

                # Same entry_point/config_path for all egos
                sampled_scenario = self.mutator.sample(
                    ctn_operator=ctn_operator,
                    ego_entry_point=self.agent_entry_point,
                    ego_config_path=self.agent_config_path,
                )

                ctn_operator.cleanup()
                self.ctn_manager.release(ctn_config)

                if sampled_scenario is not None:
                    return FuzzSeed(id="unnamed", scenario=sampled_scenario)

                logger.warning("Sampling returned None, retrying...")

            except Exception as e:
                logger.error(f"Sampling failed (attempt {attempt}): {e}")
                try:
                    self.ctn_manager.release(ctn_config)
                except Exception:
                    pass

            if attempt < max_retries:
                self.restart_containers()

        raise RuntimeError(f"Failed to sample scenario after {max_retries} attempts")

    # ──────────────────────────────────────────────────────────────
    # Checkpoint
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
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def _run(self, start_time):
        container_restart_interval = 300.0
        last_restart_time = time.time()

        while not self.termination_check(start_time):
            self.global_search_step += 1
            logger.info(f"=== Generation {self.global_search_step} ===")

            if time.time() - last_restart_time > container_restart_interval:
                self.restart_containers()
                last_restart_time = time.time()

            batch = []
            for i in range(self.population_size):
                ind_id = f"gen_{self.global_search_step}_ind_{i}"
                try:
                    seed = self.random_sample_seed()
                except RuntimeError as e:
                    logger.error(f"Skipping ind {i}: {e}")
                    continue
                ind = creator.Individual(**seed.to_deap_args())
                ind.set_id(ind_id)
                batch.append([ind])

            if not batch:
                logger.warning("No scenarios sampled, skipping.")
                continue

            batch = self.toolbox.evaluate(batch)

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
                f"F_corpus = {len(self.F_corpus)}"
            )

            self.save_checkpoint()

        logger.info("Time budget exhausted. Multi-ego fuzzing complete.")

    def close(self):
        super().close()
