"""
BehAVExplor — Coverage-guided scenario fuzzer.

Key ideas (from BehAVExplor paper):
  - Uses behavior coverage model (KMeans on ego state/action time series)
  - Seeds are added to corpus if they discover new behavior coverage
  - Energy-based seed selection: seeds that produce more violations get more energy
  - Two mutation stages: small (perturb parameters) vs large (resample from scratch)
  - Fitness is a combination of safety score (collision proximity) and diversity score
"""

import os
import copy
import time
import random
import numpy as np

from loguru import logger
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, field, fields, replace
from omegaconf import DictConfig
from deap import base, tools

from registry import FUZZER_REGISTRY
from scenario_runner.ctn_manager import CtnSimOperator

from fuzzer.runner_base import Fuzzer, FuzzSeed
from tools.recorder_tool import load_observation, load_runtime_result, visualize_trajectories

from .mutator.random_sample import RandomSampler
from .mutator.behavior_mutator import ScenarioMutator
from .feedback.behavior_feedback import BehaviorFeedbackCalculator
from .oracle.general_oracle import ScenarioOracle
from .scenario_space import ScenarioODDSpace


# ──────────────────────────────────────────────────────────────────────────
# BehSeed — extends FuzzSeed with coverage-specific fields
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BehSeed(FuzzSeed):
    is_new: bool = False
    parent_index: int = -1
    select_num: int = 0
    fail_num: int = 0
    safety_score: float = 1.0
    diversity_score: float = 0.0
    energy: float = 1.0

    @classmethod
    def from_fuzz_seed(cls, seed: FuzzSeed, **extra) -> "BehSeed":
        """Upgrade a FuzzSeed to BehSeed."""
        return cls(
            id=seed.id,
            scenario=seed.scenario,
            oracle_result=seed.oracle_result,
            feedback_result=seed.feedback_result,
            sample_result=seed.sample_result,
            is_expected=seed.is_expected,
            is_ignored=seed.is_ignored,
            scenario_dir=seed.scenario_dir,
            **extra,
        )

    @classmethod
    def load_from_dict(cls, data: dict) -> "BehSeed":
        base = FuzzSeed.load_from_dict(data)
        extra_fields = {f.name for f in fields(cls)} - {f.name for f in fields(FuzzSeed)}
        extra = {k: data.get(k, getattr(cls, k, None)) for k in extra_fields}
        return cls.from_fuzz_seed(base, **extra)

    def to_dict(self) -> dict:
        data = super().to_dict()
        for f in fields(self):
            if f.name not in data:
                data[f.name] = getattr(self, f.name)
        return data


# ──────────────────────────────────────────────────────────────────────────
# BehAVExplor Fuzzer
# ──────────────────────────────────────────────────────────────────────────

@FUZZER_REGISTRY.register("fuzzer.behavexplor")
class BehAVExplor(Fuzzer):

    def __init__(
        self,
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig,
    ):
        super().__init__(fuzzer_config, agent_config, scenario_config)

        for d in ("feedback_cache", "sample_cache", "fuzz_cache"):
            os.makedirs(os.path.join(self.output_root, d), exist_ok=True)

        # 1. Scenario space + sampler (for initial corpus generation)
        self.scenario_space = ScenarioODDSpace(self.pipeline_config['scenario_space'])
        self.sampler = RandomSampler(self.scenario_space, self.mutator_config)

        # 2. Perturbation mutator (for small/large mutations)
        self.behavior_mutator = ScenarioMutator(self.mutator_config)

        # 3. Behavior feedback (safety + coverage diversity)
        self.feedback = BehaviorFeedbackCalculator(self.feedback_config)

        # 4. Oracle
        self.oracle = ScenarioOracle(self.oracle_config)

        # Pipeline parameters
        self.initial_corpus_size = self.pipeline_config.get('initial_corpus_size', 4)
        self.batch_size = self.pipeline_config.get('batch_size', 1)

        # Internal state
        self.best_safety = 1.0
        self.best_diversity = 0.0
        self.all_seeds: List[BehSeed] = []
        self.seed_corpus: List[int] = []   # indices into all_seeds
        self.F_corpus: List[int] = []      # fault seeds
        self.new_corpus: List[int] = []    # new coverage seeds

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "safety", "diversity", "corpus_size"]

        if not self.resume:
            self.time_counter = 0.0
            logger.info("Start from scratch, time counter reset to 0.")

    # ──────────────────────────────────────────────────────────────
    # DEAP (not used for selection, but we keep toolbox for evaluate)
    # ──────────────────────────────────────────────────────────────

    def setup_deap(self):
        self.toolbox = base.Toolbox()
        # No DEAP fitness needed — we use custom energy-based selection

    def assign_feedback_to_ind(self, ind, feedback_result):
        # Not using DEAP fitness for BehAVExplor
        pass

    # ──────────────────────────────────────────────────────────────
    # Sampling (initial corpus)
    # ──────────────────────────────────────────────────────────────

    def _sample_initial_seed(self, max_retries=3) -> FuzzSeed:
        """Sample a new scenario from scratch using RandomSampler."""
        for attempt in range(1, max_retries + 1):
            try:
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
                scenario = self.sampler.sample(
                    ctn_operator=ctn_operator,
                    ego_entry_point=self.agent_entry_point,
                    ego_config_path=self.agent_config_path,
                )
                ctn_operator.cleanup()
                self.ctn_manager.release(ctn_config)
                if scenario is not None:
                    return FuzzSeed(id="unnamed", scenario=scenario)
            except Exception as e:
                logger.error(f"Initial sampling failed (attempt {attempt}): {e}")
                try:
                    self.ctn_manager.release(ctn_config)
                except Exception:
                    pass
                if attempt < max_retries:
                    self.restart_containers()
        raise RuntimeError("Failed to sample initial scenario")

    # ──────────────────────────────────────────────────────────────
    # Mutation (energy-driven: small vs large)
    # ──────────────────────────────────────────────────────────────

    def _mutate(self, source_seed: BehSeed) -> BehSeed:
        """Mutate based on energy: high energy → small perturbation, low → large resample.

        Both stages may need CARLA (small: for perturb_scenario's _setup,
        large: for generate's full resample). Connection is fully released
        before returning so execution subprocess can use the same container.
        """
        mutation_stage = 'small' if random.random() < source_seed.energy else 'large'
        logger.info(f"Mutation stage: {mutation_stage} (energy={source_seed.energy:.2f})")

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
        ctn_operator.start()

        try:
            mutated = self.behavior_mutator.step(
                source_seed=copy.deepcopy(source_seed),
                ctn_operator=ctn_operator,
                mutation_stage=mutation_stage,
            )
        finally:
            # Fully release CARLA connection before subprocess uses the container
            ctn_operator.cleanup()
            self.ctn_manager.release(ctn_config)

        return mutated

    # ──────────────────────────────────────────────────────────────
    # Selection (energy-weighted probability)
    # ──────────────────────────────────────────────────────────────

    def _select(self) -> BehSeed:
        """Select a seed from corpus proportional to energy."""
        energies = np.array([self.all_seeds[i].energy for i in self.seed_corpus]) + 1e-5
        probs = energies / energies.sum()
        corpus_idx = np.random.choice(len(self.seed_corpus), p=probs)
        seed_idx = self.seed_corpus[corpus_idx]

        selected = self.all_seeds[seed_idx]
        selected.select_num += 1

        result = copy.deepcopy(selected)
        result.parent_index = seed_idx
        return result

    # ──────────────────────────────────────────────────────────────
    # Corpus update (energy + coverage)
    # ──────────────────────────────────────────────────────────────

    def _update_corpus(self, seed: BehSeed):
        """Add seed to all_seeds and update corpus based on coverage/safety."""
        seed.energy = 1.0
        self.all_seeds.append(copy.deepcopy(seed))
        idx = len(self.all_seeds) - 1
        parent_idx = seed.parent_index

        # No parent → initial corpus seed
        if parent_idx < 0:
            self.seed_corpus.append(idx)
            return

        # Violation found
        if seed.is_expected:
            self.F_corpus.append(idx)
            self.all_seeds[parent_idx].fail_num += 1

        # Update parent energy
        parent = self.all_seeds[parent_idx]
        delta_safety = parent.safety_score - seed.safety_score
        delta_fail = parent.fail_num / (parent.select_num + 1e-5) if parent.select_num > 0 else 0.0
        delta_select = -0.1
        new_energy = parent.energy + 0.3 * np.tanh(delta_safety) + 0.5 * delta_fail + delta_select
        self.all_seeds[parent_idx].energy = float(np.clip(new_energy, 0.0, 1.0))

        # Add to corpus if improved safety or new coverage
        if not seed.is_expected:
            if seed.safety_score < parent.safety_score:
                self.seed_corpus.append(idx)
            elif seed.is_new:
                self.seed_corpus.append(idx)
                self.new_corpus.append(idx)

    # ──────────────────────────────────────────────────────────────
    # Evaluate batch
    # ──────────────────────────────────────────────────────────────

    def _evaluate_batch(self, batch: List[BehSeed]) -> List[BehSeed]:
        """Execute scenarios and compute feedback."""
        exec_results = self.execute_population(batch)

        for res in exec_results:
            ind_index = res['index']
            scenario_dir = res['scenario_dir']

            if not res['status']:
                batch[ind_index].safety_score = 1.0
                batch[ind_index].diversity_score = 0.0
                batch[ind_index].is_new = False
                continue

            try:
                visualize_trajectories(scenario_dir)
                obs = load_observation(scenario_dir)
                runtime = load_runtime_result(scenario_dir)
                oracle_result = self.oracle.evaluate(obs, runtime)
                feedback_result = self.feedback.evaluate(obs, oracle_result)
            except Exception as e:
                logger.warning(f"Evaluation failed for {scenario_dir}: {e}")
                batch[ind_index].safety_score = 1.0
                batch[ind_index].diversity_score = 0.0
                batch[ind_index].is_new = False
                continue

            ind = batch[ind_index]
            ind.oracle_result = oracle_result
            ind.feedback_result = feedback_result
            ind.is_expected = oracle_result['expected']
            ind.is_ignored = oracle_result.get('ignored', False)
            ind.safety_score = feedback_result['safety_score']
            ind.diversity_score = feedback_result['diversity_score']
            ind.is_new = feedback_result['is_new']
            ind.scenario_dir = scenario_dir
            batch[ind_index] = ind

        return batch

    # ──────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────

    def _get_checkpoint_data(self) -> dict:
        data = super()._get_checkpoint_data()
        data.update({
            'initial_corpus_size': self.initial_corpus_size,
            'best_safety': self.best_safety,
            'best_diversity': self.best_diversity,
            'all_seeds': [s.to_dict() for s in self.all_seeds],
            'seed_corpus': self.seed_corpus,
            'F_corpus_indices': self.F_corpus,
            'new_corpus': self.new_corpus,
            'feedback_model': self.feedback.save_checkpoint(
                os.path.dirname(self.checkpoint_path)),
        })
        return data

    def _restore_checkpoint_data(self, data: dict):
        super()._restore_checkpoint_data(data)
        self.initial_corpus_size = data.get('initial_corpus_size', self.initial_corpus_size)
        self.best_safety = data.get('best_safety', 1.0)
        self.best_diversity = data.get('best_diversity', 0.0)
        self.all_seeds = [BehSeed.load_from_dict(d) for d in data.get('all_seeds', [])]
        self.seed_corpus = data.get('seed_corpus', [])
        self.F_corpus = data.get('F_corpus_indices', [])
        self.new_corpus = data.get('new_corpus', [])
        self.feedback.load_checkpoint(data.get('feedback_model'))

    def save_checkpoint(self):
        super().save_checkpoint()
        # Extra: save logbook
        with open(os.path.join(self.output_root, "logbook.json"), 'w') as f:
            import json
            json.dump(self.logbook, f, indent=2, default=str)

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def _run(self, start_time):
        container_restart_interval = 300.0
        last_restart_time = time.time()

        # ── Phase 1: Build initial corpus via random sampling ──
        while len(self.seed_corpus) < self.initial_corpus_size:
            if self.termination_check(start_time):
                return

            self.global_search_step += 1
            logger.info(f"=== Initial Phase: Gen {self.global_search_step} ===")

            batch = []
            for i in range(self.batch_size):
                ind_id = f"gen_{self.global_search_step}_init_{i}"
                try:
                    seed = self._sample_initial_seed()
                    bseed = BehSeed.from_fuzz_seed(seed)
                    bseed.set_id(ind_id)
                    batch.append(bseed)
                except RuntimeError as e:
                    logger.error(f"Skipping init seed: {e}")

            if not batch:
                continue

            batch = self._evaluate_batch(batch)

            for ind in batch:
                self._update_corpus(ind)
                if ind.safety_score < self.best_safety:
                    self.best_safety = ind.safety_score
                if ind.diversity_score > self.best_diversity:
                    self.best_diversity = ind.diversity_score

            self.logbook.record(
                gen=self.global_search_step,
                safety=self.best_safety,
                diversity=self.best_diversity,
                corpus_size=len(self.seed_corpus),
            )
            self.save_checkpoint()

        # ── Initialize coverage model from corpus ──
        if not self.feedback.is_initialized:
            logger.info("Initializing coverage model from initial corpus...")
            corpus_seeds = [self.all_seeds[i] for i in self.seed_corpus]
            obs_list = []
            for seed in corpus_seeds:
                if seed.scenario_dir:
                    try:
                        obs_list.append(load_observation(seed.scenario_dir))
                    except Exception:
                        pass
            if obs_list:
                self.feedback.initialize_coverage_model(obs_list)
            self.save_checkpoint()

        logger.info(f"Initial corpus: {len(self.seed_corpus)} seeds. Starting fuzzing...")

        # ── Phase 2: Coverage-guided fuzzing ──
        while not self.termination_check(start_time):
            self.global_search_step += 1
            logger.info(f"=== Generation {self.global_search_step} ===")

            if time.time() - last_restart_time > container_restart_interval:
                self.restart_containers()
                last_restart_time = time.time()

            batch = []
            for i in range(self.batch_size):
                ind_id = f"gen_{self.global_search_step}_ind_{i}"
                try:
                    selected = self._select()
                    selected.set_id(ind_id)
                    mutated = self._mutate(selected)
                    batch.append(mutated)
                except Exception as e:
                    logger.error(f"Mutation failed: {e}")

            if not batch:
                continue

            batch = self._evaluate_batch(batch)

            for ind in batch:
                self._update_corpus(ind)
                if ind.safety_score < self.best_safety:
                    self.best_safety = ind.safety_score
                if ind.diversity_score > self.best_diversity:
                    self.best_diversity = ind.diversity_score

            self.logbook.record(
                gen=self.global_search_step,
                safety=self.best_safety,
                diversity=self.best_diversity,
                corpus_size=len(self.seed_corpus),
            )

            logger.info(
                f"[Gen {self.global_search_step}] "
                f"Safety={self.best_safety:.4f} Diversity={self.best_diversity:.4f} "
                f"Corpus={len(self.seed_corpus)} Faults={len(self.F_corpus)} "
                f"New={len(self.new_corpus)}"
            )

            self.save_checkpoint()

        logger.info("Time budget exhausted. BehAVExplor complete.")

    def close(self):
        super().close()
