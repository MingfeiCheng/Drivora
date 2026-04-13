"""
SAMOTA — Surrogate-Assisted Multi-Objective Testing Algorithm.

Adapted from ICSE-SAMOTA (Cheng et al.) to Drivora's framework.

Flow:
  1. Initialize: RandomSampler × N → simulate → build database
  2. Train surrogate ensemble (RBF + PR + Kriging) per objective
  3. Global Search: surrogate-guided NSGA-II → candidate vectors
  4. Local Search: cluster database → per-cluster RBF + GA → candidate vectors
  5. For each candidate vector:
     - Find nearest scenario in database (by vector distance)
     - Mutate that scenario (AVFuzzer-style perturbation on ScenarioConfig)
     - Simulate mutated scenario → feedback → update database
  6. Update archive, repeat

Key design:
  - Mutation is always on ScenarioConfig (not on vectors)
  - scenario_to_features() produces fixed-length vectors for surrogate training
  - Surrogate search guides WHICH scenarios to mutate, not HOW
  - Feedback uses AVFuzzer's composite score; objectives stored per-scenario
"""

import os
import copy
import time
import numpy as np

from loguru import logger
from omegaconf import DictConfig
from deap import base, tools

from registry import FUZZER_REGISTRY
from scenario_runner.ctn_manager import CtnSimOperator

from fuzzer.runner_base import Fuzzer, FuzzSeed
from tools.recorder_tool import load_observation, load_runtime_result, visualize_trajectories

from .mutator.random_sample import RandomSampler
from .mutator.avfuzzer_mutator import AVFuzzerMutator
from .feedback.samota_feedback import SAMOTAFeedbackCalculator
from .feedback.samota_feedback import N_OBJECTIVES
from .oracle.general_oracle import ScenarioOracle
from .scenario_space import ScenarioODDSpace

from .misc.samota_utils import (
    SamotaCandidate, scenario_to_features,
    global_search, local_search, update_archive as update_samota_archive,
)


@FUZZER_REGISTRY.register("fuzzer.samota")
class SAMOTAFuzzer(Fuzzer):

    def __init__(
        self,
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig,
    ):
        super().__init__(fuzzer_config, agent_config, scenario_config)

        for d in ("feedback_cache", "sample_cache", "fuzz_cache"):
            os.makedirs(os.path.join(self.output_root, d), exist_ok=True)

        # 1. Scenario space + sampler (initial random sampling)
        self.scenario_space = ScenarioODDSpace(self.pipeline_config['scenario_space'])
        self.sampler = RandomSampler(self.scenario_space, self.mutator_config)

        # 2. Perturbation mutator (AVFuzzer-style, for mutating scenarios)
        self.perturbation = AVFuzzerMutator(self.mutator_config)

        # 3. Feedback (SAMOTA's 6-objective: DfC, DfV, DfP, DfM, DT, Dest)
        self.feedback = SAMOTAFeedbackCalculator(self.feedback_config)

        # 4. Oracle
        self.oracle = ScenarioOracle(self.oracle_config)

        # SAMOTA parameters
        self.initial_db_size = self.pipeline_config.get('initial_db_size', 6)
        self.gs_pop_size = self.pipeline_config.get('gs_pop_size', 6)
        self.gs_generations = self.pipeline_config.get('gs_generations', 100)
        default_thresholds = [0.1] * N_OBJECTIVES
        self.thresholds = list(self.pipeline_config.get('thresholds', default_thresholds))

        # Internal state
        self.database_vectors = []     # list of (feature_vector, objective_values)
        self.database_seeds = []       # parallel list of FuzzSeed (for mutation)
        self.archive = []              # SamotaCandidate archive
        self.objective_uncovered = list(range(N_OBJECTIVES))
        self.best_per_objective = [float('inf')] * N_OBJECTIVES
        self.ensembles = []            # trained surrogate ensembles (cached)

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "db_size", "archive_size", "uncovered", "best_score"]

        if not self.resume:
            self.time_counter = 0.0

    # ──────────────────────────────────────────────────────────────
    # DEAP (minimal)
    # ──────────────────────────────────────────────────────────────

    def setup_deap(self):
        self.toolbox = base.Toolbox()

    def assign_feedback_to_ind(self, ind, feedback_result):
        pass

    # ──────────────────────────────────────────────────────────────
    # Sample + evaluate
    # ──────────────────────────────────────────────────────────────

    def _sample_seed(self, max_retries=3) -> FuzzSeed:
        for attempt in range(1, max_retries + 1):
            try:
                ctn_config = self.ctn_manager.acquire()
                ctn_operator = CtnSimOperator(
                    idx=ctn_config.idx, container_name=ctn_config.container_name,
                    gpu=ctn_config.gpu, random_seed=ctn_config.random_seed,
                    docker_image=ctn_config.docker_image, fps=ctn_config.fps,
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
                logger.error(f"Sampling failed (attempt {attempt}): {e}")
                try:
                    self.ctn_manager.release(ctn_config)
                except Exception:
                    pass
                if attempt < max_retries:
                    self.restart_containers()
        raise RuntimeError("Failed to sample scenario")

    def _evaluate_and_store(self, seeds: list):
        """Execute scenarios, compute feedback, store in database."""
        exec_results = self.execute_population(seeds)

        for res in exec_results:
            idx = res['index']
            scenario_dir = res['scenario_dir']
            seed = seeds[idx]

            if not res['status']:
                continue

            try:
                visualize_trajectories(scenario_dir)
                obs = load_observation(scenario_dir)
                runtime = load_runtime_result(scenario_dir)
                oracle_result = self.oracle.evaluate(obs, runtime)
                feedback_result = self.feedback.evaluate(obs, oracle_result)
            except Exception as e:
                logger.warning(f"Eval failed: {e}")
                continue

            seed.oracle_result = oracle_result
            seed.feedback_result = feedback_result
            seed.is_expected = oracle_result['expected']
            seed.is_ignored = oracle_result.get('ignored', False)
            seed.set_scenario_dir(scenario_dir or "")

            # Extract 6 objectives from SAMOTA feedback + convert scenario to vector
            obj_vals = feedback_result.get('objective_values', [1.0] * N_OBJECTIVES)
            fv = scenario_to_features(seed.scenario)

            # Store in database
            candidate = SamotaCandidate(fv)
            candidate.set_objective_values(obj_vals)
            self.database_vectors.append((fv, obj_vals))
            self.database_seeds.append(copy.deepcopy(seed))

            # Update bests
            for i in range(N_OBJECTIVES):
                if obj_vals[i] < self.best_per_objective[i]:
                    self.best_per_objective[i] = obj_vals[i]

            # Record
            self.seed_recorder.append({
                'id': seed.id, 'is_expected': seed.is_expected,
                'is_ignored': seed.is_ignored, 'oracle_result': oracle_result,
                'feedback_result': feedback_result,
            })
            if seed.is_expected and not seed.is_ignored:
                self.F_corpus.append({
                    'id': seed.id, 'is_expected': True,
                    'oracle_result': oracle_result,
                })

    # ──────────────────────────────────────────────────────────────
    # Candidate vector → find nearest scenario → mutate → new scenario
    # ──────────────────────────────────────────────────────────────

    def _candidates_to_seeds(self, candidates: list) -> list:
        """For each surrogate candidate vector, find the closest scenario
        in the database and mutate it to produce a new scenario."""
        new_seeds = []
        if not self.database_seeds:
            return new_seeds

        db_vectors = np.array([fv for fv, _ in self.database_vectors])

        for candidate in candidates:
            # Find nearest scenario by Euclidean distance in feature space
            target_fv = np.array(candidate.get_features())
            dists = np.linalg.norm(db_vectors - target_fv, axis=1)
            nearest_idx = int(np.argmin(dists))
            base_seed = copy.deepcopy(self.database_seeds[nearest_idx])

            # Mutate the scenario (AVFuzzer-style perturbation)
            base_seed.scenario = self.perturbation.perturb(base_seed.scenario)
            new_seeds.append(base_seed)

        return new_seeds

    def _get_bounds(self):
        """Get feature vector bounds from database."""
        if not self.database_vectors:
            return [0.0] * 29, [1.0] * 29
        all_fv = np.array([fv for fv, _ in self.database_vectors])
        lb = (all_fv.min(axis=0) - 1.0).tolist()
        ub = (all_fv.max(axis=0) + 1.0).tolist()
        return lb, ub

    # ──────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────

    def _get_checkpoint_data(self) -> dict:
        data = super()._get_checkpoint_data()
        # Serialize archive candidates
        archive_data = []
        for c in self.archive:
            archive_data.append({
                'features': c.get_features(),
                'objective_values': c.get_objective_values(),
                'objectives_covered': c.objectives_covered,
            })
        # Save ensembles to disk
        ensemble_paths = []
        ensemble_dir = os.path.join(os.path.dirname(self.checkpoint_path), 'ensembles')
        os.makedirs(ensemble_dir, exist_ok=True)
        for ens in self.ensembles:
            path = os.path.join(ensemble_dir, f'ensemble_obj_{ens.objective_index}.pkl')
            try:
                ens.save(path)
                ensemble_paths.append({'path': path, 'obj': ens.objective_index})
            except Exception as e:
                logger.warning(f"Failed to save ensemble obj={ens.objective_index}: {e}")

        data.update({
            'database_vectors': self.database_vectors,
            'database_seeds': [s.to_dict() for s in self.database_seeds],
            'archive': archive_data,
            'objective_uncovered': list(self.objective_uncovered),
            'best_per_objective': list(self.best_per_objective),
            'initial_db_size': self.initial_db_size,
            'ensemble_paths': ensemble_paths,
        })
        return data

    def _restore_checkpoint_data(self, data: dict):
        super()._restore_checkpoint_data(data)
        self.database_vectors = data.get('database_vectors', [])
        self.database_seeds = [FuzzSeed.load_from_dict(d) for d in data.get('database_seeds', [])]
        self.objective_uncovered = list(data.get('objective_uncovered', list(range(N_OBJECTIVES))))
        self.best_per_objective = list(data.get('best_per_objective', [float('inf')] * N_OBJECTIVES))
        self.initial_db_size = data.get('initial_db_size', self.initial_db_size)

        # Restore archive
        self.archive = []
        for entry in data.get('archive', []):
            c = SamotaCandidate(entry['features'])
            c.set_objective_values(entry['objective_values'])
            for obj in entry.get('objectives_covered', []):
                c.add_objective_covered(obj)
            self.archive.append(c)

        # Restore ensembles
        self.ensembles = []
        for entry in data.get('ensemble_paths', []):
            path = entry.get('path', '')
            if os.path.exists(path):
                try:
                    from fuzzer.misc.surrogate_models import EnsembleSurrogate
                    ens = EnsembleSurrogate.load(path)
                    self.ensembles.append(ens)
                except Exception as e:
                    logger.warning(f"Failed to load ensemble from {path}: {e}")

        logger.info(f"Restored SAMOTA: DB={len(self.database_seeds)} "
                     f"Archive={len(self.archive)} Ensembles={len(self.ensembles)} "
                     f"Uncovered={len(self.objective_uncovered)}")

    def save_checkpoint(self):
        super().save_checkpoint()
        import json
        with open(os.path.join(self.output_root, "logbook.json"), 'w') as f:
            json.dump(self.logbook, f, indent=2, default=str)

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def _run(self, start_time):
        container_restart_interval = 300.0
        last_restart_time = time.time()

        # ── Phase 1: Build initial database (skip if restored from checkpoint) ──
        if len(self.database_seeds) >= self.initial_db_size:
            logger.info(f"Initial database already has {len(self.database_seeds)} entries (restored from checkpoint).")
        else:
            logger.info(f"Building initial database ({len(self.database_seeds)}/{self.initial_db_size})...")
            while len(self.database_seeds) < self.initial_db_size:
                if self.termination_check(start_time):
                    return

                self.global_search_step += 1
                logger.info(f"=== Init {len(self.database_seeds)+1}/{self.initial_db_size} ===")

                try:
                    seed = self._sample_seed()
                    seed.set_id(f"gen_{self.global_search_step}_init_{len(self.database_seeds)}")
                except RuntimeError:
                    continue

                self._evaluate_and_store([seed])
                self.save_checkpoint()

            logger.info(f"Initial database complete: {len(self.database_seeds)} entries.")

        # Rebuild archive from full database if empty (e.g., first run or old checkpoint)
        if not self.archive and self.database_vectors:
            logger.info("Rebuilding archive from database...")
            all_candidates = []
            for fv, ov in self.database_vectors:
                c = SamotaCandidate(fv)
                c.set_objective_values(ov)
                all_candidates.append(c)
            update_samota_archive(
                all_candidates, self.objective_uncovered, self.archive,
                N_OBJECTIVES, self.thresholds)
            logger.info(f"Archive rebuilt: {len(self.archive)} entries, "
                         f"uncovered: {len(self.objective_uncovered)}")
            self.save_checkpoint()

        # ── Phase 2: SAMOTA loop ──
        while not self.termination_check(start_time):
            self.global_search_step += 1
            logger.info(f"=== SAMOTA Iteration {self.global_search_step} ===")

            if time.time() - last_restart_time > container_restart_interval:
                self.restart_containers()
                last_restart_time = time.time()

            lb, ub = self._get_bounds()

            # Build SamotaCandidate list for surrogate
            db_candidates = []
            for fv, ov in self.database_vectors:
                c = SamotaCandidate(fv)
                c.set_objective_values(ov)
                db_candidates.append(c)

            # ── Global Search (surrogate-guided NSGA-II) ──
            logger.info("Global Search...")
            gs_candidates, trained_ensembles = global_search(
                database=db_candidates,
                objective_uncovered=list(self.objective_uncovered),
                pop_size=self.gs_pop_size,
                n_generations=self.gs_generations,
                lb=lb, ub=ub,
                n_objectives=N_OBJECTIVES,
            )
            self.ensembles = trained_ensembles  # cache for checkpoint

            # Candidate vectors → nearest scenario → mutate → simulate
            gs_seeds = self._candidates_to_seeds(gs_candidates)
            for i, s in enumerate(gs_seeds):
                s.set_id(f"gen_{self.global_search_step}_gs_{i}")

            if gs_seeds:
                logger.info(f"Evaluating {len(gs_seeds)} GS scenarios...")
                self._evaluate_and_store(gs_seeds)

            # ── Local Search (cluster-based) ──
            logger.info("Local Search...")
            ls_candidates = local_search(
                database=db_candidates,
                objective_uncovered=list(self.objective_uncovered),
                lb=lb, ub=ub,
            )

            ls_seeds = self._candidates_to_seeds(ls_candidates)
            for i, s in enumerate(ls_seeds):
                s.set_id(f"gen_{self.global_search_step}_ls_{i}")

            if ls_seeds:
                logger.info(f"Evaluating {len(ls_seeds)} LS scenarios...")
                self._evaluate_and_store(ls_seeds)

            # ── Update archive ──
            new_candidates = []
            for fv, ov in self.database_vectors[-len(gs_seeds)-len(ls_seeds):]:
                c = SamotaCandidate(fv)
                c.set_objective_values(ov)
                new_candidates.append(c)
            update_samota_archive(
                new_candidates, self.objective_uncovered, self.archive,
                N_OBJECTIVES, self.thresholds)

            # ── Log ──
            self.logbook.record(
                gen=self.global_search_step,
                db_size=len(self.database_seeds),
                archive_size=len(self.archive),
                uncovered=len(self.objective_uncovered),
                best_score=min(self.best_per_objective),
            )

            logger.info(
                f"[SAMOTA Gen {self.global_search_step}] "
                f"DB={len(self.database_seeds)} Archive={len(self.archive)} "
                f"Uncovered={len(self.objective_uncovered)} "
                f"F_corpus={len(self.F_corpus)}"
            )

            self.save_checkpoint()

            if not self.objective_uncovered:
                logger.info("All objectives covered!")

        logger.info("Time budget exhausted. SAMOTA complete.")

    def close(self):
        super().close()
