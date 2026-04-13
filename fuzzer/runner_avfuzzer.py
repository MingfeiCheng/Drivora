"""
AVFuzzer — GA-based scenario fuzzer following the original AVFuzzer paper.

Faithful reimplementation of the AVFuzzer genetic algorithm adapted to
Drivora's framework. Key algorithmic elements preserved:

  - Chromosome = scenario (NPC configs with speeds/trigger times)
  - Crossover: swap one NPC between two parent scenarios
  - Mutation: randomly perturb one NPC's speed or trigger time at one waypoint
  - Selection: roulette wheel (fitness-proportional)
  - Restart: if no improvement for `stagnation_gens` generations, re-initialize
    with diverse population around best-so-far
  - Local Iterative Search (LIS): periodic local GA around global best
    with increased mutation probability

Reference: Li et al., "AV-FUZZER: Finding Safety Violations of
Autonomous Vehicles" (ISSRE 2020)
"""

import os
import copy
import time
import random

import numpy as np
from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from deap import base, creator, tools

from registry import FUZZER_REGISTRY
from scenario_runner.ctn_manager import CtnSimOperator

from fuzzer.runner_base import Fuzzer, FuzzSeed

from .mutator.random_sample import RandomSampler
from .feedback.avfuzzer_feedback import AVFuzzerFeedbackCalculator
from .oracle.general_oracle import ScenarioOracle
from .scenario_space import ScenarioODDSpace


@FUZZER_REGISTRY.register("fuzzer.avfuzzer")
class AVFuzzer(Fuzzer):
    """GA-based scenario fuzzer — faithful to the original AVFuzzer algorithm."""

    def __init__(
        self,
        fuzzer_config: DictConfig,
        agent_config: DictConfig,
        scenario_config: DictConfig,
    ):
        super().__init__(fuzzer_config, agent_config, scenario_config)

        for d in ("feedback_cache", "sample_cache", "fuzz_cache"):
            os.makedirs(os.path.join(self.output_root, d), exist_ok=True)

        # 1. Scenario space + sampler
        self.scenario_space = ScenarioODDSpace(self.pipeline_config['scenario_space'])
        self.sampler = RandomSampler(self.scenario_space, self.mutator_config)

        # 2. Feedback (collision proximity — AVFuzzer uses single objective)
        self.feedback = AVFuzzerFeedbackCalculator(self.feedback_config)

        # 3. Oracle
        self.oracle = ScenarioOracle(self.oracle_config)

        # GA parameters (from original AVFuzzer)
        self.population_size = self.pipeline_config.get('population_size', 4)
        self.pm = self.pipeline_config.get('mutation_prob', 0.6)     # mutation probability
        self.pc = self.pipeline_config.get('crossover_prob', 0.6)    # crossover probability
        self.selection_method = self.pipeline_config.get('selection', 'roulette')  # 'roulette' or 'top2'
        self.stagnation_gens = self.pipeline_config.get('stagnation_gens', 5)     # restart threshold
        self.lis_interval = self.pipeline_config.get('lis_interval', 3)           # LIS trigger interval
        self.lis_generations = self.pipeline_config.get('lis_generations', 5)      # LIS inner GA generations
        self.lis_pm_factor = self.pipeline_config.get('lis_pm_factor', 1.5)       # LIS mutation boost

        # Perturbation bounds
        self.speed_bounds = self.pipeline_config.get('speed_bounds', [1.0, 15.0])
        self.trigger_bounds = self.pipeline_config.get('trigger_bounds', [0.0, 10.0])

        # Internal state
        self.best_score = float('inf')  # lower is better (collision proximity)
        self.pop = []                   # current population: list of FuzzSeed with .fitness
        self.g_best = None              # global best FuzzSeed
        self.gen_bests = []             # best fitness per generation (for stagnation detection)
        self.last_restart_gen = 0

        self.setup_deap()

        if not self.resume:
            self.time_counter = 0.0
            logger.info("Start from scratch.")

    # ──────────────────────────────────────────────────────────────
    # DEAP setup
    # ──────────────────────────────────────────────────────────────

    def setup_deap(self):
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", FuzzSeed, fitness=creator.FitnessMin)
        super().setup_deap()

    def assign_feedback_to_ind(self, ind, feedback_result):
        ind.fitness.values = (feedback_result['score'],)
        return ind

    # ──────────────────────────────────────────────────────────────
    # Sampling (initial population)
    # ──────────────────────────────────────────────────────────────

    def _sample_seed(self, max_retries=3) -> FuzzSeed:
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
                logger.error(f"Sampling failed (attempt {attempt}): {e}")
                try:
                    self.ctn_manager.release(ctn_config)
                except Exception:
                    pass
                if attempt < max_retries:
                    self.restart_containers()
        raise RuntimeError("Failed to sample scenario")

    # ──────────────────────────────────────────────────────────────
    # AVFuzzer Crossover: swap one NPC between two scenarios
    # ──────────────────────────────────────────────────────────────

    def _crossover(self) -> set:
        """Crossover: for random pairs, swap one NPC vehicle config.
        Returns set of indices that were modified."""
        touched = set()
        for _ in range(len(self.pop) // 2):
            if random.random() > self.pc:
                continue
            i, j = 0, 0
            while i == j:
                i = random.randint(0, len(self.pop) - 1)
                j = random.randint(0, len(self.pop) - 1)

            npcs_i = self.pop[i].scenario.npc_vehicles or []
            npcs_j = self.pop[j].scenario.npc_vehicles or []
            if not npcs_i or not npcs_j:
                continue

            min_len = min(len(npcs_i), len(npcs_j))
            swap_idx = random.randint(0, min_len - 1)

            # Swap NPC configs
            temp = copy.deepcopy(npcs_j[swap_idx])
            npcs_j[swap_idx] = copy.deepcopy(npcs_i[swap_idx])
            npcs_i[swap_idx] = temp
            touched.add(i)
            touched.add(j)
        return touched

    # ──────────────────────────────────────────────────────────────
    # AVFuzzer Mutation: randomly perturb one NPC's speed or trigger
    # ──────────────────────────────────────────────────────────────

    def _mutate_population(self, pm=None):
        """Mutate: for each individual, with probability pm, change one
        NPC's speed at a random waypoint OR its trigger time."""
        if pm is None:
            pm = self.pm

        mutated_indices = set()
        for i in range(len(self.pop)):
            if random.random() > pm:
                continue

            scenario = self.pop[i].scenario
            npcs = scenario.npc_vehicles or []
            if not npcs:
                continue

            npc_idx = random.randint(0, len(npcs) - 1)
            npc = npcs[npc_idx]

            action = random.randint(0, 1)
            if action == 0 and npc.route:
                # Mutate speed at a random waypoint
                wp_idx = random.randint(0, len(npc.route) - 1)
                npc.route[wp_idx].speed = random.uniform(self.speed_bounds[0], self.speed_bounds[1])
            else:
                # Mutate trigger time
                npc.trigger_time = random.uniform(self.trigger_bounds[0], self.trigger_bounds[1])

            mutated_indices.add(i)

        return mutated_indices

    # ──────────────────────────────────────────────────────────────
    # AVFuzzer Selection: roulette wheel or top-2
    # ──────────────────────────────────────────────────────────────

    def _select_roulette(self):
        """Roulette wheel selection (fitness-proportional). Higher fitness = more likely."""
        # AVFuzzer maximizes fitness (closer to collision = higher).
        # Our fitness is score from feedback (lower = better), so we invert.
        fitnesses = []
        for ind in self.pop:
            f = ind.fitness.values[0] if ind.fitness.valid else 1.0
            fitnesses.append(max(0.001, 1.0 - f))  # invert: lower score → higher selection prob

        total = sum(fitnesses)
        probs = [f / total for f in fitnesses]

        # Build cumulative distribution
        cum = []
        s = 0
        for p in probs:
            s += p
            cum.append(s)

        new_pop = []
        for _ in range(len(self.pop)):
            r = random.random()
            for j, c in enumerate(cum):
                if r <= c:
                    new_pop.append(copy.deepcopy(self.pop[j]))
                    break
            else:
                new_pop.append(copy.deepcopy(self.pop[-1]))

        self.pop = new_pop

    def _select_top2(self):
        """Keep top-2 individuals, fill population by duplicating them."""
        sorted_pop = sorted(self.pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
        new_pop = []
        for i in range(len(self.pop)):
            new_pop.append(copy.deepcopy(sorted_pop[i % 2]))
        self.pop = new_pop

    # ──────────────────────────────────────────────────────────────
    # AVFuzzer Restart: re-initialize with diverse population
    # ──────────────────────────────────────────────────────────────

    def _check_stagnation(self, gen: int) -> bool:
        """Check if fitness has stagnated over last `stagnation_gens` generations."""
        if gen < self.last_restart_gen + self.stagnation_gens:
            return False
        recent = self.gen_bests[-(self.stagnation_gens):]
        if not recent:
            return False
        avg = sum(recent) / len(recent)
        current = self.gen_bests[-1] if self.gen_bests else float('inf')
        return avg <= current  # no improvement → stagnation

    def _restart_population(self):
        """Restart: keep global best, re-sample the rest."""
        logger.info("Restarting population due to stagnation...")
        new_pop = []
        if self.g_best is not None:
            new_pop.append(copy.deepcopy(self.g_best))
        while len(new_pop) < self.population_size:
            try:
                seed = self._sample_seed()
                ind = creator.Individual(**seed.to_deap_args())
                new_pop.append(ind)
            except RuntimeError:
                break
        self.pop = new_pop

    # ──────────────────────────────────────────────────────────────
    # AVFuzzer LIS: Local Iterative Search around best
    # ──────────────────────────────────────────────────────────────

    def _local_iterative_search(self, global_gen: int) -> FuzzSeed:
        """LIS: run a short local GA around g_best with boosted mutation rate."""
        logger.info(f"Starting Local Iterative Search ({self.lis_generations} generations)...")
        lis_pop = [copy.deepcopy(self.g_best) for _ in range(self.population_size)]

        # Initial diversification: mutate all with pm=1.0
        old_pop = self.pop
        self.pop = lis_pop
        self._mutate_population(pm=1.0)

        # Set IDs for initial LIS population
        for i in range(len(self.pop)):
            self.pop[i].set_id(f"local_gen_{global_gen}_lis_init_{i}")

        # Evaluate
        wrapped = [[ind] for ind in self.pop]
        wrapped = self.toolbox.evaluate(wrapped)
        self.pop = [item[0] for item in wrapped]

        lis_best = min(self.pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))

        for lis_gen in range(self.lis_generations):
            crossed = self._crossover()
            mutated = self._mutate_population(pm=self.pm * self.lis_pm_factor)
            touched = crossed | mutated

            # Set IDs for touched individuals
            for i in touched:
                self.pop[i].set_id(f"local_gen_{global_gen}_lis_{lis_gen}_ind_{i}")
                if self.pop[i].fitness.valid:
                    del self.pop[i].fitness.values

            # Evaluate touched
            to_eval_indices = [i for i in touched if not self.pop[i].fitness.valid]
            to_eval = [self.pop[i] for i in to_eval_indices]
            if to_eval:
                wrapped = [[ind] for ind in to_eval]
                wrapped = self.toolbox.evaluate(wrapped)
                for idx, item in zip(to_eval_indices, wrapped):
                    self.pop[idx] = item[0]

            if self.selection_method == 'roulette':
                self._select_roulette()
            else:
                self._select_top2()

            gen_best = min(self.pop, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf'))
            if gen_best.fitness.valid and gen_best.fitness.values[0] < lis_best.fitness.values[0]:
                lis_best = copy.deepcopy(gen_best)

        self.pop = old_pop  # restore original population
        logger.info(f"LIS complete. Best fitness: {lis_best.fitness.values[0]:.4f}")
        return lis_best

    # ──────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────

    def _get_checkpoint_data(self) -> dict:
        data = super()._get_checkpoint_data()
        data.update({
            'population_size': self.population_size,
            'best_score': self.best_score,
            'pop': [s.to_dict() for s in self.pop],
            'g_best': self.g_best.to_dict() if self.g_best else None,
            'gen_bests': self.gen_bests,
            'last_restart_gen': self.last_restart_gen,
        })
        return data

    def _restore_checkpoint_data(self, data: dict):
        super()._restore_checkpoint_data(data)
        self.population_size = data.get('population_size', self.population_size)
        self.best_score = data.get('best_score', float('inf'))
        self.gen_bests = data.get('gen_bests', [])
        self.last_restart_gen = data.get('last_restart_gen', 0)

        pop_data = data.get('pop', [])
        self.pop = []
        for d in pop_data:
            seed = FuzzSeed.load_from_dict(d)
            ind = creator.Individual(**seed.to_deap_args())
            score = d.get('feedback_result', {}).get('score')
            if score is not None:
                ind.fitness.values = (score,)
            self.pop.append(ind)

        g_best_data = data.get('g_best')
        if g_best_data:
            seed = FuzzSeed.load_from_dict(g_best_data)
            self.g_best = creator.Individual(**seed.to_deap_args())
            score = g_best_data.get('feedback_result', {}).get('score')
            if score is not None:
                self.g_best.fitness.values = (score,)

    # ──────────────────────────────────────────────────────────────
    # Main loop — faithful to original AVFuzzer
    # ──────────────────────────────────────────────────────────────

    def _run(self, start_time):
        container_restart_interval = 300.0
        last_restart_time = time.time()

        # ── Phase 1: Initialize population ──
        if len(self.pop) < self.population_size:
            logger.info(f"Initializing population ({len(self.pop)}/{self.population_size})...")
            while len(self.pop) < self.population_size:
                if self.termination_check(start_time):
                    return
                self.global_search_step += 1
                ind_id = f"global_gen_{self.global_search_step}_init_{len(self.pop)}"
                try:
                    seed = self._sample_seed()
                except RuntimeError as e:
                    logger.error(f"Init sampling failed: {e}")
                    continue
                ind = creator.Individual(**seed.to_deap_args())
                ind.set_id(ind_id)
                self.pop.append(ind)

            # Evaluate initial population
            wrapped = [[ind] for ind in self.pop]
            wrapped = self.toolbox.evaluate(wrapped)
            self.pop = [item[0] for item in wrapped]

            # Find initial best
            valid_pop = [ind for ind in self.pop if ind.fitness.valid]
            if valid_pop:
                best = min(valid_pop, key=lambda x: x.fitness.values[0])
                self.g_best = copy.deepcopy(best)
                self.best_score = best.fitness.values[0]

            self.save_checkpoint()
            logger.info(f"Init complete. Best fitness: {self.best_score:.4f}")

        # ── Phase 2: GA evolution (AVFuzzer main loop) ──
        while not self.termination_check(start_time):
            self.global_search_step += 1
            gen = self.global_search_step
            logger.info(f"=== Generation {gen} ===")

            if time.time() - last_restart_time > container_restart_interval:
                self.restart_containers()
                last_restart_time = time.time()

            # Crossover (returns indices of touched individuals)
            crossed = self._crossover()

            # Mutation (returns indices of touched individuals)
            mutated = self._mutate_population()

            # All modified individuals need new IDs and re-evaluation
            touched = crossed | mutated
            logger.info(f"Touched {len(touched)} individuals (crossed={len(crossed)}, mutated={len(mutated)})")

            for i in touched:
                self.pop[i].set_id(f"global_gen_{gen}_ind_{i}")
                if self.pop[i].fitness.valid:
                    del self.pop[i].fitness.values

            # Evaluate only touched individuals
            to_eval_indices = [i for i in touched if not self.pop[i].fitness.valid]
            to_eval = [self.pop[i] for i in to_eval_indices]
            if to_eval:
                wrapped = [[ind] for ind in to_eval]
                wrapped = self.toolbox.evaluate(wrapped)
                for idx, item in zip(to_eval_indices, wrapped):
                    self.pop[idx] = item[0]

            # Selection
            if self.selection_method == 'roulette':
                self._select_roulette()
            else:
                self._select_top2()

            # Find generation best
            valid_pop = [ind for ind in self.pop if ind.fitness.valid]
            if valid_pop:
                gen_best = min(valid_pop, key=lambda x: x.fitness.values[0])
                gen_best_val = gen_best.fitness.values[0]
            else:
                gen_best_val = float('inf')

            self.gen_bests.append(gen_best_val)

            # Update global best
            if gen_best_val < self.best_score:
                self.best_score = gen_best_val
                self.g_best = copy.deepcopy(gen_best)

            # ── Restart check ──
            if self._check_stagnation(gen):
                self._restart_population()
                self.last_restart_gen = gen

                # Evaluate restarted population
                wrapped = [[ind] for ind in self.pop]
                wrapped = self.toolbox.evaluate(wrapped)
                self.pop = [item[0] for item in wrapped]

            # ── LIS (Local Iterative Search) ──
            if gen > self.last_restart_gen + self.lis_interval:
                lis_best = self._local_iterative_search(global_gen=gen)
                if lis_best.fitness.valid and lis_best.fitness.values[0] < self.best_score:
                    logger.info(f"LIS improved global best: {self.best_score:.4f} → {lis_best.fitness.values[0]:.4f}")
                    self.best_score = lis_best.fitness.values[0]
                    self.g_best = copy.deepcopy(lis_best)
                    # Inject LIS best into population
                    worst_idx = max(range(len(self.pop)),
                                    key=lambda i: self.pop[i].fitness.values[0] if self.pop[i].fitness.valid else float('inf'))
                    self.pop[worst_idx] = copy.deepcopy(lis_best)

            self.record_logbook(gen, [[ind] for ind in self.pop])

            logger.info(
                f"[Gen {gen}] Best of gen = {gen_best_val:.4f}, "
                f"Global best = {self.best_score:.4f}, "
                f"F_corpus = {len(self.F_corpus)}"
            )

            self.save_checkpoint()

        logger.info("Time budget exhausted. AVFuzzer complete.")

    def close(self):
        super().close()
