"""
SAMOTA Utilities — core algorithm helpers adapted from ICSE-SAMOTA.

Provides:
  - SamotaCandidate: wraps a feature vector with objective values + uncertainty
  - Feature extraction: ScenarioConfig → flat feature vector
  - GA operations: dominance, preference sort, crowding distance
  - Archive management: multi-objective threshold-based archive
  - Global Search (GS) and Local Search (LS) procedures
"""

import copy
import random
import numpy as np
import hdbscan

from typing import List, Tuple, Optional
from loguru import logger

from fuzzer.misc.surrogate_models import EnsembleSurrogate


# ═══════════════════════════════════════════════════════════════════════════
#  SamotaCandidate
# ═══════════════════════════════════════════════════════════════════════════

class SamotaCandidate:
    """Wraps a scenario feature vector with multi-objective fitness."""

    def __init__(self, feature_vector: list):
        self.feature_vector = list(feature_vector)
        self.objective_values = []       # list of float, one per objective
        self.uncertainty_values = []     # list of float, one per objective
        self.crowding_distance = 0.0
        self.objectives_covered = []     # which objectives are satisfied

    def get_features(self) -> list:
        return self.feature_vector

    def get_objective_values(self) -> list:
        return self.objective_values

    def get_objective_value(self, idx) -> float:
        return self.objective_values[idx]

    def set_objective_values(self, vals):
        self.objective_values = list(vals)

    def get_uncertainty_value(self, idx) -> float:
        return self.uncertainty_values[idx] if idx < len(self.uncertainty_values) else 0.0

    def set_uncertainty_values(self, vals):
        self.uncertainty_values = list(vals)

    def set_crowding_distance(self, cd):
        self.crowding_distance = cd

    def get_crowding_distance(self):
        return self.crowding_distance

    def add_objective_covered(self, obj_idx):
        if obj_idx not in self.objectives_covered:
            self.objectives_covered.append(obj_idx)

    def is_objective_covered(self, obj_idx) -> bool:
        return obj_idx in self.objectives_covered


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Extraction (ScenarioConfig → flat vector)
# ═══════════════════════════════════════════════════════════════════════════

def scenario_to_features(scenario) -> list:
    """Extract a flat numeric feature vector from a ScenarioConfig.

    Features include:
      - NPC vehicle: count, speeds, trigger times
      - NPC walker: count, trigger times
      - NPC static: count
      - Weather: all 10 parameters
      - Traffic light: timing
    Ego route is NOT included (fixed per scenario space).
    """
    features = []

    # NPC vehicles
    npcs = scenario.npc_vehicles or []
    features.append(len(npcs))
    npc_speeds = []
    npc_triggers = []
    for npc in npcs[:5]:  # cap at 5 NPCs for fixed-length vector
        avg_speed = np.mean([wp.speed for wp in npc.route]) if npc.route else 0.0
        npc_speeds.append(avg_speed)
        npc_triggers.append(npc.trigger_time)
    # Pad to 5
    while len(npc_speeds) < 5:
        npc_speeds.append(0.0)
        npc_triggers.append(0.0)
    features.extend(npc_speeds)
    features.extend(npc_triggers)

    # NPC walkers
    walkers = scenario.npc_walkers or []
    features.append(len(walkers))
    walker_triggers = [w.trigger_time for w in walkers[:3]]
    while len(walker_triggers) < 3:
        walker_triggers.append(0.0)
    features.extend(walker_triggers)

    # NPC statics
    statics = scenario.npc_statics or []
    features.append(len(statics))

    # Weather
    if scenario.weather:
        w = scenario.weather
        features.extend([
            w.cloudiness, w.precipitation, w.precipitation_deposits,
            w.wind_intensity, w.sun_azimuth_angle, w.sun_altitude_angle,
            w.fog_density, w.fog_distance, w.wetness, w.fog_falloff,
        ])
    else:
        features.extend([0.0] * 10)

    # Traffic light
    if scenario.traffic_light:
        tl = scenario.traffic_light
        features.extend([
            tl.yellow_time, tl.red_time,
            tl.green_time if tl.green_time else tl.red_time,
        ])
    else:
        features.extend([0.0] * 3)

    return features


def get_feature_dim() -> int:
    """Return the dimension of the feature vector."""
    # 1(npc_count) + 5(speeds) + 5(triggers) + 1(walker_count) + 3(walker_triggers)
    # + 1(static_count) + 10(weather) + 3(tl) = 29
    return 29


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-objective helpers
# ═══════════════════════════════════════════════════════════════════════════

def dominates(vals1: list, vals2: list, objectives: list) -> bool:
    """Returns True if vals1 dominates vals2 on the given objectives."""
    better_in_any = False
    for obj in objectives:
        if vals1[obj] < vals2[obj]:
            better_in_any = True
        elif vals1[obj] > vals2[obj]:
            return False
    return better_in_any


def preference_sort(population: list, size: int, objective_uncovered: list) -> list:
    """SAMOTA's preference sort: pick best per objective, then non-dominated sort."""
    result = []
    remaining = list(population)

    for obj in objective_uncovered:
        if not remaining:
            break
        best = min(remaining, key=lambda c: c.get_objective_value(obj))
        result.append(best)
        remaining.remove(best)

    if len(result) >= size:
        result.extend(remaining)
    else:
        # Non-dominated sort on remaining
        fronts = fast_non_dominated_sort(remaining, objective_uncovered)
        for front in fronts:
            result.extend(front)

    return result


def fast_non_dominated_sort(population: list, objectives: list) -> list:
    """Return list of fronts (each front is a list of candidates)."""
    remaining = list(population)
    fronts = []
    while remaining:
        front = []
        for p in remaining:
            is_dominated = False
            for q in remaining:
                if p is q:
                    continue
                if dominates(q.get_objective_values(), p.get_objective_values(), objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                front.append(p)
        if not front:
            fronts.append(remaining)
            break
        fronts.append(front)
        for f in front:
            remaining.remove(f)
    return fronts


def compute_crowding_distance(population: list, n_obj: int) -> list:
    """Compute and assign crowding distance to each candidate."""
    n = len(population)
    if n <= 2:
        for c in population:
            c.set_crowding_distance(float('inf'))
        return population

    for c in population:
        c.set_crowding_distance(0.0)

    for m in range(n_obj):
        population.sort(key=lambda c: c.get_objective_value(m))
        population[0].set_crowding_distance(float('inf'))
        population[-1].set_crowding_distance(float('inf'))

        obj_min = population[0].get_objective_value(m)
        obj_max = population[-1].get_objective_value(m)
        obj_range = obj_max - obj_min if obj_max > obj_min else 1.0

        for i in range(1, n - 1):
            dist = (population[i + 1].get_objective_value(m) - population[i - 1].get_objective_value(m)) / obj_range
            population[i].set_crowding_distance(population[i].get_crowding_distance() + dist)

    return population


# ═══════════════════════════════════════════════════════════════════════════
#  Archive management
# ═══════════════════════════════════════════════════════════════════════════

def update_archive(population: list, objective_uncovered: list, archive: list,
                   n_objectives: int, thresholds: list):
    """Update archive: add candidates that satisfy objective thresholds."""
    for obj in range(n_objectives):
        for candidate in population:
            if not candidate.get_objective_values():
                continue
            val = candidate.get_objective_value(obj)
            if val <= thresholds[obj]:
                existing = None
                existing_idx = None
                for i, arc in enumerate(archive):
                    if arc.is_objective_covered(obj):
                        existing = arc
                        existing_idx = i
                        break

                if existing is not None:
                    if val < existing.get_objective_value(obj):
                        candidate.add_objective_covered(obj)
                        archive[existing_idx] = candidate
                        if obj in objective_uncovered:
                            objective_uncovered.remove(obj)
                else:
                    candidate.add_objective_covered(obj)
                    archive.append(candidate)
                    if obj in objective_uncovered:
                        objective_uncovered.remove(obj)


# ═══════════════════════════════════════════════════════════════════════════
#  Population generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_random_candidates(size: int, lb: list, ub: list) -> list:
    """Generate random candidates within bounds."""
    candidates = []
    for _ in range(size):
        fv = [random.uniform(lb[i], ub[i]) for i in range(len(lb))]
        candidates.append(SamotaCandidate(fv))
    return candidates


def generate_adaptive_random_candidates(size: int, lb: list, ub: list) -> list:
    """Adaptive random testing: maximize minimum distance between candidates."""
    pop = [generate_random_candidates(1, lb, ub)[0]]
    while len(pop) < size:
        batch = generate_random_candidates(size, lb, ub)
        best = max(batch, key=lambda c: min(
            np.linalg.norm(np.array(c.get_features()) - np.array(p.get_features()))
            for p in pop
        ))
        pop.append(best)
    return pop


# ═══════════════════════════════════════════════════════════════════════════
#  GA offspring generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_offspring(parent_pop: list, objectives: list, lb: list, ub: list) -> list:
    """Generate offspring via tournament selection + crossover + mutation."""
    offspring = []
    pop_size = len(parent_pop)

    for _ in range(pop_size):
        # Tournament selection (size 2)
        t = random.sample(parent_pop, min(2, pop_size))
        parent1 = min(t, key=lambda c: c.get_objective_values()[objectives[0]] if c.get_objective_values() else float('inf'))
        t = random.sample(parent_pop, min(2, pop_size))
        parent2 = min(t, key=lambda c: c.get_objective_values()[objectives[0]] if c.get_objective_values() else float('inf'))

        # Single-point crossover
        fv1 = parent1.get_features()
        fv2 = parent2.get_features()
        point = random.randint(1, len(fv1) - 1)
        child_fv = fv1[:point] + fv2[point:]

        # Uniform mutation
        for i in range(len(child_fv)):
            if random.random() < 0.1:
                child_fv[i] = random.uniform(lb[i], ub[i])
            child_fv[i] = max(lb[i], min(ub[i], child_fv[i]))

        offspring.append(SamotaCandidate(child_fv))

    return offspring


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate with ensemble
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_with_ensembles(ensembles: List[EnsembleSurrogate], candidates: list):
    """Evaluate candidates using ensemble surrogate models."""
    n_obj = len(ensembles)
    for candidate in candidates:
        obj_vals = [1.0] * n_obj
        unc_vals = [0.0] * n_obj
        for ens in ensembles:
            idx = ens.objective_index
            pred, unc = ens.predict(np.array(candidate.get_features()))
            obj_vals[idx] = pred
            unc_vals[idx] = unc
        candidate.set_objective_values(obj_vals)
        candidate.set_uncertainty_values(unc_vals)


# ═══════════════════════════════════════════════════════════════════════════
#  Global Search (GS) — SAMOTA's surrogate-guided NSGA-II
# ═══════════════════════════════════════════════════════════════════════════

def global_search(database: list, objective_uncovered: list, pop_size: int,
                  n_generations: int, lb: list, ub: list, n_objectives: int):
    """Run surrogate-guided evolutionary search.

    Returns: (candidates, ensembles) — best candidates + trained ensemble models.
    """
    # Train ensemble for each uncovered objective
    logger.info(f"[GS] Training ensembles for {len(objective_uncovered)} uncovered objectives "
                f"(database size={len(database)})...")
    ensembles = []
    for obj in objective_uncovered:
        X = np.array([c.get_features() for c in database])
        y = np.array([c.get_objective_value(obj) for c in database])
        ens = EnsembleSurrogate(objective_index=obj)
        try:
            ens.train(X, y)
            logger.info(f"[GS] Ensemble obj={obj} trained — "
                        f"weights: RBF={ens.w_rbf:.3f} PR={ens.w_poly:.3f} KR={ens.w_kriging:.3f}")
        except Exception as e:
            logger.warning(f"[GS] Ensemble training failed for obj {obj}: {e}")
            continue
        ensembles.append(ens)

    if not ensembles:
        logger.warning("[GS] No ensembles trained, skipping global search.")
        return [], []

    # Initialize population
    P = generate_random_candidates(pop_size, lb, ub)
    evaluate_with_ensembles(ensembles, P)

    T_b = [None] * n_objectives  # best per objective
    T_n = [None] * n_objectives  # most uncertain per objective

    logger.info(f"[GS] Running NSGA-II for {n_generations} generations (pop_size={pop_size})...")
    for gen in range(n_generations):
        Q = generate_offspring(P, objective_uncovered, lb, ub)
        evaluate_with_ensembles(ensembles, Q)

        R = P + Q

        # Update per-objective bests
        for obj in objective_uncovered:
            for c in R:
                if T_b[obj] is None or c.get_objective_value(obj) < T_b[obj].get_objective_value(obj):
                    T_b[obj] = copy.deepcopy(c)
                if T_n[obj] is None or c.get_uncertainty_value(obj) > T_n[obj].get_uncertainty_value(obj):
                    T_n[obj] = copy.deepcopy(c)

        # Preference sort + crowding distance selection
        sorted_pop = preference_sort(R, pop_size, objective_uncovered)
        P = sorted_pop[:pop_size]

        if (gen + 1) % 20 == 0 or gen == n_generations - 1:
            best_vals = {obj: f"{T_b[obj].get_objective_value(obj):.4f}" if T_b[obj] else "N/A"
                         for obj in objective_uncovered}
            logger.info(f"[GS] Gen {gen+1}/{n_generations} — best per obj: {best_vals}")

    # Collect results
    results = []
    for c in T_b:
        if c is not None:
            results.append(c)
    for c in T_n:
        if c is not None:
            results.append(c)
    logger.info(f"[GS] Complete — returning {len(results)} candidates")
    return results, ensembles


# ═══════════════════════════════════════════════════════════════════════════
#  Local Search (LS) — cluster-based RBF + GA
# ═══════════════════════════════════════════════════════════════════════════

def local_search(database: list, objective_uncovered: list, lb: list, ub: list,
                 n_clusters: int = 20) -> list:
    """Run local search: cluster the database, train per-cluster RBF, run GA."""
    from fuzzer.misc.surrogate_models import RBFSurrogate

    logger.info(f"[LS] Starting local search for {len(objective_uncovered)} objectives "
                f"(database size={len(database)})...")
    results = []
    X_all = np.array([c.get_features() for c in database])

    for obj in objective_uncovered:
        y_all = np.array([c.get_objective_value(obj) for c in database])

        # Cluster
        try:
            min_size = max(3, len(database) // n_clusters)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
            labels = clusterer.fit_predict(X_all)
        except Exception:
            labels = np.zeros(len(X_all), dtype=int)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise label
        logger.info(f"[LS] Obj {obj}: {len(unique_labels)} clusters found")

        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 3:
                continue

            X_cluster = X_all[mask]
            y_cluster = y_all[mask]

            # Train local RBF
            model = RBFSurrogate()
            try:
                model.train(X_cluster, y_cluster)
            except Exception:
                continue

            # Find local bounds
            local_lb = X_cluster.min(axis=0).tolist()
            local_ub = X_cluster.max(axis=0).tolist()
            # Widen slightly
            for i in range(len(local_lb)):
                margin = max(0.1, (local_ub[i] - local_lb[i]) * 0.1)
                local_lb[i] = max(lb[i], local_lb[i] - margin)
                local_ub[i] = min(ub[i], local_ub[i] + margin)

            # Simple GA to find minimum
            best_fv = None
            best_val = float('inf')
            for _ in range(200):
                fv = [random.uniform(local_lb[i], local_ub[i]) for i in range(len(lb))]
                val = model.predict(np.array(fv))
                if val < best_val:
                    best_val = val
                    best_fv = fv

            if best_fv is not None:
                results.append(SamotaCandidate(best_fv))

    logger.info(f"[LS] Complete — returning {len(results)} candidates")
    return results
