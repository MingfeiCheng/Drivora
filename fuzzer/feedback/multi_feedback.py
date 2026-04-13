"""
Multi-ADS Feedback Calculator — computes per-ego and aggregate fitness
for scenarios with multiple ego vehicles (potentially different ADS agents).

Metrics per ego:
  - collision_feedback: min distance between ego bbox and any NPC bbox
  - stuck_feedback: max blocked duration (inverted)
  - destination_feedback: remaining distance to destination (inverted)

Aggregate score: average across all egos (lower = more dangerous = better for fuzzer).
Also exposes per-ego scores for multi-objective optimization.
"""

import numpy as np

from tqdm import tqdm
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d


class MultiADSFeedbackCalculator:
    """Feedback calculator for multi-ADS scenarios."""

    def __init__(self, config):
        self.config = config
        self.scale_collision = config.get("scale_collision", 10.0)
        self.scale_stuck = config.get("scale_stuck", 180.0)
        self.scale_destination = config.get("scale_destination", 20.0)

    def get_default_feedback(self):
        return {
            "score": 1.0,
            "single_score": 1.0,
            "mutliple_scores": [1.0, 1.0, 1.0],
            "per_ego": {},
            "details": {
                "collision_feedback": 1.0,
                "stuck_feedback": 1.0,
                "destination_feedback": 1.0,
            },
        }

    def evaluate(self, observation_data, oracle_result):
        runtime_results = oracle_result.get("runtime_results", {})

        # ── Per-ego metrics ──
        per_ego = {}
        ego_ids = self._collect_ego_ids(observation_data)

        for ego_id in ego_ids:
            collision_fb = self._ego_collision_feedback(observation_data, ego_id)
            collision_fb = float(np.clip(collision_fb / self.scale_collision, 0, 1))

            max_dist, max_stuck = self._ego_runtime_metrics(runtime_results, ego_id)
            stuck_fb = 1 - float(np.clip(max_stuck / self.scale_stuck, 0, 1))
            dest_fb = 1 - float(np.clip(max_dist / self.scale_destination, 0, 1))

            ego_score = (collision_fb + stuck_fb + dest_fb) / 3.0

            per_ego[ego_id] = {
                "score": ego_score,
                "collision_feedback": collision_fb,
                "stuck_feedback": stuck_fb,
                "destination_feedback": dest_fb,
            }

        # ── Also compute ego-to-ego min distance ──
        ego_ego_dist = self._ego_ego_min_distance(observation_data, ego_ids)

        # ── Aggregate score: average across all egos ──
        if per_ego:
            all_collision = [v["collision_feedback"] for v in per_ego.values()]
            all_stuck = [v["stuck_feedback"] for v in per_ego.values()]
            all_dest = [v["destination_feedback"] for v in per_ego.values()]

            agg_collision = min(all_collision)  # worst-case collision proximity
            agg_stuck = min(all_stuck)          # worst-case stuck
            agg_dest = min(all_dest)            # worst-case destination

            # Include ego-ego distance in collision metric
            if ego_ego_dist < float("inf"):
                ego_ego_fb = float(np.clip(ego_ego_dist / self.scale_collision, 0, 1))
                agg_collision = min(agg_collision, ego_ego_fb)

            score = (agg_collision + agg_stuck + agg_dest) / 3.0
        else:
            agg_collision = 1.0
            agg_stuck = 1.0
            agg_dest = 1.0
            score = 1.0

        return {
            "score": score,
            "single_score": score,
            "mutliple_scores": [agg_collision, agg_stuck, agg_dest],
            "per_ego": per_ego,
            "ego_ego_min_distance": ego_ego_dist,
            "details": {
                "collision_feedback": agg_collision,
                "stuck_feedback": agg_stuck,
                "destination_feedback": agg_dest,
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _collect_ego_ids(self, observation_data):
        """Collect all ego IDs seen across all frames."""
        ids = set()
        for frame in observation_data:
            for eid in frame.get("egos", {}):
                ids.add(eid)
        return sorted(ids)

    def _ego_collision_feedback(self, observation_data, ego_id):
        """Min distance between a specific ego and any NPC across all frames."""
        min_dist = float("inf")
        for frame in observation_data:
            ego_obs = frame.get("egos", {}).get(ego_id)
            if not ego_obs:
                continue
            ego_poly = self._actor_polygon(ego_obs)
            if ego_poly is None:
                continue

            npc_actors = (
                frame.get("other_actors", {}).get("vehicles", [])
                + frame.get("other_actors", {}).get("walkers", [])
                + frame.get("other_actors", {}).get("static_props", [])
            )
            for npc in npc_actors:
                npc_poly = self._actor_polygon(npc)
                if npc_poly is None:
                    continue
                dist = ego_poly.distance(npc_poly)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def _ego_ego_min_distance(self, observation_data, ego_ids):
        """Min distance between any pair of ego vehicles across all frames."""
        if len(ego_ids) < 2:
            return float("inf")

        min_dist = float("inf")
        for frame in observation_data:
            egos = frame.get("egos", {})
            polys = {}
            for eid in ego_ids:
                obs = egos.get(eid)
                if obs:
                    p = self._actor_polygon(obs)
                    if p:
                        polys[eid] = p

            ego_list = list(polys.keys())
            for i in range(len(ego_list)):
                for j in range(i + 1, len(ego_list)):
                    dist = polys[ego_list[i]].distance(polys[ego_list[j]])
                    if dist < min_dist:
                        min_dist = dist

        return min_dist

    def _ego_runtime_metrics(self, runtime_results, ego_id):
        """Extract destination distance and stuck time for a specific ego."""
        max_dist = 0.0
        max_stuck = 0.0

        for criteria_name, criteria_result in runtime_results.items():
            if "_group_criteria" not in criteria_name:
                continue
            actor_result = criteria_result.get(ego_id, {})
            dist = (
                actor_result.get("reach_destination", {})
                .get("details", {})
                .get("distance_to_destination", 0.0)
            )
            if dist > max_dist:
                max_dist = dist
            stuck = (
                actor_result.get("stuck", {})
                .get("details", {})
                .get("max_blocked_duration", 0.0)
            )
            if stuck > max_stuck:
                max_stuck = stuck

        return max_dist, max_stuck

    @staticmethod
    def _actor_polygon(actor_obs):
        bbox = actor_obs.get("bounding_box")
        if not bbox:
            return None
        try:
            corners = calculate_bbox_polygon_2d(
                actor_location_x=actor_obs["location"][0],
                actor_location_y=actor_obs["location"][1],
                actor_yaw=actor_obs["rotation"][2],
                bbox_extent_x=bbox["extent"][0],
                bbox_extent_y=bbox["extent"][1],
                bbox_rotation_yaw=bbox["rotation"][2],
            )
            return Polygon(corners)
        except (KeyError, IndexError):
            return None
