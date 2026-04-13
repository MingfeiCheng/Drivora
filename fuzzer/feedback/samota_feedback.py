"""
SAMOTA Feedback Calculator — 6-objective fitness.

Extracts per-type collision distances and binary violation flags
matching the 6 objectives from the SAMOTA paper:

  0. DfC — Distance from lane Center (offroad/wrong_lane proximity)
  1. DfV — Distance from Vehicles (min ego-vehicle distance)
  2. DfP — Distance from Pedestrians (min ego-pedestrian distance)
  3. DfM — Distance from static Misc objects (min ego-static distance)
  4. DT  — Traffic rule compliance (red light / stop sign)
  5. Dest — Remaining distance to destination

All values normalized to [0, 1]. Lower = more dangerous = better for fuzzer.
"""

import numpy as np
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d


N_OBJECTIVES = 6


class SAMOTAFeedbackCalculator:

    def __init__(self, config):
        self.config = config
        self.scale_vehicle = config.get("scale_vehicle", 10.0)
        self.scale_pedestrian = config.get("scale_pedestrian", 10.0)
        self.scale_static = config.get("scale_static", 10.0)
        self.scale_destination = config.get("scale_destination", 20.0)

    def get_default_feedback(self):
        return {
            "score": 1.0,
            "single_score": 1.0,
            "mutliple_scores": [1.0] * N_OBJECTIVES,
            "objective_values": [1.0] * N_OBJECTIVES,
            "details": {},
        }

    def evaluate(self, observation_data, oracle_result):
        criteria_summary = oracle_result.get("criteria_summary", {})
        runtime_results = oracle_result.get("runtime_results", {})

        # ── Per-type min distances ──
        dfv_raw = self._min_distance_by_type(observation_data, "vehicles")
        dfp_raw = self._min_distance_by_type(observation_data, "walkers")
        dfm_raw = self._min_distance_by_type(observation_data, "static_props")

        dfv = float(np.clip(dfv_raw / self.scale_vehicle, 0, 1))
        dfp = float(np.clip(dfp_raw / self.scale_pedestrian, 0, 1))
        dfm = float(np.clip(dfm_raw / self.scale_static, 0, 1))

        # ── DfC: lane keeping ──
        dfc = 1.0
        for _, summary in criteria_summary.items():
            if summary.get("offroad", {}).get("occurred", False):
                dfc = 0.0
            if summary.get("wrong_lane", {}).get("occurred", False):
                dfc = 0.0

        # ── DT: traffic rules ──
        dt = 1.0
        for _, summary in criteria_summary.items():
            if summary.get("running_red_light", {}).get("occurred", False):
                dt = 0.0
            if summary.get("running_stop", {}).get("occurred", False):
                dt = 0.0

        # ── Dest: remaining route distance ──
        dest_raw = self._remaining_distance(runtime_results)
        dest = float(np.clip(dest_raw / self.scale_destination, 0, 1))

        obj_vals = [dfc, dfv, dfp, dfm, dt, dest]
        score = sum(obj_vals) / len(obj_vals)

        return {
            "score": score,
            "single_score": score,
            "mutliple_scores": obj_vals,
            "objective_values": obj_vals,
            "details": {
                "DfC": dfc, "DfV": dfv, "DfP": dfp,
                "DfM": dfm, "DT": dt, "Dest": dest,
                "DfV_raw": dfv_raw, "DfP_raw": dfp_raw,
                "DfM_raw": dfm_raw, "Dest_raw": dest_raw,
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _min_distance_by_type(self, observation_data, actor_type: str) -> float:
        min_dist = float("inf")
        for frame in observation_data:
            actors = frame.get("other_actors", {}).get(actor_type, [])
            if not actors:
                continue
            for ego_id, ego_obs in frame.get("egos", {}).items():
                ego_poly = self._polygon(ego_obs)
                if ego_poly is None:
                    continue
                for actor in actors:
                    actor_poly = self._polygon(actor)
                    if actor_poly is None:
                        continue
                    d = ego_poly.distance(actor_poly)
                    if d < min_dist:
                        min_dist = d
        return min_dist

    def _remaining_distance(self, runtime_results: dict) -> float:
        max_dist = 0.0
        for name, result in runtime_results.items():
            if "_group_criteria" not in name:
                continue
            for _, actor_result in result.items():
                d = (actor_result.get("reach_destination", {})
                     .get("details", {})
                     .get("distance_to_destination", 0.0))
                if d > max_dist:
                    max_dist = d
        return max_dist

    @staticmethod
    def _polygon(obs):
        bbox = obs.get("bounding_box")
        if not bbox:
            return None
        try:
            corners = calculate_bbox_polygon_2d(
                actor_location_x=obs["location"][0],
                actor_location_y=obs["location"][1],
                actor_yaw=obs["rotation"][2],
                bbox_extent_x=bbox["extent"][0],
                bbox_extent_y=bbox["extent"][1],
                bbox_rotation_yaw=bbox["rotation"][2],
            )
            return Polygon(corners)
        except (KeyError, IndexError):
            return None
