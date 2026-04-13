import numpy as np

from tqdm import tqdm
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d


class AVFuzzerFeedbackCalculator:
    """Feedback calculator for AVFuzzer — multi-objective composite scoring."""

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
            "details": {
                "collision_feedback": 1.0,
                "stuck_feedback": 1.0,
                "destination_feedback": 1.0,
            },
        }

    def _calculate_collision_feedback(self, scenario_observation):
        min_distance = float("inf")

        for current_scene in tqdm(scenario_observation, desc="Collision feedback"):
            ego_obs_group = current_scene.get("egos", {})
            for ego_id, ego_obs in ego_obs_group.items():
                ego_poly = self._actor_polygon(ego_obs)
                if ego_poly is None:
                    continue

                npc_actors = (
                    current_scene.get("other_actors", {}).get("vehicles", [])
                    + current_scene.get("other_actors", {}).get("walkers", [])
                    + current_scene.get("other_actors", {}).get("static_props", [])
                )

                for npc_actor in npc_actors:
                    npc_poly = self._actor_polygon(npc_actor)
                    if npc_poly is None:
                        continue
                    dist = ego_poly.distance(npc_poly)
                    if dist < min_distance:
                        min_distance = dist

        return min_distance

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

    def _extract_runtime_metrics(self, runtime_results):
        max_distance_to_destination = 0.0
        max_stuck_time = 0.0

        for criteria_name, criteria_result in runtime_results.items():
            if "_group_criteria" not in criteria_name:
                continue
            for actor_id, actor_result in criteria_result.items():
                dist = (
                    actor_result.get("reach_destination", {})
                    .get("details", {})
                    .get("distance_to_destination", 0.0)
                )
                if dist > max_distance_to_destination:
                    max_distance_to_destination = dist

                stuck = (
                    actor_result.get("stuck", {})
                    .get("details", {})
                    .get("max_blocked_duration", 0.0)
                )
                if stuck > max_stuck_time:
                    max_stuck_time = stuck

        return max_distance_to_destination, max_stuck_time

    def evaluate(self, observation_data, oracle_result):
        runtime_results = oracle_result.get("runtime_results", {})
        max_dist, max_stuck = self._extract_runtime_metrics(runtime_results)

        collision_feedback = self._calculate_collision_feedback(observation_data)
        collision_feedback = float(np.clip(collision_feedback / self.scale_collision, 0, 1))

        stuck_feedback = 1 - float(np.clip(max_stuck / self.scale_stuck, 0, 1))
        destination_feedback = 1 - float(np.clip(max_dist / self.scale_destination, 0, 1))

        # AVFuzzer: multi-objective composite score
        score = (collision_feedback + stuck_feedback + destination_feedback) / 3.0

        return {
            "score": score,
            "single_score": score,
            "mutliple_scores": [collision_feedback, stuck_feedback, destination_feedback],
            "details": {
                "collision_feedback": collision_feedback,
                "stuck_feedback": stuck_feedback,
                "destination_feedback": destination_feedback,
            },
        }
