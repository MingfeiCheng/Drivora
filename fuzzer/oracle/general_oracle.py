from typing import List
from tqdm import tqdm
from loguru import logger
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d

# All criteria produced by RuntimeSingleTest
ALL_CRITERIA = [
    "collision", "stuck", "reach_destination", "offroad",
    "overspeed", "running_stop", "running_red_light", "wrong_lane",
]

# Criteria that indicate a safety violation (trigger expected=True)
SAFETY_CRITERIA = [
    "collision", "stuck", "offroad", "overspeed",
    "running_stop", "running_red_light", "wrong_lane",
]


class ScenarioOracle:
    def __init__(self, config):
        self.config = config
        self.collision_recheck = config.get("collision_recheck", True)

    def evaluate(self, scenario_observation: List[dict], runtime_results: dict):

        oracle_result = {
            "expected": False,
            "ignored": False,
            "runtime_results": runtime_results,
            "criteria_summary": {},
            "offline_results": {},
        }

        # Extract all criteria per actor from runtime results
        for criteria_name, criteria_result in runtime_results.items():
            if "_group_criteria" not in criteria_name:
                continue

            for actor_id, actor_result in criteria_result.items():
                actor_summary = {}
                for criterion in ALL_CRITERIA:
                    criterion_data = actor_result.get(criterion, {})
                    occurred = criterion_data.get("occurred", False)
                    details = criterion_data.get("details", {})
                    actor_summary[criterion] = {
                        "occurred": occurred,
                        "details": details,
                    }

                    # Mark expected if any safety criterion fired
                    if occurred and criterion in SAFETY_CRITERIA:
                        oracle_result["expected"] = True

                oracle_result["criteria_summary"][actor_id] = actor_summary

        # Offline collision recheck via bounding-box intersection
        if self.collision_recheck:
            collision_recheck_result = self._recheck_collisions(scenario_observation)
            oracle_result["offline_results"]["collision_recheck"] = collision_recheck_result

            for ego_id, collided in collision_recheck_result.items():
                if collided:
                    oracle_result["expected"] = True
                    break

        return oracle_result

    def _recheck_collisions(self, scenario_observation: List[dict]) -> dict:
        """Recheck collisions via 2D bounding-box polygon intersection."""
        collision_recheck_result = {}

        for current_scene in tqdm(scenario_observation, desc="Collision recheck"):
            ego_obs_group = current_scene.get("egos", {})

            for ego_id, ego_obs in ego_obs_group.items():
                if collision_recheck_result.get(ego_id, False):
                    continue  # already detected

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
                    if ego_poly.intersects(npc_poly):
                        collision_recheck_result[ego_id] = True
                        break

        return collision_recheck_result

    @staticmethod
    def _actor_polygon(actor_obs: dict):
        """Build a 2D polygon from an actor observation dict."""
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
