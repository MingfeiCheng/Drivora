import numpy as np

from tqdm import tqdm
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d

class FeedbackCalculator:
    def __init__(self, config):
        self.config = config
        # Initialize feedback parameters based on config
        
        self.scale_collision = config.get("scale_collision", 10.0)
        self.scale_stuck = config.get("scale_stuck", 180.0)
        self.scale_destination = config.get("scale_destination", 20.0)
        
    def _calculate_collision_feedback(self, scenario_observation):
        min_distance = float('inf')
        
        scenario_length = len(scenario_observation)
        
        for i in tqdm(range(scenario_length), desc="Evaluating scenario"):
            
            current_scene = scenario_observation[i]
            
            ego_obs_group = current_scene['egos']
            for ego_id, ego_obs in ego_obs_group.items():
                
                ego_location = ego_obs['location']
                ego_yaw = ego_obs['rotation'][2]
                ego_bbox = ego_obs['bounding_box']
                
                ego_bbox_polygon = calculate_bbox_polygon_2d(
                    actor_location_x=ego_location[0],
                    actor_location_y=ego_location[1],
                    actor_yaw=ego_yaw,
                    bbox_extent_x=ego_bbox['extent'][0],
                    bbox_extent_y=ego_bbox['extent'][1],
                    bbox_rotation_yaw=ego_bbox['rotation'][2]
                )
                ego_bbox_polygon = Polygon(ego_bbox_polygon)
                
                # check collision with others 
                # npc vehicles
                npc_vehicles_list = current_scene['other_actors']['vehicles']
                
                # npc walkers
                npc_walkers_list = current_scene['other_actors']['walkers']
                
                # static props
                static_props_list = current_scene['other_actors']['static_props']
                
                npc_actors = npc_vehicles_list + npc_walkers_list + static_props_list
                for npc_actor in npc_actors:
                    npc_location = npc_actor['location']
                    npc_yaw = npc_actor['rotation'][2]
                    npc_bbox = npc_actor['bounding_box']
                    
                    npc_bbox_polygon = calculate_bbox_polygon_2d(
                        actor_location_x=npc_location[0],
                        actor_location_y=npc_location[1],
                        actor_yaw=npc_yaw,
                        bbox_extent_x=npc_bbox['extent'][0],
                        bbox_extent_y=npc_bbox['extent'][1],
                        bbox_rotation_yaw=npc_bbox['rotation'][2]
                    )
                    npc_bbox_polygon = Polygon(npc_bbox_polygon)
                    
                    dist_ego_npc = ego_bbox_polygon.distance(npc_bbox_polygon)
                    if dist_ego_npc < min_distance:
                        min_distance = dist_ego_npc
                        
        return min_distance

    def evaluate(self, observation_data, oracle_result):
        # Implement your feedback evaluation logic here
                
        runtime_results = oracle_result.get("runtime_results", {})
        max_distance_to_destination = 0.0
        max_stuck_time = 0.0
        for crietria_name, crietria_result in runtime_results.items():
            if "_group_criteria" in crietria_name:
                for actor_id, actor_result in crietria_result.items():
                    distance_to_destination = actor_result.get('reach_destination', {}).get('details', {}).get('distance_to_destination', 0.0)
                    if distance_to_destination > max_distance_to_destination:
                        max_distance_to_destination = distance_to_destination
                        
                    stuck_time = actor_result.get('stuck', {}).get('details', {}).get('max_blocked_duration', 0.0)
                    if stuck_time > max_stuck_time:
                        max_stuck_time = stuck_time
        
        collision_feedback = self._calculate_collision_feedback(observation_data)
        collision_feedback = float(np.clip(collision_feedback / self.scale_collision, 0, 1))
        
        stuck_feedback = 1 - float(np.clip(max_stuck_time / self.scale_stuck, 0, 1)) # min is better
        
        destination_feedback = 1 - float(np.clip(max_distance_to_destination / self.scale_destination, 0, 1)) # min is better
        
        score = (collision_feedback + stuck_feedback + destination_feedback) / 3.0
        
        return {
            "score": score,  # Example score
            "details": {
                "collision_feedback": collision_feedback,
                "stuck_feedback": stuck_feedback,
                "destination_feedback": destination_feedback
            }  # Additional feedback details
        }