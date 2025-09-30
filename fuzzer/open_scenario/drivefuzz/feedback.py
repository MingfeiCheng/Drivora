import os
import numpy as np

from tqdm import tqdm
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d

from .behavior_model import CoverageModel

class FeedbackCalculator:
    
    def __init__(self, config):
        self.config = config
        # Initialize feedback parameters based on config
        
        self.scale_collision = config.get("scale_collision", 10.0)
        self.scale_stuck = config.get("scale_stuck", 180.0)
        self.scale_destination = config.get("scale_destination", 20.0)
        
        self.coverage_step = config.get("coverage_step", 2) # 1 / 20 -> 0.05 -> 0.1
        self.coverage_cluster_num = config.get("coverage_cluster_num", 50)
        self.coverage_window_size = config.get("coverage_window_size", 10) # 1 seconds
        self.coverage_threshold = config.get("coverage_threshold", 0.4)
        self.coverage_dynamic_threshold = config.get("coverage_dynamic_threshold", False)
        
        self.coverage_model = CoverageModel(
            window_size=self.coverage_window_size,
            cluster_num=self.coverage_cluster_num,
            threshold_coverage=self.coverage_threshold,
            use_dynamic_threshold=self.coverage_dynamic_threshold
        )
        
        self.is_initialized = False
    
    def save_checkpoint(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        save_path = f"{checkpoint_dir}/coverage_model.pkl"
        self.coverage_model.save(save_path)
        return {
            "coverage_model_path": save_path,
            "is_initialized": self.is_initialized
        }
        
    def load_checkpoint(self, load_info):
        if load_info is None:
            return
        
        coverage_model_path = load_info.get("coverage_model_path", None)
        if coverage_model_path is not None and os.path.exists(coverage_model_path):
            self.coverage_model = CoverageModel.load(coverage_model_path)
            self.is_initialized = load_info.get("is_initialized", False)

    def initialize_coverage_model(self, scenario_observation_list):
        # load observation data
        X = []
        for scenario_observation in scenario_observation_list:
            x = self._extract_coverage_obs(scenario_observation)
            X.append(x)
        self.coverage_model.initialize(X)
        self.is_initialized = True
    
    def _extract_coverage_obs(self, scenario_observation):
        # convert to x
        x = []
        for i in range(0, len(scenario_observation), self.coverage_step):
            current_scene = scenario_observation[i]
            ego_obs_group = current_scene['egos']
            if len(ego_obs_group.keys()) != 1:
                raise ValueError("Currently only support one ego vehicle.")
            
            for ego_id, ego_obs in ego_obs_group.items():
                ego_yaw = ego_obs['rotation'][2]
                ego_velocity = ego_obs['velocity']
                ego_acceleration = ego_obs['acceleration']
                ego_control = [ego_obs['control']['throttle'], ego_obs['control']['brake'], ego_obs['control']['steer']]
                
                ego_state_vector = np.array([
                    ego_velocity[0], ego_velocity[1], ego_velocity[2],
                    ego_acceleration[0], ego_acceleration[1], ego_acceleration[2],
                    ego_yaw
                ])
                ego_action = np.array(ego_control)
                frame_vector = np.concatenate((ego_state_vector, ego_action), axis=0)
                x.append(frame_vector)
                
        x = np.array(x) # (n, 10)
        return x
    
    def _calculate_diversity(self, scenario_observation):
        if not self.is_initialized:
            return True, 1.0
        
        x = self._extract_coverage_obs(scenario_observation)
        is_new, cov_score, _ = self.coverage_model.feedback_coverage_behavior(x)
        return is_new, cov_score            
        
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
    
    def _calculate_safety_score(self, scenario_observation, oracle_result):
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
        
        collision_feedback = self._calculate_collision_feedback(scenario_observation)
        collision_feedback = float(np.clip(collision_feedback / self.scale_collision, 0, 1))
        
        stuck_feedback = 1 - float(np.clip(max_stuck_time / self.scale_stuck, 0, 1)) # min is better
        
        destination_feedback = 1 - float(np.clip(max_distance_to_destination / self.scale_destination, 0, 1)) # min is better
        
        score = (collision_feedback + stuck_feedback + destination_feedback) / 3.0
        
        safety_score_log = {
            "collision_feedback": collision_feedback,
            "stuck_feedback": stuck_feedback,
            "destination_feedback": destination_feedback
        }
        
        return score, safety_score_log

    def evaluate(self, observation_data, oracle_result):
        # Implement your feedback evaluation logic here
                
        safety_score, safety_score_log = self._calculate_safety_score(observation_data, oracle_result)
        is_new, diversity_score = self._calculate_diversity(observation_data)
        
        return {
            "safety_score": safety_score,
            "diversity_score": diversity_score,
            "is_new": is_new,
            "details": {
                "is_new_coverage": is_new,
                **safety_score_log
            }  # Additional feedback details
        }