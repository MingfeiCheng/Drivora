from typing import List
from tqdm import tqdm
from loguru import logger
from shapely.geometry import Polygon

from scenario_runner.misc import calculate_bbox_polygon_2d

class ScenarioOracle:
    def __init__(self, config):
        self.config = config
        
        self.collision_recheck = config.get("collision_recheck", True)

    def evaluate(self, scenario_observation: List[dict], runtime_results: dict):

        oracle_result = {
            "expected": False,
            "runtime_results": runtime_results,
            "offline_results": {}
        }
        
        # obtain collision info
        for crietria_name, crietria_result in runtime_results.items():
            if "_group_criteria" in crietria_name:
                for actor_id, actor_result in crietria_result.items():
                    if actor_result['collision']['occurred']:
                        oracle_result['expected'] = True
                    elif actor_result['stuck']['occurred']:
                        # if any actor reach destination, we consider it as expected
                        oracle_result['expected'] = True
                    else:
                        pass
        
        if self.collision_recheck:
            
            scenario_length = len(scenario_observation)
            collision_recheck_result = {}
            
            for i in tqdm(range(scenario_length), desc="Evaluating scenario"):
                
                current_scene = scenario_observation[i]
                            
                ego_obs_group = current_scene['egos']
                for ego_id, ego_obs in ego_obs_group.items():
                    
                    ego_recheck_result = collision_recheck_result.get(ego_id, False)
                    
                    if ego_recheck_result:
                        # already detected collision, skip
                        continue
                    
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
                        if ego_bbox_polygon.intersects(npc_bbox_polygon):
                            collision_recheck_result[ego_id] = True
                            break
                        
        for ego_id, collision in collision_recheck_result.items():
            if collision:
                oracle_result['expected'] = True
                break
            
        return oracle_result
        
                    
                    
                