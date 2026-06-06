from __future__ import annotations

import os
import time
import carla
import random
import traceback
import numpy as np

from tqdm import tqdm
from loguru import logger
from omegaconf import DictConfig
from typing import List, Tuple, Optional
from shapely.geometry import LineString

from scenario_corpus.openscenario.config import (
    ScenarioConfig, EgoConfig, MapConfig, WeatherConfig,
    StaticObstacleConfig,
)
from scenario_elements.config import Waypoint
from scenario_elements.behavior.waypoint_vehicle.behavior import WaypointVehicleConfig
from scenario_elements.behavior.ai_walker.behavior import AIWalkerConfig
from scenario_elements.behavior.traffic_light.behavior import TrafficLightBehaviorConfig
from agent_corpus.atomic.route_manipulation import GlobalRoutePlanner
from scenario_runner.ctn_operator import CtnSimOperator, get_carla_version, OLD_CARLA_VERSION

from ..scenario_space import ScenarioODDSpace

# HERE, we only focus on single task
TASK_SCOPE = [
    'LANEFOLLOW',
    'LEFT',
    'RIGHT',
    'STRAIGHT',
    'CHANGELANELEFT',
    'CHANGELANERIGHT'
]

class RandomSampler:
    
    MAX_RETRY = 50
    
    def __init__(
        self,
        scenario_space: ScenarioODDSpace,
        mutation_config: DictConfig
    ):
        self.mutation_config = mutation_config
        self.scenario_space = scenario_space
        
        # define some global parameters
        self.ctn_operator = None # should be set outside
        self._world = None
        self._client = None
        self._map = None
        
        self.carla_version = get_carla_version()        
        
        self.map_region_space = self.scenario_space.map_region_space
        self.ego_space = self.scenario_space.ego_space
        self.npc_vehicle_space = self.scenario_space.npc_vehicle_space
        self.npc_static_space = self.scenario_space.npc_static_space
        self.npc_pedestrian_space = self.scenario_space.npc_pedestrian_space
        self.weather_space = self.scenario_space.weather_space
        self.traffic_light_space = self.scenario_space.traffic_light_space
        
        self.town = self.scenario_space.map_region_space.town

        # Auto-generate cache file path from town name (shared across all fuzzers)
        cache_dir = "cache"
        self.map_cache_file = self.mutation_config.get(
            "map_cache_file",
            os.path.join(cache_dir, f"{self.town.lower()}_map_cache.json")
        )

        # DEBUG fast path: when set (int), the region-route builder stops after
        # collecting roughly this many ego AND npc routes instead of enumerating
        # every waypoint pair, and the (partial) result is NOT persisted to the
        # map cache file. Lets you sample one scenario almost instantly for
        # debugging. Default None = full enumeration + persistent cache.
        self.debug_max_routes = self.mutation_config.get("debug_max_routes", None)

        # local buffer
        # self.ego_route_buffer = [] # if task assign, keep the buffer
        if self.map_cache_file is not None:
            map_cache_parent_dir = os.path.dirname(self.map_cache_file)
            if map_cache_parent_dir and not os.path.exists(map_cache_parent_dir):
                os.makedirs(map_cache_parent_dir)
                logger.info(f"Created map cache directory: {map_cache_parent_dir}")
        
    ####### Basic Tools #######
    def _setup(self, town):
        if not self.ctn_operator.is_connected:
            self.ctn_operator.start()

        self._client = self.ctn_operator.client
        try:
            self._world = self._client.load_world(town)
        except Exception as e:
            logger.error(traceback.print_exc())

        settings = self._world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.ctn_operator.fps
        settings.synchronous_mode = True
        self._world.apply_settings(settings)
        self._world.reset_all_traffic_lights()
        self._world.tick()
        self._map = self._world.get_map()
        
        # update cache
        if not self.map_region_space.map_driving_waypoints_cache or not self.map_region_space.map_ego_routes_cache or not self.map_region_space.map_npc_routes_cache:
            # debug fast path bypasses the persistent cache (load + save) so it
            # always does a quick, partial build
            if (self.debug_max_routes is None
                    and os.path.isfile(self.map_cache_file)
                    and os.path.getsize(self.map_cache_file) > 0):
                self.map_region_space.load_from_file(self.map_cache_file)
            else:
                cache_wps, cache_ego_routes, cache_npc_routes = self._get_region_driving_wps_routes(
                    region_x_min=self.map_region_space.region_x_min,
                    region_x_max=self.map_region_space.region_x_max,
                    region_y_min=self.map_region_space.region_y_min,
                    region_y_max=self.map_region_space.region_y_max
                )
                self.map_region_space.map_driving_waypoints_cache = cache_wps
                self.map_region_space.map_ego_routes_cache = cache_ego_routes
                self.map_region_space.map_npc_routes_cache = cache_npc_routes
                if self.debug_max_routes is None:
                    self.map_region_space.save_to_file(self.map_cache_file)

        # if len(self.ego_route_buffer) == 0:
        #     self._load_assigned_routes(self.ego_space.ego_task_assignment)
        #     self.map_region_space
        if self.map_region_space.cache_ego_route_list is None:
            self.map_region_space.build_route_indexes(self.ego_space.ego_task_assignment)
                
    def _cleanup(self):
        """Clean up actors and disconnect from CARLA."""
        try:
            if self._world is not None:
                for actor in self._world.get_actors():
                    if actor and actor.is_alive:
                        try:
                            actor.destroy()
                        except Exception:
                            pass
        except Exception:
            pass

        # Release CARLA references so C++ destructors don't fire later
        self._world = None
        self._client = None
        self._map = None
        if self.ctn_operator is not None:
            self.ctn_operator.client = None
            self.ctn_operator.world = None
    
    # def _load_assigned_routes(self, ego_assigned_task: Optional[str]) -> bool:
    #     self.ego_route_buffer = []
    #     if ego_assigned_task is None:
    #         for ego_route_task, ego_route_pool, in self.map_region_space.map_ego_routes_cache.items():
    #             # expected str
    #             self.ego_route_buffer.extend(ego_route_pool)
                
    #     else:
    #         ego_assigned_task = self.ego_space.ego_task_assignment
    #         if ego_assigned_task not in TASK_SCOPE:
    #             raise ValueError(f"Unsupported ego task assignment: {ego_assigned_task}, supported: {TASK_SCOPE}")
            
    #         ego_task_str = f"LANEFOLLOW-{ego_assigned_task}-LANEFOLLOW"
    #         if ego_task_str not in self.map_region_space.map_ego_routes_cache.keys():
    #             logger.error(f"Ego assigned task {ego_task_str} not found in cached ego routes. Only have: {list(self.map_region_space.map_ego_routes_cache.keys())}")
    #             raise ValueError(f"Ego assigned task {ego_task_str} not found in cached ego routes.")
            
    #         self.ego_route_buffer = self.map_region_space.map_ego_routes_cache[ego_task_str]
            
    #     if len(self.ego_route_buffer) == 0:
    #         logger.error(f"No available ego routes in the buffer after loading assigned routes. Your task: {ego_assigned_task}")
    #         raise ValueError("No available ego routes in the buffer after loading assigned routes.")
            
        
    def _put_actor(self, actor_model, x, y, z, pitch, yaw, roll):
        
        try:
            bp_filter = actor_model
            blueprint = random.choice(self._world.get_blueprint_library().filter(bp_filter))
        except Exception as e:
            logger.error(f"Failed to get blueprint for model {actor_model}")
            logger.error(traceback.print_exc())
            return None
        
        # Get the spawn point
        agent_initial_transform = carla.Transform(
            location=carla.Location(
                x=x,
                y=y,
                z=z
            ),
            rotation=carla.Rotation(
                pitch=pitch,
                yaw=yaw,
                roll=roll
            )
        )

        _spawn_point = carla.Transform()
        _spawn_point.rotation = agent_initial_transform.rotation
        _spawn_point.location.x = agent_initial_transform.location.x
        _spawn_point.location.y = agent_initial_transform.location.y
        _spawn_point.location.z = agent_initial_transform.location.z + 1.321  # + 0.2

        actor = self._world.try_spawn_actor(blueprint, _spawn_point)
        self._world.tick()

        if actor is None:
            return None

        return actor

    def _check_ego(
        self, 
        agents_info
    ):
        # TODO: need more fine-grained checking, no need to put first
        def clean(actors):
            for _actor in actors:
                if _actor:
                    _actor.destroy()

        added_actors = list()
        for agent in agents_info:
            
            carla_actor = self._put_actor(
                agent['model'],
                agent['x'],
                agent['y'],
                agent['z'],
                agent['pitch'],
                agent['yaw'],
                agent['roll']
            )
            if carla_actor is None:
                clean(added_actors)
                return False

            # carla_actor.set_simulate_physics(enabled=set_physics)
            added_actors.append(carla_actor)
        clean(added_actors)
        return True

    def _interpolate_trajectory(self, waypoints_trajectory, hop_resolution=2.0):
        """
        NOTE: here we use fixed 1.0 as estimation -> TODO: set the senario also 2.0 pls
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
        returns the full interpolated route both in GPS coordinates and also in its original form.
        
        Args:
            - waypoints_trajectory: the current coarse trajectory
            - hop_resolution: distance between the trajectory's waypoints
        """
        # Cache the GlobalRoutePlanner: its constructor rebuilds the full map
        # topology graph (_build_topology + _build_graph). This method is called
        # O(n^2) times by the region-route builder, so reconstructing it every
        # call dominated runtime. Same (map, hop_resolution) -> identical results.
        cache = getattr(self, "_grp_cache", None)
        if cache is None or cache.get("map") is not self._map:
            cache = {"map": self._map, "grp": {}}
            self._grp_cache = cache
        grp = cache["grp"].get(hop_resolution)
        if grp is None:
            grp = GlobalRoutePlanner(self._map, hop_resolution)
            cache["grp"][hop_resolution] = grp
        route = []

        for i in range(len(waypoints_trajectory) - 1):
            waypoint = waypoints_trajectory[i]
            waypoint_next = waypoints_trajectory[i + 1]
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            for wp, connection in interpolated_trace:
                route.append([wp.transform, connection]) # connection is a type of RoadOption
        return route

    @staticmethod
    def _check_route_overlapping(route1: List[Waypoint], route2: List[Waypoint], front_length_ratio=0.2):
        """
        Return True if:
        - The *front part* of the two routes does not overlap
        - The remaining part may overlap, merge, cross, or touch

        front_length_ratio defines how much of the route front we inspect.
        """
        if not route1 or not route2:
            return False

        # Convert to LineString
        line1 = LineString([(wp.x, wp.y) for wp in route1])
        line2 = LineString([(wp.x, wp.y) for wp in route2])

        # Compute partial lengths
        L1 = line1.length
        L2 = line2.length

        # Inspect front X% portion (default 30%)
        cut1 = line1.interpolate(L1 * front_length_ratio, normalized=False)
        cut2 = line2.interpolate(L2 * front_length_ratio, normalized=False)

        # Extract the early parts
        front_part_1 = LineString([line1.coords[0], (cut1.x, cut1.y)])
        front_part_2 = LineString([line2.coords[0], (cut2.x, cut2.y)])

        # ❌ If the **front part** overlaps → invalid
        if front_part_1.overlaps(front_part_2):
            return False

        # ❌ If the front part touches/merges → also not allowed (no merging at start)
        if front_part_1.touches(front_part_2):
            return False

        # From here on, everything is acceptable
        # crossing / merging / overlapping later are ALL ok
        return True

    def _check_close_to_route(self, route1: List[Waypoint], route2: List[Waypoint], distance_threshold: float = 1.0) -> bool:
        if not route1 or not route2:
            return False

        # Convert waypoints to numpy arrays for efficient distance computation
        pts1 = np.array([[wp.x, wp.y] for wp in route1])
        pts2 = np.array([[wp.x, wp.y] for wp in route2])

        # Option 1: brute-force pairwise distance check (fine for short routes)
        for p1 in pts1:
            dists = np.linalg.norm(pts2 - p1, axis=1)
            if np.any(dists < distance_threshold):
                return True
            
        return False
    
    ####### Unit Mutation Operators #########            
    def follow_lane(
        self,
        wp,
        steps=10,
        step_dist=2.0,
        not_junction: bool = True
    ) -> List[carla.Waypoint]:
        """Go forward along the lane. If `not_junction=True`, avoid ending in a junction."""
        route = []
        for _ in range(steps):
            route.append(wp)
            next_wps = wp.next(step_dist)
            if not next_wps:
                break
            wp = random.choice(next_wps)

        if not_junction and route and route[-1].is_junction:
            for _ in range(5): 
                next_wps = wp.next(step_dist)
                if not next_wps:
                    break
                wp = random.choice(next_wps)
                route.append(wp)
                if not wp.is_junction:
                    break 
        return route

    ##########################################
    # Sampling Tools
    ##########################################    
    def sample_ego_route(
        self,
        conflict_vehicle_wps: List[Waypoint] = [],
        conflict_walker_wps: List[Waypoint] = [], # we can ignore the walker for ego route
        conflict_static_wps: List[Waypoint] = []
    ) -> Tuple:
        
        hop_resolution = 2.0
        max_tries = self.MAX_RETRY
        tries = 0
        
        while tries < max_tries:

            tries += 1

            # start wps
            if self.ego_space.ego_start_point is not None and self.ego_space.ego_end_point is not None:
                logger.info("Use the user fixed ego start and end points.")
                ego_start_wp = self._map.get_waypoint(
                    carla.Location(
                        x=self.ego_space.ego_start_point[0], 
                        y=self.ego_space.ego_start_point[1], 
                        z=self.ego_space.ego_start_point[2]),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                ego_end_wp = self._map.get_waypoint(
                    carla.Location(
                        x=self.ego_space.ego_end_point[0], 
                        y=self.ego_space.ego_end_point[1], 
                        z=self.ego_space.ego_end_point[2]),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
            else:
                if not self.map_region_space.cache_ego_route_list:
                    logger.error("No available ego routes in the buffer to sample from.")
                    raise ValueError("No available ego routes in the buffer to sample from.")
                
                sample_route = random.choice(self.map_region_space.cache_ego_route_list)
                ego_start_wp_lst = sample_route[0]
                ego_end_wp_lst = sample_route[1]
                
                ego_start_wp = self._map.get_waypoint(
                    carla.Location(
                        x=ego_start_wp_lst[0],
                        y=ego_start_wp_lst[1],
                        z=ego_start_wp_lst[2]
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                ego_end_wp = self._map.get_waypoint(
                    carla.Location(
                        x=ego_end_wp_lst[0],
                        y=ego_end_wp_lst[1],
                        z=ego_end_wp_lst[2]
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                        
            if ego_start_wp is None or ego_end_wp is None:
                logger.warning("Failed to get valid ego start or end waypoint, resample.")
                continue
            
            ego_start_wp_loc = [
                ego_start_wp.transform.location.x,
                ego_start_wp.transform.location.y,
                ego_start_wp.transform.location.z
            ]
            
            # check if already covered
            is_valid = True
            for covered_wp in conflict_vehicle_wps:
                covered_wp_carla = self._map.get_waypoint(
                    carla.Location(
                        x=covered_wp.x,
                        y=covered_wp.y,
                        z=covered_wp.z
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                
                covered_wp_loc = [
                    covered_wp.x,
                    covered_wp.y,
                    covered_wp.z
                ]
                dist = np.linalg.norm(np.array(ego_start_wp_loc) - np.array(covered_wp_loc))
                
                if ego_start_wp.lane_id == covered_wp_carla.lane_id and ego_start_wp.section_id == covered_wp_carla.section_id and ego_start_wp.road_id == covered_wp_carla.road_id:
                    if dist <= self.ego_space.dist2vehicle_same_lane: # min distance to existing start point
                        is_valid = False
                        break
                else:
                    if dist <= self.ego_space.dist2vehicle_other_lane: # min distance to existing start point
                        is_valid = False
                        break
                    
            if not is_valid:
                continue
            
            # check with existing walkers
            for covered_wp in conflict_walker_wps:
                dist = np.linalg.norm(
                    np.array(ego_start_wp_loc) - np.array([
                        covered_wp.x,
                        covered_wp.y,
                        covered_wp.z
                    ])
                )
                if dist <= self.ego_space.dist2pedestrian: # min distance to existing walker
                    is_valid = False
                    break
                
            if not is_valid:
                continue
            
            # check with existing static obstacles
            for covered_wp in conflict_static_wps:
                covered_wp_carla = self._map.get_waypoint(
                    carla.Location(
                        x=covered_wp.x,
                        y=covered_wp.y,
                        z=covered_wp.z
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                
                covered_wp_loc = [
                    covered_wp.x,
                    covered_wp.y,
                    covered_wp.z
                ]
                dist = np.linalg.norm(np.array(ego_start_wp_loc) - np.array(covered_wp_loc))
                
                if ego_start_wp.lane_id == covered_wp_carla.lane_id and ego_start_wp.section_id == covered_wp_carla.section_id and ego_start_wp.road_id == covered_wp_carla.road_id:                    
                    if dist <= self.ego_space.dist2static_same_lane: # min distance to existing static obstacle
                        is_valid = False
                        break
                else:
                    if dist <= self.ego_space.dist2static_other_lane: # min distance to existing static obstacle
                        is_valid = False
                        break
            
            if not is_valid:
                continue
            
            ego_route_pairs = self._interpolate_trajectory(
                [ego_start_wp.transform.location, ego_end_wp.transform.location], 
                hop_resolution=hop_resolution
            )
            ego_route_len = len(ego_route_pairs) * hop_resolution
            
            if self.ego_space.route_length[0] <= ego_route_len <= self.ego_space.route_length[1]:
                
                # process ego route
                ego_route_task = []
                ego_route = []
                ego_route_detail = []
                ego_estimated_route_length = 0.0
                ego_route_min_x = float('inf')
                ego_route_max_x = float('-inf')
                ego_route_min_y = float('inf')
                ego_route_max_y = float('-inf')
                prev_wp_loc = None
                prev_wp_info = None
                for wp_index, wp_pair in enumerate(ego_route_pairs):
            
                    wp_transform, wp_task = wp_pair
                    
                    wp = self._map.get_waypoint(
                        location=wp_transform.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving
                    )
                    wp_x = wp.transform.location.x
                    wp_y = wp.transform.location.y
                    if wp_x < ego_route_min_x:
                        ego_route_min_x = wp_x
                    if wp_x > ego_route_max_x:
                        ego_route_max_x = wp_x
                    if wp_y < ego_route_min_y:
                        ego_route_min_y = wp_y
                    if wp_y > ego_route_max_y:
                        ego_route_max_y = wp_y
                        
                    # update estimated route length
                    if prev_wp_loc is not None:
                        ego_estimated_route_length += wp.transform.location.distance(prev_wp_loc)
                        
                    prev_wp_loc = wp.transform.location
                    
                    ego_wp_cfg = Waypoint(
                        x=wp.transform.location.x,
                        y=wp.transform.location.y,
                        z=wp.transform.location.z,
                        pitch=wp.transform.rotation.pitch,
                        yaw=wp.transform.rotation.yaw,
                        roll=wp.transform.rotation.roll,
                        speed=0.0,
                        road_option=wp_task.name
                    )
                    
                    ego_route_detail.append(ego_wp_cfg)
                    
                    curr_wp_info = f"{wp.road_id}_{wp.lane_id}_{wp.section_id}"
                    if curr_wp_info != prev_wp_info or wp_index == 0 or wp_index == (len(ego_route_pairs) - 1):
                        ego_route.append(ego_wp_cfg)
                        prev_wp_info = curr_wp_info
                        
                    wp_task_name = wp_task.name
                    if len(ego_route_task) == 0 or wp_task_name != ego_route_task[-1]:
                        ego_route_task.append(wp_task_name)

                region_min_x = ego_route_min_x - self.ego_space.region_buffer
                region_max_x = ego_route_max_x + self.ego_space.region_buffer
                region_min_y = ego_route_min_y - self.ego_space.region_buffer
                region_max_y = ego_route_max_y + self.ego_space.region_buffer

                ego_limited_region = [region_min_x, region_max_x, region_min_y, region_max_y]
                
                # return
                if len(ego_route) < 2:
                    logger.warning("Sampled ego route has less than 2 waypoints after processing, resample.")
                    continue
                
                return ego_route, ego_route_detail, ego_route_task, ego_limited_region, ego_estimated_route_length
            
        return None, None, None, None, None
    
    def sample_npc_route(
        self,
        ego_routes: List[List[Waypoint]] = [],
        conflict_vehicle_wps: List[Waypoint] = [],
        conflict_walker_wps: List[Waypoint] = [],
        conflict_static_wps: List[Waypoint] = [],
        required_overlapping_with_ego: bool = False
    ) -> List[List[Waypoint]]:
        # I think we need filter out the potential waypoints
        
        max_tries = self.MAX_RETRY
        tries = 0
        step_dist = 2.0
        
        while tries < max_tries:
            tries += 1
            
            npc_routes_cache_pool = random.choice(list(self.map_region_space.map_npc_routes_cache.values()))
            # randomly select two points
            sampled_wps = random.choice(npc_routes_cache_pool)
            
            wp_start_list = sampled_wps[0]
            wp_end_list = sampled_wps[1]
            
            wp_start = self._map.get_waypoint(
                carla.Location(
                    x=wp_start_list[0],
                    y=wp_start_list[1],
                    z=wp_start_list[2]
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            wp_end = self._map.get_waypoint(
                carla.Location(
                    x=wp_end_list[0],
                    y=wp_end_list[1],
                    z=wp_end_list[2]
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            wp_start_loc = [
                wp_start.transform.location.x,
                wp_start.transform.location.y,
                wp_start.transform.location.z
            ]
            
            # check start point with existing vehicles
            is_valid = True
            for ex_wp in conflict_vehicle_wps: # config waypoint
                # NOTE: some big one may contain conflicts
                ex_wp_carla = self._map.get_waypoint(
                    carla.Location(
                        x=ex_wp.x,
                        y=ex_wp.y,
                        z=ex_wp.z
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                
                ex_wp_loc = [
                    ex_wp.x,
                    ex_wp.y, 
                    ex_wp.z
                ]
                dist = np.linalg.norm(np.array(wp_start_loc) - np.array(ex_wp_loc))
                if wp_start.lane_id == ex_wp_carla.lane_id and wp_start.section_id == ex_wp_carla.section_id and wp_start.road_id == ex_wp_carla.road_id:
                    if dist <= self.npc_vehicle_space.dist2vehicle_same_lane: # min distance to ego start point, in case spawn too close
                        is_valid = False
                        break
                else:
                    if dist <= self.npc_vehicle_space.dist2vehicle_other_lane: # min distance to ego start point, in case spawn too close
                        is_valid = False
                        break
                # NOTE: we do not consider end, as the npc can be destroyed when standing too long
                
            if not is_valid:
                continue
            
            # check start point with existing walkers
            for ex_wp in conflict_walker_wps:
                dist = np.linalg.norm(
                    np.array(wp_start_loc) - np.array([
                        ex_wp.x,
                        ex_wp.y,
                        ex_wp.z
                    ])
                )
                if dist <= self.npc_vehicle_space.dist2pedestrian: # min distance to ego start point, in case spawn too close
                    is_valid = False
                    break
                
            if not is_valid:
                continue
            
            # check start point with existing static obstacles
            for ex_wp in conflict_static_wps:
                ex_wp_carla = self._map.get_waypoint(
                    carla.Location(
                        x=ex_wp.x,
                        y=ex_wp.y,
                        z=ex_wp.z
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                
                ex_wp_loc = [
                    ex_wp.x,
                    ex_wp.y, 
                    ex_wp.z
                ]
                dist = np.linalg.norm(np.array(wp_start_loc) - np.array(ex_wp_loc))
                if wp_start.lane_id == ex_wp_carla.lane_id and wp_start.section_id == ex_wp_carla.section_id and wp_start.road_id == ex_wp_carla.road_id:                    
                    if dist <= self.npc_vehicle_space.dist2static_same_lane: # min distance to ego start point, in case spawn too close
                        is_valid = False
                        break
                else:
                    if dist <= self.npc_vehicle_space.dist2static_other_lane: # min distance to ego start point, in case spawn too close
                        is_valid = False
                        break
            
            if not is_valid:
                continue
            
            # extend the end route if necessary
            wp_end_loc = [
                wp_end.transform.location.x,
                wp_end.transform.location.y,
                wp_end.transform.location.z
            ]
            for ego_route in ego_routes: # config waypoint
                ego_wp = self._map.get_waypoint(
                    carla.Location(
                        x=ego_route[-1].x,
                        y=ego_route[-1].y,
                        z=ego_route[-1].z
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                
                ego_loc = [
                    ego_wp.transform.location.x,
                    ego_wp.transform.location.y, 
                    ego_wp.transform.location.z
                ]
                
                dist = np.linalg.norm(np.array(wp_end_loc) - np.array(ego_loc))
                while wp_end.lane_id == ego_wp.lane_id and wp_end.section_id == ego_wp.section_id and wp_end.road_id == ego_wp.road_id and dist <= 8.0: # min distance to ego start point, in case spawn too close
                    # extend the end point
                    exteded_route_wps = self.follow_lane(
                        wp_end,
                        steps=5,
                        step_dist=step_dist
                    )
                    wp_end = exteded_route_wps[-1]
                    wp_end_loc = [
                        wp_end.transform.location.x,
                        wp_end.transform.location.y,
                        wp_end.transform.location.z
                    ]
                    dist = np.linalg.norm(np.array(wp_end_loc) - np.array(ego_loc))
        
            route_opt = self._interpolate_trajectory(
                [wp_start.transform.location, wp_end.transform.location], 
                hop_resolution=step_dist
            )
                        
            final_npc_route = list()
            target_speed = random.uniform(self.npc_vehicle_space.speed[0], self.npc_vehicle_space.speed[1])
            for _, wp_pair in enumerate(route_opt):
                final_npc_route.append(
                    Waypoint(
                        x=wp_pair[0].location.x,
                        y=wp_pair[0].location.y,
                        z=wp_pair[0].location.z,
                        pitch=wp_pair[0].rotation.pitch,
                        yaw=wp_pair[0].rotation.yaw,
                        roll=wp_pair[0].rotation.roll,
                        speed=float(target_speed),
                        road_option=wp_pair[1].name
                    )
                )
                            
            if len(final_npc_route) < 2:
                continue
            
            # check with ego routes
            if required_overlapping_with_ego:
                is_overlapping = False
                for ego_route in ego_routes:
                    if self._check_route_overlapping(final_npc_route, ego_route):
                        is_overlapping = True
                        break
                if not is_overlapping:
                    continue
            
            return final_npc_route
        
        return None
    
    def sample_walker_route(
        self,
        ego_routes: List[List[Waypoint]] = [],
        conflict_vehicle_wps: List[Waypoint] = [],
        conflict_walker_wps: List[Waypoint] = [],
        conflict_static_wps: List[Waypoint] = [],
        required_overlapping_with_ego: bool = False
    ) -> List[List[Waypoint]]:
        
        max_tries = self.MAX_RETRY
        tries = 0
                
        while tries < max_tries:
            tries += 1
            
            sample_region = [
                self.map_region_space.region_x_min - abs(self.ego_space.region_buffer),
                self.map_region_space.region_x_max + abs(self.ego_space.region_buffer),
                self.map_region_space.region_y_min - abs(self.ego_space.region_buffer),
                self.map_region_space.region_y_max + abs(self.ego_space.region_buffer)
            ]
                
            # sample region firstly
            sample_region_x = [sample_region[0], sample_region[1]]
            sample_region_y = [sample_region[2], sample_region[3]]
            
            # start point - NOTE: this is TOO SPARSE!!!!!
            x1 = random.uniform(sample_region_x[0], sample_region_x[1])
            y1 = random.uniform(sample_region_y[0], sample_region_y[1])
            loc1 = carla.Location(x=x1, y=y1, z=0.1)

            wp_start = self._map.get_waypoint(
                loc1,
                project_to_road=True,
                lane_type=(carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Parking),
            )
            
            if wp_start is None:
                continue

            if not (sample_region_x[0] <= wp_start.transform.location.x <= sample_region_x[1] and \
                sample_region_y[0] <= wp_start.transform.location.y <= sample_region_y[1]):
                continue
            
            # distance
            is_valid = True
            for _ex_wp in conflict_walker_wps:
                dist = np.linalg.norm(
                    np.array([
                        wp_start.transform.location.x,
                        wp_start.transform.location.y,
                        wp_start.transform.location.z
                    ]) - np.array([
                        _ex_wp.x,
                        _ex_wp.y,
                        _ex_wp.z
                    ])
                )
                if dist <= self.npc_pedestrian_space.dist2pedestrian: # min distance to existing walker
                    is_valid = False
                    break
                
            if not is_valid:
                continue
            
            # distance to existing vehicles
            for ex_wp in conflict_vehicle_wps: # config waypoint                
                ex_wp_loc = [
                    ex_wp.x,
                    ex_wp.y, 
                    ex_wp.z
                ]
                dist = np.linalg.norm(np.array([
                    wp_start.transform.location.x,
                    wp_start.transform.location.y,
                    wp_start.transform.location.z
                ]) - np.array(ex_wp_loc))
                if dist <= self.npc_pedestrian_space.dist2vehicle: # min distance to existing vehicle
                    is_valid = False
                    break
                
            if not is_valid:
                continue
            
            # distance to existing statics
            for ex_wp in conflict_static_wps: # config waypoint
                ex_wp_loc = [
                    ex_wp.x,
                    ex_wp.y, 
                    ex_wp.z
                ]
                dist = np.linalg.norm(np.array([
                    wp_start.transform.location.x,
                    wp_start.transform.location.y,
                    wp_start.transform.location.z
                ]) - np.array(ex_wp_loc))
                if dist <= self.npc_pedestrian_space.dist2static: # min distance to existing static
                    is_valid = False
                    break
                
            if not is_valid:
                continue
                            
            # end point
            x2 = random.uniform(sample_region_x[0], sample_region_x[1])
            y2 = random.uniform(sample_region_y[0], sample_region_y[1])
            loc2 = carla.Location(x=x2, y=y2, z=0.1)

            wp_end = self._map.get_waypoint(
                loc2,
                project_to_road=True,
                lane_type=(carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Parking),
            )
            
            if wp_end is None:
                continue
            
            if not (sample_region_x[0] <= wp_end.transform.location.x <= sample_region_x[1] and \
                sample_region_y[0] <= wp_end.transform.location.y <= sample_region_y[1]):
                continue

            final_route = list()
            for wp in [wp_start, wp_end]:
                final_route.append(
                    Waypoint(
                        x=wp.transform.location.x,
                        y=wp.transform.location.y,
                        z=wp.transform.location.z,
                        pitch=wp.transform.rotation.pitch,
                        yaw=wp.transform.rotation.yaw,
                        roll=wp.transform.rotation.roll,
                        speed=0.0,
                        road_option="NONE"
                    )
                )
                
            # check with ego routes
            if required_overlapping_with_ego:
                is_overlapping = False
                for ego_route in ego_routes:
                    if self._check_route_overlapping(final_route, ego_route):
                        is_overlapping = True
                        break
                if not is_overlapping:
                    continue
            
            return final_route
    
        return None
    
    
    def _check_static_location_valid(
        self,
        wp: carla.Waypoint,
        ego_routes: List[List[Waypoint]] = [],
        conflict_vehicle_wps : List[Waypoint] = [],
        conflict_walker_wps: List[Waypoint] = [],
        conflict_static_wps: List[Waypoint] = []
    ) -> bool:
        wp_loc = [
            wp.transform.location.x,
            wp.transform.location.y,
            wp.transform.location.z
        ]
        is_valid = True
        for ex_wp in conflict_vehicle_wps: # config waypoint
            ex_wp_carla = self._map.get_waypoint(
                carla.Location(
                    x=ex_wp.x,
                    y=ex_wp.y,
                    z=ex_wp.z
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            ex_wp_loc = [
                ex_wp.x,
                ex_wp.y, 
                ex_wp.z
            ]
            dist = np.linalg.norm(np.array(wp_loc) - np.array(ex_wp_loc))
            
            if wp.lane_id == ex_wp_carla.lane_id and wp.section_id == ex_wp_carla.section_id and wp.road_id == ex_wp_carla.road_id:
                if dist <= self.npc_static_space.dist2vehicle_same_lane: # min distance to existing vehicle
                    is_valid = False
                    break
            else:
                if dist <= self.npc_static_space.dist2vehicle_other_lane: # min distance to existing vehicle
                    is_valid = False
                    break
                
        if not is_valid:
            return False
        
        for ex_wp in conflict_walker_wps: # config waypoint
            dist = np.linalg.norm(
                np.array(wp_loc) - np.array([
                    ex_wp.x,
                    ex_wp.y,
                    ex_wp.z
                ])
            )
            if dist <= self.npc_static_space.dist2pedestrian: # min distance to existing walker
                is_valid = False
                break
            
        if not is_valid:
            return False
            
        for ex_wp in conflict_static_wps: # config waypoint
            ex_wp_carla = self._map.get_waypoint(
                carla.Location(
                    x=ex_wp.x,
                    y=ex_wp.y,
                    z=ex_wp.z
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            ex_wp_loc = [
                ex_wp.x,
                ex_wp.y, 
                ex_wp.z
            ]
            dist = np.linalg.norm(np.array(wp_loc) - np.array(ex_wp_loc))
            
            if wp.lane_id == ex_wp_carla.lane_id and wp.section_id == ex_wp_carla.section_id and wp.road_id == ex_wp_carla.road_id:                    
                if dist <= self.npc_static_space.dist2static_same_lane: # min distance to existing vehicle
                    is_valid = False
                    break
            else:
                if dist <= self.npc_static_space.dist2static_other_lane: # min distance to existing vehicle
                    is_valid = False
                    break
                
        if not is_valid:
            return False
        
        for ego_route in ego_routes:
            ego_start_wp_cfg = ego_route[0] # config waypoint
            ego_start_wp = self._map.get_waypoint(
                carla.Location(
                    x=ego_start_wp_cfg.x,
                    y=ego_start_wp_cfg.y,
                    z=ego_start_wp_cfg.z
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            ego_end_wp_cfg = ego_route[-1]
            ego_end_wp = self._map.get_waypoint(
                carla.Location(
                    x=ego_end_wp_cfg.x,
                    y=ego_end_wp_cfg.y,
                    z=ego_end_wp_cfg.z
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            
            ego_start_loc = [
                ego_start_wp.transform.location.x, 
                ego_start_wp.transform.location.y, 
                ego_start_wp.transform.location.z
            ]
            dist = np.linalg.norm(np.array(wp_loc) - np.array(ego_start_loc))
            
            if wp.lane_id == ego_start_wp.lane_id and wp.section_id == ego_start_wp.section_id and wp.road_id == ego_start_wp.road_id:
                if dist <= 8.0: # min distance to ego start point, in case spawn too close
                    is_valid = False
                    break
            
            # Do not block ego end point
            ego_end_loc = [
                ego_end_wp.transform.location.x, 
                ego_end_wp.transform.location.y, 
                ego_end_wp.transform.location.z
            ]
            dist = np.linalg.norm(np.array(wp_loc) - np.array(ego_end_loc))
            if wp.lane_id == ego_end_wp.lane_id and wp.section_id == ego_end_wp.section_id and wp.road_id == ego_end_wp.road_id:
                if dist <= 8.0: # min distance to ego end point, in case spawn too close
                    is_valid = False
                    break
        
        return is_valid
    
    def sample_static_location(
        self,
        ego_routes: List[List[Waypoint]] = [],
        conflict_vehicle_wps : List[Waypoint] = [],
        conflict_walker_wps: List[Waypoint] = [],
        conflict_static_wps: List[Waypoint] = [],
        required_overlapping_with_ego: bool = False,
        on_ego_route: bool = False
    ) -> List[Waypoint]:
        
        max_tries = self.MAX_RETRY
        tries = 0
        
        while tries < max_tries:
            tries += 1
            
            # sample region firstly
            sample_region = [
                self.map_region_space.region_x_min - abs(self.ego_space.region_buffer),
                self.map_region_space.region_x_max + abs(self.ego_space.region_buffer),
                self.map_region_space.region_y_min - abs(self.ego_space.region_buffer),
                self.map_region_space.region_y_max + abs(self.ego_space.region_buffer)
            ]
            
            # sample region firstly
            sample_region_x = [sample_region[0], sample_region[1]]
            sample_region_y = [sample_region[2], sample_region[3]]
            
            # location point
            x = random.uniform(sample_region_x[0], sample_region_x[1])
            y = random.uniform(sample_region_y[0], sample_region_y[1])
            loc = carla.Location(x=x, y=y, z=0.1)

            # NOTE: why we only use driving lane? -> extend to sidewalk etc
            wp = self._map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=(carla.LaneType.Shoulder | carla.LaneType.Parking | carla.LaneType.Driving),
            )

            if wp is None or wp.is_junction:
                continue
            
            # check init distance
            wp_loc = [
                wp.transform.location.x, 
                wp.transform.location.y,
                wp.transform.location.z
            ]
            is_valid = self._check_static_location_valid(
                wp,
                ego_routes=ego_routes,
                conflict_vehicle_wps=conflict_vehicle_wps,
                conflict_walker_wps=conflict_walker_wps,
                conflict_static_wps=conflict_static_wps
            )
            
            if not is_valid:
                continue
            
            static_wp_cfg = Waypoint(
                x=wp.transform.location.x,
                y=wp.transform.location.y,
                z=wp.transform.location.z,
                pitch=wp.transform.rotation.pitch,
                yaw=wp.transform.rotation.yaw,
                roll=wp.transform.rotation.roll,
                speed=0.0,
                road_option="NONE"
            )
            
            # # check with ego routes, we remove this, leave it as random on all space
            if on_ego_route:
                is_on_ego_route = False
                for ego_route in ego_routes:
                    if self._check_close_to_route([static_wp_cfg], ego_route, distance_threshold=2.0):
                        is_on_ego_route = True
                        break
                if not is_on_ego_route:
                    continue
            return static_wp_cfg
        return None
    
    def _change_to_weather_constraints(self, weather: WeatherConfig) -> WeatherConfig:
        s = self.weather_space
        
        wetness = weather.wetness
        precipitation = weather.precipitation
        precipitation_deposits = weather.precipitation_deposits
        cloudiness = weather.cloudiness
        wind_intensity = weather.wind_intensity
        sun_altitude_angle = weather.sun_altitude_angle
        # ===================================================
        #                APPLY REALISM CONSTRAINTS
        # ===================================================

        # (C1) Rain → Wetness
        wetness = max(wetness, precipitation * s.ratio_wetness_precipitation)

        # (C2) Rain → Deposits
        precipitation_deposits = max(precipitation_deposits, precipitation * s.ratio_precipitation_deposits_precipitation)

        # (C4) Thick clouds block sunlight
        if cloudiness > s.cloudiness_sun_threshold:
            sun_altitude_angle = min(sun_altitude_angle, s.cloudiness_sun_altitude_angle)

        # (C5) Strong wind typically implies more clouds
        if wind_intensity > s.wind_intensity_threshold:
            cloudiness = max(cloudiness, s.cloudiness_intensity)
            
        weather.wetness = round(wetness, 4)
        weather.precipitation = round(precipitation, 4)
        weather.precipitation_deposits = round(precipitation_deposits, 4)
        weather.cloudiness = round(cloudiness, 4)  
        weather.wind_intensity = round(wind_intensity, 4)
        weather.sun_altitude_angle = round(sun_altitude_angle, 4)
        return weather
    
    def sample_weather(self) -> WeatherConfig:
        weather_pattern = "sampled"
        rnd = random.uniform
        s = self.weather_space

        # --- raw random sample ---
        cloudiness = rnd(*s.cloudiness)
        precipitation = rnd(*s.precipitation)
        precipitation_deposits = rnd(*s.precipitation_deposits)
        wind_intensity = rnd(*s.wind_intensity)
        sun_azimuth_angle = rnd(*s.sun_azimuth_angle)
        sun_altitude_angle = rnd(*s.sun_altitude_angle)
        fog_density = rnd(*s.fog_density)
        fog_distance = rnd(*s.fog_distance)
        fog_falloff = rnd(*s.fog_falloff)
        wetness = rnd(*s.wetness)


        # ===================================================
        # Build WeatherConfig
        # ===================================================
        sampled_weather = WeatherConfig(
            pattern=weather_pattern,
            cloudiness=round(cloudiness, 4),
            precipitation=round(precipitation, 4),
            precipitation_deposits=round(precipitation_deposits, 4),
            wind_intensity=round(wind_intensity, 4),
            sun_azimuth_angle=round(sun_azimuth_angle, 4),
            sun_altitude_angle=round(sun_altitude_angle, 4),
            fog_density=round(fog_density, 4),
            fog_distance=round(fog_distance, 4),
            fog_falloff=round(fog_falloff, 4),
            wetness=round(wetness, 4)
        )
        return self._change_to_weather_constraints(sampled_weather)

    def sample_traffic_light(self) -> TrafficLightBehaviorConfig:
        traffic_light_pattern = random.choice(self.traffic_light_space.pattern)
        green_time = random.uniform(self.traffic_light_space.green_duration[0], self.traffic_light_space.green_duration[1])
        yellow_time = random.uniform(self.traffic_light_space.yellow_duration[0], self.traffic_light_space.yellow_duration[1])
        red_time = random.uniform(self.traffic_light_space.red_duration[0], self.traffic_light_space.red_duration[1])
        
        traffic_light = TrafficLightBehaviorConfig(
            pattern=traffic_light_pattern,
            yellow_time=yellow_time,
            red_time=red_time,
            green_time=green_time
        )
        return traffic_light
    
    ####### Intergarted Scenario Sample #########
    def _get_region_driving_wps_routes(
        self,
        region_x_min: float,
        region_x_max: float,
        region_y_min: float,
        region_y_max: float,
    ):
        """Return driving waypoints + valid ego/npc route endpoints within the region."""
        # --------------------------
        # Helper functions
        # --------------------------
        def in_region_xy(x, y):
            return (region_x_min <= x <= region_x_max) and (region_y_min <= y <= region_y_max)

        def wp_in_region(wp: carla.Waypoint):
            loc = wp.transform.location
            return in_region_xy(loc.x, loc.y)

        # --------------------------
        # 1. Filter driving waypoints inside region
        # --------------------------
        all_wps = self._map.generate_waypoints(distance=2.0)

        cached_wps = [
            wp for wp in all_wps
            if (not wp.is_junction)
            and (wp.lane_type == carla.LaneType.Driving)
            and wp_in_region(wp)
        ]

        logger.info(f"[Region] {len(cached_wps)} driving waypoints in region.")

        # debug fast path: randomize start order so the early-stop below collects
        # a varied handful of routes instead of whatever sits in one corner
        if self.debug_max_routes is not None:
            random.shuffle(cached_wps)

        # extract bounds only once
        ego_min, ego_max = self.ego_space.route_length
        npc_min, npc_max = self.npc_vehicle_space.route_length
        sample_res = 2.0
        
        # start and end must be LANEFOLLOW
        
        ENDPOINT_TASK = 'LANEFOLLOW'
        
        ego_task_count_dict = {}
        npc_task_count_dict = {}
        # --------------------------
        # 2. Pairwise route generation with pruning
        # --------------------------
        for wp_i, wp_start in tqdm(
            enumerate(cached_wps),
            total=len(cached_wps),
            desc="Building region routes",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        ):
            # debug fast path: stop once we have enough routes (skip full O(n^2))
            if self.debug_max_routes is not None:
                ego_n = sum(len(v) for v in ego_task_count_dict.values())
                npc_n = sum(len(v) for v in npc_task_count_dict.values())
                if ego_n >= self.debug_max_routes and npc_n >= self.debug_max_routes:
                    logger.info(f"[Region][debug] early stop after {wp_i} starts: "
                                f"{ego_n} ego / {npc_n} npc routes collected")
                    break

            start_loc = wp_start.transform.location

            for wp_end in cached_wps:
                if wp_end is wp_start:
                    continue  # skip same waypoint

                end_loc = wp_end.transform.location

                # --- (A) Cheap distance pruning ---
                dx = start_loc.x - end_loc.x
                dy = start_loc.y - end_loc.y
                approx_dist = abs(dx) + abs(dy)

                # below min → route length不可能够
                if approx_dist < ego_min * 0.5:
                    continue

                # --- (B) Interpolate trajectory (real route building) ---
                route = self._interpolate_trajectory(
                    [start_loc, end_loc],
                    hop_resolution=sample_res
                )

                # --- (C) check all points inside region ---
                valid = True
                route_len = 0.0
                route_task = []
                prev_wp_loc = None
                for wp_transform, wp_task in route:
                    loc = wp_transform.location
                    if not in_region_xy(loc.x, loc.y):
                        valid = False
                        break
                    
                    wp_task_name = wp_task.name
                    if len(route_task) == 0:
                        if wp_task_name != ENDPOINT_TASK:
                            valid = False
                            break
                        else:
                            route_task.append(wp_task_name)
                            
                    elif wp_task_name != route_task[-1]:
                        route_task.append(wp_task_name)
                    
                    if prev_wp_loc is None:
                        prev_wp_loc = wp_transform.location
                        continue
                    else:
                        route_len += wp_transform.location.distance(prev_wp_loc)
                        prev_wp_loc = wp_transform.location
                    
                if len(route_task) == 0 or route_task[-1] != ENDPOINT_TASK:
                    valid = False
                
                if not valid:
                    continue
                
                # --- (D) Save valid start→end pair ---
                if ego_min <= route_len <= ego_max:
                    
                    ego_route = None
                    if self.ego_space.corssing_junction:
                        # check if crossing junction
                        crosses_junction = False
                        for task in route_task:
                            if task != "LANEFOLLOW":
                                crosses_junction = True
                                break
                            
                        if crosses_junction:
                            ego_route = [wp_start, wp_end]
                    else:
                        ego_route = [wp_start, wp_end]

                    if ego_route is not None:
                        route_task_str = "-".join(route_task)
                        if route_task_str not in ego_task_count_dict:
                            ego_task_count_dict[route_task_str] = []
                        ego_task_count_dict[route_task_str].append({
                            'route': ego_route,
                            'length': route_len
                        })
                        
                if npc_min <= route_len <= npc_max:
                    npc_route = [wp_start, wp_end]
                    route_task_str = "-".join(route_task)
                    if route_task_str not in npc_task_count_dict:
                        npc_task_count_dict[route_task_str] = []
                    npc_task_count_dict[route_task_str].append({
                        'route': npc_route,
                        'length': route_len
                    })
                    
        # calculate cache
        ego_route_cache = {}       
        for task_str, route_list in ego_task_count_dict.items():

            # route_list structure: [{'route': ..., 'length': ...}, ...]

            if not route_list:
                continue

            # 1. Sort by length desc
            sorted_routes = sorted(route_list, key=lambda x: x['length'], reverse=True)

            # 2. Build picks (avoid repetition)
            picks = []

            # ---- Top-1 ----
            picks.append(sorted_routes[0])

            n = len(sorted_routes)

            # ---- Top-half ----
            if n >= 2:
                half_idx = n // 2
                candidate = sorted_routes[half_idx]
                if candidate not in picks:
                    picks.append(candidate)

            # ---- Min ----
            if n >= 3:
                candidate = sorted_routes[-1]
                if candidate not in picks:
                    picks.append(candidate)

            # if still not enough (e.g., only 1 unique), fill remaining from sorted list
            i = 1
            while len(picks) < 3 and i < n:
                if sorted_routes[i] not in picks:
                    picks.append(sorted_routes[i])
                i += 1

            if task_str not in ego_route_cache:
                ego_route_cache[task_str] = []
            
            for p in picks:
                save_format = [
                    [p['route'][0].transform.location.x, p['route'][0].transform.location.y, p['route'][0].transform.location.z],
                    [p['route'][-1].transform.location.x, p['route'][-1].transform.location.y, p['route'][-1].transform.location.z]
                ]
                ego_route_cache[task_str].append(save_format)

        npc_route_cache = {}
        for task_str, route_list in npc_task_count_dict.items():

            if not route_list:
                continue

            # 1. Sort by length desc
            sorted_routes = sorted(route_list, key=lambda x: x['length'], reverse=True)
            n = len(sorted_routes)

            # 2. Build unique picks list
            picks = []

            # Helper: add candidate if not duplicated
            def add_candidate(idx):
                if 0 <= idx < n:
                    cand = sorted_routes[idx]
                    if cand not in picks:
                        picks.append(cand)

            # ---- Max (0%) ----
            add_candidate(0)

            # ---- 1/4 position ----
            idx_q1 = n // 4
            add_candidate(idx_q1)

            # ---- Median (1/2) ----
            idx_q2 = n // 2
            add_candidate(idx_q2)

            # ---- 3/4 position ----
            idx_q3 = (3 * n) // 4
            add_candidate(idx_q3)

            # ---- Min (last) ----
            add_candidate(n - 1)

            # ---- Fallback: if <5 unique picks, fill from sorted_routes ----
            i = 1
            while len(picks) < 5 and i < n:
                if sorted_routes[i] not in picks:
                    picks.append(sorted_routes[i])
                i += 1

            # extend npc route cache
            if task_str not in npc_route_cache:
                npc_route_cache[task_str] = []
                
            for p in picks:
                save_format = [
                    [p['route'][0].transform.location.x, p['route'][0].transform.location.y, p['route'][0].transform.location.z],
                    [p['route'][-1].transform.location.x, p['route'][-1].transform.location.y, p['route'][-1].transform.location.z]
                ]
                npc_route_cache[task_str].append(save_format)
                
        logger.info(
            f"[Region] valid: ego routes={len(ego_route_cache)}, npc routes={len(npc_route_cache)}"
        )
        
        # convert format
        cached_wps_floats = []
        for wp in cached_wps:
            cached_wps_floats.append(
                [wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]
            )

        return cached_wps_floats, ego_route_cache, npc_route_cache

    def sample(
        self,
        ctn_operator: CtnSimOperator,
        ego_entry_point,
        ego_config_path,
        require_setup: bool = True
    ) -> Optional[ScenarioConfig]:
        """Sample a scenario. For multi-ADS support, ego_entry_point and
        ego_config_path can be either a single string (applied to all egos)
        or a list of strings (one per ego, cycled if shorter than ego_num)."""
        
        # setup CTN operator
        self.ctn_operator = ctn_operator
        if not self.ctn_operator.is_running:
            self.ctn_operator.start()
        
        # load town & info
        if require_setup:
            try:
                # reset every time
                self._setup(self.town)
            except Exception as e:
                self.ctn_operator.start()
                try:
                    self._setup(self.town)
                except Exception as e:
                    logger.error(traceback.print_exc())
                    return None
        
        # sample ego configs
        # NOTE:!!! the sampling is sequence task, ego -> npc vehicles -> npc walkers -> npc statics
        ego_num = random.randint(self.ego_space.num[0], self.ego_space.num[1])
        ego_vehicles = []
        ego_routes_details = []
        existing_vehicle_wps = []
        existing_walker_wps = []
        existing_static_wps = []
        
        start_time = time.time()
        # ego parameters
        # 1. trigger time
        # 2. model (fixed)
        # 3. route
        while len(ego_vehicles) < ego_num:
            if time.time() - start_time > 120:
                if len(ego_vehicles) == 0:
                    logger.error("Failed to sample any ego vehicles within time limit. Please check the map.")
                    raise RuntimeError("Failed to sample any ego vehicles within time limit.")
                else:
                    break
                
            ego_route, ego_route_detail, ego_route_task, ego_region, ego_routes_length = self.sample_ego_route(
                conflict_vehicle_wps=existing_vehicle_wps,
                conflict_walker_wps=existing_walker_wps,
                conflict_static_wps=existing_static_wps
            )
            
            if ego_route is None:
                continue
            
            ego_trigger_time = float(random.uniform(self.ego_space.trigger_time[0], self.ego_space.trigger_time[1]))
            ego_model = random.choice(self.ego_space.models)

            # Multi-ADS: pick entry_point/config_path for this ego index
            idx = len(ego_vehicles)
            if isinstance(ego_entry_point, list):
                _ep = ego_entry_point[idx % len(ego_entry_point)]
            else:
                _ep = ego_entry_point
            if isinstance(ego_config_path, list):
                _cp = ego_config_path[idx % len(ego_config_path)]
            else:
                _cp = ego_config_path

            ego_config = EgoConfig(
                id=f"ego_{idx}",
                model=ego_model,
                rolename="hero",
                color=None,
                category="vehicle",
                route=ego_route,
                trigger_time=ego_trigger_time,
                route_func_coverage=ego_route_task,
                route_estimated_length=ego_routes_length,
                entry_point=_ep,
                config_path=_cp,
            )
            
            # checker
            checking_actor = [
                {
                    'model': ego_config.model,
                    'x': ego_config.route[0].x,
                    'y': ego_config.route[0].y,
                    'z': ego_config.route[0].z,
                    'pitch': ego_config.route[0].pitch,
                    'yaw': ego_config.route[0].yaw,
                    'roll': ego_config.route[0].roll
                }
            ]
            check_pass = self._check_ego(
                agents_info=checking_actor
            )
            
            if check_pass:
                ego_vehicles.append(ego_config)
                existing_vehicle_wps.append(ego_config.route[0])
                ego_routes_details.append(ego_route_detail)
                
        logger.debug(f"Target ego number {ego_num} , actual added {len(ego_vehicles)}.")
        
        # add map region filed
        map_region_config = MapConfig(
            town=self.town,
            region=[
                self.map_region_space.region_x_min,
                self.map_region_space.region_x_max,
                self.map_region_space.region_y_min,
                self.map_region_space.region_y_max,
            ]
        )
            
        # sample NPC vehicles
        # 1. num
        # 2. route
        # 3. model
        # 4. trigger time
        npc_vehicles = []      
        npc_vehicle_num = random.randint(self.npc_vehicle_space.num[0], self.npc_vehicle_space.num[1])
        for i in range(npc_vehicle_num):            
            npc_route = self.sample_npc_route(
                ego_routes=ego_routes_details,
                conflict_vehicle_wps=existing_vehicle_wps,
                conflict_walker_wps=existing_walker_wps,
                conflict_static_wps=existing_static_wps,
                required_overlapping_with_ego=False
            )
            if npc_route is None:
                continue
            
            npc_model = random.choice(self.npc_vehicle_space.models)
            npc_trigger_time = float(random.uniform(self.npc_vehicle_space.trigger_time[0], self.npc_vehicle_space.trigger_time[1]))
            npc_config = WaypointVehicleConfig(
                id=f"npc_vehicle_{i}",
                model=npc_model,
                rolename="npc_vehicle",
                color=None,
                category="vehicle",
                route=npc_route,
                trigger_time=npc_trigger_time
            )
                        
            # TODO: do we need put checker? I think no.
            npc_vehicles.append(npc_config)
            existing_vehicle_wps.append(npc_config.route[0])
                
        logger.debug(f"Sampled {npc_vehicle_num} NPC vehicles, actual added {len(npc_vehicles)}.")
            
        # sample NPC walkers
        # 1. num
        # 2. route
        # 3. model
        # 4. trigger time
        npc_walkers = [] # to be filled
        npc_walker_num = random.randint(self.npc_pedestrian_space.num[0], self.npc_pedestrian_space.num[1])
        for i in range(npc_walker_num):
            
            npc_route = self.sample_walker_route(
                ego_routes=ego_routes_details,
                conflict_vehicle_wps=existing_vehicle_wps,
                conflict_walker_wps=existing_walker_wps,
                conflict_static_wps=existing_static_wps,
                required_overlapping_with_ego=False
            )
            if npc_route is None:
                continue
            
            npc_model = random.choice(self.npc_pedestrian_space.models)
            npc_trigger_time = float(random.uniform(self.npc_pedestrian_space.trigger_time[0], self.npc_pedestrian_space.trigger_time[1]))
            npc_config = AIWalkerConfig(
                id=f"npc_walker_{i}",
                model=npc_model,
                rolename="npc_walker",
                category="walker",
                route=npc_route,
                trigger_time=npc_trigger_time,
            )
            
            npc_walkers.append(npc_config)
            existing_walker_wps.append(npc_config.route[0])
        
        logger.debug(f"Sampled {npc_walker_num} NPC walkers, actual added {len(npc_walkers)}.")
        
        # sample NPC statics
        # 1. num
        # 2. location
        # 3. model
        npc_statics = [] # to be filled     
        npc_num = random.randint(self.npc_static_space.num[0], self.npc_static_space.num[1]) 
        for i in range(npc_num):
            static_location = self.sample_static_location(
                ego_routes=ego_routes_details,
                conflict_vehicle_wps=existing_vehicle_wps,
                conflict_walker_wps=existing_walker_wps,
                conflict_static_wps=existing_static_wps,
                required_overlapping_with_ego=False,
                on_ego_route=False
            )
            if static_location is None:
                continue
            
            static_model = random.choice(self.npc_static_space.models)
            static_config = StaticObstacleConfig(
                id=f"npc_static_{i}",
                model=static_model,
                rolename="npc_static",
                category="static",
                location=static_location,
            )
            npc_statics.append(static_config)
            existing_static_wps.append(static_location)
                
        logger.debug(f"Sampled {npc_num} NPC statics, actual added {len(npc_statics)}.")
    
        # sample weather
        # 1. all parameters
        weather = self.sample_weather()
        logger.debug(f"Sampled weather: {weather}")
        
        # sample traffic light
        # 1. pattern
        # 2. green time
        # 3. yellow time
        # 4. red time
        traffic_light = self.sample_traffic_light()
        logger.debug(f"Sampled traffic light behavior: {traffic_light}")
        
        sampled_scenario = ScenarioConfig(
            id=f"scenario_sampled",
            scenario_type="open_scenario",
            ego_vehicles=ego_vehicles,
            npc_vehicles=npc_vehicles,
            npc_walkers=npc_walkers,
            npc_statics=npc_statics,
            weather=weather,
            traffic_light=traffic_light,
            map_region=map_region_config
        )
        
        self._cleanup()
        return sampled_scenario