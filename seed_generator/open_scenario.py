from __future__ import annotations

import os
import time
import carla
import json
import random
import traceback

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version # Python 2 fallback
    
try:
    # requires Python 3.8+
    from importlib.metadata import metadata
    def get_carla_version():
        return Version(metadata("carla")["Version"])
except ModuleNotFoundError:
    # backport checking for older Python versions; module is deprecated
    import pkg_resources
    def get_carla_version():
        return Version(pkg_resources.get_distribution("carla").version)

OLD_CARLA_VERSION = Version("0.9.12")

from typing import List
from loguru import logger

from scenario_corpus.open_scenario.config import ScenarioConfig, Waypoint, MapConfig, EgoConfig
from agent_corpus.atomic.route_manipulation import GlobalRoutePlanner
from scenario_runner.ctn_operator import CtnSimOperator
from scenario_runner.ctn_manager import CtnSimOperator

"""
Only provide initial map region config and ego config
"""

class ScenarioSpace:
    """Define the mutation space for OpenSCENARIO scenarios."""
    # map
    # TOWN_LIST = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
    
    MAP_BUFFER = 20.0 # meter, for map region extension
    
    
class ScenarioGenerator:
        
    def __init__(
        self,
        ctn_operator: CtnSimOperator = None,
        town: str = 'Town03'
    ):
        # define some global parameters
        self.ctn_operator = ctn_operator # should be set outside
        self.town = town
        self._world = None
        self._client = None
        self._map = None
        
        self.start_end_coverage = [] # list of float
        
        self._setup(town)
        
    ####### Basic Tools #######
    def _setup(self, town):
        self._world = self.ctn_operator.get_world()
        self._client = self.ctn_operator.client
        self._map = self._world.get_map()
        
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
                
    def _cleanup(self):
        # TODO: clean all actors
        # Reset to asynchronous mode
        settings = self._world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self._world.apply_settings(settings)
        
    def _put_actor(self, actor_model, x, y, z, pitch, yaw, roll):

        bp_filter = actor_model
        blueprint = random.choice(self._world.get_blueprint_library().filter(bp_filter))

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

    def _location_check(
        self, 
        agents_info
    ):
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

            carla_actor.set_simulate_physics(enabled=True)
            added_actors.append(carla_actor)
        clean(added_actors)
        return True

    def _interpolate_trajectory(self, waypoints_trajectory, hop_resolution=2.0):
        """
        Input is carla.location
        """
        grp = GlobalRoutePlanner(self._map, hop_resolution)
        route = []

        for i in range(len(waypoints_trajectory) - 1):
            waypoint = waypoints_trajectory[i]
            waypoint_next = waypoints_trajectory[i + 1]
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            for wp, connection in interpolated_trace:
                route.append([wp.transform, connection]) # connection is a type of RoadOption
        return route

    def _remove_internal_junctions(self, route: List[carla.Waypoint]) -> List[carla.Waypoint]:
        if len(route) <= 2:
            return route  # nothing to remove
        filtered = [route[0]]  # keep first
        for wp in route[1:-1]:
            if not wp.is_junction:
                filtered.append(wp)
        filtered.append(route[-1])  # keep last
        return filtered

    def sample_one(
        self,
        min_length: float = 200.0,
        max_length: float = 2000.0
    ) -> ScenarioConfig: 
                        
        # sample
        segments = self._map.get_topology()
        
        valid_entry_wps = []
        valid_exit_wps = []
        for seg in segments:
            entry_wp = seg[0]
            exit_wp = seg[1]
            if not entry_wp.is_junction:
                valid_entry_wps.append(entry_wp)
            if not exit_wp.is_junction:
                valid_exit_wps.append(exit_wp)
        
        is_covered = True
        while is_covered:
            ego_start_wp = random.choice(valid_entry_wps)
            ego_end_wp = random.choice(valid_exit_wps)
            
            ego_route_pairs = self._interpolate_trajectory([ego_start_wp.transform.location, ego_end_wp.transform.location], hop_resolution=2.0) # algin with scenario runner
            ego_route_len = len(ego_route_pairs)
            
            if not (min_length / 2.0 <= ego_route_len <= max_length / 2.0):
                continue

            if len(self.start_end_coverage) == 0:
                is_covered = False
                break
            else:
                for ex_wp_pair in self.start_end_coverage:
                    dist_start = ego_start_wp.transform.location.distance(ex_wp_pair[0].transform.location)
                    dist_end = ego_end_wp.transform.location.distance(ex_wp_pair[1].transform.location)
                    if dist_start < 1.0 and dist_end < 1.0:
                        is_covered = True
                        break
                    else:
                        is_covered = False
                    
                if not is_covered:
                    # put checker
                    carla_version = get_carla_version()
        
                    if carla_version < OLD_CARLA_VERSION:
                        ego_model = "vehicle.lincoln.mkz2017" # 0.9.11 and before
                    else:
                        ego_model = "vehicle.lincoln.mkz_2017" # 0.9.12 and after
                        
                    check_pass = self._location_check(
                        agents_info=[
                            {
                                'model': ego_model,
                                'x': ego_start_wp.transform.location.x,
                                'y': ego_start_wp.transform.location.y,
                                'z': ego_start_wp.transform.location.z,
                                'pitch': ego_start_wp.transform.rotation.pitch,
                                'yaw': ego_start_wp.transform.rotation.yaw,
                                'roll': ego_start_wp.transform.rotation.roll
                            }
                        ]
                    )
                    if not check_pass:
                        logger.warning("Failed to put ego vehicle, retry...")
                        is_covered = True
                        
            time.sleep(0.1) # avoid dead loop
        
        # NOTE: always in loop
        ego_route_pairs = self._interpolate_trajectory([ego_start_wp.transform.location, ego_end_wp.transform.location], hop_resolution=2.0) # algin with scenario runner
        
        ego_route_len = len(ego_route_pairs)
        
        
        ego_route_task = []
        ego_route = []
        ego_route_min_x = float('inf')
        ego_route_max_x = float('-inf')
        ego_route_min_y = float('inf')
        ego_route_max_y = float('-inf')
        prev_wp_info = None
        
        estimated_route_length = 0.0
        prev_wp_loc = None
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
                estimated_route_length += wp.transform.location.distance(prev_wp_loc)
            prev_wp_loc = wp.transform.location
            
            curr_wp_info = f"{wp.road_id}_{wp.lane_id}_{wp.section_id}"
            if curr_wp_info != prev_wp_info or wp_index == 0 or wp_index == (ego_route_len - 1):
                ego_route.append(
                    Waypoint(
                        x=wp.transform.location.x,
                        y=wp.transform.location.y,
                        z=wp.transform.location.z,
                        pitch=wp.transform.rotation.pitch,
                        yaw=wp.transform.rotation.yaw,
                        roll=wp.transform.rotation.roll,
                        speed=0.0
                    )
                )
                prev_wp_info = curr_wp_info
                
            wp_task_name = wp_task.name
            if len(ego_route_task) == 0 or wp_task_name != ego_route_task[-1]:
                ego_route_task.append(wp_task_name)
        
        # extend_map region 
        region_min_x = ego_route_min_x - ScenarioSpace.MAP_BUFFER
        region_max_x = ego_route_max_x + ScenarioSpace.MAP_BUFFER
        region_min_y = ego_route_min_y - ScenarioSpace.MAP_BUFFER
        region_max_y = ego_route_max_y + ScenarioSpace.MAP_BUFFER
        
        map_config = MapConfig(
            town=self.town,
            region=[region_min_x, region_max_x, region_min_y, region_max_y]
        )
        
        ego_config = EgoConfig(
            id="ego_0",
            model=None,
            rolename="hero",
            color=None,
            category="vehicle",
            route=ego_route,
            route_func_coverage=ego_route_task,
            route_esitmated_length=estimated_route_length,
            trigger_time=0.0,
            entry_point=None,
            config_path=None
        )
        
        scenario_config = ScenarioConfig(
            id=f"{self.town}", # to be filled in your fuzzing
            scenario_type="open_scenario",
            ego_vehicles=[ego_config],
            npc_vehicles=[], # to be filled
            npc_walkers=[], # to be filled
            npc_statics=[], # to be filled
            weather=None,
            traffic_light=None,
            map_region=map_config,
        )
        
        self.start_end_coverage.append([ego_start_wp, ego_end_wp])
        
        self._cleanup()
        return scenario_config
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Scenario Generator")
    parser.add_argument('--ctn_name', type=str, default='seed-gen', help='Container name')
    parser.add_argument('--ctn_gpu', type=int, default=0, help='Container GPU id')
    parser.add_argument('--image', type=str, default='carlasim/carla:0.9.10.1', help='Ego agent entry point  file')
    parser.add_argument('--town', type=str, default='Town01', help='Town name')
    parser.add_argument('--num', type=int, default=1, help='Number of scenarios to generate')
    parser.add_argument('--max_length', type=int, default=100, help='Random seed')
    parser.add_argument('--min_length', type=int, default=50, help='Random seed')
    parser.add_argument('--out_dir', type=str, default='.', help='Output directory for generated scenarios')
    args = parser.parse_args()  
    
    carla_version = args.image.split(':')[-1]
    
    out_dir = os.path.join(args.out_dir, carla_version, f"route_{int(args.min_length)}_{int(args.max_length)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ctn_op = CtnSimOperator(
        idx=0,
        container_name=f"{args.ctn_name}_{carla_version}",
        gpu=args.ctn_gpu,
        docker_image=args.image
    )
    ctn_op.start()
    
    scen_gen = ScenarioGenerator(ctn_operator=ctn_op, town=args.town)
    
    for i in range(args.num):
        scen_config = scen_gen.sample_one(args.min_length, args.max_length)
        scen_config.id = f"{scen_config.id}_{i:04d}"
        out_path = os.path.join(out_dir, f"{scen_config.id}.json")
        
        with open(out_path, "w") as f:
            json.dump(scen_config.model_dump(), f, indent=4)
            
        logger.info(f"Scenario {scen_config.id} saved to {out_path}")