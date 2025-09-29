from __future__ import annotations

import copy
import carla
import random
import traceback
import numpy as np

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

from omegaconf import DictConfig
from typing import List
from loguru import logger

from scenario_corpus.open_scenario.config import Waypoint, WaypointVehicleConfig, AIWalkerConfig, TrafficLightBehaviorConfig, WeatherConfig, StaticObstacleConfig
from agent_corpus.atomic.route_manipulation import GlobalRoutePlanner
from scenario_runner.ctn_operator import CtnSimOperator

from fuzzer.open_scenario.base import FuzzSeed

class ScenarioSpace:
    """Define the mutation space for OpenSCENARIO scenarios."""
    # common
    MIN_TRIGGER_TIME = 0.0  # seconds
    MAX_TRIGGER_TIME = 0.0  # seconds
    
    # npc vehicles
    MIN_VEHICLE_NUM = 2
    MAX_VEHICLE_NUM = 10
    MIN_VEHICLE_SPEED = 0.0  # m/s
    MAX_VEHICLE_SPEED = 15.0  # m/s
    MIN_VEHICLE_ROUTE_LEN = 10  # waypoints
    MAX_VEHICLE_ROUTE_LEN = 5000  # waypoints
    DELTA_VEHICLE_SPEED = 1.0  # m/s
        
    # npc walkers
    MIN_WALKER_NUM = 2
    MAX_WALKER_NUM = 5
    
    # npc statics
    MIN_STATIC_NUM = 1
    MAX_STATIC_NUM = 5
    
    # weather
    WEATHER_SPACE ={
        "cloudiness": (0, 80),
        "precipitation": (0, 70),
        "precipitation_deposits": (0, 50),
        "wind_intensity": (0, 50),
        "sun_azimuth_angle": (0, 360),
        "sun_altitude_angle": (0, 70),
        "fog_density": (0, 40),
        "fog_distance": (100, 500),
        "wetness": (0, 60),
        "fog_falloff": (1.0, 5.0)
    }

    
    # traffic light
    TRAFFIC_LIGHT_PATTERN = ['none', 'S7left', 'S7right', 'S7opposite', 'S8left', 'S9right']
    MIN_YELLOW_TIME = 1.0  # seconds
    MAX_YELLOW_TIME = 5.0  # seconds
    MIN_RED_TIME = 5.0  # seconds
    MAX_RED_TIME = 20.0  # seconds
    MIN_GREEN_TIME = 5.0  # seconds
    MAX_GREEN_TIME = 20.0  # seconds

    # relationships
    MIN_DISTANCE_BETWEEN_AGENTS = 2.0  # meters
    MAX_DISTANCE_BETWEEN_AGENTS = 50.0  # meters
    MIN_START_DISTANCE_FROM_EGO = 5.0  # meters
    MAX_START_DISTANCE_FROM_EGO = 50.0  # meters
    
    # some fix ego info
    EGO_MODEL = "vehicle.lincoln.mkz2017" # 0.9.10, after mkz_2017
    EGO_ROLENAME = "hero"
    EGO_COLOR = None

class ScenarioMutator:
    
    MAX_RETRY = 20
    
    def __init__(
        self,
        mutation_config: DictConfig
    ):
        # define some global parameters
        self.ctn_operator = None # should be set outside
        self._world = None
        self._client = None
        self._map = None
        
        # update mutataion space
        mutation_space = mutation_config.get("mutation_space", {})

        ScenarioSpace.MIN_TRIGGER_TIME = float(mutation_space.get("MIN_TRIGGER_TIME", ScenarioSpace.MIN_TRIGGER_TIME))
        ScenarioSpace.MAX_TRIGGER_TIME = float(mutation_space.get("MAX_TRIGGER_TIME", ScenarioSpace.MAX_TRIGGER_TIME))

        ScenarioSpace.MIN_VEHICLE_NUM = int(mutation_space.get("MIN_VEHICLE_NUM", ScenarioSpace.MIN_VEHICLE_NUM))
        ScenarioSpace.MAX_VEHICLE_NUM = int(mutation_space.get("MAX_VEHICLE_NUM", ScenarioSpace.MAX_VEHICLE_NUM))

        ScenarioSpace.MIN_VEHICLE_SPEED = float(mutation_space.get("MIN_VEHICLE_SPEED", ScenarioSpace.MIN_VEHICLE_SPEED))
        ScenarioSpace.MAX_VEHICLE_SPEED = float(mutation_space.get("MAX_VEHICLE_SPEED", ScenarioSpace.MAX_VEHICLE_SPEED))

        ScenarioSpace.MIN_VEHICLE_ROUTE_LEN = int(mutation_space.get("MIN_VEHICLE_ROUTE_LEN", ScenarioSpace.MIN_VEHICLE_ROUTE_LEN))
        ScenarioSpace.MAX_VEHICLE_ROUTE_LEN = float(mutation_space.get("MAX_VEHICLE_ROUTE_LEN", ScenarioSpace.MAX_VEHICLE_ROUTE_LEN))  # allow inf

        ScenarioSpace.DELTA_VEHICLE_SPEED = float(mutation_space.get("DELTA_VEHICLE_SPEED", ScenarioSpace.DELTA_VEHICLE_SPEED))

        ScenarioSpace.MIN_WALKER_NUM = int(mutation_space.get("MIN_WALKER_NUM", ScenarioSpace.MIN_WALKER_NUM))
        ScenarioSpace.MAX_WALKER_NUM = int(mutation_space.get("MAX_WALKER_NUM", ScenarioSpace.MAX_WALKER_NUM))

        ScenarioSpace.MIN_STATIC_NUM = int(mutation_space.get("MIN_STATIC_NUM", ScenarioSpace.MIN_STATIC_NUM))
        ScenarioSpace.MAX_STATIC_NUM = int(mutation_space.get("MAX_STATIC_NUM", ScenarioSpace.MAX_STATIC_NUM))

        ScenarioSpace.MIN_YELLOW_TIME = float(mutation_space.get("MIN_YELLOW_TIME", ScenarioSpace.MIN_YELLOW_TIME))
        ScenarioSpace.MAX_YELLOW_TIME = float(mutation_space.get("MAX_YELLOW_TIME", ScenarioSpace.MAX_YELLOW_TIME))
        ScenarioSpace.MIN_RED_TIME = float(mutation_space.get("MIN_RED_TIME", ScenarioSpace.MIN_RED_TIME))
        ScenarioSpace.MAX_RED_TIME = float(mutation_space.get("MAX_RED_TIME", ScenarioSpace.MAX_RED_TIME))
        ScenarioSpace.MIN_GREEN_TIME = float(mutation_space.get("MIN_GREEN_TIME", ScenarioSpace.MIN_GREEN_TIME))
        ScenarioSpace.MAX_GREEN_TIME = float(mutation_space.get("MAX_GREEN_TIME", ScenarioSpace.MAX_GREEN_TIME))

        ScenarioSpace.MIN_DISTANCE_BETWEEN_AGENTS = float(mutation_space.get("MIN_DISTANCE_BETWEEN_AGENTS", ScenarioSpace.MIN_DISTANCE_BETWEEN_AGENTS))
        ScenarioSpace.MAX_DISTANCE_BETWEEN_AGENTS = float(mutation_space.get("MAX_DISTANCE_BETWEEN_AGENTS", ScenarioSpace.MAX_DISTANCE_BETWEEN_AGENTS))
        ScenarioSpace.MIN_START_DISTANCE_FROM_EGO = float(mutation_space.get("MIN_START_DISTANCE_FROM_EGO", ScenarioSpace.MIN_START_DISTANCE_FROM_EGO))
        ScenarioSpace.MAX_START_DISTANCE_FROM_EGO = float(mutation_space.get("MAX_START_DISTANCE_FROM_EGO", ScenarioSpace.MAX_START_DISTANCE_FROM_EGO))
        
        self.carla_version = get_carla_version()
        
        if self.carla_version < OLD_CARLA_VERSION:
            ScenarioSpace.EGO_MODEL = "vehicle.lincoln.mkz2017" # 0.9.11 and before
        else:
            ScenarioSpace.EGO_MODEL = "vehicle.lincoln.mkz_2017" # 0.9.12 and after
        
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
        NOTE: here we use fixed 1.0 as estimation -> TODO: set the senario also 2.0 pls
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
        returns the full interpolated route both in GPS coordinates and also in its original form.
        
        Args:
            - waypoints_trajectory: the current coarse trajectory
            - hop_resolution: distance between the trajectory's waypoints
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

        # 如果不允许终点在 junction，且最后一个点是 junction，就往前继续走
        if not_junction and route and route[-1].is_junction:
            for _ in range(5):  # 最多尝试 5 步往前找非 junction 点
                next_wps = wp.next(step_dist)
                if not next_wps:
                    break
                wp = random.choice(next_wps)
                route.append(wp)
                if not wp.is_junction:
                    break  # 找到非路口点就停
        return route

    def reverse_follow_lane(
        self,
        wp,
        steps=10,
        step_dist=2.0,
        not_junction: bool = True
    ) -> List[carla.Waypoint]:
        """Go backward along the lane. If `not_junction=True`, avoid ending in a junction."""
        route = []
        for _ in range(steps):
            route.insert(0, wp)
            prev_wps = wp.previous(step_dist)
            if not prev_wps:
                break
            wp = random.choice(prev_wps)

        if not_junction and route and route[0].is_junction:
            for _ in range(5):
                prev_wps = wp.previous(step_dist)
                if not prev_wps:
                    break
                
                wp = random.choice(prev_wps)
                route.insert(0, wp)
                if not wp.is_junction:
                    break
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

    def sample_npc_route(
        self,
        sample_num: int = 1,
        spawn_points: List[Waypoint] = [],
        ego_start_wps: List[Waypoint] = [],
    ) -> List[dict]:
                
        # filter too far spawn points
        valid_spawn_points = []
        for sp in spawn_points:
            sp_loc = [sp.transform.location.x, sp.transform.location.y, sp.transform.location.z]
            is_valid = True
            for ego_wp in ego_start_wps:
                ego_loc = [ego_wp.transform.location.x, ego_wp.transform.location.y, ego_wp.transform.location.z]
                dist = np.linalg.norm(np.array(sp_loc) - np.array(ego_loc))
                if dist < ScenarioSpace.MIN_START_DISTANCE_FROM_EGO or dist > ScenarioSpace.MAX_START_DISTANCE_FROM_EGO:
                    is_valid = False
                    break
                
            if is_valid:
                valid_spawn_points.append(sp)
        
        # sample near ones
        if len(valid_spawn_points) > sample_num:
            npc_start_wps = random.sample(valid_spawn_points, sample_num)
        else:
            npc_start_wps = valid_spawn_points
            
        # sample routes
        npc_routes = []
        for i in range(len(npc_start_wps)):
            center_waypoint = self._map.get_waypoint(
                carla.Location(x=npc_start_wps[i].transform.location.x, y=npc_start_wps[i].transform.location.y, z=npc_start_wps[i].transform.location.z),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            npc_route = self.reverse_follow_lane(center_waypoint, steps=random.randint(ScenarioSpace.MIN_VEHICLE_ROUTE_LEN//2, ScenarioSpace.MAX_VEHICLE_ROUTE_LEN//2)) \
                + [center_waypoint] \
                + self.follow_lane(center_waypoint, steps=random.randint(ScenarioSpace.MIN_VEHICLE_ROUTE_LEN//2, ScenarioSpace.MAX_VEHICLE_ROUTE_LEN//2))
            
            npc_route = self._remove_internal_junctions(npc_route)
        
            npc_route_loc = [wp.transform.location for wp in npc_route]
            route_opt = self._interpolate_trajectory(npc_route_loc) # 2 meter per point
            
            final_route = list()
            prev_speed = random.uniform(ScenarioSpace.MIN_VEHICLE_SPEED, ScenarioSpace.MAX_VEHICLE_SPEED)
            for _, wp_pair in enumerate(route_opt):
                delta_speed = np.clip(
                    ScenarioSpace.DELTA_VEHICLE_SPEED * random.uniform(-1, 1),
                    0.0, ScenarioSpace.DELTA_VEHICLE_SPEED
                )
                
                target_speed = np.clip(
                    prev_speed + delta_speed,
                    ScenarioSpace.MIN_VEHICLE_SPEED, ScenarioSpace.MAX_VEHICLE_SPEED
                )
                
                final_route.append(
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
                # final_route.append(
                #     {
                #         'x': wp_pair[0].location.x,
                #         'y': wp_pair[0].location.y,
                #         'z': wp_pair[0].location.z,
                #         'pitch': wp_pair[0].rotation.pitch,
                #         'yaw': wp_pair[0].rotation.yaw,
                #         'roll': wp_pair[0].rotation.roll,
                #         'speed': float(target_speed),
                #         'road_option': wp_pair[1].name
                #     }
                # )
                prev_speed = target_speed
            
            npc_routes.append(final_route)
        return npc_routes
    
    def sample_walker_route(
        self,
        sample_num: int = 1,
        spawn_points: List[Waypoint] = [],
    ):
        pass
    
    def sample_static_location(
        self,
        sample_num: int = 1,
        spawn_points: List[Waypoint] = [],
    ):
        pass
    
    # Weather Tool
    def sample_weather(
        self
    ) -> WeatherConfig:
        ranges = ScenarioSpace.WEATHER_SPACE

        def sample_param(key):
            low, high = ranges[key]
            return random.uniform(low, high)

        weather_pattern = "sampled" 
        weather = WeatherConfig(
            pattern=weather_pattern,
            cloudiness=sample_param("cloudiness"),
            precipitation=sample_param("precipitation"),
            precipitation_deposits=sample_param("precipitation_deposits"),
            wind_intensity=sample_param("wind_intensity"),
            sun_azimuth_angle=sample_param("sun_azimuth_angle"),
            sun_altitude_angle=sample_param("sun_altitude_angle"),
            fog_density=sample_param("fog_density"),
            fog_distance=sample_param("fog_distance"),
            wetness=sample_param("wetness"),
            fog_falloff=sample_param("fog_falloff"),
        )
        return weather
    
    # Traffic Light Tool
    def sample_traffic_light(self) -> TrafficLightBehaviorConfig:
        traffic_light_pattern = 'none' # random.choice(ScenarioSpace.TRAFFIC_LIGHT_PATTERN)
        green_time = random.uniform(ScenarioSpace.MIN_GREEN_TIME, ScenarioSpace.MAX_GREEN_TIME)
        yellow_time = random.uniform(ScenarioSpace.MIN_YELLOW_TIME, ScenarioSpace.MAX_YELLOW_TIME)
        red_time = random.uniform(ScenarioSpace.MIN_RED_TIME, ScenarioSpace.MAX_RED_TIME)
        traffic_light = TrafficLightBehaviorConfig(
            pattern=traffic_light_pattern,
            yellow_time=yellow_time,
            red_time=red_time,
            green_time=green_time
        )
        return traffic_light
    
    ####### Main Interface #########
    def generate(
        self, 
        source_seed: FuzzSeed,
        ctn_operator: CtnSimOperator,
        ego_entry_point: str, 
        ego_config_path: str
    ) -> FuzzSeed:
        
        mutated_seed = copy.deepcopy(source_seed)
        mutated_scenario = mutated_seed.scenario
        
        # some parameters
        # setup ctn
        self.ctn_operator = ctn_operator
        map_region = mutated_scenario.map_region
        map_region_min_x = map_region.region[0]
        map_region_max_x = map_region.region[1]
        map_region_min_y = map_region.region[2]
        map_region_max_y = map_region.region[3]

        existing_actor_info = [] # for location checker
        
        # load town & info
        try:
            self._setup(map_region.town)
        except Exception as e:
            self.ctn_operator.start()
            try:
                self._setup(map_region.town)
            except Exception as e:
                logger.error(traceback.print_exc())
                return None
        
        
        # get spawn points
        all_waypoints = self._map.generate_waypoints(2.0)
        driving_waypoints = []
        walker_waypoints = []
        for wp in all_waypoints:
            
            if wp.transform.location.x < map_region_min_x or wp.transform.location.x > map_region_max_x \
                or wp.transform.location.y < map_region_min_y or wp.transform.location.y > map_region_max_y:
                continue
            
            if wp.lane_type == carla.LaneType.Driving and (not wp.is_junction):
                driving_waypoints.append(wp)
                
            right_lane_wp = wp.get_right_lane()
            # NOTE: we do not remove repeated ones, as we do not have too many spawn points
            if right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Sidewalk:
                walker_waypoints.append(right_lane_wp)
                
            left_lane_wp = wp.get_left_lane()
            if left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Sidewalk:
                walker_waypoints.append(left_lane_wp)
        
        # update ego configs
        ego_configs = mutated_scenario.ego_vehicles
        ego_waypoints = []
        for ego_config in ego_configs:
            ego_config.model = ScenarioSpace.EGO_MODEL
            ego_config.rolename = ScenarioSpace.EGO_ROLENAME
            ego_config.color = ScenarioSpace.EGO_COLOR
            ego_config.trigger_time = float(random.uniform(ScenarioSpace.MIN_TRIGGER_TIME, ScenarioSpace.MAX_TRIGGER_TIME))
            
            existing_actor_info.append(
                {
                    'model': ego_config.model,
                    'x': ego_config.route[0].x,
                    'y': ego_config.route[0].y, 
                    'z': ego_config.route[0].z,
                    'pitch': ego_config.route[0].pitch,
                    'yaw': ego_config.route[0].yaw,
                    'roll': ego_config.route[0].roll
                }
            )
            ego_waypoints.append(
                self._map.get_waypoint(
                    carla.Location(
                        x=ego_config.route[0].x,
                        y=ego_config.route[0].y,
                        z=ego_config.route[0].z
                    ),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
            )
            
            # assign entry point and config path
            ego_config.entry_point = ego_entry_point
            ego_config.config_path = ego_config_path
        
            
        # sample NPC vehicles
        npc_vehicles = []
        for _ in range(self.MAX_RETRY):
            
            npc_routes = self.sample_npc_route(
                sample_num=random.randint(ScenarioSpace.MIN_VEHICLE_NUM, ScenarioSpace.MAX_VEHICLE_NUM),
                spawn_points=driving_waypoints,
                ego_start_wps=ego_waypoints
            )
            
            npc_vehicles = []
            for i in range(len(npc_routes)):
                npc_blueprint = random.choice(self._world.get_blueprint_library().filter('vehicle.*'))
                npc_config = WaypointVehicleConfig(
                    id=f"npc_vehicle_{i}",
                    model=npc_blueprint.id,
                    rolename="npc_vehicle",
                    color=None,
                    category="vehicle",
                    route=npc_routes[i],
                    trigger_time=float(random.uniform(ScenarioSpace.MIN_TRIGGER_TIME, ScenarioSpace.MAX_TRIGGER_TIME)),
                )
                npc_vehicles.append(npc_config)
                
            # checker
            checking_actors = [
                {
                    'model': npc_config.model,
                    'x': npc_config.route[0].x,
                    'y': npc_config.route[0].y,
                    'z': npc_config.route[0].z,
                    'pitch': npc_config.route[0].pitch,
                    'yaw': npc_config.route[0].yaw,
                    'roll': npc_config.route[0].roll
                } for npc_config in npc_vehicles
            ]
            check_pass = self._location_check(
                agents_info=existing_actor_info + checking_actors
            )
            if check_pass:
                break
        
        # update existing actor info
        existing_actor_info.extend(
            [
                {
                    'model': npc_config.model,
                    'x': npc_config.route[0].x,
                    'y': npc_config.route[0].y,
                    'z': npc_config.route[0].z,
                    'pitch': npc_config.route[0].pitch,
                    'yaw': npc_config.route[0].yaw,
                    'roll': npc_config.route[0].roll
                } for npc_config in npc_vehicles
            ]
        )
            
        # sample NPC walkers
        npc_walkers = [] # to be filled
        
        # sample NPC statics
        npc_statics = [] # to be filled
        
        # sample weather
        weather = self.sample_weather()
        
        # sample traffic light
        traffic_light = self.sample_traffic_light()
                
        # generate scenario config
        mutated_scenario.npc_vehicles = npc_vehicles
        mutated_scenario.npc_walkers = npc_walkers
        mutated_scenario.npc_statics = npc_statics
        mutated_scenario.weather = weather
        mutated_scenario.traffic_light = traffic_light
        
        mutated_seed.scenario = mutated_scenario
        
        self._cleanup()
        return mutated_seed