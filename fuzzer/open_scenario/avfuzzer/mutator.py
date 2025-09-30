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
from typing import List, Tuple
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
    TRAFFIC_LIGHT_PATTERN = ['none', 'rule']
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

    # large mutation
    def sample_npc_route(
        self,
        sample_num: int = 1,
        spawn_points: List[Waypoint] = [],
        ego_start_wps: List[Waypoint] = [],
    ) -> List[List[Waypoint]]:
                
        # filter too far spawn points
        valid_spawn_points = []
        for sp in spawn_points:
            sp_loc = [sp.transform.location.x, sp.transform.location.y, sp.transform.location.z]
            is_valid = True
            for ego_wp in ego_start_wps:
                ego_loc = [ego_wp.transform.location.x, ego_wp.transform.location.y, ego_wp.transform.location.z]
                dist = np.linalg.norm(np.array(sp_loc) - np.array(ego_loc))
                
                if sp.lane_id == ego_wp.lane_id and sp.section_id == ego_wp.section_id and sp.road_id == ego_wp.road_id:
                    if dist < ScenarioSpace.MIN_START_DISTANCE_FROM_EGO:
                        is_valid = False
                        break
                
                if dist > ScenarioSpace.MAX_START_DISTANCE_FROM_EGO:
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
                
                prev_speed = target_speed
            
            npc_routes.append(final_route)
        return npc_routes
    
    def sample_walker_route(
        self,
        sample_num: int = 1,
        spawn_region_x: List[float] = [],
        spawn_region_y: List[float] = [],
    ) -> List[List[Waypoint]]:
        
        npc_routes = []
        max_tries = self.MAX_RETRY * sample_num
        tries = 0
                
        while len(npc_routes) < sample_num and tries < max_tries:
            tries += 1
            
            # start point
            x1 = random.uniform(spawn_region_x[0], spawn_region_x[1])
            y1 = random.uniform(spawn_region_y[0], spawn_region_y[1])
            loc1 = carla.Location(x=x1, y=y1, z=0.1)

            wp_start = self._map.get_waypoint(
                loc1,
                project_to_road=True,
                lane_type=(carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Parking),
            )

            if wp_start is None:
                continue
                
            
            # end point
            x2 = random.uniform(spawn_region_x[0], spawn_region_x[1])
            y2 = random.uniform(spawn_region_y[0], spawn_region_y[1])
            loc2 = carla.Location(x=x2, y=y2, z=0.1)

            wp_end = self._map.get_waypoint(
                loc2,
                project_to_road=True,
                lane_type=(carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Parking),
            )

            if wp_end is None:
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
            npc_routes.append(final_route)
        return npc_routes
    
    def sample_static_location(
        self,
        sample_num: int = 1,
        spawn_region_x: List[float] = [],
        spawn_region_y: List[float] = [],
    ) -> List[Waypoint]:
        
        npc_locations = []
        max_tries = self.MAX_RETRY * sample_num
        tries = 0
                
        while len(npc_locations) < sample_num and tries < max_tries:
            tries += 1
            
            # location point
            x = random.uniform(spawn_region_x[0], spawn_region_x[1])
            y = random.uniform(spawn_region_y[0], spawn_region_y[1])
            loc = carla.Location(x=x, y=y, z=0.1)

            wp = self._map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=(carla.LaneType.Sidewalk | carla.LaneType.Shoulder | carla.LaneType.Parking | carla.LaneType.Driving),
            )

            if wp is None:
                continue
                
            npc_locations.append(
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
        return npc_locations
    
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
        all_waypoints = self._map.generate_waypoints(1.0)
        driving_waypoints = []
        for wp in all_waypoints:
            
            if wp.transform.location.x < map_region_min_x or wp.transform.location.x > map_region_max_x \
                or wp.transform.location.y < map_region_min_y or wp.transform.location.y > map_region_max_y:
                continue
            
            if wp.lane_type == carla.LaneType.Driving and (not wp.is_junction):
                driving_waypoints.append(wp)
                
        logger.info(f"Found {len(driving_waypoints)} driving spawn points")
        
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
                        
            # checker
            checking_actor = [
                {
                    'model': npc_config.model,
                    'x': npc_config.route[0].x,
                    'y': npc_config.route[0].y,
                    'z': npc_config.route[0].z,
                    'pitch': npc_config.route[0].pitch,
                    'yaw': npc_config.route[0].yaw,
                    'roll': npc_config.route[0].roll
                }
            ]
            check_pass = self._location_check(
                agents_info=existing_actor_info + checking_actor
            )
            
            if check_pass:
                npc_vehicles.append(npc_config)
                # update existing actor info
                existing_actor_info.extend(checking_actor)
                
        logger.info(f"Sampled {len(npc_routes)} NPC vehicles, actual added {len(npc_vehicles)}.")
            
        # sample NPC walkers
        npc_walkers = [] # to be filled
        npc_routes = self.sample_walker_route(
            sample_num=random.randint(ScenarioSpace.MIN_WALKER_NUM, ScenarioSpace.MAX_WALKER_NUM),
            spawn_region_x=[map_region_min_x, map_region_max_x],
            spawn_region_y=[map_region_min_y, map_region_max_y],
        )
        for i in range(len(npc_routes)):
            npc_blueprint = random.choice(self._world.get_blueprint_library().filter('walker.pedestrian.*'))
            npc_config = AIWalkerConfig(
                id=f"npc_walker_{i}",
                model=npc_blueprint.id,
                rolename="npc_walker",
                category="walker",
                route=npc_routes[i],
                trigger_time=float(random.uniform(ScenarioSpace.MIN_TRIGGER_TIME, ScenarioSpace.MAX_TRIGGER_TIME)),
            )
                        
            # checker
            checking_actor = [
                {
                    'model': npc_config.model,
                    'x': npc_config.route[0].x,
                    'y': npc_config.route[0].y,
                    'z': npc_config.route[0].z,
                    'pitch': npc_config.route[0].pitch,
                    'yaw': npc_config.route[0].yaw,
                    'roll': npc_config.route[0].roll
                }
            ]
            check_pass = self._location_check(
                agents_info=existing_actor_info + checking_actor
            )
            if check_pass:
                npc_walkers.append(npc_config)
                # update existing actor info
                existing_actor_info.extend(checking_actor)
        
        logger.info(f"Sampled {len(npc_routes)} NPC walkers, actual added {len(npc_walkers)}.")
        
        # sample NPC statics
        npc_statics = [] # to be filled      
        static_locations = self.sample_static_location(
            sample_num=random.randint(ScenarioSpace.MIN_STATIC_NUM, ScenarioSpace.MAX_STATIC_NUM),
            spawn_region_x=[map_region_min_x, map_region_max_x],
            spawn_region_y=[map_region_min_y, map_region_max_y]
        )
        
        for i in range(len(static_locations)):
            static_blueprint = random.choice(self._world.get_blueprint_library().filter('static.prop.*'))
            static_config = StaticObstacleConfig(
                id=f"npc_static_{i}",
                model=static_blueprint.id,
                rolename="npc_static",
                category="static",
                location=static_locations[i],
            )
            
            # checker
            checking_actor = [
                {
                    'model': static_config.model,
                    'x': static_config.location.x,
                    'y': static_config.location.y,
                    'z': static_config.location.z,
                    'pitch': static_config.location.pitch,
                    'yaw': static_config.location.yaw,
                    'roll': static_config.location.roll
                }
            ]
            check_pass = self._location_check(
                agents_info=existing_actor_info + checking_actor
            )
            if check_pass:
                npc_statics.append(static_config)
                # update existing actor info
                existing_actor_info.extend(checking_actor)
                
        logger.info(f"Sampled {len(static_locations)} NPC statics, actual added {len(npc_statics)}.")
    
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
    
    # small mutation
    def perturb_npc(
        self,
        npc_vehicles: List[WaypointVehicleConfig],
    ):
        # add small perturbations to existing npc vehicles
        # generate first, if not ok, do not replace
        new_npc_vehicles = copy.deepcopy(npc_vehicles)
        for i in range(len(new_npc_vehicles)):
            # perturb speed
            for j in range(len(new_npc_vehicles[i].route)):
                delta_speed = np.clip(
                    ScenarioSpace.DELTA_VEHICLE_SPEED * random.uniform(-1, 1),
                    0.0, ScenarioSpace.DELTA_VEHICLE_SPEED
                )
                
                target_speed = np.clip(
                    new_npc_vehicles[i].route[j].speed + delta_speed,
                    ScenarioSpace.MIN_VEHICLE_SPEED, ScenarioSpace.MAX_VEHICLE_SPEED
                )
                
                new_npc_vehicles[i].route[j].speed = float(target_speed)
        return new_npc_vehicles
    
    def perturb_walker(
        self,
        npc_walkers: List[AIWalkerConfig],
        region_x: List[float],
        region_y: List[float],
    ):
        # add small perturbations on existing npc walkers with small position noise
        # TODO: some may go out of region, but should be ok, may need wp checker
        new_npc_walkers = copy.deepcopy(npc_walkers)
        for i in range(len(new_npc_walkers)):
            # perturb start point a bit
            new_npc_walkers[i].route[0].x += random.uniform(-1.0, 1.0)
            new_npc_walkers[i].route[0].y += random.uniform(-1.0, 1.0)
            new_npc_walkers[i].route[0].x = np.clip(new_npc_walkers[i].route[0].x, region_x[0], region_x[1])
            new_npc_walkers[i].route[0].y = np.clip(new_npc_walkers[i].route[0].y, region_y[0], region_y[1])
            
            # perturb end point a bit
            new_npc_walkers[i].route[-1].x += random.uniform(-1.0, 1.0)
            new_npc_walkers[i].route[-1].y += random.uniform(-1.0, 1.0)
            new_npc_walkers[i].route[-1].x = np.clip(new_npc_walkers[i].route[-1].x, region_x[0], region_x[1])
            new_npc_walkers[i].route[-1].y = np.clip(new_npc_walkers[i].route[-1].y, region_y[0], region_y[1])
        return new_npc_walkers
    
    def perturb_static(
        self,
        npc_statics: List[StaticObstacleConfig],
        retion_x: List[float],
        region_y: List[float],
    ):
        # add small perturbations on existing npc statics with small position noise
        # TODO: same as walker, may need checker
        new_npc_statics = copy.deepcopy(npc_statics)
        for i in range(len(new_npc_statics)):
            # perturb location a bit
            new_npc_statics[i].location.x += random.uniform(-1.0, 1.0)
            new_npc_statics[i].location.y += random.uniform(-1.0, 1.0)
            new_npc_statics[i].location.x = np.clip(new_npc_statics[i].location.x, retion_x[0], retion_x[1])
            new_npc_statics[i].location.y = np.clip(new_npc_statics[i].location.y, region_y[0], region_y[1])
        return new_npc_statics
    
    def perturb_weather(self, weather: WeatherConfig):
        mutated_weather = copy.deepcopy(weather)
        # perturb each param a bit
        def gaussian_perturb(value, low, high, sigma=0.1):
            new_value = value + random.gauss(0, sigma * (high - low))
            new_value = np.clip(new_value, low, high)
            return float(new_value)     
        
        for key in ScenarioSpace.WEATHER_SPACE.keys():
            low, high = ScenarioSpace.WEATHER_SPACE[key]
            cur_value = getattr(mutated_weather, key)
            new_value = gaussian_perturb(cur_value, low, high, sigma=0.1)
            setattr(mutated_weather, key, new_value) 
            
        return mutated_weather
    
    def perturb_traffic_light(self, traffic_light: TrafficLightBehaviorConfig):
        # perturb traffic light timings a bit
        mutated_traffic_light = copy.deepcopy(traffic_light)
        # gaussian perturb
        def gaussian_perturb(value, low, high, sigma=0.1):
            new_value = value + random.gauss(0, sigma * (high - low))
            new_value = np.clip(new_value, low, high)
            return float(new_value)
        
        mutated_traffic_light.green_time = gaussian_perturb(
            mutated_traffic_light.green_time,
            ScenarioSpace.MIN_GREEN_TIME,
            ScenarioSpace.MAX_GREEN_TIME,
            sigma=0.1
        )
        mutated_traffic_light.yellow_time = gaussian_perturb(
            mutated_traffic_light.yellow_time,
            ScenarioSpace.MIN_YELLOW_TIME,
            ScenarioSpace.MAX_YELLOW_TIME,
            sigma=0.1
        )
        mutated_traffic_light.red_time = gaussian_perturb(
            mutated_traffic_light.red_time,
            ScenarioSpace.MIN_RED_TIME,
            ScenarioSpace.MAX_RED_TIME,
            sigma=0.1
        )
        return mutated_traffic_light
    
    def perturb_scenario(
        self,
        source_seed: FuzzSeed,
        ctn_operator: CtnSimOperator
    ) -> FuzzSeed:
        # We currently implement a simple mutation strategies
        
        mutated_seed = copy.deepcopy(source_seed)
        mutated_scenario = mutated_seed.scenario
        
        # setup ctn
        self.ctn_operator = ctn_operator
        map_region = mutated_scenario.map_region
        map_region_min_x = map_region.region[0]
        map_region_max_x = map_region.region[1]
        map_region_min_y = map_region.region[2]
        map_region_max_y = map_region.region[3]

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
        
        # perturb npc vehicles
        if len(mutated_scenario.npc_vehicles) > 0:
            mutated_scenario.npc_vehicles = self.perturb_npc(
                npc_vehicles=mutated_scenario.npc_vehicles
            )
        
        # perturb npc walkers
        if len(mutated_scenario.npc_walkers) > 0:
            mutated_scenario.npc_walkers = self.perturb_walker(
                npc_walkers=mutated_scenario.npc_walkers,
                region_x=[map_region_min_x, map_region_max_x],
                region_y=[map_region_min_y, map_region_max_y],
            )
        
        # perturb npc statics
        if len(mutated_scenario.npc_statics) > 0:
            mutated_scenario.npc_statics = self.perturb_static(
                npc_statics=mutated_scenario.npc_statics,
                retion_x=[map_region_min_x, map_region_max_x],
                region_y=[map_region_min_y, map_region_max_y],
            )
        
        # perturb weather
        if mutated_scenario.weather is not None:
            mutated_scenario.weather = self.perturb_weather(
                weather=mutated_scenario.weather
            )
        
        # perturb traffic light
        if mutated_scenario.traffic_light is not None:
            mutated_scenario.traffic_light = self.perturb_traffic_light(
                traffic_light=mutated_scenario.traffic_light
            )
        
        mutated_seed.scenario = mutated_scenario
        
        self._cleanup()
        return mutated_seed
    
    def crossover_scenarios(
        self,
        seed1: FuzzSeed,
        seed2: FuzzSeed,
        ctn_operator: CtnSimOperator
    ) -> Tuple[FuzzSeed, FuzzSeed]:
        # TODO: refine and add checker
        
        # crossover two scenarios by exchanging npc vehicles
        
        if seed1.scenario.map_region.town != seed2.scenario.map_region.town:
            logger.warning("Crossover failed due to different towns.")
            raise ValueError("Crossover failed due to different towns.")
        
        mutated_seed1 = copy.deepcopy(seed1)
        mutated_seed2 = copy.deepcopy(seed2)
        
        mutated_scenario = mutated_seed1.scenario
        
        # setup ctn
        self.ctn_operator = ctn_operator
        map_region = mutated_scenario.map_region

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
        
        # crossover npc vehicles
        npc_vehicles_1 = seed1.scenario.npc_vehicles
        npc_vehicles_2 = seed2.scenario.npc_vehicles
        
        if len(npc_vehicles_1) != len(npc_vehicles_2):
            raise ValueError("Crossover failed due to different npc vehicle numbers.")
        
        split_point = random.randint(1, len(npc_vehicles_1))        
        new_npc_vehicles_1 = npc_vehicles_1[:split_point] + npc_vehicles_2[split_point:]
        new_npc_vehicles_2 = npc_vehicles_2[:split_point] + npc_vehicles_1[split_point:]
        
        # crossover npc walkers
        npc_walkers_1 = seed1.scenario.npc_walkers
        npc_walkers_2 = seed2.scenario.npc_walkers
        
        if len(npc_walkers_1) != len(npc_walkers_2):
            raise ValueError("Crossover failed due to different npc walker numbers.")
        
        split_point = random.randint(1, len(npc_walkers_1))        
        new_npc_walkers_1 = npc_walkers_1[:split_point] + npc_walkers_2[split_point:]
        new_npc_walkers_2 = npc_walkers_2[:split_point] + npc_walkers_1[split_point:]
        
        # crossover npc statics
        npc_statics_1 = seed1.scenario.npc_statics
        npc_statics_2 = seed2.scenario.npc_statics
        
        if len(npc_statics_1) != len(npc_statics_2):
            raise ValueError("Crossover failed due to different npc static numbers.")
        
        split_point = random.randint(1, len(npc_statics_1))        
        new_npc_statics_1 = npc_statics_1[:split_point] + npc_statics_2[split_point:]
        new_npc_statics_2 = npc_statics_2[:split_point] + npc_statics_1[split_point:]
        
        # crossover weather
        weather_1 = seed1.scenario.weather
        weather_2 = seed2.scenario.weather
        if random.random() < 0.5:
            new_weather_1 = weather_1
            new_weather_2 = weather_2
        else:
            new_weather_1 = weather_2
            new_weather_2 = weather_1
            
        # crossover traffic light
        traffic_light_1 = seed1.scenario.traffic_light
        traffic_light_2 = seed2.scenario.traffic_light
        if random.random() < 0.5:
            new_traffic_light_1 = traffic_light_1
            new_traffic_light_2 = traffic_light_2
        else:
            new_traffic_light_1 = traffic_light_2
            new_traffic_light_2 = traffic_light_1
        
        # assign to seeds
        # new seed 1
        mutated_seed1.scenario.npc_vehicles = new_npc_vehicles_1
        mutated_seed1.scenario.npc_walkers = new_npc_walkers_1
        mutated_seed1.scenario.npc_statics = new_npc_statics_1
        mutated_seed1.scenario.weather = new_weather_1
        mutated_seed1.scenario.traffic_light = new_traffic_light_1
        # new seed 2
        mutated_seed2.scenario.npc_vehicles = new_npc_vehicles_2
        mutated_seed2.scenario.npc_walkers = new_npc_walkers_2
        mutated_seed2.scenario.npc_statics = new_npc_statics_2
        mutated_seed2.scenario.weather = new_weather_2
        mutated_seed2.scenario.traffic_light = new_traffic_light_2      
        
        self._cleanup()
        return mutated_seed1, mutated_seed2
    