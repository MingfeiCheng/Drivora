"""
Overall flow:
1. create ego first
2. create sub-scenarios then
"""
import carla
import copy
import traceback

from loguru import logger
from typing import List, Optional

from scenario_runner.ctn_operator import CtnSimOperator
from scenario_corpus.base.scenario import BasicScenario
from scenario_elements.config import Waypoint
from scenario_elements.criteria.runtime_single import RuntimeSingleTest
from scenario_elements.criteria.runtime_group import RuntimeGroupTest

from agent_corpus.atomic.route_manipulation import interpolate_trajectory

from .waypoint_vehicle import WaypointVehicleScenario
from .ai_walker import AIWalkerScenario
from .traffic_light import TrafficLightScenario

from .config import ScenarioConfig

class OpenScenario(BasicScenario):
    """
    For crietria, we define at fuzzing concrete scenarios, thanks
    """
    def __init__(
        self, 
        config: ScenarioConfig,
        ctn_operator: CtnSimOperator,
        terminate_on_failure: bool = True,
        criteria_enable: bool = True,
        debug_mode: bool = False,
        timeout: float = 600.0,
    ):
        self.config = config
        
        super(OpenScenario, self).__init__(
            name=config.id,
            ctn_operator=ctn_operator,
            terminate_on_failure=terminate_on_failure,
            criteria_enable=criteria_enable,
            debug_mode=debug_mode,
            timeout=timeout
        )
        
    def _calculate_ego_route(self, waypoints: List[Waypoint]):
        # prepare route's trajectory (interpolate and add the GPS route)
        keypoints_location = []
        for keypoint in [waypoints[0], waypoints[-1]]:  # only use start and end point for now
            keypoints_location.append(carla.Location(
                x=keypoint.x,
                y=keypoint.y,
                z=keypoint.z
            ))
        # interpolate the trajectory
        gps_route, route = interpolate_trajectory(self.ctn_operator, keypoints_location, 2.0)
        return gps_route, route
    
    def _setup_ego_vehicles(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        ego_configs = self.config.ego_vehicles

        self.ego_vehicles = {} # to store the ego vehicles/ carla actors
        self.ego_gps_routes = {} # to store the ego gps routes if any
        self.ego_routes = {} # to store the ego routes if any
        self.ego_destinations = {} # to store the ego destinations if any
        self.ego_triggers = {} # to store the ego triggers if any

        _map = self.world.get_map()
        for _ego_cfg in ego_configs:
            _ego_init_wp = _ego_cfg.route[0]
            elevate_waypoint = _map.get_waypoint(
                carla.Location(
                    x=_ego_init_wp.x,
                    y=_ego_init_wp.y,
                    z=_ego_init_wp.z
                )
            )
            elevate_transform = elevate_waypoint.transform
            elevate_transform.location.z += 1.0
            # logger.debug(f"{_ego_cfg.id} waypoint: {elevate_waypoint.lane_type}, {elevate_waypoint.road_id}, {elevate_waypoint.lane_id} {elevate_transform.location}")

            _ego_cfg.model = 'vehicle.lincoln.mkz2017'
            _ego_cfg.rolename = 'hero'
            _ego_cfg.route[0].x = elevate_transform.location.x
            _ego_cfg.route[0].y = elevate_transform.location.y
            _ego_cfg.route[0].z = elevate_transform.location.z
            _ego_cfg.route[0].pitch = elevate_transform.rotation.pitch
            _ego_cfg.route[0].yaw = elevate_transform.rotation.yaw
            _ego_cfg.route[0].roll = elevate_transform.rotation.roll

            ego_actor = self.ctn_operator.request_new_actor(
                model=_ego_cfg.model,
                spawn_point=elevate_transform,
                rolename=_ego_cfg.rolename,
                autopilot=False,
                color=None,
                actor_category='vehicle',
                attribute_filter=None,
                tick=True
            )
            if ego_actor is None:
                raise RuntimeError(f"Ego vehicle {_ego_cfg.id} could not be spawned")

            if _ego_cfg.id not in self.ego_vehicles:
                self.ego_vehicles[_ego_cfg.id] = ego_actor
            else:
                raise RuntimeError(f"Ego vehicle id {_ego_cfg.id} already exists")
            
            # calculate route
            ego_gps_route, ego_route = self._calculate_ego_route(_ego_cfg.route)
            self.ego_gps_routes[_ego_cfg.id] = ego_gps_route
            self.ego_routes[_ego_cfg.id] = ego_route
            self.ego_destinations[_ego_cfg.id] = copy.deepcopy(_ego_cfg.route[-1])
            self.ego_trigger_times[_ego_cfg.id] = _ego_cfg.trigger_time
    
    def _setup_scenarios(self):
        self.list_scenarios = []
        
        # create vehicle scenario
        try:
            vehicle_scenario_instance = WaypointVehicleScenario(
                self.config.id + '_vehicle',
                self.config.npc_vehicles,
                self.ctn_operator
            )
            vehicle_scenario_instance.initialize()
            if self.ctn_operator.is_sync_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception as e:
            logger.warning("Skipping scenario '{}' due to setup error: {}".format('Vehicle', e))
            traceback.print_exc()
            vehicle_scenario_instance = None
        
        if vehicle_scenario_instance:
            self.list_scenarios.append(vehicle_scenario_instance)

        # create walker scenario
        try:
            walker_scenario_instance = AIWalkerScenario(
                self.config.id + '_walker',
                self.config.npc_walkers,
                self.ctn_operator
            )
            walker_scenario_instance.initialize()
            if self.ctn_operator.is_sync_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception as e:
            logger.warning("Skipping scenario '{}' due to setup error: {}".format('Walker', e))
            traceback.print_exc()
            walker_scenario_instance = None
        
        if walker_scenario_instance:
            self.list_scenarios.append(walker_scenario_instance)

        # set traffic lights to green
        # set traffic light
        traffic_light_type = self.config.traffic_light.pattern
        if traffic_light_type is None or traffic_light_type == 'none':
            list_actor = self.world.get_actors()
            selected_actors = list_actor.filter('*' + 'traffic_light' + '*')
            for actor_ in selected_actors:
                if isinstance(actor_, carla.TrafficLight):
                    # for any light, first set the light state, then set time. for yellow it is
                    # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                    actor_.set_state(carla.TrafficLightState.Green)
                    actor_.set_green_time(100000.0)
        else:
            try:
                traffic_light_scenario_instance = TrafficLightScenario(
                    self.config.id + '_traffic_light',
                    self.config.traffic_light,
                    self.ctn_operator
                )
                traffic_light_scenario_instance.initialize()
                if self.ctn_operator.is_sync_mode:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
            except Exception as e:
                logger.warning("Skipping scenario '{}' due to setup error: {}".format('Walker', e))
                traceback.print_exc()
                traffic_light_scenario_instance = None
            
            if traffic_light_scenario_instance:
                self.list_scenarios.append(traffic_light_scenario_instance)

        logger.info(f"Total {len(self.list_scenarios)} sub-scenarios are created in the random scenario.")
    
    def _initialize_environment(self):
        world = self.world
        weather_config = self.config.weather
        carla_weather = carla.WeatherParameters(
            cloudiness=weather_config.cloudiness,
            precipitation=weather_config.precipitation,
            precipitation_deposits=weather_config.precipitation_deposits,
            wind_intensity=weather_config.wind_intensity,
            sun_azimuth_angle=weather_config.sun_azimuth_angle,
            sun_altitude_angle=weather_config.sun_altitude_angle,
            fog_density=weather_config.fog_density,
            fog_distance=weather_config.fog_distance,
            wetness=weather_config.wetness,
            fog_falloff=weather_config.fog_falloff
        )
        world.set_weather(carla_weather)

    def _initialize_actors(self):
        self.other_actors = {}
        for scenario in self.list_scenarios:
            for key, value in scenario.other_actors.items():
                if key in self.other_actors:
                    raise ValueError(f"Duplicate actor key found: {key}")
                self.other_actors[key] = value
    
    def _create_test_criteria(self) -> Optional[list]:
        """
        Create a runtime monitor for the sub-behavior tree.
        This is a placeholder for future implementation.
        """
        ego_num = len(list(self.ego_vehicles.keys()))
        
        single_criterias = []
        for ego_id in self.ego_vehicles.keys():
            criteria = RuntimeSingleTest(
                id=ego_id,
                name=f"{self.name}_criteria_{ego_id}",
                actor=self.ego_vehicles[ego_id],
                actor_route=self.ego_routes[ego_id],
                actor_trigger_time=self.ego_trigger_times[ego_id],
                min_speed=0.1, # 1 m/s
                max_time=180.0,
                terminate_on_failure=self.terminate_on_failure,
                ctn_operator=self.ctn_operator
            )
            single_criterias.append(criteria)
        
        if len(single_criterias) == 1:
            return single_criterias
        else:
            group_criteria = RuntimeGroupTest(
                name=f"{self.name}_group_criteria",
                detectors=single_criterias,
                terminate_on_failure=self.terminate_on_failure
            )
            return [group_criteria]
    
    def remove_all_actors(self):
        super(OpenScenario, self).remove_all_actors() # as we also record in this scenarios
        
        # remove ego vehicles
        if hasattr(self, 'ego_vehicles'):
            for _, ego_actor in self.ego_vehicles.items():
                if ego_actor is not None:
                    self.ctn_operator.remove_actor(ego_actor)
            self.ego_vehicles = dict()
        