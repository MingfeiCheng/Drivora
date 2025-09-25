#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide BasicScenario, the basic class of all the scenarios.
"""

from __future__ import print_function

import os
import cv2
import sys
import time
import json
import gzip
import carla
import signal
import py_trees
import traceback

from PIL import Image
from tqdm import tqdm
from loguru import logger
from typing import TypeVar
from importlib import import_module
from pydantic import BaseModel

from tools.watchdog import Watchdog
from tools.timer import GameTime

from scenario_runner.ctn_operator import CtnSimOperator
from agent_corpus.atomic.sensor_interface import SensorInterface, CallBack
from agent_corpus.atomic.base_agent_wrapper import AgentWrapper
from tools.recorder_tool import images_to_video
from tools.result_writer import ResultOutputProvider

from registry import MANAGER_REGISTRY

ScenarioConfigType = TypeVar('ScenarioConfigType', bound='BaseModel')

def load_entry_point(name):
    mod_name, attr_name = name.split(":")
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

@MANAGER_REGISTRY.register("scenario_manager.default")
class ScenarioManager(object):

    """
    Base class for user-defined scenario
    Here, the input should be the scenario tree, and directly run the scenario tree
    You should better define all the scenario in your template, including sensors, records
    """

    def __init__(
        self, 
        scenario_entry_point: str,
        scenario_config: ScenarioConfigType, # directly pass the scenario tree instance here
        ctn_operator: CtnSimOperator,
        scenario_dir: str = None,
        debug: bool = False,
        pytree_debug: bool = False,
        require_vis: bool = False,
        max_sim_time: float = 300.0,
    ):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        
        This class aims to run the scenario behavior tree
        """        
        # basic configuration of the scenario, you can define your own scenario config by inheriting from MetaScenarioConfig
        self.scenario_entry_point = scenario_entry_point
        self.scenario_config = scenario_config
        self.ctn_operator = ctn_operator
        self.debug = debug
        self.pytree_debug = pytree_debug  # Whether to enable py_trees debug mode
        self.require_vis = require_vis  # Whether to enable visualization
        self.scenario_dir = scenario_dir  # directory to save the scenario results
        self.max_sim_time = max_sim_time if max_sim_time > 0 else 600.0
        
        # If no timeout was provided, set it to 60 seconds
        if self.pytree_debug:
            py_trees.logging.level = py_trees.logging.Level.DEBUG
                    
        # internal parameters
        self._running = False  # Flag to indicate if the scenario is running
        
        # time counters
        self.start_system_time = 0.0  # System time when the scenario started
        self.start_game_time = 0.0  # Game time when the scenario started
        self.end_system_time = 0.0  # System time when the scenario ended
        self.end_game_time = 0.0
        self.step = 0  # Step counter for the scenario
        self.scenario_duration_system = 0.0  # System time when the scenario ended
        self.scenario_duration_game = 0.0  # Game time when the scenario ended
        self._timestamp_last_run = 0.0  # Last run timestamp of the scenario
        
        # watchdogs
        self._watchdog_tick = Watchdog(
            60.0,  # 60 seconds for tick watchdog
        )
        self._watchdog_agent = Watchdog(
            120.0,  # 60 seconds for initialization watchdog
        )
        
        # carla info
        self.client = self.ctn_operator.client
        self.world = None
        self.map = None
        
        # scenario parameters
        self.scenario_class = load_entry_point(self.scenario_entry_point)
        self.scenario_instance = None
        
        # agent instances
        self.agent_instances = dict()  # can be set in the scenario instance
        self.agent_runners = dict()   # wrapper for the agent instances
        
        # some recorder sensors
        self.recorder_sensor_list = []
        self.recorder_observation = []
        
        # NOTE: can be changed to scenario or config
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam   

    def _reset_parameters(self):
        """
        This method is used to initialize the parameters of the scenario.
        It can be overridden by user-defined scenarios to provide custom initialization.
        reset some internal hyper-parameters
        """
        GameTime.restart_with_server() # NOTE: this should be careful, as it will reset the game time
        
        self.start_system_time = 0.0  # System time when the scenario started
        self.start_game_time = 0.0  # Game time when the scenario started
        self.end_system_time = 0.0  # System time when the scenario ended
        self.end_game_time = 0.0
        self.step = 0  # Step counter for the scenario
        self.scenario_duration_system = 0.0  # System time when the scenario ended
        self.scenario_duration_game = 0.0  # Game time when the scenario ended
        self._timestamp_last_run = 0.0  # Last run timestamp of the scenario
        
        self._running = False
        
        self.recorder_sensor_list = []
        self.recorder_observation = []
    
    def _load_wait_for_world(self, town):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """
        
        # TODO: check if we need this
        tm = self.client.get_trafficmanager(int(self.ctn_operator.tm_port))
        tm.set_random_device_seed(int(self.ctn_operator.random_seed))
        if self.ctn_operator.is_sync_mode:
            tm.set_synchronous_mode(True)
            
        try:
            self.world = self.client.load_world(town)
        except Exception as e:
            logger.error(traceback.print_exc())
            logger.error("> {}\033[0m\n".format(e))

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.ctn_operator.fps # use here
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()

        # Wait for the world to be ready
        if self.ctn_operator.is_sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if os.path.basename(self.world.get_map().name) != town:
            raise Exception(f"The CARLA server uses the wrong map {self.world.get_map().name}!"
                            "This scenario requires to use map {}".format(town))

    def run(self):
        logger.info(f"[ScenarioManager] Running scenario {self.scenario_config.id}")
        
        # 1. initialize hyperparameters
        self._reset_parameters()
        
        # 2. load world first
        town = self.scenario_config.get_town()
        logger.info(f"Load town {town} for scenario {self.scenario_config.id}")
        try:
            self._load_wait_for_world(town)
        except Exception as e:
            logger.error(f"Failed to load town {town} for scenario {self.scenario_config.id}: {e}")
            traceback.print_exc()
            self.stop()
            return False
        
        # 3. load scenario instance
        logger.info(f"Load scenario class {self.scenario_class} for scenario {self.scenario_config.id}")
        try:
            self.scenario_instance = self.scenario_class(
                config=self.scenario_config,
                ctn_operator=self.ctn_operator,
                terminate_on_failure=True,
                criteria_enable=True,
                debug_mode=self.pytree_debug,
                timeout=self.max_sim_time,
            )
            self.scenario_instance.initialize()
            
            # open vehicle light
            if self.scenario_config.weather.sun_altitude_angle < 0.0:
                for vehicle_id, vehicle in self.scenario_instance.ego_vehicles.items():
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
                    
                for actor_id, actor in self.scenario_instance.other_actors:
                    if isinstance(actor, carla.Vehicle) and actor.is_alive:
                        actor.set_light_state(carla.VehicleLightState(self._vehicle_lights))
            
        except Exception as e:
            logger.error(f"Failed to initialize scenario instance: {e}")
            traceback.print_exc()
            self.stop()
            return False
        
        # 4. setup the ads under test
        logger.info(f"Setup ADS agents for scenario {self.scenario_config.id}")
        try:
            logger.info(f"Target number of ego vehicles: {len(self.scenario_config.ego_vehicles)}")
            
            for ego_config in self.scenario_config.ego_vehicles:
                ego_id = ego_config.id
                
                logger.info(f"Setting up agent for ego vehicle {ego_id} with config: {ego_config}")
                
                agent_class = load_entry_point(ego_config.entry_point)
                
                self._watchdog_agent.start()
                # setup the agent instance
                agent_instance = agent_class()
                agent_instance.set_global_plan(
                    global_plan_gps=self.scenario_instance.ego_gps_routes[ego_id], 
                    global_plan_world_coord=self.scenario_instance.ego_routes[ego_id]
                )
                agent_instance.setup_env(
                    id=ego_id,
                    vehicle=self.scenario_instance.ego_vehicles[ego_id],
                    ctn_operator=self.ctn_operator,
                    scenario_dir=self.scenario_dir,
                    ego_config = ego_config.config_path
                ) 
                
                self.agent_instances[ego_id] = agent_instance
                logger.info(f"Successfully set up agent for ego vehicle {ego_id}")
                
                self._watchdog_agent.stop()
                
        except Exception as e:
            logger.error(f"Failed to set up ADS agents: {e}")
            traceback.print_exc()
            self.stop()
            return False
        
        logger.info(f"Success loading scenario")
        
        # 5. running the scenarios
        try:
            self._run_scenario_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, stopping scenario...")
            self.stop()
        except Exception as e:
            logger.error(f"Exception during scenario run: {e}")
            traceback.print_exc()
            self.stop()
            return False
    
        # 6. anaylze and stop the scenario
        self.stop()
        logger.info(f"Scenario {self.scenario_config.id} finished")
        return True
    
    def stop(self, signum=None, frame=None):
        """
        Stop the scenario and all sub-scenarios
        """
        self._running = False
        self._watchdog_tick.stop()
        
        if self._watchdog_agent:
            self._watchdog_agent.stop()
        
        # time counter
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()
        
        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time
        
        # clean up sensors
        for i, _ in enumerate(self.recorder_sensor_list):
            if self.recorder_sensor_list[i] is not None:
                self.recorder_sensor_list[i].stop()
                self.recorder_sensor_list[i].destroy()
                self.recorder_sensor_list[i] = None
        self.recorder_sensor_list.clear()
        
        # cleean the agent runners
        for ego_id, agent_runner in self.agent_runners.items():
            if agent_runner is not None:
                try:
                    agent_runner.cleanup()
                except Exception as e:
                    logger.warning(f"[WARN] Failed to stop agent runner for {ego_id}: {e}")
        self.agent_runners.clear()
        
        # clean up the agent instances
        if self.agent_instances:
            for agent_id, agent_ins in self.agent_instances.items(): # this is agent wrapper right?
                if agent_ins:
                    try:
                        agent_ins.destroy()
                        agent_ins = None
                    except Exception as e:
                        logger.warning(f"[WARN] Failed to destroy agent {agent_id}: {e}")
                        
            self.agent_instances.clear()
        
 
        if self.scenario_instance:       
            self.scenario_instance.terminate()
            self.scenario_instance.remove_all_actors()
            self.analyze_scenario() # save results
            
            self.scenario_instance = None
            
        if signum == signal.SIGINT:
            pass # May need some special handling for SIGINT
            # GlobalConfig.get_world().apply_settings(carla.WorldSettings(synchronous_mode=False))

        # TODO: check if we need this
        if self.world is not None and self.ctn_operator.is_sync_mode:
            try:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.client.get_trafficmanager(int(self.ctn_operator.tm_port)).set_synchronous_mode(False)
            except RuntimeError:
                logger.error("Failed to reset CARLA world to asynchronous mode. "
                             "This might be due to the scenario not being properly terminated.")
                sys.exit(-1)
    
    def _run_scenario_loop(self):
        
        # 1.start some time counters & conditions
        GameTime.restart_with_server() # restart the game time IMPORTANT
        
        # 2. wrapper the agent instances
        for ego_id, agent_instance in self.agent_instances.items():
            if ego_id in self.scenario_instance.ego_vehicles:
                agent_runner = None
                try:
                    # setup the agent wrapper
                    agent_runner = AgentWrapper(agent_instance, self.ctn_operator)
                    agent_runner.setup_sensors()
                    self.agent_runners[ego_id] = agent_runner
                except Exception as e:
                    logger.error(f"Failed to wrap agent for ego vehicle {ego_id}: {e}")
                    traceback.print_exc()
                    self.stop()
                    raise RuntimeError(f"Ego vehicle {ego_id} not found in scenario instance.")
            else:
                logger.error(f"Ego vehicle {ego_id} not found in scenario instance.")
                self.stop()
                raise RuntimeError(f"Ego vehicle {ego_id} not found in scenario instance.")
        
        # 3. setup some recorder sensors
        if self.require_vis:
            logger.info(f"Setting up recorder sensors for scenario {self.scenario_config.id}")
            self._setup_recorder_sensors()
            
        if self.ctn_operator.is_sync_mode:
            self.world.tick()  # Tick once to ensure everything is set up
        else:
            self.world.wait_for_tick()
            
        self.recorder_observation = [] # reset the recorder observation
        
        # 4. run - run bar time counter
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()
        
        self._watchdog_tick.start()
        self._running = True
        
        bar = tqdm()
        wallclock_t0 = GameTime.get_wallclocktime()
        
        # debug info
        logger.debug(f"Total vehicle actors: {len(self.world.get_actors().filter('vehicle.*'))}")
        logger.debug(f"Total walker actors: {len(self.world.get_actors().filter('walker.*'))}")
        logger.debug(f"Total traffic light actors: {len(self.world.get_actors().filter('traffic.traffic_light*'))}")
        logger.debug(f"Total traffic sign actors: {len(self.world.get_actors().filter('traffic.traffic_sign*'))}")
        logger.debug(f"Total static prop actors: {len(self.world.get_actors().filter('static.prop.*'))}")
        logger.debug(f"Total sensor actors: {len(self.world.get_actors().filter('sensor.*'))}")
        
        while self._running:
            timestamp = None
            if self.world:
                snapshot = self.world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
                    
            if timestamp:
                self._tick_scenario(timestamp)
                
            # time counter
            game_timestamp = GameTime.get_time()
            
            if isinstance(wallclock_t0, int) or isinstance(wallclock_t0, float):
                wallclock_t0 = GameTime.get_wallclocktime()
            else:
                wallclock = GameTime.get_wallclocktime()
                wallclock_diff = (wallclock - wallclock_t0).total_seconds()
                bar.set_description(
                    f"-> Scenario {self.scenario_config.id} | "
                    f"System: {wallclock_diff:.3f} | "
                    f"Game: {game_timestamp:.3f} "
                )
        
        # 5. run finished
        # save the video
        recorder_video_dir = os.path.join(self.scenario_dir, 'agent/video')
        if os.path.exists(recorder_video_dir):
            ego_ids = os.listdir(recorder_video_dir)
            for ego_id in ego_ids:
                ego_video_dir = os.path.join(recorder_video_dir, ego_id)
                if os.path.isdir(ego_video_dir):
                    images_to_video(ego_video_dir, f"{recorder_video_dir}/{ego_id}.mp4", fps=30, delete_folder=True)
                    
        # save the scenario record
        observation_file = os.path.join(self.scenario_dir, 'observation.jsonl.gz')
        # with open(observation_file, 'w') as f:
        #     json.dump(self.recorder_observation, f, indent=4)
        with gzip.open(observation_file, "wt", encoding="utf-8") as f:
            for record in self.recorder_observation:
                f.write(json.dumps(record) + "\n")
        
    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._running and self._timestamp_last_run < timestamp.elapsed_seconds:
            self._timestamp_last_run = timestamp.elapsed_seconds
            self.step += 1
            
            self._watchdog_tick.update()
            GameTime.on_carla_tick(timestamp)
            
            exec_timestamp = GameTime.get_time()
            agent_log_data = {}
            for _agent_id, _agent in self.agent_runners.items():
                try:
                    _agent_action, _agent_log_data = _agent()
                    agent_log_data[_agent_id] = _agent_log_data
                except Exception as e:
                    traceback.print_exc()
                    raise RuntimeError(e)
                
                # Check if the time trigger is reached
                _agent_trigger_time = self.scenario_instance.ego_trigger_times.get(_agent_id, 0.0)
                if exec_timestamp >= _agent_trigger_time:
                    self.scenario_instance.ego_vehicles[_agent_id].apply_control(_agent_action)

            try:
                self._tick_recorder(agent_log_data)
            except Exception as e:
                traceback.print_exc()
                raise RuntimeError(e)

            self.scenario_instance.tick()  
            
            if self.pytree_debug:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_instance.scenario_tree, show_status=True)
                sys.stdout.flush()

            # whole status, but may not stop, we consider criteria tree
            if self.scenario_instance.scenario_tree.status != py_trees.common.Status.RUNNING:
                logger.info(f"Scenario {self.scenario_config.id} finished with status {self.scenario_instance.scenario_tree.status}")
                self._running = False
            
        if self._running and self._watchdog_tick.get_status():
            if self.ctn_operator.is_sync_mode:
                self.world.tick()
            else:
                self.world.wait_for_tick()
                     
    def _setup_recorder_sensors(self):
        self.recorder_sensor_interface = SensorInterface()
        for ego_vehicle_id, ego_vehicle in self.scenario_instance.ego_vehicles.items():
            # recorder_sensor = {
            #     'type': 'sensor.camera.rgb',
            #     'x': - 8.0, 'y': 0.0, 'z': 2.2,
            #     'roll': 0.0, 'pitch': -10.0, 'yaw': 0.0,
            #     'width': 1600, 'height': 800, 'fov': 70,
            #     'id': f'{ego_vehicle_id}'
            # }
            recorder_sensor = {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 20.0,     # 高度适中，清晰看到车和周围路口
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 640, 'height': 480,       # 小分辨率，保证快
                'fov': 90,                         # 适中视野，不会太远
                'id': f'{ego_vehicle_id}_bird' # config id 
            }
            
            bp_library = self.ctn_operator.get_world().get_blueprint_library()

            bp = bp_library.find(str(recorder_sensor['type']))
            bp.set_attribute('image_size_x', str(recorder_sensor['width']))
            bp.set_attribute('image_size_y', str(recorder_sensor['height']))
            bp.set_attribute('fov', str(recorder_sensor['fov']))
            bp.set_attribute('lens_circle_multiplier', str(3.0))
            bp.set_attribute('lens_circle_falloff', str(3.0))
            bp.set_attribute('chromatic_aberration_intensity', str(0.5))
            bp.set_attribute('chromatic_aberration_offset', str(0))

            sensor_location = carla.Location(
                x=recorder_sensor['x'], 
                y=recorder_sensor['y'],
                z=recorder_sensor['z']
            )
            sensor_rotation = carla.Rotation(
                pitch=recorder_sensor['pitch'],
                roll=recorder_sensor['roll'],
                yaw=recorder_sensor['yaw']
            )
            
            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = self.ctn_operator.get_world().spawn_actor(bp, sensor_transform, ego_vehicle)
            # setup callback
            sensor.listen(CallBack(recorder_sensor['id'], recorder_sensor['type'], sensor, self.recorder_sensor_interface))
            self.recorder_sensor_list.append(sensor)

            # Tick once to spawn the sensors
            # Wait for the world to be ready
            if self.ctn_operator.is_sync_mode:
                self.ctn_operator.get_world().tick()
            else:
                self.ctn_operator.get_world().wait_for_tick()
                
        logger.info(f"Successfully setup {len(self.recorder_sensor_list)} recorder sensors")
        
    def _tick_recorder(self, agent_log_data: dict):
        
        recorder_save_dir = os.path.join(self.scenario_dir, 'agent')
        if not os.path.exists(recorder_save_dir):
            os.makedirs(recorder_save_dir, exist_ok=True)
            
        recorder_video_dir = os.path.join(recorder_save_dir, 'video')
        if not os.path.exists(recorder_video_dir):
            os.makedirs(recorder_video_dir, exist_ok=True)
        
        ######### Save the visualization #########
        if self.require_vis:
            input_data = self.recorder_sensor_interface.get_data(GameTime.get_frame())
            for ego_id, ego_data in input_data.items():
                
                ego_save_video_dir = os.path.join(recorder_video_dir, ego_id)
                os.makedirs(ego_save_video_dir, exist_ok=True)
                    
                # We add some visualization here, can be deleted if not neccesary
                # save observation image
                view_image = cv2.cvtColor(input_data[ego_id][1][:, :, :3], cv2.COLOR_BGR2RGB)
                view_image = Image.fromarray(view_image)
                # Image.fromarray(view_image).save(os.path.join(self._dir_path, view_image_path))
                # Get the original size
                view_image_size = view_image.size  # (width, height)
                # Calculate the new size (half of the original size)
                new_size = (view_image_size[0] // 2, view_image_size[1] // 2)
                # Resize the image
                view_image = view_image.resize(new_size, Image.ANTIALIAS)
                view_image.save(os.path.join(ego_save_video_dir, f"{self.step:06d}.png"))
        
        
        ###### save scene observation ######  
        scene_dict = {
            "step": self.step,
            "timestamp": GameTime.get_time(),
            "frame": GameTime.get_frame(),
            "carla_time": GameTime.get_carla_time(),
            "town": self.ctn_operator.get_world().get_map().name,
            "agent_logs": agent_log_data,  # {ego_id: {log_data}
            "egos": {},
            "other_actors": {
                "vehicles": [],
                "walkers": [],
                "static_props": [],
                "traffic_lights": [],
                "traffic_signs": [],
                "map_statics": []
            },
            "weather": {
              "cloudiness": self.world.get_weather().cloudiness,
              "precipitation": self.world.get_weather().precipitation,
              "precipitation_deposits": self.world.get_weather().precipitation_deposits,
              "wind_intensity": self.world.get_weather().wind_intensity,
              "sun_azimuth_angle": self.world.get_weather().sun_azimuth_angle,
              "sun_altitude_angle": self.world.get_weather().sun_altitude_angle,
              "fog_density": self.world.get_weather().fog_density,
              "fog_distance": self.world.get_weather().fog_distance,
              "wetness": self.world.get_weather().wetness,
              "fog_falloff": self.world.get_weather().fog_falloff,
            },
        }
        
        covered_actor_ids = []
        for ego_id, ego_vehicle in self.scenario_instance.ego_vehicles.items():  
            # get meta data
            ego_vehicle = self.scenario_instance.ego_vehicles[ego_id]            
            ego_obs = self._recorder_carla_actor(ego_vehicle)
            ego_obs['config_id'] = ego_id
            scene_dict['egos'][ego_id] = ego_obs
            covered_actor_ids.append(ego_vehicle.id)
            
        # other actors
        config_other_actors = self.scenario_instance.other_actors
        actor_config_mapper = {actor.id: config_id for config_id, actor in config_other_actors.items() if actor is not None}
        
        # get vehicles
        other_vehicles = self.ctn_operator.get_world().get_actors().filter('*vehicle.*')
        for other_vehicle in other_vehicles:            
            if other_vehicle.id in covered_actor_ids:
                continue
            other_vehicle_info = self._recorder_carla_actor(other_vehicle)
            other_vehicle_info['config_id'] = actor_config_mapper.get(other_vehicle.id, None)
            scene_dict['other_actors']['vehicles'].append(other_vehicle_info)
            covered_actor_ids.append(other_vehicle.id)
            
        # get walkers
        other_walkers = self.ctn_operator.get_world().get_actors().filter('*walker*')
        for other_walker in other_walkers:
            if other_walker.id in covered_actor_ids:
                continue
            other_walker_info = self._recorder_carla_actor(other_walker)
            other_walker_info['config_id'] = actor_config_mapper.get(other_walker.id, None)
            scene_dict['other_actors']['walkers'].append(other_walker_info)
            covered_actor_ids.append(other_walker.id)

        # get traffic lights
        traffic_lights = self.ctn_operator.get_world().get_actors().filter("traffic.traffic_light*")
        for traffic_light in traffic_lights:
            if traffic_light.id in covered_actor_ids:
                continue
            traffic_light_info = self._recorder_carla_actor(traffic_light)
            traffic_light_info['config_id'] = actor_config_mapper.get(traffic_light.id, None)
            scene_dict['other_actors']['traffic_lights'].append(traffic_light_info)
            covered_actor_ids.append(traffic_light.id)
        
        # get traffic signs
        traffic_signs = self.ctn_operator.get_world().get_actors().filter("traffic.traffic_sign*")
        for traffic_sign in traffic_signs:
            if traffic_sign.id in covered_actor_ids:
                continue
            traffic_sign_info = self._recorder_carla_actor(traffic_sign)
            traffic_sign_info['config_id'] = actor_config_mapper.get(traffic_sign.id, None)
            scene_dict['other_actors']['traffic_signs'].append(traffic_sign_info)
            covered_actor_ids.append(traffic_sign.id)
        
        # get static props
        static_props_info = self._recorder_static_actor(self.ctn_operator.get_world(), actor_config_mapper)
        scene_dict['other_actors']['static_props'] = static_props_info
        
        # only save background in the first frame
        if len(self.recorder_observation) == 0:
            map_statics = self._recorder_map_static(self.ctn_operator.get_world())
            scene_dict['other_actors']['map_statics'] = map_statics
        
        self.recorder_observation.append(scene_dict)
        
    def _recorder_carla_actor(self, actor: carla.Actor):
        actor_info = {
            'actor_id': actor.id,
            'attributes': actor.attributes,
            'type_id': actor.type_id,
            'location': [
                actor.get_transform().location.x,
                actor.get_transform().location.y,
                actor.get_transform().location.z
            ],
            'rotation': [
                actor.get_transform().rotation.pitch,
                actor.get_transform().rotation.roll,
                actor.get_transform().rotation.yaw
            ],
            'velocity': [
                actor.get_velocity().x,
                actor.get_velocity().y,
                actor.get_velocity().z
            ],
            'acceleration': [
                actor.get_acceleration().x,
                actor.get_acceleration().y,
                actor.get_acceleration().z
            ],
            'angular_velocity': [
                actor.get_angular_velocity().x,
                actor.get_angular_velocity().y,
                actor.get_angular_velocity().z
            ],
        }
        
        if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
            # bounding box
            actor_info['bounding_box'] = {
                'location': [
                    actor.bounding_box.location.x,
                    actor.bounding_box.location.y,
                    actor.bounding_box.location.z
                ],
                'extent': [
                    actor.bounding_box.extent.x,
                    actor.bounding_box.extent.y,
                    actor.bounding_box.extent.z
                ],
                'rotation': [
                    actor.bounding_box.rotation.pitch,
                    actor.bounding_box.rotation.roll,
                    actor.bounding_box.rotation.yaw
                ],
            }
        
        # vehicle
        if isinstance(actor, carla.Vehicle):
            actor_info['category'] = 'vehicle'
            actor_info['control'] = {
               'throttle': actor.get_control().throttle,
               'steer': actor.get_control().steer,
               'brake': actor.get_control().brake,
               'reverse': actor.get_control().reverse,
               'hand_brake': actor.get_control().hand_brake,
               'manual_gear_shift': actor.get_control().manual_gear_shift,
               'gear': actor.get_control().gear,
            }
            actor_info['light_state'] = actor.get_light_state()
            actor_info['speed_limit'] = actor.get_speed_limit()
            actor_info['is_at_traffic_light'] = actor.is_at_traffic_light()
            # traffic light
            actor_traffic_light = actor.get_traffic_light()
            if actor_traffic_light:
                actor_info['traffic_light'] = {
                    'id': actor_traffic_light.id,
                    'location': [
                        actor_traffic_light.get_transform().location.x,
                        actor_traffic_light.get_transform().location.y,
                        actor_traffic_light.get_transform().location.z
                    ],
                    'state': str(actor_traffic_light.get_state()),
                    'trigger_volume': {
                        'location': [
                            actor_traffic_light.trigger_volume.location.x,
                            actor_traffic_light.trigger_volume.location.y,
                            actor_traffic_light.trigger_volume.location.z
                        ],
                        'extent': [
                            actor_traffic_light.trigger_volume.extent.x,
                            actor_traffic_light.trigger_volume.extent.y,
                            actor_traffic_light.trigger_volume.extent.z
                        ],
                        'rotation': [
                            actor_traffic_light.trigger_volume.rotation.pitch,
                            actor_traffic_light.trigger_volume.rotation.roll,
                            actor_traffic_light.trigger_volume.rotation.yaw
                        ],
                    },
                    'pole_index': actor_traffic_light.get_pole_index(),
                    'group_ids': [_tmp_actor.id for _tmp_actor in actor_traffic_light.get_group_traffic_lights()],
                    'get_elapsed_time': actor_traffic_light.get_elapsed_time(),
                    'green_time': actor_traffic_light.get_green_time(),
                    'yellow_time': actor_traffic_light.get_yellow_time(),
                    'red_time': actor_traffic_light.get_red_time()
                }
            else:
                actor_info['traffic_light'] = None
                
        elif isinstance(actor, carla.Walker):
            actor_info['category'] = 'walker'
            actor_info['control'] = {
                'direction': [
                    actor.get_control().direction.x,
                    actor.get_control().direction.y,
                    actor.get_control().direction.z
                ],
                'speed': actor.get_control().speed,
                'jump': actor.get_control().jump,
                'walk_on_spot': actor.get_control().walk_on_spot,
            }
            
        elif isinstance(actor, carla.TrafficLight):
            actor_info['category'] = 'traffic_light'
            actor_info['trigger_volume'] = {
                'location': [
                    actor.trigger_volume.location.x,
                    actor.trigger_volume.location.y,
                    actor.trigger_volume.location.z
                ],
                'extent': [
                    actor.trigger_volume.extent.x,
                    actor.trigger_volume.extent.y,
                    actor.trigger_volume.extent.z
                ],
                'rotation': [
                    actor.trigger_volume.rotation.pitch,
                    actor.trigger_volume.rotation.roll,
                    actor.trigger_volume.rotation.yaw
                ],
            }
            actor_info['state'] = str(actor.get_state())
            actor_info['pole_index'] = actor.get_pole_index()
            actor_info['group_ids'] = [_tmp_actor.id for _tmp_actor in actor.get_group_traffic_lights()]
            actor_info['get_elapsed_time'] = actor.get_elapsed_time()
            actor_info['green_time'] = actor.get_green_time()
            actor_info['yellow_time'] = actor.get_yellow_time()
            actor_info['red_time'] = actor.get_red_time()
        elif isinstance(actor, carla.TrafficSign) and (not isinstance(actor, carla.TrafficLight)):
            actor_info['category'] = 'traffic_sign'
            actor_info['trigger_volume'] = {
                'location': [
                    actor.trigger_volume.location.x,
                    actor.trigger_volume.location.y,
                    actor.trigger_volume.location.z
                ],
                'extent': [
                    actor.trigger_volume.extent.x,
                    actor.trigger_volume.extent.y,
                    actor.trigger_volume.extent.z
                ],
                'rotation': [
                    actor.trigger_volume.rotation.pitch,
                    actor.trigger_volume.rotation.roll,
                    actor.trigger_volume.rotation.yaw
                ],
            }
            
        return actor_info
    
    def _recorder_static_actor(self, world, config_actors):
        """
        # we only save required static objects to reduce the size of the observation file
        Collect static objects:
        - For static props (actors): return id + type_id + transform + matched bounding box
        - For map statics (non-actors): return label + bounding box only
        """
        static_props_info = []

        # 1. Collect level bounding boxes for matching
        static_bbs = world.get_level_bbs(carla.CityObjectLabel.Static)

        # 2. Collect static prop actors
        static_props = world.get_actors().filter("static.prop.*")
        for actor in static_props:
            
            if actor.id not in config_actors.keys():
                continue
            
            transform = actor.get_transform()

            # Try native bounding_box (>=0.9.12 supports this)
            bb = getattr(actor, "bounding_box", None)

            if bb is None or (bb.extent.x == 0 and bb.extent.y == 0 and bb.extent.z == 0):
                # Fallback: match nearest level_bb by location
                nearest_bb, min_dist = None, float("inf")
                for cand in static_bbs:
                    dist = transform.location.distance(cand.location)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_bb = cand
                bb = nearest_bb

            actor_info = {
                'config_id': config_actors.get(actor.id, None),
                'actor_id': actor.id,
                'attributes': actor.attributes,
                'type_id': actor.type_id,
                'category': 'static_prop',
                'location': [
                    actor.get_transform().location.x,
                    actor.get_transform().location.y,
                    actor.get_transform().location.z
                ],
                'rotation': [
                    actor.get_transform().rotation.pitch,
                    actor.get_transform().rotation.roll,
                    actor.get_transform().rotation.yaw
                ],
                'velocity': [
                    actor.get_velocity().x,
                    actor.get_velocity().y,
                    actor.get_velocity().z
                ],
                'acceleration': [
                    actor.get_acceleration().x,
                    actor.get_acceleration().y,
                    actor.get_acceleration().z
                ],
                'angular_velocity': [
                    actor.get_angular_velocity().x,
                    actor.get_angular_velocity().y,
                    actor.get_angular_velocity().z
                ],
                "bounding_box": {
                    "center": [bb.location.x, bb.location.y, bb.location.z],
                    "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                    "rotation": [bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll]
                } if bb else None
            }
            static_props_info.append(actor_info)

        return static_props_info
    
    def _recorder_map_static(self, world):
        """
        # we only save required static objects to reduce the size of the observation file
        Collect static objects:
        - For static props (actors): return id + type_id + transform + matched bounding box
        - For map statics (non-actors): return label + bounding box only
        """
        map_statics = []
        
        # 1. Collect level bounding boxes for matching
        static_bbs = world.get_level_bbs(carla.CityObjectLabel.Static)

        # 2. Collect static prop actors
        static_props = world.get_actors().filter("static.prop.*")
        for actor in static_props:
            transform = actor.get_transform()

            # Try native bounding_box (>=0.9.12 supports this)
            bb = getattr(actor, "bounding_box", None)

            if bb is None or (bb.extent.x == 0 and bb.extent.y == 0 and bb.extent.z == 0):
                # Fallback: match nearest level_bb by location
                nearest_bb, min_dist = None, float("inf")
                for cand in static_bbs:
                    dist = transform.location.distance(cand.location)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_bb = cand
                bb = nearest_bb

            actor_info = {
                'actor_id': actor.id,
                'attributes': actor.attributes,
                'type_id': actor.type_id,
                'category': 'static_prop',
                'location': [
                    actor.get_transform().location.x,
                    actor.get_transform().location.y,
                    actor.get_transform().location.z
                ],
                'rotation': [
                    actor.get_transform().rotation.pitch,
                    actor.get_transform().rotation.roll,
                    actor.get_transform().rotation.yaw
                ],
                'velocity': [
                    actor.get_velocity().x,
                    actor.get_velocity().y,
                    actor.get_velocity().z
                ],
                'acceleration': [
                    actor.get_acceleration().x,
                    actor.get_acceleration().y,
                    actor.get_acceleration().z
                ],
                'angular_velocity': [
                    actor.get_angular_velocity().x,
                    actor.get_angular_velocity().y,
                    actor.get_angular_velocity().z
                ],
                "bounding_box": {
                    "center": [bb.location.x, bb.location.y, bb.location.z],
                    "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                    "rotation": [bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll]
                } if bb else None
            }
            map_statics.append(actor_info)

        # # 3. Collect map-level statics (Buildings, Fences, Vegetation)
        labels = [
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Fences,
            carla.CityObjectLabel.Poles,
            carla.CityObjectLabel.Vegetation,
            carla.CityObjectLabel.Static
        ]
        
        for label in labels:
            for bb in world.get_level_bbs(label):
                map_statics.append({
                    "category": "map_static",
                    "label": str(label),
                    "bounding_box": {
                        "center": [bb.location.x, bb.location.y, bb.location.z],
                        "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                        "rotation": [bb.rotation.pitch, bb.rotation.yaw, bb.rotation.roll],
                    }
                })

        return map_statics

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        TODO: Modify this and check this
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'

        logger.debug(f"self.scenario.get_criteria(): {self.scenario_instance.get_criteria()}")
        ## added
        st_detail_overview = {}
        for criterion in self.scenario_instance.get_criteria():
            try:
                if criterion.test_status != "SUCCESS":
                    global_result = '\033[91m'+'FAILURE'+'\033[0m'
                
                if hasattr(criterion, 'st_detail'):
                    st_detail_overview[criterion.name] = criterion.st_detail
            except Exception as e:
                # skip non criteria nodes # as some nodes do not have test_status
                pass

        if self.scenario_instance.timeout_node and self.scenario_instance.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'

        overall_result = os.path.join(self.scenario_dir, 'result.txt')
        rp = ResultOutputProvider(self, global_result, stdout=True, filename=overall_result)
        rp.write()
        
        # we add a results of st_details
        for k, v in st_detail_overview.items():
            logger.info(f"Criterion {k} details: {v}")
        st_result_file = os.path.join(self.scenario_dir, 'result.json')
        # obtain st_results
        with open(st_result_file, 'w') as f:
            json.dump(st_detail_overview, f, indent=4)