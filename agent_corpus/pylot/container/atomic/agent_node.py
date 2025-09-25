from __future__ import print_function

import carla

import json

from .timer import GameTime
from .sensor_interface import SensorInterface
from .route_manipulation import downsample_route
from .data_provider import DataProvider

class AgentNode(object):
    """
    Base classes for the ADS under test,
    To support multiple ADSs, we add extra API interfaces:
    (1) insert identifier: insert the identifier of the agent
    (2) get_observation(): get the observation of the agent at the current step
    (3) set_extra_commands(): set extra commands to the agent, i.e., commands from others
    """
    def __init__(
            self
    ):
        self.id = None
        self.carla_actor = None
        
         #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()
    
    def bid_actor(self, actor_id):
        """
        Bid the actor id in the carla world
        NOTE: MUST be called before setup() to set the actor id
        """
        self.id = actor_id
        self.carla_actor = DataProvider.get_world().get_actor(actor_id) # carla actors
        
    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        log_data = {}
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control, log_data

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        # how to access the host timer?
        
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        
        timestamp = GameTime.get_time()

        control, log_data = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control, log_data

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        gps_route, route
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 1)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        
    def apply_control(self, control):
        """
        Apply the control to the carla actor
        """
        if self.carla_actor is not None:
            self.carla_actor.apply_control(control)
        else:
            raise RuntimeError("Carla actor is not set. Please call bid_actor() before apply_control().")