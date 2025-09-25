import carla

from .data_provider import DataProvider
from .sensor_interface import OpenDriveMapReader, SpeedometerReader, CallBack
from loguru import logger

class AgentWrapper(object):
    # This class wraps the agent and its sensors
    """
    This class wraps the agent and its sensors
    TO further support Multiple agents, we need to provide more APIs to:
    (1) Set id from agent instance
    (2) Obtain the agent's observation
    (3) Accept the control or decisions from others
    """
    

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent
        self._sensors_list = []

    def __call__(self, *args, **kwargs):
        """
        Pass the call directly to the agent
        """
        return self._agent()
    
    def setup_route(self, plan_gps, plan_world_coord):
        """
        Start the agent with a global plan
        :param plan_gps: the global plan in GPS coordinates
        :param plan_world_coord: the global plan in world coordinates
        """
        # add converter
        
        self._agent.set_global_plan(plan_gps, plan_world_coord)

    def setup_sensors(self):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        vehicle = self._agent.carla_actor
        
        bp_library = DataProvider.get_world().get_blueprint_library()
        logger.info("Obtained blueprint library for sensors")
        logger.info(f"Spawning sensors number: {len(self._agent.sensors())} sensor list length: {len(self._sensors_list)}")
        
        for sensor_spec in self._agent.sensors():
            # These are the pseudosensors (not spawned)
            if sensor_spec['type'].startswith('sensor.opendrive_map'):
                # The HDMap pseudo sensor is created directly here
                sensor = OpenDriveMapReader(vehicle, sensor_spec['reading_frequency'])
            elif sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = DataProvider.get_world().get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(vehicle, frame_rate)
            # These are the sensors spawned on the carla world
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera.semantic_segmentation'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.camera.depth'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['width']))
                    bp.set_attribute('image_size_y', str(sensor_spec['height']))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))

                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar.ray_cast_semantic'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 for old lidar models
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(600000))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0.45))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])

                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    # bp.set_attribute('noise_alt_stddev', str(0.000005))
                    # bp.set_attribute('noise_lat_stddev', str(0.000005))
                    # bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation()

                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                    sensor_location = carla.Location(x=sensor_spec['x'],
                                                     y=sensor_spec['y'],
                                                     z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                     roll=sensor_spec['roll'],
                                                     yaw=sensor_spec['yaw'])
                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = DataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
                logger.info(f"Spawned sensor {sensor_spec['id']} of type {sensor_spec['type']} at location {sensor_location} with rotation {sensor_rotation}")
                
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._agent.sensor_interface))
            self._sensors_list.append(sensor)

        logger.info(f"Spawned {len(self._sensors_list)} sensors for agent {self._agent.id}")
        # Tick once to spawn the sensors
        if DataProvider.is_sync_mode():
            DataProvider.get_world().tick()
        else:
            DataProvider.get_world().wait_for_tick()
        logger.info("All sensors are set up successfully.")

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        logger.info("Destroying sensors!")
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []
        logger.info(f"sensor list length: {len(self._sensors_list)}")
        
        self._agent.destroy()