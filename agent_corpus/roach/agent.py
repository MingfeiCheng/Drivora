import cv2
import os
import time
import carla
import numpy as np
import torch

from PIL import Image
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from .utils import transforms as trans_utils
from .criteria import run_stop_sign
from .obs_manager.birdview.chauffeurnet import ObsManager
from .utils.config_utils import load_entry_point
from .utils.traffic_light import TrafficLightHandler
from .planner import RoutePlanner

from agent_corpus.atomic.navigation.local_planner import RoadOption
from agent_corpus.atomic.base_agent import AutonomousAgent

def get_config_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'config_agent.yaml')
    return config_path

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])
    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)
    return result

def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))

def get_xyz(_):
    return np.array([_.x, _.y, _.z])

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1
    if abs(np.linalg.det(A)) < 1e-3:
        return False, None
    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1

def get_nearby_object(vehicle_position, actor_list, radius):
    nearby_objects = []
    for actor in actor_list:
        trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
        trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
        if trigger_box_global_pos.distance(vehicle_position) < radius:
            nearby_objects.append(actor)
    return nearby_objects

class RoachAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        
        self._render_dict = None
        self.supervision_dict = None
        cfg = OmegaConf.load(path_to_conf_file)
        cfg = OmegaConf.to_container(cfg)
        self.cfg = cfg

        self._ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['test']['weight'])
        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        self._policy_kwargs = cfg['policy']['kwargs']
        if self._ckpt is None:
            self._policy = None
        else:
            self._policy, self._train_cfg['kwargs'] = self._policy_class.load(self._ckpt)
            self._policy = self._policy.eval()

        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.prev_lidar = None ## The frequency of lidar is 10 Hz while the frequency of simulation is 20Hz -> In each frame, the sensor only returns half lidar points.
        self._active_traffic_light = None

    def _init(self):
        self._global_route = self._global_plan_world_coord # added
        
        # logger.debug("global route: {}".format(self._global_route))
        
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self._world = self.ctn_operator.get_world()
        self._map = self._world.get_map()

        # NOTE: this is changed
        self._ego_vehicle = self.carla_actor  #DataProvider.get_ego(self.id) # world.get_actor(self.id) is ok

        self._last_route_location = self._ego_vehicle.get_location()
        self._criteria_stop = run_stop_sign.RunStopSign(self._world) # todo: check this meaning

        self.birdview_obs_manager = ObsManager(self.cfg['obs_configs']['birdview'], self._criteria_stop) # save bev view information
        self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)

        self.navigation_idx = -1
        # for stop signs
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign
        TrafficLightHandler.reset(self._world)

        self.initialized = True

        logger.info("initialized")

    def _truncate_global_route_till_local_target(self, windows_size=5):
        """
        This function aims to: truncate global route from current location, find future waypoints
        :param windows_size: set the max comparison for saving time
        :return:
        """
        ev_location = self._ego_vehicle.get_location()
        closest_idx = 0
        for i in range(len(self._global_route)-1):
            if i > windows_size:
                break
            loc0 = self._global_route[i][0].location
            loc1 = self._global_route[i+1][0].location
            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z
            if dot_ve_wp > 0:
                closest_idx = i+1

        if closest_idx > 0:
            self._last_route_location = carla.Location(self._global_route[0][0].location)

        self._global_route = self._global_route[closest_idx:]

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps

    def sensors(self):
        roach_senors = [
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
                },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
                },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
                }
        ]
        
        return roach_senors

    def tick(self, input_data, timestamp):
        # process input_data -> return many attributes

        self._truncate_global_route_till_local_target()

        # obs_dict = {'rendered': image, 'masks': masks} masks (c, h, w)
        birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)

        control = self._ego_vehicle.get_control()
        throttle = np.array([control.throttle], dtype=np.float32)
        steer = np.array([control.steer], dtype=np.float32)
        brake = np.array([control.brake], dtype=np.float32)
        gear = np.array([control.gear], dtype=np.float32)

        ev_transform = self._ego_vehicle.get_transform()
        vel_w = self._ego_vehicle.get_velocity()
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)
        vel_xy = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)

        # todo: check this criteria
        self._criteria_stop.tick(self._ego_vehicle, timestamp)

        state_list = []
        state_list.append(throttle)
        state_list.append(steer)
        state_list.append(brake)
        state_list.append(gear)
        state_list.append(vel_xy)
        state = np.concatenate(state_list)
        obs_dict = {
            'state': state.astype(np.float32),
            'birdview': birdview_obs['masks'],
        }
        # Roach Input:
        # Road Mask - 0,
        # Route Mask - 1,
        # Lane_Mask (broken lane - 0.5) -2, 4 * vehicle - 3456, 4 * walker, 4 * traffic_light [-16, -11, -6, -1] 10hz
        bev_seg_label = birdview_obs['masks'].copy().astype(np.float32)/255.0

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        acceleration = input_data['imu'][1][:3]
        angular_velocity = input_data['imu'][1][3:6]
        target_gps, target_command = self.get_target_gps(input_data['gps'][1], compass)
        weather = self._weather_to_dict(self._world.get_weather())

        result = {
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'weather': weather,
            "acceleration":acceleration,
            "angular_velocity":angular_velocity,
            "bev_seg_label": bev_seg_label
        }
        next_wp, next_cmd = self._route_planner.run_step(self._get_position(result))
                    
        result['next_command'] = next_cmd.value
        result['x_target'] = next_wp[0]
        result['y_target'] = next_wp[1]
        # obs_dict: policy input
        return result, obs_dict, birdview_obs['rendered'], target_gps, target_command, None

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        
        # logger.debug(f"{self.id} Roach run step")

        # TODO: check this
        log_data = {
        }
        
        if not self.initialized:
            self._init()

        self.step += 1

        if self.step < 10:
            # wait 1s at the beginning
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            self.last_control = control
            self.prev_matrix = self._ego_vehicle.get_transform().get_matrix()
            return control, log_data

        if self.step % 2 != 0: # collect each 2 steps
            self.prev_matrix = self._ego_vehicle.get_transform().get_matrix()
            return self.last_control, log_data

        tick_data, policy_input, rendered, target_gps, target_command, _ = self.tick(input_data, timestamp)

        gps = self._get_position(tick_data)
        ## Roach forward
        actions, values, log_probs, mu, sigma, features, cnn_feature = self._policy.forward(policy_input, deterministic=True, clip_action=True)

        control = self.process_act(actions)

        ## Rules for emergency brake
        should_brake = self.collision_detect()
        if should_brake:
            control.steer = control.steer * 0.5
            control.throttle = 0.0
            control.brake = 1.0

        steer = control.steer
        control.steer = steer + 1e-2 * np.random.randn() ## Random noise for robustness
        self.last_control = control
        self.prev_matrix = self._ego_vehicle.get_transform().get_matrix()
        
        return control, log_data

    def collision_detect(self):
        actors = self._world.get_actors()
        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        return any(x is not None for x in [vehicle, walker])

    def _is_walker_hazard(self, walkers_list):
        z = self._ego_vehicle.get_location().z
        p1 = _numpy(self._ego_vehicle.get_location())
        v1 = 10.0 * _orientation(self._ego_vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))
            if s2 < 0.05:
                v2_hat *= s2
            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat
            collides, collision_point = get_collision(p1, v1, p2, v2)
            if collides:
                return walker
        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._ego_vehicle.get_location().z
        o1 = _orientation(self._ego_vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._ego_vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._ego_vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._ego_vehicle.id:
                continue
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat
            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue
            return target_vehicle
        return None

    def get_target_gps(self, gps, compass):
        # target gps
        def gps_to_location(gps):
            lat, lon, z = gps
            lat = float(lat)
            lon = float(lon)
            z = float(z)

            location = carla.Location(z=z)
            xy =  (gps[:2] - self._command_planner.mean) * self._command_planner.scale
            location.x = xy[0]
            location.y = -xy[1] ## Dose not matter, because no explicit judge left or right
            return location

        global_plan_gps = self._global_plan
        next_gps, _ = global_plan_gps[self.navigation_idx+1]
        next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
        next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
        loc_in_ev = trans_utils.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
            self.navigation_idx += 1

        self.navigation_idx = min(self.navigation_idx, len(global_plan_gps)-2)

        _, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
        gps_point, road_option_1 = global_plan_gps[self.navigation_idx+1]
        gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0

        return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)


    def process_act(self, action):
        acc = action[0][0]
        steer = action[0][1]
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control

    def _weather_to_dict(self, carla_weather):
        weather = {
            'cloudiness': carla_weather.cloudiness,
            'precipitation': carla_weather.precipitation,
            'precipitation_deposits': carla_weather.precipitation_deposits,
            'wind_intensity': carla_weather.wind_intensity,
            'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
            'sun_altitude_angle': carla_weather.sun_altitude_angle,
            'fog_density': carla_weather.fog_density,
            'fog_distance': carla_weather.fog_distance,
            'wetness': carla_weather.wetness,
            'fog_falloff': carla_weather.fog_falloff,
        }
        return weather

    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix