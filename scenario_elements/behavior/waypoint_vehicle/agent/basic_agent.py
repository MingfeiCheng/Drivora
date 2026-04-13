# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
import numpy as np

from typing import List
from enum import Enum

from .navigation.misc import get_speed, is_within_distance, get_trafficlight_trigger_location

from .navigation.local_planner import LocalPlanner
from .navigation.global_route_planner import GlobalRoutePlanner

from scenario_elements.config import Waypoint

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])
    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)
    return result

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

STATIC_LIST = [
    "static.prop.streetbarrier",
    "static.prop.constructioncone",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.trafficwarning"
]

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, target_speed=20, opt_dict={}):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 10.0  # meters
        self._max_brake = 0.5

        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_brake' in opt_dict:
            self._max_steering = opt_dict['max_brake']

        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict)
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        
        route_trace_dict = []
        #  {x, y, z, pitch, roll, yaw, speed, road_option}
        for waypoint in route_trace:
            route_trace_dict.append({
                'x': waypoint[0].transform.location.x,
                'y': waypoint[0].transform.location.y,
                'z': waypoint[0].transform.location.z,
                'pitch': waypoint[0].transform.rotation.pitch,
                'roll': waypoint[0].transform.rotation.roll,
                'yaw': waypoint[0].transform.rotation.yaw,
                'speed': self._target_speed if self._target_speed else 15.0,
                'road_option': waypoint[1]
            })
        
        # self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)
        self._local_planner.set_global_plan(route_trace_dict, clean_queue=clean_queue)

    def set_global_plan(self, plan: List[Waypoint], stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.
        our speed in waypoint is m/s

            :param plan:  List of route waypoints to be followed by the agent.
                including, x, y, z, pitch, roll, yaw, speed, road_option
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def is_task_finished(self):
        """
        Check if the agent has finished its task.
        This is used to check if the agent has reached its destination
        """
        return self._local_planner.is_task_finished()
    
    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self, debug=False):
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        
        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True
            
        # NOTE: we add static here
        for static_name in STATIC_LIST:
            static_list = actor_list.filter(f"*{static_name}*")
            affected_by_static, _ = self._vehicle_obstacle_detected(static_list, max_vehicle_distance)
            if affected_by_static:
                hazard_detected = True
                break

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        third_part_collision_detector = self.collision_detect()
        if third_part_collision_detector:
            hazard_detected = True
            
        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location)
            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if not next_wpt:
                    continue
                if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id:
                    continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_vehicle.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [0, 90]):
                return (True, target_vehicle)
        return (False, None)
    
    def _static_obstacle_detected(self, static_list=None, max_distance=None):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None)
        
        if not static_list:
            return (False, None)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        for target_static in static_list:
            target_transform = target_static.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location)
            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if not next_wpt:
                    continue
                if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id:
                    continue
            
            dist = np.linalg.norm(_numpy(target_transform.location) - _numpy(self._vehicle.get_location()))
            if dist < max_distance:
                return (True, target_static)
            
        return (False, None)
    
    def collision_detect(self):
        actors = self._world.get_actors()
        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))
        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0
        return any(x is not None for x in [vehicle, walker])
        # return any(x is not None for x in [vehicle]) # we ignore pedestrians for now, as we care about vehicle collisions mostly

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

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
        z = self._vehicle.get_location().z
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(10.0, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))) # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat
        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
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

            if angle_between_heading > 60.0 and not (angle_to_car < 20 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue
            return target_vehicle
        return None