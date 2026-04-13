import math
import carla
import py_trees
import numpy as np  

from .atomic.base import Criterion, to_numpy
from tools.timer import GameTime
from scenario_runner.ctn_operator import CtnSimOperator

class OffRoadTest(Criterion):
    """
    Atomic containing a test to detect sidewalk invasions of a specific actor. This atomic can
    fail when actor has spent a specific time outside driving lanes (defined by OpenDRIVE).

    Args:
        actor (carla.Actor): CARLA actor to be used for this test
        duration (float): Time spent at sidewalks before the atomic fails.
            If terminate_on_failure isn't active, this is ignored.
        optional (bool): If True, the result is not considered for an overall pass/fail result
            when using the output argument
        terminate_on_failure (bool): If True, the atomic will fail when the duration condition has been met.
    """

    def __init__(
        self, 
        actor,
        ctn_operator: CtnSimOperator,
        duration=0, 
        terminate_on_failure=False, 
        name="OffRoadTest"
    ):
        """
        Construction with sensor setup
        """
        super().__init__(name, actor, False, terminate_on_failure)

        self.ctn_operator = ctn_operator
        
        self._world = self.ctn_operator.get_world()
        self._map = self.ctn_operator.get_map()
        
        self._onsidewalk_active = False
        self._outside_lane_active = False

        self._actor_location = self.actor.get_location()
        self._wrong_sidewalk_distance = 0
        self._wrong_outside_lane_distance = 0
        self._sidewalk_start_location = None
        self._outside_lane_start_location = None
        self._duration = duration
        self._prev_time = None
        self._time_outside_lanes = 0
        
        self.st_detail = {
            "occurred": False,
            "details": []
        }

    def update(self):
        """
        First, transforms the actor's current position as well as its four corners to their
        corresponding waypoints. Depending on their lane type, the actor will be considered to be
        outside (or inside) driving lanes.

        returns:
            py_trees.common.Status.FAILURE: when the actor has spent a given duration outside
                driving lanes and terminate_on_failure is active
            py_trees.common.Status.RUNNING: the rest of the time
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE

        # Some of the vehicle parameters
        current_tra = self.actor.get_transform()
        current_loc = current_tra.location
        current_wp = self._map.get_waypoint(current_loc, lane_type=carla.LaneType.Any)

        # Case 1) Car center is at a sidewalk
        if current_wp.lane_type == carla.LaneType.Sidewalk:
            if not self._onsidewalk_active:
                self._onsidewalk_active = True
                self._sidewalk_start_location = current_loc

        # Case 2) Not inside allowed zones (Driving and Parking)
        elif current_wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
            # Get the vertices of the vehicle
            heading_vec = current_tra.get_forward_vector()
            heading_vec.z = 0
            heading_vec = heading_vec / math.sqrt(math.pow(heading_vec.x, 2) + math.pow(heading_vec.y, 2))
            perpendicular_vec = carla.Vector3D(-heading_vec.y, heading_vec.x, 0)

            extent = self.actor.bounding_box.extent
            x_boundary_vector = heading_vec * extent.x
            y_boundary_vector = perpendicular_vec * extent.y

            bbox = [
                current_loc + carla.Location(x_boundary_vector - y_boundary_vector),
                current_loc + carla.Location(x_boundary_vector + y_boundary_vector),
                current_loc + carla.Location(-1 * x_boundary_vector - y_boundary_vector),
                current_loc + carla.Location(-1 * x_boundary_vector + y_boundary_vector),
            ]

            bbox_wp = [
                self._map.get_waypoint(bbox[0], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[1], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[2], lane_type=carla.LaneType.Any),
                self._map.get_waypoint(bbox[3], lane_type=carla.LaneType.Any),
            ]

            lane_type_list = [bbox_wp[0].lane_type, bbox_wp[1].lane_type, bbox_wp[2].lane_type, bbox_wp[3].lane_type]

            # Case 2.1) Not quite outside yet
            allowed_types = (carla.LaneType.Driving, carla.LaneType.Parking)
            if (
                bbox_wp[0].lane_type in allowed_types
                or bbox_wp[1].lane_type in allowed_types
                or bbox_wp[2].lane_type in allowed_types
                or bbox_wp[3].lane_type in allowed_types
            ):
                self._onsidewalk_active = False
                self._outside_lane_active = False

            # Case 2.2) At the mini Shoulders between Driving and Sidewalk
            elif carla.LaneType.Sidewalk in lane_type_list:
                if not self._onsidewalk_active:
                    self._onsidewalk_active = True
                    self._sidewalk_start_location = current_loc

            else:
                
                current_loc_np = to_numpy(current_loc)
                current_wp_np = to_numpy(current_wp.transform.location)
                distance_vehicle_wp = np.linalg.norm(current_loc_np - current_wp_np)
                # distance_vehicle_wp = current_loc.distance(current_wp.transform.location)

                # Case 2.3) Outside lane
                if distance_vehicle_wp >= current_wp.lane_width / 2:
                    if not self._outside_lane_active:
                        self._outside_lane_active = True
                        self._outside_lane_start_location = current_loc

                # Case 2.4) Very very edge case (but still inside driving lanes)
                else:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False

        # Case 3) Driving and Parking conditions
        else:
            # Check for false positives at junctions
            if current_wp.is_junction:
                distance_vehicle_wp = math.sqrt(
                    math.pow(current_wp.transform.location.x - current_loc.x, 2)
                    + math.pow(current_wp.transform.location.y - current_loc.y, 2)
                )

                if distance_vehicle_wp <= current_wp.lane_width / 2:
                    self._onsidewalk_active = False
                    self._outside_lane_active = False
                # Else, do nothing, the waypoint is too far to consider it a correct position
            else:
                self._onsidewalk_active = False
                self._outside_lane_active = False

        # Counts the time offroad
        if self._onsidewalk_active or self._outside_lane_active:
            if self._prev_time is None:
                self._prev_time = GameTime.get_time()
            else:
                curr_time = GameTime.get_time()
                self._time_outside_lanes += curr_time - self._prev_time
                self._prev_time = curr_time
        else:
            self._prev_time = None

        if self._time_outside_lanes > self._duration:
            self.test_status = "FAILURE"

        # Update the distances
        actor_current_location = self.actor.get_location()
        distance_vector = actor_current_location - self._actor_location
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if distance >= 0.02:  # Used to avoid micro-changes adding to considerable sums
            self._actor_location = actor_current_location

            if self._onsidewalk_active:
                self._wrong_sidewalk_distance += distance
            elif self._outside_lane_active:
                # Only add if car is outside the lane but ISN'T in a junction
                self._wrong_outside_lane_distance += distance

        # Register the sidewalk event
        if not self._onsidewalk_active and self._wrong_sidewalk_distance > 0:
            self.actual_value += 1
            
            self.st_detail["occurred"] = True
            self.st_detail["details"].append({
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "type": "onsidewalk",
                "start_location": {
                    "x": self._sidewalk_start_location.x,
                    "y": self._sidewalk_start_location.y,
                    "z": self._sidewalk_start_location.z
                },
                "distance": round(self._wrong_sidewalk_distance, 3)
            })
            
            self._onsidewalk_active = False
            self._wrong_sidewalk_distance = 0

        # Register the outside of a lane event
        if not self._outside_lane_active and self._wrong_outside_lane_distance > 0:
            self.actual_value += 1
            
            self.st_detail["occurred"] = True
            self.st_detail["details"].append({
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "type": "outsidelane",
                "start_location": {
                    "x": self._outside_lane_start_location.x,
                    "y": self._outside_lane_start_location.y,
                    "z": self._outside_lane_start_location.z
                },
                "distance": round(self._wrong_outside_lane_distance, 3)
            })
            self._outside_lane_active = False
            self._wrong_outside_lane_distance = 0

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        # If currently at a sidewalk, register the event
        if self._onsidewalk_active:
            self.actual_value += 1

            self.st_detail["occurred"] = True
            self.st_detail["details"].append({
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "type": "onsidewalk",
                "start_location": {
                    "x": self._sidewalk_start_location.x,
                    "y": self._sidewalk_start_location.y,
                    "z": self._sidewalk_start_location.z
                },
                "distance": round(self._wrong_sidewalk_distance, 3)
            })
            
            
            self._onsidewalk_active = False
            self._wrong_sidewalk_distance = 0

        # If currently outside of our lane, register the event
        if self._outside_lane_active:
            self.actual_value += 1

            self.st_detail["occurred"] = True
            self.st_detail["details"].append({
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "type": "outsidelane",
                "start_location": {
                    "x": self._outside_lane_start_location.x,
                    "y": self._outside_lane_start_location.y,
                    "z": self._outside_lane_start_location.z
                },
                "distance": round(self._wrong_outside_lane_distance, 3)
            })
            self._outside_lane_active = False
            self._wrong_outside_lane_distance = 0

        super().terminate(new_status)
