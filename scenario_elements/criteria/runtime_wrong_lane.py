import math
import py_trees
import numpy as np

from .atomic.base import Criterion
from tools.timer import GameTime
from scenario_runner.ctn_operator import CtnSimOperator

class WrongLaneTest(Criterion):
    """
    This class contains an atomic test to detect invasions to wrong direction lanes.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    MAX_ALLOWED_ANGLE = 120.0
    MAX_WAYPOINT_ANGLE = 150.0

    def __init__(
        self, 
        actor, 
        ctn_operator: CtnSimOperator,
        name="WrongLaneTest", 
        terminate_on_failure=False
    ):
        """
        Construction with sensor setup
        """
        super().__init__(name, actor, False, terminate_on_failure)

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.ctn_operator = ctn_operator
        
        self._world = self.ctn_operator.get_world()
        self._map = self.ctn_operator.get_map()
        
        self._last_lane_id = None
        self._last_road_id = None

        self._in_lane = True
        self._wrong_distance = 0
        self._actor_location = self.actor.get_location()
        self._previous_lane_waypoint = self._map.get_waypoint(self.actor.get_location())
        self._wrong_lane_start_location = None
        
        self.st_detail = {
            "occurred": False,
            "details": []
        }

    def update(self):
        """
        Check lane invasion count
        """

        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        lane_waypoint = self._map.get_waypoint(self.actor.get_location())
        current_lane_id = lane_waypoint.lane_id
        current_road_id = lane_waypoint.road_id

        if (
            self._last_road_id != current_road_id or self._last_lane_id != current_lane_id
        ) and not lane_waypoint.is_junction:
            next_waypoint = lane_waypoint.next(2.0)[0]

            if not next_waypoint:
                return new_status

            # The waypoint route direction can be considered continuous.
            # Therefore just check for a big gap in waypoint directions.
            previous_lane_direction = self._previous_lane_waypoint.transform.get_forward_vector()
            current_lane_direction = lane_waypoint.transform.get_forward_vector()

            p_lane_vector = np.array([previous_lane_direction.x, previous_lane_direction.y])
            c_lane_vector = np.array([current_lane_direction.x, current_lane_direction.y])

            waypoint_angle = math.degrees(
                math.acos(
                    np.clip(
                        np.dot(p_lane_vector, c_lane_vector)
                        / (np.linalg.norm(p_lane_vector) * np.linalg.norm(c_lane_vector)),
                        -1.0,
                        1.0,
                    )
                )
            )

            if waypoint_angle > self.MAX_WAYPOINT_ANGLE and self._in_lane:
                self.test_status = "FAILURE"
                self._in_lane = False
                self.actual_value += 1
                self._wrong_lane_start_location = self._actor_location

            else:
                # Reset variables
                self._in_lane = True

            # Continuity is broken after a junction so check vehicle-lane angle instead
            if self._previous_lane_waypoint.is_junction:
                vector_wp = np.array(
                    [
                        next_waypoint.transform.location.x - lane_waypoint.transform.location.x,
                        next_waypoint.transform.location.y - lane_waypoint.transform.location.y,
                    ]
                )

                vector_actor = np.array(
                    [
                        math.cos(math.radians(self.actor.get_transform().rotation.yaw)),
                        math.sin(math.radians(self.actor.get_transform().rotation.yaw)),
                    ]
                )

                vehicle_lane_angle = math.degrees(
                    math.acos(np.clip(np.dot(vector_actor, vector_wp) / (np.linalg.norm(vector_wp)), -1.0, 1.0))
                )

                if vehicle_lane_angle > self.MAX_ALLOWED_ANGLE:
                    self.test_status = "FAILURE"
                    self._in_lane = False
                    self.actual_value += 1
                    self._wrong_lane_start_location = self.actor.get_location()

        # Keep adding "meters" to the counter
        distance_vector = self.actor.get_location() - self._actor_location
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if distance >= 0.02:  # Used to avoid micro-changes adding add to considerable sums
            self._actor_location = self.actor.get_location()

            if not self._in_lane and not lane_waypoint.is_junction:
                self._wrong_distance += distance

        # Register the event
        if self._in_lane and self._wrong_distance > 0:
            self.st_detail["occurred"] = True
            self.st_detail["details"].append({
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "wrong_lane_start_location": {
                    "x": self._wrong_lane_start_location.x,
                    "y": self._wrong_lane_start_location.y,
                    "z": self._wrong_lane_start_location.z
                },
                "wrong_distance": self._wrong_distance,
                "current_road_id": current_road_id,
                "current_lane_id":current_lane_id
            })
            self._wrong_distance = 0

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._previous_lane_waypoint = lane_waypoint

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        If there is currently an event running, it is registered
        """
        if not self._in_lane:
            lane_waypoint = self._map.get_waypoint(self.actor.get_location())
            current_lane_id = lane_waypoint.lane_id
            current_road_id = lane_waypoint.road_id
            
            self.st_detail["occurred"] = True
            self.st_detail["details"].append({
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "wrong_lane_start_location": {
                    "x": self._wrong_lane_start_location.x,
                    "y": self._wrong_lane_start_location.y,
                    "z": self._wrong_lane_start_location.z
                },
                "wrong_distance": self._wrong_distance,
                "current_road_id": current_road_id,
                "current_lane_id":current_lane_id
            })
            self._wrong_distance = 0
            self._in_lane = True

        super().terminate(new_status)
