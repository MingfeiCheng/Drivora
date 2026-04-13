import math
import carla
import py_trees
import numpy as np

from .atomic.base import Criterion, to_numpy
from tools.timer import GameTime
from scenario_runner.ctn_operator import CtnSimOperator

class RunningStopTest(Criterion):
    """
    Check if an actor is running a stop sign

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    PROXIMITY_THRESHOLD = 4.0  # Stops closer than this distance will be detected [m]
    SPEED_THRESHOLD = 0.1  # Minimum speed to consider the actor has stopped [m/s]
    WAYPOINT_STEP = 0.5  # m

    def __init__(
        self, 
        actor, 
        ctn_operator: CtnSimOperator,
        name="RunningStopTest", 
        terminate_on_failure=False
    ):
        """ """
        super().__init__(name, actor, False, terminate_on_failure=terminate_on_failure)
        self.ctn_operator = ctn_operator
        
        self._world = self.ctn_operator.get_world()
        self._map = self.ctn_operator.get_map()
        
        self._list_stop_signs = []
        self._target_stop_sign = None
        self._stop_completed = False

        self._last_failed_stop = None

        all_actors = self._world.get_actors()
        for _actor in all_actors:  # pylint: disable=not-an-iterable
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

        self.st_detail = {
            "occurred": False,
            "details": []
        }
        
    def point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.2):
        """Checks whether or not a point is inside a bounding box."""

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad  # pylint: disable=chained-comparison

    def is_actor_affected_by_stop(self, wp_list, stop):
        """
        Check if the given actor is affected by the stop.
        Without using waypoints, a stop might not be detected if the actor is moving at the lane edge.
        """
        # Quick distance test
        stop_location = stop.get_transform().transform(stop.trigger_volume.location)
        actor_location = wp_list[0].transform.location
        
        stop_location_np = to_numpy(stop_location)
        actor_location_np = to_numpy(actor_location)
        dist = np.linalg.norm(stop_location_np - actor_location_np)
        
        if dist > self.PROXIMITY_THRESHOLD:
            return False

        # Check if the any of the actor wps is inside the stop's bounding box.
        # Using more than one waypoint removes issues with small trigger volumes and backwards movement
        stop_extent = stop.trigger_volume.extent
        for actor_wp in wp_list:
            if self.point_inside_boundingbox(actor_wp.transform.location, stop_location, stop_extent):
                return True

        return False

    def _scan_for_stop_sign(self, actor_transform, wp_list):
        """
        Check the stop signs to see if any of them affect the actor.
        Ignore all checks when going backwards or through an opposite direction
        """

        actor_direction = to_numpy(actor_transform.get_forward_vector())

        actor_velocity = to_numpy(self.actor.get_velocity())
        if actor_velocity.dot(actor_direction) < -0.17:  # 100º, just in case
            return None

        # Ignore all when going in the opposite direction
        lane_direction = to_numpy(wp_list[0].transform.get_forward_vector())
        if actor_direction.dot(lane_direction) < -0.17:  # 100º, just in case
            return None

        for stop in self._list_stop_signs:
            if self.is_actor_affected_by_stop(wp_list, stop):
                return stop
        return None

    def _get_waypoints(self, actor):
        """Returns a list of waypoints starting from the ego location and a set amount forward"""
        wp_list = []
        steps = int(self.PROXIMITY_THRESHOLD / self.WAYPOINT_STEP)

        # Add the actor location
        wp = self._map.get_waypoint(actor.get_location())
        wp_list.append(wp)

        # And its forward waypoints
        next_wp = wp
        for _ in range(steps):
            next_wps = next_wp.next(self.WAYPOINT_STEP)
            if not next_wps:
                break
            next_wp = next_wps[0]
            wp_list.append(next_wp)

        return wp_list

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        actor_transform = self.actor.get_transform()
        check_wps = self._get_waypoints(self.actor)

        if not self._target_stop_sign:
            self._target_stop_sign = self._scan_for_stop_sign(actor_transform, check_wps)
            return new_status

        if not self._stop_completed:
            current_vel =self.actor.get_velocity()
            current_speed = math.sqrt(current_vel.x**2 + current_vel.y**2 + current_vel.z**2)
            if current_speed < self.SPEED_THRESHOLD:
                self._stop_completed = True

        if not self.is_actor_affected_by_stop(check_wps, self._target_stop_sign):
            if not self._stop_completed and self._last_failed_stop != self._target_stop_sign.id:
                # did we stop?
                self.actual_value += 1
                self.test_status = "FAILURE"
                stop_location = self._target_stop_sign.get_transform().location
                
                
                self.st_detail["occurred"] = True
                self.st_detail["details"].append({
                    "id": self.actor.id,
                    "frame": GameTime.get_frame(),
                    "timestamp": GameTime.get_time(),
                    "other_id": self._target_stop_sign.id,
                    "location": {
                        "x": stop_location.x,
                        "y": stop_location.y,
                        "z": stop_location.z
                    }
                })

                self._last_failed_stop = self._target_stop_sign.id

            # Reset state
            self._target_stop_sign = None
            self._stop_completed = False

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status