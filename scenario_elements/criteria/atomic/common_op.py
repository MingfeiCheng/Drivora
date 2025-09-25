import math
import carla
import shapely
import py_trees
import numpy as np

from tools.timer import GameTime

from .base import Criterion, to_numpy
from .traffic_events import TrafficEvent, TrafficEventType
from scenario_runner.ctn_operator import CtnSimOperator # some needs the carla 

class SimulationTimeCondition(py_trees.behaviour.Behaviour):

    """
    This class contains an atomic simulation time condition behavior.
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.

    Returns, if the provided success_rule (greaterThan, lessThan, equalTo)
    was successfully evaluated
    """

    def __init__(self, timeout, success_rule="greaterThan", name="SimulationTimeCondition"):
        """
        Setup timeout
        """
        super(SimulationTimeCondition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._timeout_value = timeout
        self._start_time = 0.0
        self._success_rule = success_rule
        self._ops = {"greaterThan": (lambda x, y: x > y),
                     "equalTo": (lambda x, y: x == y),
                     "lessThan": (lambda x, y: x < y)}

    def initialise(self):
        """
        Set start_time to current GameTime
        """
        self._start_time = GameTime.get_time()
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def update(self):
        """
        Get current game time, and compare it to the timeout value
        Upon successfully comparison using the provided success_rule operator,
        the status changes to SUCCESS
        """

        elapsed_time = GameTime.get_time() - self._start_time

        if not self._ops[self._success_rule](elapsed_time, self._timeout_value):
            new_status = py_trees.common.Status.RUNNING
        else:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

class TimeOut(SimulationTimeCondition):

    """
    This class contains an atomic timeout behavior.
    It uses the CARLA game time, not the system time which is used by
    the py_trees timer.
    """

    def __init__(self, timeout, name="TimeOut"):
        """
        Setup timeout
        """
        super(TimeOut, self).__init__(timeout, name=name)
        self.timeout = False

    def update(self):
        """
        Upon reaching the timeout value the status changes to SUCCESS
        """

        new_status = super(TimeOut, self).update()

        if new_status == py_trees.common.Status.SUCCESS:
            self.timeout = True
        
        if self.timeout:
            new_status = py_trees.common.Status.FAILURE

        return new_status

class CollisionTest(Criterion):

    """
    This class contains an atomic test for collisions.

    Args:
    - actor (carla.Actor): CARLA actor to be used for this test
    - other_actor (carla.Actor): only collisions with this actor will be registered
    - other_actor_type (str): only collisions with actors including this type_id will count.
        Additionally, the "miscellaneous" tag can also be used to include all static objects in the scene
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    COLLISION_RADIUS = 5  # Two collisions that happen within this distance count as one
    MAX_ID_TIME = 5  # Two collisions with the same id that happen within this time count as one
    EPSILON = 0.1  # Collisions at lower this speed won't be counted as the actor's fault

    def __init__(self, actor, ctn_operator: CtnSimOperator, other_actor=None, other_actor_type=None,
                 optional=False, terminate_on_failure=False, name="CollisionTest"):
        """
        Construction with sensor setup
        """
        self.ctn_operator = ctn_operator
        
        super(CollisionTest, self).__init__(name, actor, optional, terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._other_actor_type = other_actor_type

        # Attributes to store the last collisions's data
        self._collision_sensor = None
        self._collision_id = None
        self._collision_time = None
        self._collision_location = None
        

    def initialise(self):
        """
        Creates the sensor and callback"""
        world = self.ctn_operator.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._collision_sensor.listen(lambda event: self._count_collisions(event))
        super(CollisionTest, self).initialise()

    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        actor_location = self.actor.get_location()

        # Check if the last collision can be ignored
        if self._collision_location:
            distance_vector = actor_location - self._collision_location
            # if distance_vector.length() > self.COLLISION_RADIUS:
            if np.linalg.norm(to_numpy(distance_vector)) > self.COLLISION_RADIUS:
                self._collision_location = None
        if self._collision_id:
            elapsed_time = GameTime.get_time() - self._collision_time
            if elapsed_time > self.MAX_ID_TIME:
                self._collision_id = None

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
        self._collision_sensor = None
        super(CollisionTest, self).terminate(new_status)

    def _count_collisions(self, event):     # pylint: disable=too-many-return-statements
        """Update collision count"""
        actor_location = self.actor.get_location()

        # Check if the care about the other actor
        if self._other_actor and self._other_actor.id != event.other_actor.id:
            return

        if self._other_actor_type:
            if self._other_actor_type == "miscellaneous":  # Special OpenScenario case
                if "traffic" not in event.other_actor.type_id and "static" not in event.other_actor.type_id:
                    return
            elif self._other_actor_type not in event.other_actor.type_id:
                    return

        # To avoid multiple counts of the same collision, filter some of them.
        if self._collision_id == event.other_actor.id:
            return
        if self._collision_location:
            distance_vector = actor_location - self._collision_location
            if distance_vector.length() <= self.COLLISION_RADIUS:
                return

        # If the actor speed is 0, the collision isn't its fault
        velocity_vec = self.actor.get_velocity()
        if np.linalg.norm(to_numpy(velocity_vec)) < self.EPSILON:
            return

        # The collision is valid, save the data
        self.test_status = "FAILURE"
        self.actual_value += 1

        self._collision_time = GameTime.get_time()
        self._collision_location = actor_location
        if event.other_actor.id != 0: # Number 0: static objects -> ignore it
            self._collision_id = event.other_actor.id

        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) \
                and 'sidewalk' not in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN
        else:
            return

        collision_event = TrafficEvent(event_type=actor_type, frame=GameTime.get_frame())
        collision_event.set_dict({'other_actor': event.other_actor, 'location': actor_location})
        collision_event.set_message(
            "Agent collided against object with type={} and id={} at (x={}, y={}, z={})".format(
                event.other_actor.type_id,
                event.other_actor.id,
                round(actor_location.x, 3),
                round(actor_location.y, 3),
                round(actor_location.z, 3)))
        self.events.append(collision_event)
        
        
class ActorBlockedTest(Criterion):

    """
    This test will fail if the actor has had its linear velocity lower than a specific value for
    a specific amount of time
    Important parameters:
    - actor: CARLA actor to be used for this test
    - min_speed: speed required [m/s]
    - max_time: Maximum time (in seconds) the actor can remain under the speed threshold
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    def __init__(self, actor, min_speed, max_time, name="ActorBlockedTest", optional=False, terminate_on_failure=False):
        """
        Class constructor
        """
        super().__init__(name, actor, optional, terminate_on_failure)
        self._min_speed = min_speed
        self._max_time = max_time
        self._time_last_valid_state = None
        self._active = True
        self.units = None  # We care about whether or not it fails, no units attached
        self.success_value = self._max_time
        
    def update(self):
        """
        Check if the actor speed is above the min_speed
        """
        new_status = py_trees.common.Status.RUNNING

        # Deactivate/Activate checking by blackboard message
        active = py_trees.blackboard.Blackboard().get('AC_SwitchActorBlockedTest')
        if active is not None:
            self._active = active
            self._time_last_valid_state = GameTime.get_time()
            py_trees.blackboard.Blackboard().set("AC_SwitchActorBlockedTest", None, overwrite=True)

        if self._active:
            actor_velocity_vec = self.actor.get_velocity()
            linear_speed = np.linalg.norm(to_numpy(actor_velocity_vec))
            if linear_speed is not None:
                if linear_speed < self._min_speed and self._time_last_valid_state:
                    delta_time = GameTime.get_time() - self._time_last_valid_state
                    if delta_time > self.actual_value:
                        self.actual_value = delta_time
                    if delta_time > self._max_time:
                        # The actor has been "blocked" for too long, save the data
                        self.test_status = "FAILURE"

                        vehicle_location = self.actor.get_location() # CarlaDataProvider.get_location(self.actor)
                        event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED, frame=GameTime.get_frame())
                        event.set_message('Agent got blocked at (x={}, y={}, z={})'.format(
                            round(vehicle_location.x, 3),
                            round(vehicle_location.y, 3),
                            round(vehicle_location.z, 3))
                        )
                        event.set_dict({'location': vehicle_location})
                        self.events.append(event)
                else:
                    self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status
    
class RouteCompletionTest(Criterion):

    """
    Check at which stage of the route is the actor at each tick

    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    WINDOWS_SIZE = 2

    # Thresholds to return that a route has been completed
    DISTANCE_THRESHOLD = 10.0  # meters
    PERCENTAGE_THRESHOLD = 99  # %

    def __init__(self, actor, route, ctn_operator: CtnSimOperator, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        self.ctn_operator = ctn_operator
        super(RouteCompletionTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.units = "%"
        self.success_value = 100
        self._route = route
        self._map = self.ctn_operator.get_map()

        self._index = 0
        self._route_length = len(self._route)
        self._route_transforms, _ = zip(*self._route)
        self._route_accum_perc = self._get_acummulated_percentages()

        self.target_location = self._route_transforms[-1].location

        self._traffic_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETION, frame=0)
        self._traffic_event.set_dict({'route_completed': self.actual_value})
        self._traffic_event.set_message("Agent has completed {} of the route".format(self.actual_value))
        self.events.append(self._traffic_event)

    def _get_acummulated_percentages(self):
        """Gets the accumulated percentage of each of the route transforms"""
        accum_meters = []
        prev_loc = self._route_transforms[0].location
        for i, tran in enumerate(self._route_transforms):
            d = tran.location.distance(prev_loc)
            new_d = 0 if i == 0 else accum_meters[i - 1]

            accum_meters.append(d + new_d)
            prev_loc = tran.location

        max_dist = accum_meters[-1]
        return [x / max_dist * 100 for x in accum_meters]

    def update(self):
        """
        Check if the actor location is within trigger region
        """
        new_status = py_trees.common.Status.RUNNING

        location = self.actor.get_location()
        if location is None:
            return new_status

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        elif self.test_status in ('RUNNING', 'INIT'):

            for index in range(self._index, min(self._index + self.WINDOWS_SIZE + 1, self._route_length)):
                # Get the dot product to know if it has passed this location
                route_transform = self._route_transforms[index]
                route_location = route_transform.location
                wp_dir = to_numpy(route_transform.get_forward_vector())      # Waypoint's forward vector
                wp_veh = to_numpy(location - route_location)                   # vector route - vehicle

                if wp_veh.dot(wp_dir) > 0:
                    self._index = index
                    self.actual_value = self._route_accum_perc[self._index]

            self.actual_value = round(self.actual_value, 2)
            self._traffic_event.set_dict({'route_completed': self.actual_value})
            self._traffic_event.set_message("Agent has completed {} of the route".format(self.actual_value))

            if self.actual_value > self.PERCENTAGE_THRESHOLD \
                    and location.distance(self.target_location) < self.DISTANCE_THRESHOLD:
                self.test_status = "SUCCESS"
                self.actual_value = 100

        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """
        Set test status to failure if not successful and terminate
        """
        self.actual_value = round(self.actual_value, 2)

        self._traffic_event.set_dict({'route_completed': self.actual_value})
        self._traffic_event.set_message("Agent has completed {} of the route".format(self.actual_value))

        if self.test_status == "INIT":
            self.test_status = "FAILURE"
        super(RouteCompletionTest, self).terminate(new_status)
        
class RunningStopTest(Criterion):

    """
    Check if an actor is running a stop sign

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    PROXIMITY_THRESHOLD = 4.0  # Stops closer than this distance will be detected [m]
    SPEED_THRESHOLD = 0.1 # Minimum speed to consider the actor has stopped [m/s]
    WAYPOINT_STEP = 0.5  # m

    def __init__(self, actor, ctn_operator: CtnSimOperator, name="RunningStopTest", terminate_on_failure=False):
        """
        """
        self.ctn_operator = ctn_operator
        super(RunningStopTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self._world = self.ctn_operator.get_world()
        self._map = self.ctn_operator.get_map()
        self._list_stop_signs = []
        self._target_stop_sign = None
        self._stop_completed = False

        self._last_failed_stop = None
    
        for _actor in self._world.get_actors():
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

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
        if stop_location.distance(actor_location) > self.PROXIMITY_THRESHOLD:
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
        Ignore all checks when going backwards or through an opposite direction"""

        actor_direction = to_numpy(actor_transform.get_forward_vector())

        # Ignore all when going backwards
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
            _actor_velocity_vec = self.actor.get_velocity()
            current_speed = np.linalg.norm(to_numpy(_actor_velocity_vec))
            # current_speed = GlobalConfig.get_velocity(self.actor)
            if current_speed < self.SPEED_THRESHOLD:
                self._stop_completed = True

        if not self.is_actor_affected_by_stop(check_wps, self._target_stop_sign):
            if not self._stop_completed and self._last_failed_stop != self._target_stop_sign.id:
                # did we stop?
                self.actual_value += 1
                self.test_status = "FAILURE"
                stop_location = self._target_stop_sign.get_transform().location
                running_stop_event = TrafficEvent(event_type=TrafficEventType.STOP_INFRACTION, frame=GameTime.get_frame())
                running_stop_event.set_message(
                    "Agent ran a stop with id={} at (x={}, y={}, z={})".format(
                        self._target_stop_sign.id,
                        round(stop_location.x, 3),
                        round(stop_location.y, 3),
                        round(stop_location.z, 3)))
                running_stop_event.set_dict({'id': self._target_stop_sign.id, 'location': stop_location})

                self.events.append(running_stop_event)

                self._last_failed_stop = self._target_stop_sign.id

            # Reset state
            self._target_stop_sign = None
            self._stop_completed = False

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status


class RunningRedLightTest(Criterion):

    """
    Check if an actor is running a red light

    Important parameters:
    - actor: CARLA actor to be used for this test
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    DISTANCE_LIGHT = 15  # m

    def __init__(self, actor, ctn_operator: CtnSimOperator, name="RunningRedLightTest", terminate_on_failure=False):
        """
        Init
        """
        self.ctn_operator = ctn_operator
        super(RunningRedLightTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self._world = self.ctn_operator.get_world()
        self._map = self.ctn_operator.get_map()
        self._list_traffic_lights = []
        self._last_red_light_id = None
        self.debug = False

        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))

    # pylint: disable=no-self-use
    def is_vehicle_crossing_line(self, seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

    def update(self):
        """
        Check if the actor is running a red light
        """
        new_status = py_trees.common.Status.RUNNING

        transform = self.actor.get_transform()
        location = transform.location
        if location is None:
            return new_status

        veh_extent = self.actor.bounding_box.extent.x

        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0, 0), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0, 0), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:

            if self.debug:
                z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(155, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 155, 0)
                else:
                    color = carla.Color(155, 155, 0)
                self._world.debug.draw_point(center + carla.Location(z=z), size=0.2, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(
                        wp.transform.location + carla.Location(x=1, z=z), text, color=color, life_time=0.01)
                    self._world.debug.draw_point(
                        wp.transform.location + carla.Location(z=z), size=0.1, color=color, life_time=0.01)

            center_loc = carla.Location(center)

            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:

                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = to_numpy(self.actor.get_transform().get_forward_vector())
                wp_dir = to_numpy(wp.transform.get_forward_vector())

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and ve_dir.dot(wp_dir) > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.6 * lane_width, 0, 0), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):

                        self.test_status = "FAILURE"
                        self.actual_value += 1
                        location = traffic_light.get_transform().location
                        red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION, frame=GameTime.get_frame())
                        red_light_event.set_message(
                            "Agent ran a red light {} at (x={}, y={}, z={})".format(
                                traffic_light.id,
                                round(location.x, 3),
                                round(location.y, 3),
                                round(location.z, 3)))
                        red_light_event.set_dict({'id': traffic_light.id, 'location': location})

                        self.events.append(red_light_event)
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps
