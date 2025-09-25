import math
import carla
import py_trees

from loguru import logger

from tools.timer import GameTime
from pydantic import BaseModel, Field
from typing import Optional, List
try:
    from typing import Literal  # Python ≥3.8
except ImportError:
    from typing_extensions import Literal  # Python 3.7

from ..atomic import AtomicBehavior

class TrafficLightBehaviorConfig(BaseModel):
    pattern: Optional[
        Literal["none", "S7left", "S7right", "S7opposite", "S8left", "S9right"]
    ] = Field(
        None,
        description=(
            "Traffic light subtype (intersection configuration).\n"
            "- none: Always green (no control).\n"
            "- S7left: Intersection type S7, priority order = ['left', 'opposite', 'right'].\n"
            "- S7right: Intersection type S7, priority order = ['left', 'opposite'].\n"
            "- S7opposite: Intersection type S7, priority order = ['right', 'left', 'opposite'].\n"
            "- S8left: Intersection type S8, priority order = ['opposite'].\n"
            "- S9right: Intersection type S9, priority order = ['left', 'opposite']."
        ),
    )
    yellow_time: float = Field(..., gt=0, description="Duration of yellow light (seconds)")
    red_time: float = Field(..., gt=0, description="Duration of red light (seconds)")
    green_time: Optional[float] = Field(
        None, gt=0, description="Duration of green light (seconds). If None, always green."
    )

    def priority_order(self) -> Optional[List[str]]:
        """Return the direction priority list for the given pattern."""
        SUBTYPE_CONFIG_TRANSLATION = {
            "S7left":     ["left", "opposite", "right"],
            "S7right":    ["left", "opposite"],
            "S7opposite": ["right", "left", "opposite"],
            "S8left":     ["opposite"],
            "S9right":    ["left", "opposite"],
        }
        return SUBTYPE_CONFIG_TRANSLATION.get(self.pattern, None)


class TrafficLightBehavior(AtomicBehavior):
    
    """
    Atomic behavior that manipulates traffic lights around the ego_vehicle to trigger scenarios 7 to 10.
    This is done by setting 2 of the traffic light at the intersection to green (with some complex precomputation
    to set everything up).

    Important parameters:
    - ego_vehicle: CARLA actor that controls this behavior
    - subtype: string that gathers information of the route and scenario number
      (check SUBTYPE_CONFIG_TRANSLATION below)
    """

    RED = carla.TrafficLightState.Red
    YELLOW = carla.TrafficLightState.Yellow
    GREEN = carla.TrafficLightState.Green

    # Time constants
    RED_TIME = 1.5  # Minimum time the ego vehicle waits in red (seconds)
    YELLOW_TIME = 2  # Time spent at yellow state (seconds)
    RESET_TIME = 6  # Time waited before resetting all the junction (seconds)

    # Experimental values
    TRIGGER_DISTANCE = 10  # Distance that makes all vehicles in the lane enter the junction (meters)
    DIST_TO_WAITING_TIME = 0.04  # Used to wait longer at larger intersections (s/m)

    INT_CONF_OPP1 = {'ego': RED, 'ref': RED, 'left': RED, 'right': RED, 'opposite': GREEN}
    INT_CONF_OPP2 = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': RED, 'opposite': GREEN}
    INT_CONF_LFT1 = {'ego': RED, 'ref': RED, 'left': GREEN, 'right': RED, 'opposite': RED}
    INT_CONF_LFT2 = {'ego': GREEN, 'ref': GREEN, 'left': GREEN, 'right': RED, 'opposite': RED}
    INT_CONF_RGT1 = {'ego': RED, 'ref': RED, 'left': RED, 'right': GREEN, 'opposite': RED}
    INT_CONF_RGT2 = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': GREEN, 'opposite': RED}

    INT_CONF_REF1 = {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': RED, 'opposite': RED}
    INT_CONF_REF2 = {'ego': YELLOW, 'ref': YELLOW, 'left': RED, 'right': RED, 'opposite': RED}

    # Depending on the scenario, IN ORDER OF IMPORTANCE, the traffic light changed
    # The list has to contain only items of the INT_CONF
    SUBTYPE_CONFIG_TRANSLATION = {
        'S7left': ['left', 'opposite', 'right'],
        'S7right': ['left', 'opposite'],
        'S7opposite': ['right', 'left', 'opposite'],
        'S8left': ['opposite'],
        'S9right': ['left', 'opposite']
    }

    CONFIG_TLM_TRANSLATION = {
        'left': [INT_CONF_LFT1, INT_CONF_LFT2],
        'right': [INT_CONF_RGT1, INT_CONF_RGT2],
        'opposite': [INT_CONF_OPP1, INT_CONF_OPP2]
    }

    def __init__(
        self, 
        ctn_operator,
        subtype, 
        yellow_time = 2.0, 
        red_time = 1.0, 
        debug=False, 
        name="TrafficLightBehavior"
    ):
        super(TrafficLightBehavior, self).__init__(name)
        self.subtype = subtype
        self.current_step = 1
        self.debug = debug
        self.ctn_operator = ctn_operator
        self.world = self.ctn_operator.get_world()
        self.map = self.world.get_map()

        self.traffic_light = None
        self.annotations = None
        self.configuration = None
        self.prev_junction_state = None
        self.junction_location = None
        self.seconds_waited = 0
        self.prev_time = None
        self.max_trigger_distance = None
        self.waiting_time = None
        self.inside_junction = False

        self.YELLOW_TIME = yellow_time
        self.RED_TIME = red_time
        
        self._traffic_light_map = {} 
        self.prepare_traffic_light_map()

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        
    def prepare_traffic_light_map(self):
        """
        This function set the current map and loads all traffic lights for this map to
        _traffic_light_map
        """
        # Parse all traffic lights
        self._traffic_light_map.clear()
        for traffic_light in self.world.get_actors().filter('*traffic_light*'):
            if traffic_light not in list(self._traffic_light_map):
                self._traffic_light_map[traffic_light] = traffic_light.get_transform()
            else:
                raise KeyError("Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

    # def get_next_traffic_light(self, actor):
    #     """
    #     returns the next relevant traffic light for the provided actor
    #     """
    #     location = actor.get_transform().location

    #     waypoint = self.map.get_waypoint(location)
    #     # Create list of all waypoints until next intersection
    #     list_of_waypoints = []
    #     while waypoint and not waypoint.is_junction:
    #         list_of_waypoints.append(waypoint)
    #         waypoint = waypoint.next(2.0)[0]

    #     # If the list is empty, the actor is in an intersection
    #     if not list_of_waypoints:
    #         return None

    #     relevant_traffic_light = None
    #     distance_to_relevant_traffic_light = float("inf")

    #     for traffic_light in self._traffic_light_map:
    #         if hasattr(traffic_light, 'trigger_volume'):
    #             tl_t = self._traffic_light_map[traffic_light]
    #             transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)
    #             distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

    #             if distance < distance_to_relevant_traffic_light:
    #                 relevant_traffic_light = traffic_light
    #                 distance_to_relevant_traffic_light = distance


    #     return relevant_traffic_light
    
    @staticmethod
    def get_trafficlight_trigger_location(traffic_light):  # pylint: disable=invalid-name
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """

        def rotate_point(point, angle):
            """
            rotate a given point by a given angle
            """
            x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y

            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)
    
    def annotate_trafficlight_in_group(self, traffic_light):
        # type: (carla.TrafficLight) -> dict[str, list[carla.TrafficLight]]
        """
        Get dictionary with traffic light group info for a given traffic light
        """
        dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # Get the waypoints
        ref_location = self.get_trafficlight_trigger_location(traffic_light)
        ref_waypoint = self.map.get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw

        group_tl = traffic_light.get_group_traffic_lights()

        for target_tl in group_tl:
            if traffic_light.id == target_tl.id:
                dict_annotations['ref'].append(target_tl)
            else:
                # Get the angle between yaws
                target_location = self.get_trafficlight_trigger_location(target_tl)
                target_waypoint = self.map.get_waypoint(target_location)
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360

                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations['right'].append(target_tl)
                elif diff > 135.0:
                    dict_annotations['opposite'].append(target_tl)
                elif diff > 30:
                    dict_annotations['left'].append(target_tl)

        return dict_annotations
    
    @staticmethod
    def reset_lights(reset_params):
        """
        Reset traffic lights
        """
        for param in reset_params:
            param['light'].set_state(param['state'])
            param['light'].set_green_time(param['green_time'])
            param['light'].set_red_time(param['red_time'])
            param['light'].set_yellow_time(param['yellow_time'])
    
    @staticmethod
    def update_light_states(ego_light, annotations, states, freeze=False, timeout=1000000000):
        # type: (carla.TrafficLight, dict[str, list[carla.TrafficLight]], dict[str, carla.TrafficLightState], bool, float) -> list[dict[str, carla.TrafficLight | carla.TrafficLightState | float]] # pylint: disable=line-too-long
        """
        Update traffic light states
        """
        reset_params = []  # type: list[dict]

        for state in states:
            relevant_lights = []
            if state == 'ego':
                relevant_lights = [ego_light]
            else:
                relevant_lights = annotations[state]
            for light in relevant_lights:
                prev_state = light.get_state()
                prev_green_time = light.get_green_time()
                prev_red_time = light.get_red_time()
                prev_yellow_time = light.get_yellow_time()
                reset_params.append(
                    {
                        'light': light,
                        'state': prev_state,
                        'green_time': prev_green_time,
                        'red_time': prev_red_time,
                        'yellow_time': prev_yellow_time,
                    }
                )

                light.set_state(states[state])
                if freeze:
                    light.set_green_time(timeout)
                    light.set_red_time(timeout)
                    light.set_yellow_time(timeout)

        return reset_params
         
    def update(self):

        new_status = py_trees.common.Status.RUNNING

        # 1) Set up the parameters
        if self.current_step == 1:

            # Traffic light affecting the ego vehicle
            # self.traffic_light = self.get_next_traffic_light(self.ego_vehicle)
            traffic_light_ids = list(self._traffic_light_map.keys())
            self.traffic_light = self._traffic_light_map[traffic_light_ids[0]]
            if not self.traffic_light:
                # nothing else to do in this iteration...
                return new_status

            # "Topology" of the intersection
            self.annotations = self.annotate_trafficlight_in_group(self.traffic_light)

            # Which traffic light will be modified (apart from the ego lane)
            self.configuration = self.get_traffic_light_configuration(self.subtype, self.annotations)
            if self.configuration is None:
                self.current_step = 0  # End the behavior
                return new_status

            # Modify the intersection. Store the previous state
            self.prev_junction_state = self.set_intersection_state(self.INT_CONF_REF1)

            self.current_step += 1
            if self.debug:
                logger.debug("--- All set up")

        # 2) Modify the ego lane to yellow when closeby
        elif self.current_step == 2:

            ego_location = self.ego_vehicle.get_location()

            if self.junction_location is None:
                ego_waypoint = self.map.get_waypoint(ego_location)
                junction_waypoint = ego_waypoint.next(0.5)[0]
                while not junction_waypoint.is_junction:
                    next_wp = junction_waypoint.next(0.5)
                    if len(next_wp) > 0:
                        next_wp = next_wp[0]
                        junction_waypoint = next_wp
                    else:
                        break
                self.junction_location = junction_waypoint.transform.location

            distance = ego_location.distance(self.junction_location)

            # Failure check
            if self.max_trigger_distance is None:
                self.max_trigger_distance = distance + 1
            if distance > self.max_trigger_distance:
                self.current_step = 0

            elif distance < self.TRIGGER_DISTANCE:
                _ = self.set_intersection_state(self.INT_CONF_REF2)
                self.current_step += 1

            if self.debug:
                logger.debug("--- Distance until traffic light changes: {}".format(distance))

        # 3) Modify the ego lane to red and the chosen one to green after several seconds
        elif self.current_step == 3:

            if self.passed_enough_time(self.YELLOW_TIME):
                _ = self.set_intersection_state(self.CONFIG_TLM_TRANSLATION[self.configuration][0])

                self.current_step += 1

        # 4) Wait a bit to let vehicles enter the intersection, then set the ego lane to green
        elif self.current_step == 4:

            # Get the time in red, dependent on the intersection dimensions
            if self.waiting_time is None:
                self.waiting_time = self.get_waiting_time(self.annotations, self.configuration)

            if self.passed_enough_time(self.waiting_time):
                _ = self.set_intersection_state(self.CONFIG_TLM_TRANSLATION[self.configuration][1])

                self.current_step += 1

        # 5) Wait for the end of the intersection
        elif self.current_step == 5:
            # the traffic light has been manipulated, wait until the vehicle finsihes the intersection
            ego_location = self.ego_vehicle.get_location()
            ego_waypoint = self.map.get_waypoint(ego_location)

            if not self.inside_junction:
                if ego_waypoint.is_junction:
                    # Wait for the ego_vehicle to enter a junction
                    self.inside_junction = True
                else:
                    if self.debug:
                        logger.debug("--- Waiting to ENTER a junction")

            else:
                if ego_waypoint.is_junction:
                    if self.debug:
                        logger.debug("--- Waiting to EXIT a junction")
                else:
                    # And to leave it
                    self.inside_junction = False
                    self.current_step += 1

        # 6) At the end (or if something failed), reset to the previous state
        else:
            if self.prev_junction_state:
                self.reset_lights(self.prev_junction_state)
                if self.debug:
                    logger.debug("--- Returning the intersection to its previous state")

            self.variable_cleanup()
            new_status = py_trees.common.Status.SUCCESS

        return new_status

    def passed_enough_time(self, time_limit):
        """
        Returns true or false depending on the time that has passed from the
        first time this function was called
        """
        # Start the timer
        if self.prev_time is None:
            self.prev_time = GameTime.get_time()

        timestamp = GameTime.get_time()
        self.seconds_waited += (timestamp - self.prev_time)
        self.prev_time = timestamp

        if self.debug:
            logger.debug("--- Waited seconds: {}".format(self.seconds_waited))

        if self.seconds_waited >= time_limit:
            self.seconds_waited = 0
            self.prev_time = None

            return True
        return False

    def set_intersection_state(self, choice):
        """
        Changes the intersection to the desired state
        """
        prev_state = self.update_light_states(
            self.traffic_light,
            self.annotations,
            choice,
            freeze=True)

        return prev_state

    def get_waiting_time(self, annotation, direction):
        """
        Calculates the time the ego traffic light will remain red
        to let vehicles enter the junction
        """

        tl = annotation[direction][0]
        ego_tl = annotation["ref"][0]

        tl_location = self.get_trafficlight_trigger_location(tl)
        ego_tl_location = self.get_trafficlight_trigger_location(ego_tl)

        distance = ego_tl_location.distance(tl_location)

        return self.RED_TIME + distance * self.DIST_TO_WAITING_TIME

    def get_traffic_light_configuration(self, subtype, annotations):
        """
        Checks the list of possible altered traffic lights and gets
        the first one that exists in the intersection

        Important parameters:
        - subtype: Subtype of the scenario
        - annotations: list of the traffic light of the junction, with their direction (right, left...)
        """
        configuration = None

        if subtype in self.SUBTYPE_CONFIG_TRANSLATION:
            possible_configurations = self.SUBTYPE_CONFIG_TRANSLATION[self.subtype]
            while possible_configurations:
                # Chose the first one and delete it
                configuration = possible_configurations[0]
                possible_configurations = possible_configurations[1:]
                if configuration in annotations:
                    if annotations[configuration]:
                        # Found a valid configuration
                        break
                    else:
                        # The traffic light doesn't exist, get another one
                        configuration = None
                else:
                    if self.debug:
                        logger.debug("This configuration name is wrong")
                    configuration = None

            if configuration is None and self.debug:
                logger.debug("This subtype has no traffic light available")
        else:
            if self.debug:
                logger.debug("This subtype is unknown")

        return configuration

    def variable_cleanup(self):
        """
        Resets all variables to the intial state
        """
        self.current_step = 1
        self.traffic_light = None
        self.annotations = None
        self.configuration = None
        self.prev_junction_state = None
        self.junction_location = None
        self.max_trigger_distance = None
        self.waiting_time = None
        self.inside_junction = False

