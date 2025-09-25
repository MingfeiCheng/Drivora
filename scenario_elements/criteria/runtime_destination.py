import py_trees
import numpy as np

from .atomic.base import Criterion
from .atomic.traffic_events import TrafficEvent, TrafficEventType

from scenario_runner.ctn_operator import CtnSimOperator

def to_numpy(vec):
    """
    Convert a carla.Vector3D to a numpy array
    if use Carla version < 0.9.14, uncomment the return line
    """
    return np.array([vec.x, vec.y, vec.z])
    # return vec
    
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
    PERCENTAGE_THRESHOLD = 95  # %

    def __init__(self, actor, route, ctn_operator: CtnSimOperator, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        super(RouteCompletionTest, self).__init__(name, actor, terminate_on_failure=terminate_on_failure)
        self.units = "%"
        self.success_value = 100
        self._route = route
        self.ctn_operator = ctn_operator
        
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
        
        self.st_detail = {
            "occurred": False,
            "details": {}
        }

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

        if self.test_status == "FAILURE":
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

            dist2dest = location.distance(self.target_location)
            
            if self.actual_value > self.PERCENTAGE_THRESHOLD \
                    and dist2dest < self.DISTANCE_THRESHOLD:
                self.test_status = "SUCCESS"
                self.actual_value = 100
            
            self.st_detail = {
                "occurred": self.test_status == "SUCCESS",
                "details": {
                    "final_location": {
                        'x': location.x,
                        'y': location.y,
                        'z': location.z
                    },
                    "route_completed": self.actual_value,
                    "distance_to_destination": dist2dest
                }
            }

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