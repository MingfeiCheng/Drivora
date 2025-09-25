import py_trees
import numpy as np

from .atomic.base import Criterion
from .atomic.traffic_events import TrafficEvent, TrafficEventType

from tools.timer import GameTime
from scenario_runner.ctn_operator import CtnSimOperator

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

    def __init__(self, actor, min_speed, max_time, ctn_operator: CtnSimOperator, trigger_time, name="ActorBlockedTest", optional=False, terminate_on_failure=False):
        """
        Class constructor
        """
        super().__init__(name, actor, optional, terminate_on_failure)
        self._min_speed = min_speed
        self._max_time = max_time
        self.ctn_operator = ctn_operator
        
        self._trigger_time = trigger_time
        self._time_last_valid_state = None
        self._best_time = 0.0
        self._active = True
        self.units = None  # We care about whether or not it fails, no units attached
        self.success_value = self._max_time
        self.st_detail = {
            "occurred": False,
            "details": {}
        }
        
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

        if self._active and GameTime.get_time() > self._trigger_time:
            linear_speed_vec = self.actor.get_velocity()
            linear_speed = np.linalg.norm([linear_speed_vec.x, linear_speed_vec.y, linear_speed_vec.z])
            if linear_speed is not None:
                if linear_speed < self._min_speed and self._time_last_valid_state:
                    delta_time = GameTime.get_time() - self._time_last_valid_state
                    if delta_time > self.actual_value:
                        self.actual_value = delta_time
                        
                    if delta_time > self._best_time:
                        self._best_time = delta_time
                        if not self.st_detail['occurred']:
                            self.st_detail['details'] = {
                                "timestamp": GameTime.get_time(),
                                "location": {
                                    "x": self.actor.get_location().x,
                                    "y": self.actor.get_location().y,
                                    "z": self.actor.get_location().z
                                },
                                "blocked_start_timestamp": self._time_last_valid_state,
                                "blocked_duration": delta_time,
                                "max_blocked_duration": self._best_time
                            }
                    
                    if delta_time > self._max_time:
                        vehicle_location = self.actor.get_location()
                        # The actor has been "blocked" for too long, save the data
                        self.test_status = "FAILURE"

                        event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED, frame=GameTime.get_frame())
                        event.set_message('Agent got blocked at (x={}, y={}, z={})'.format(
                            round(vehicle_location.x, 3),
                            round(vehicle_location.y, 3),
                            round(vehicle_location.z, 3))
                        )
                        event.set_dict({'location': vehicle_location})
                        self.events.append(event)
                        
                        self.st_detail = {
                            "occurred": True,
                            "details": {
                                "timestamp": GameTime.get_time(),
                                "location": {
                                    "x": vehicle_location.x,
                                    "y": vehicle_location.y,
                                    "z": vehicle_location.z
                                },
                                "blocked_start_timestamp": self._time_last_valid_state,
                                "blocked_duration": delta_time,
                                "max_blocked_duration": self._best_time
                            }
                        }                    
                else:
                    self._time_last_valid_state = GameTime.get_time()

        if self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status