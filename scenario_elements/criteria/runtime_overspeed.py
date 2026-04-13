import math
import py_trees

from .atomic.base import Criterion
from tools.timer import GameTime
from scenario_runner.ctn_operator import CtnSimOperator

class OverSpeedTest(Criterion):

    """
    This class contains an atomic test for maximum velocity.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - max_velocity_allowed: maximum allowed velocity in m/s
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(
        self, 
        actor,
        ctn_operator: CtnSimOperator, 
        name="OverSpeedTest",
        terminate_on_failure=False
    ):
        """
        Setup actor and maximum allowed velovity
        """
        super().__init__(name, actor, False, terminate_on_failure)
        
        self.ctn_operator = ctn_operator
        self.success_value = self.actor.get_speed_limit() # m/s
        
        self.max_diff = 0.0
        self.max_details = {}
                
        self.st_detail = {
            "occurred": False,
            "details": {}
        }

    def update(self):
        """
        Check velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        velocity = self.actor.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_limit = self.actor.get_speed_limit() # m/s
        
        self.success_value = speed_limit

        self.actual_value = max(speed, self.actual_value)
        
        over_diff = speed - speed_limit
        if over_diff > self.max_diff:
            self.max_diff = over_diff
            self.max_details = {
                "id": self.actor.id,
                "frame": GameTime.get_frame(),
                "timestamp": GameTime.get_time(),
                "speed": speed,
                "speed_limit": speed_limit,
                "max_diff": self.max_diff,
            }

        if speed > self.success_value:
            self.test_status = "FAILURE"
            
            self.st_detail["occurred"] = True
            self.st_detail["details"] = self.max_details
            
        else:
            self.test_status = "SUCCESS"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status