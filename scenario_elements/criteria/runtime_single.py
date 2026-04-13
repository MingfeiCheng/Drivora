import carla
import py_trees
import traceback

from loguru import logger
from typing import List

from .atomic.base import Criterion
from .runtime_collision import CollisionTest
from .runtime_destination import RouteCompletionTest
from .runtime_offroad import OffRoadTest
from .runtime_overspeed import OverSpeedTest
from .runtime_stop_sign import RunningStopTest
from .runtime_stuck import ActorBlockedTest
from .runtime_traffic_light import RunningRedLightTest
from .runtime_wrong_lane import WrongLaneTest

from scenario_runner.ctn_operator import CtnSimOperator

class RuntimeSingleTest(Criterion):
    
    """
    Immediate stop:
    1. Collision -> Failure
    Wait for others:
    1. Stuck -> Success
    2. Reach the destination -> Success
    
    We need an attribute to note the failures
    event:
    {
        "id": "vehicle.id",
        "collision": {
            "occurred": True/False,
            "details": "detailed info"
        },
        "stuck": {
            "occurred": True/False,
            "details": "detailed info"
        },
        "reach_destination": {
            "occurred": True/False,
            "details": "detailed info"
        }
    }
    """
    
    def __init__(
        self, 
        id: str,
        actor: carla.Actor,
        actor_route: List, # NOTE: here is the route after interpolation, which is a list of waypoints pair
        actor_trigger_time: float,
        ctn_operator: CtnSimOperator,
        min_speed: float = 0.1,
        max_time: float = 20.0, # max time for move under min_speed
        terminate_on_failure: bool = True,
        name: str = "RuntimeSingleTest",
    ):
        super().__init__(name, actor, False)
        self.id = id
        self.actor = actor
        self.actor_id = actor.id
        # logger.debug(f"Creating RuntimeSingleTest for actor id: {self.actor_id}, name: {self.name}")
        self.actor_route = actor_route
        self.actor_trigger_time = actor_trigger_time
        self.min_speed = min_speed
        self.max_time = max_time
        self.terminate_on_failure = terminate_on_failure
        self.ctn_operator = ctn_operator
        
        # inner parameters
        self.st_detail = {
            "id": self.actor_id, # TODO: check the observation saver!!!!!, we use actor id here
            "config_id": self.id,
            "collision": {
                "occurred": False,
                "details": {}
            },
            "stuck": {
                "occurred": False,
                "details": {}
            },
            "reach_destination": {
                "occurred": False,
                "details": {}
            },
            "offroad": {
                "occurred": False,
                "details": {}
            },
            "overspeed": {
                "occurred": False,
                "details": {}
            },
            "running_stop": {
                "occurred": False,
                "details": {}
            },
            "running_red_light": {
                "occurred": False,
                "details": {}
            },
            "wrong_lane": {
                "occurred": False,
                "details": {}
            }
        }
        
        
        self.events = []
        self.actor_destination = [
            self.actor_route[-1][0].location.x, 
            self.actor_route[-1][0].location.y
        ]
        
        self.test_status = "RUNNING"
        self.success_value = 1 # this defines the success value of the criterion, expected to be 1
        self.actual_value = 0 # this is the actual value of the criterion, updated during the update phase
        self.already_terminate = False
        
        # create sub-criteria
        self.collision_test = CollisionTest(
            name=f"{self.id}_collision",
            actor=self.actor,
            terminate_on_failure=self.terminate_on_failure,
            ctn_operator=self.ctn_operator
        )
        self.stuck_test = ActorBlockedTest(
            name=f"{self.id}_stuck",
            actor=self.actor,
            min_speed=self.min_speed,
            max_time=self.max_time,
            trigger_time=self.actor_trigger_time,
            terminate_on_failure=self.terminate_on_failure,
            ctn_operator=self.ctn_operator
        )
        self.reach_destination_test = RouteCompletionTest(
            name=f"{self.id}_reach_destination",
            actor=self.actor,
            route=self.actor_route,
            terminate_on_failure=self.terminate_on_failure,
            ctn_operator=self.ctn_operator
        )
        # not need to termination oracle
        self.offroad_test = OffRoadTest(
            name=f"{self.id}_offroad",
            actor=self.actor,
            ctn_operator=self.ctn_operator,
            duration=1.0,
            terminate_on_failure=False
        )
        self.overspeed_test = OverSpeedTest(
            name=f"{self.id}_overspeed",
            actor=self.actor,
            ctn_operator=self.ctn_operator,
            terminate_on_failure=False
        )
        self.stop_sign_test = RunningStopTest(
            name=f"{self.id}_stop_sign",
            actor=self.actor,
            ctn_operator=self.ctn_operator,
            terminate_on_failure=False
        )
        self.red_light_test = RunningRedLightTest(
            name=f"{self.id}_red_light",
            actor=self.actor,
            ctn_operator=self.ctn_operator,
            terminate_on_failure=False
        )
        self.wrong_lane_test = WrongLaneTest(
            name=f"{self.id}_wrong_lane",
            actor=self.actor,
            ctn_operator=self.ctn_operator,
            terminate_on_failure=False
        )

    def initialise(self):
        super().initialise()
        if not self.already_terminate:
            self.collision_test.initialise()
            self.stuck_test.initialise()
            self.reach_destination_test.initialise()
            self.offroad_test.initialise()
            self.overspeed_test.initialise()
            self.stop_sign_test.initialise()
            self.red_light_test.initialise()
            self.wrong_lane_test.initialise()
            
    def get_stop(self) -> bool:
        return self.st_detail["collision"]["occurred"] or self.st_detail["stuck"]["occurred"] or self.st_detail["reach_destination"]["occurred"]
        
    def update(self):
        
        if self.already_terminate:
            new_status = py_trees.common.Status.SUCCESS
            return new_status
        
        if self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE
            return new_status
        
        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS
            return new_status
        
        new_status = py_trees.common.Status.RUNNING
            
        try:
            collision_status = self.collision_test.update()
            stuck_status = self.stuck_test.update()
            reach_status = self.reach_destination_test.update()
            offroad_status = self.offroad_test.update()
            overspeed_status = self.overspeed_test.update()
            stop_sign_status = self.stop_sign_test.update()
            red_light_status = self.red_light_test.update()
            wrong_lane_status = self.wrong_lane_test.update()
        except Exception as e:
            logger.error(f"RuntimeSingleTest {self.name} update error: {e}")
            traceback.print_exc()
            self.test_status = "FAILURE"
            self.actual_value = 0
            new_status = py_trees.common.Status.FAILURE
        
        self.st_detail["collision"] = self.collision_test.st_detail
        self.st_detail["stuck"] = self.stuck_test.st_detail
        self.st_detail["reach_destination"] = self.reach_destination_test.st_detail
        self.st_detail["offroad"] = self.offroad_test.st_detail
        self.st_detail["overspeed"] = self.overspeed_test.st_detail
        self.st_detail["running_stop"] = self.stop_sign_test.st_detail
        self.st_detail["running_red_light"] = self.red_light_test.st_detail
        self.st_detail["wrong_lane"] = self.wrong_lane_test.st_detail
        
        if reach_status == py_trees.common.Status.SUCCESS or reach_status == py_trees.common.Status.FAILURE:
            if "reach_destination" not in self.events:
                self.events.append(["reach_destination", self.id])
                logger.warning(f"Vehicle {self.id} has reached its destination.")
            self.actual_value = 1
            self.test_status = "SUCCESS"   
            
        if collision_status == py_trees.common.Status.FAILURE:
            self.test_status = "FAILURE"
            self.actual_value = 0
            if "collision" not in self.events:
                self.events.append(["collision", self.id])
                logger.warning(f"Vehicle {self.id} has a collision.")
            
        
        if stuck_status == py_trees.common.Status.FAILURE:
            # check the distance to the route destination
            # route: route.append((wp.transform, connection))
            curr_loca = [self.actor.get_transform().location.x, self.actor.get_transform().location.y]
            dist2dest = ((curr_loca[0] - self.actor_destination[0]) ** 2 + (curr_loca[1] - self.actor_destination[1]) ** 2) ** 0.5
            if dist2dest < 5.0:
                self.actual_value = 1
                self.test_status = "SUCCESS"
                if "reach_destination" not in self.events:
                    self.events.append(["reach_destination", self.id])
                    logger.warning(f"Vehicle {self.id} has reached its destination.")
            else:
                self.test_status = "FAILURE"
                self.actual_value = 0
                if "stuck" not in self.events:
                    self.events.append(["stuck", self.id])
                    logger.warning(f"Vehicle {self.id} is stuck.")        
        
        if self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE  
                  
        elif self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS
        
        # logger.debug(f"Criterion {self.name} new_status: {new_status}, actual_value: {self.actual_value}, success_value: {self.success_value}")
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        """Cleanup all sub-criteria."""
        sub_tests = [
            "collision_test", "stuck_test", "reach_destination_test",
            "offroad_test", "overspeed_test", "stop_sign_test",
            "red_light_test", "wrong_lane_test",
        ]
        for attr_name in sub_tests:
            test = getattr(self, attr_name, None)
            if test is not None:
                test.terminate(new_status)
                setattr(self, attr_name, None)

        self.already_terminate = True

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        super().terminate(new_status)
