import carla
import py_trees
import traceback

from loguru import logger
from typing import List

from .atomic.base import Criterion
from .runtime_collision import CollisionTest
from .runtime_stuck import ActorBlockedTest
from .runtime_destination import RouteCompletionTest

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
        super(RuntimeSingleTest, self).__init__(name, actor, False)
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

    def initialise(self):
        super(RuntimeSingleTest, self).initialise()
        if not self.already_terminate:
            self.collision_test.initialise()
            self.stuck_test.initialise()
            self.reach_destination_test.initialise()
            
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
        except Exception as e:
            logger.error(f"RuntimeSingleTest {self.name} update error: {e}")
            self.test_status = "FAILURE"
            self.actual_value = 0
            new_status = py_trees.common.Status.FAILURE
            traceback.print_exc()
            # return new_status
        
        self.st_detail["collision"] = self.collision_test.st_detail
        self.st_detail["stuck"] = self.stuck_test.st_detail
        self.st_detail["reach_destination"] = self.reach_destination_test.st_detail
        
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
        """
        Cleanup sensor
        """
        if self.collision_test is not None:
            self.collision_test.terminate(new_status)
            self.collision_test = None
        
        if self.stuck_test is not None:
            self.stuck_test.terminate(new_status)
            self.stuck_test = None
            
        if self.reach_destination_test is not None:
            self.reach_destination_test.terminate(new_status)
            self.reach_destination_test = None
        
        self.already_terminate = True
        
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        super(RuntimeSingleTest, self).terminate(new_status)
        