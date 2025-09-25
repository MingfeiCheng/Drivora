import time
import math
import copy
import carla
import py_trees
import operator

from typing import List, Optional
from pydantic import BaseModel, Field

from ..atomic import AtomicBehavior
from .agent import BehaviorWaypointAgent, get_basic_config

from loguru import logger
from scenario_elements.config import Waypoint

class WaypointVehicleConfig(BaseModel):
    
    id: str = Field(..., description="Unique identifier of the NPC vehicle")
    model: str = Field(..., description="Vehicle model name")
    rolename: str = Field(..., description="Role name, e.g., 'ego' or 'npc'")
    color: Optional[str] = Field(
        None, description="Vehicle color in format '(r,g,b)'"
    )
    category: Optional[str] = Field("car", description="Vehicle category, e.g., 'car'")
    trigger_time: float = Field(..., ge=0, description="Time to start moving (seconds)")
    route: List[Waypoint] = Field(
        ..., description="List of waypoints describing the route"
    )
    
    def get_initial_waypoint(self) -> Waypoint:
        return self.route[0]
    
class WaypointVehicleBehavior(AtomicBehavior):
    """
    Behavior for NPC vehicles in the scenario.
    This class extends AtomicBehavior to define specific behaviors for NPC vehicles.
    """

    def __init__(
        self, 
        actor: carla.Actor,
        actor_config: WaypointVehicleConfig,
        name: str = "WaypointVehicleBehavior"
    ):
        """
        Initialize the WaypointVehicleBehavior with a given NPCVehicle instance.

        :param vehicle: An instance of NPCVehicle that this behavior will control.
        """
        super(WaypointVehicleBehavior, self).__init__(name, actor)
        
        self._actor_config = actor_config
        self._route = self._actor_config.route
        
        self._local_planner = None
        self._unique_id = 0

    def initialise(self):
        super(WaypointVehicleBehavior, self).initialise()
        self._unique_id = int(round(time.time() * 1e9))
        try:
            # running_WF_actor_ -> update for terminate
            # check whether WF for this actor is already running and add new WF to running_WF list
            check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
            running = check_attr(py_trees.blackboard.Blackboard())
            active_wf = copy.copy(running)
            active_wf.append(self._unique_id)
            py_trees.blackboard.Blackboard().set("running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
        except AttributeError:
            # no WF is active for this actor
            py_trees.blackboard.Blackboard().set("terminate_WF_actor_{}".format(self._actor.id), [], overwrite=True)
            py_trees.blackboard.Blackboard().set("running_WF_actor_{}".format(self._actor.id), [self._unique_id], overwrite=True)

        self._apply_local_planner(self._actor)
        return True

    def _apply_local_planner(self, actor: carla.Actor):

        local_planner = BehaviorWaypointAgent(
            actor,
            opt_dict=get_basic_config(),
            behavior='normal'
        )
        local_planner.set_global_plan(self._route, stop_waypoint_creation=False)
        self._local_planner = local_planner

    def update(self):
        """
        Compute next control step for the given waypoint plan, obtain and apply control to actor
        """
        new_status = py_trees.common.Status.RUNNING

        # self._global_tracker.update_agent(self._actor)

        # check thread conflicts.
        check_term = operator.attrgetter("terminate_WF_actor_{}".format(self._actor.id))
        terminate_wf = check_term(py_trees.blackboard.Blackboard())
        check_run = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
        active_wf = check_run(py_trees.blackboard.Blackboard())
        # Termination of WF if the WFs unique_id is listed in terminate_wf
        # only one WF should be active, therefore all previous WF have to be terminated
        # this is to make sure that new added agent overwrite previous ones, so, skip this
        if self._unique_id in terminate_wf:
            terminate_wf.remove(self._unique_id)
            if self._unique_id in active_wf:
                active_wf.remove(self._unique_id)
            py_trees.blackboard.Blackboard().set("terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            py_trees.blackboard.Blackboard().set("running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
            new_status = py_trees.common.Status.SUCCESS
            return new_status

        # start update action - replace by localplanner output
        success = True
        if self._actor is not None and self._actor.is_alive and self._local_planner is not None:
            control = self._local_planner.run_step(debug=False)
            # logger.debug("actor {} status {}", self._actor.id, control)
            self._actor.apply_control(control)
            # Check if the actor reached the end of the plan
            #  replace access to private _waypoints_queue with public getter
            if not self._local_planner.is_task_finished():  # pylint: disable=protected-access
                success = False

        if success :
            current_velocity = self._actor.get_velocity()
            current_speed = math.sqrt(current_velocity.x ** 2 + current_velocity.y ** 2 + current_velocity.z ** 2)
            if current_speed < 0.01:
                new_status = py_trees.common.Status.SUCCESS

        return new_status

    def terminate(self, new_status):
        """
        On termination of this behavior,
        the controls should be set back to 0.
        """
        if self._actor is not None and self._actor.is_alive:
            control = self._actor.get_control()
            control.throttle = 0
            control.steering = 0
            control.brake = 1.0
            self._actor.apply_control(control)

        super(WaypointVehicleBehavior, self).terminate(new_status)