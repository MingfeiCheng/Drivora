import carla
import py_trees

from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from ..atomic import AtomicBehavior

from scenario_elements.config import Waypoint
from scenario_runner.ctn_operator import CtnSimOperator

class AIWalkerConfig(BaseModel):
    
    id: str = Field(..., description="Unique identifier of the NPC walker")
    model: str = Field(..., description="walker model name")
    rolename: str = Field(..., description="Role name, e.g., 'ego' or 'npc'")
    color: Optional[str] = Field(
        None, description="Walker color in format '(r,g,b)'"
    )
    category: Optional[str] = Field("pedestrian", description="walker category, e.g., 'walker'")
    trigger_time: float = Field(..., ge=0, description="Time to start moving (seconds)")
    route: List[Waypoint] = Field(
        ..., description="Behavior configuration for the walker"
    )
    
    def get_initial_waypoint(self) -> Waypoint:
        return self.route[0]
    
    def get_sink_location(self) -> carla.Location:
        if not self.route:
            raise ValueError("Route is empty, cannot get sink location")
        last_wp = self.route[-1]
        return carla.Location(x=last_wp.x, y=last_wp.y, z=last_wp.z)
    
    
class AIWalkerBehavior(AtomicBehavior):
    """
    Behavior that creates a walker controlled by AI Walker controller.
    The walker go from source location to sink location.
    A parallel termination behavior has to be used.

    Important parameters:
    - source_location (carla.Location): Location at which the actor will be spawned
    - sink_location (carla.Location): Location at which the actor will be deleted
    """

    def __init__(
        self, 
        ctn_operator: CtnSimOperator, 
        actor: carla.Actor,
        config: AIWalkerConfig,
        name="AIWalkerBehavior"
    ):
        """
        Setup class members
        """
        super(AIWalkerBehavior, self).__init__(name, actor=actor)

        self.ctn_operator = ctn_operator
        self._world = self.ctn_operator.get_world()
        self._controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')

        self._sink_location = config.get_sink_location()
        self._sink_dist = 2

        self._walker = self._actor
        self._controller = None

    def initialise(self):
        """
        Spawn the walker at source location.
        Setup the AI controller.

        May throw RuntimeError if the walker can not be
        spawned at given location.
        """
        # Use ai.walker to controll the walker
        self._controller = self._world.try_spawn_actor(
            self._controller_bp, carla.Transform(), self._walker)
        self._controller.start()
        self._controller.go_to_location(self._sink_location)

        super(AIWalkerBehavior, self).initialise()

    def update(self):
        """Controls the created walker"""
        # Remove walkers when needed
        if self._walker is not None:
            loc = self._walker.get_location() # CarlaDataProvider.get_location(self._walker)
            # At the very beginning of the scenario, the get_location may return None
            if loc is not None:
                if loc.distance(self._sink_location) < self._sink_dist:
                    self.terminate(py_trees.common.Status.SUCCESS)

        return py_trees.common.Status.RUNNING

    def _destroy_walker(self, walker, controller):
        if controller:
            controller.stop()
            controller.destroy()
        # if walker:
        #     self.ctn_operator.remove_actor(walker)
        # keep walker

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        try:
            self._destroy_walker(self._walker, self._controller)
        except RuntimeError:
            pass  # Actor was already destroyed