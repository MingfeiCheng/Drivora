from __future__ import print_function

import copy
import carla
import operator
import py_trees

from tools.timer import GameTime

from .base import AtomicBehavior

def calculate_distance(location, other_location, global_planner=None):
    """
    Method to calculate the distance between to locations

    Note: It uses the direct distance between the current location and the
          target location to estimate the time to arrival.
          To be accurate, it would have to use the distance along the
          (shortest) route between the two locations.
    """
    if global_planner:
        distance = 0

        # Get the route
        route = global_planner.trace_route(location, other_location)

        # Get the distance of the route
        for i in range(1, len(route)):
            curr_loc = route[i][0].transform.location
            prev_loc = route[i - 1][0].transform.location

            distance += curr_loc.distance(prev_loc)

        return distance

    return location.distance(other_location)

class Idle(AtomicBehavior):

    """
    This class contains an idle behavior scenario

    Important parameters:
    - duration[optional]: Duration in seconds of this behavior

    A termination can be enforced by providing a duration value.
    Alternatively, a parallel termination behavior has to be used.
    """

    def __init__(self, duration=float("inf"), name="Idle"):
        """
        Setup actor
        """
        super(Idle, self).__init__(name)
        self._duration = duration
        self._start_time = 0
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        """
        Set start time
        """
        self._start_time = GameTime.get_time()
        super(Idle, self).initialise()

    def update(self):
        """
        Keep running until termination condition is satisfied
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        return new_status


class ActorTransformSetter(AtomicBehavior):

    """
    This class contains an atomic behavior to set the transform
    of an actor.

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - transform: New target transform (position + orientation) of the actor
    - physics [optional]: Change the physics of the actors to true / false. To not change the physics, use None.

    The behavior terminates when actor is set to the new actor transform (closer than 1 meter)

    NOTE:
    It is very important to ensure that the actor location is spawned to the new transform because of the
    appearence of a rare runtime processing error. WaypointFollower with LocalPlanner,
    might fail if new_status is set to success before the actor is really positioned at the new transform.
    Therefore: calculate_distance(actor, transform) < 1 meter
    """

    def __init__(self, actor, transform, physics=True, name="ActorTransformSetter"):
        """
        Init
        """
        super(ActorTransformSetter, self).__init__(name, actor)
        self._transform = transform
        self._physics = physics
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def initialise(self):
        if self._actor.is_alive:
            self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            self._actor.set_transform(self._transform)
        super(ActorTransformSetter, self).initialise()

    def update(self):
        """
        Transform actor
        """
        new_status = py_trees.common.Status.RUNNING

        if not self._actor.is_alive:
            new_status = py_trees.common.Status.FAILURE

        if self._actor and self._actor.is_alive and self._transform:
            if calculate_distance(self._actor.get_location(), self._transform.location) < 1.0:
                if self._physics is not None:
                    self._actor.set_simulate_physics(self._physics)
                new_status = py_trees.common.Status.SUCCESS

        return new_status
    
class ActorDestroy(AtomicBehavior):

    """
    This class contains an actor destroy behavior.
    Given an actor this behavior will delete it.

    Important parameters:
    - actor: CARLA actor to be deleted

    The behavior terminates after removing the actor
    """

    def __init__(self, actor, name="ActorDestroy"):
        """
        Setup actor
        """
        super(ActorDestroy, self).__init__(name, actor)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        if self._actor:
            self._actor = None
            new_status = py_trees.common.Status.SUCCESS

        return new_status
    
class UpdateAllActorControls(AtomicBehavior):

    """
    Atomic to update (run one control loop step) all actor controls.

    The behavior is always in RUNNING state.

    Args:
        name (string): Name of the behavior.
            Defaults to 'UpdateAllActorControls'.
    """

    def __init__(self, name="UpdateAllActorControls"):
        """
        Constructor
        """
        super().__init__(name)

    def update(self):
        """
        Execute one control loop step for all actor controls.

        returns:
            py_trees.common.Status.RUNNING
        """

        actor_dict = {}

        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass

        for actor_id in actor_dict:
            actor_dict[actor_id].run_step()

        return py_trees.common.Status.RUNNING