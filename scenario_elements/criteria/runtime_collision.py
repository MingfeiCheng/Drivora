import carla
import py_trees
import numpy as np

from .atomic.base import Criterion
from .atomic.traffic_events import TrafficEvent, TrafficEventType

from tools.timer import GameTime

from scenario_runner.ctn_operator import CtnSimOperator

def to_numpy(vec):
    """
    Convert a carla.Vector3D to a numpy array
    if use Carla version < 0.9.14, uncomment the return line
    """
    return np.array([vec.x, vec.y, vec.z])
    # return vec
    
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
        super(CollisionTest, self).__init__(name, actor, optional, terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._other_actor = other_actor
        self._other_actor_type = other_actor_type

        # Attributes to store the last collisions's data
        self._collision_sensor = None
        self._collision_id = None
        self._collision_time = None
        self._collision_location = None
        
        self.ctn_operator = ctn_operator
        self.world = self.ctn_operator.get_world()
        
        self.st_detail = {
            "occurred": False,
            "details": {}
        }

    def initialise(self):
        """
        Creates the sensor and callback"""
        world = self.world
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._collision_sensor.listen(lambda event: self._count_collisions(event))
        super(CollisionTest, self).initialise()

    def update(self):
        """
        Check collision count
        """
        new_status = py_trees.common.Status.RUNNING

        if self.test_status == "FAILURE":
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
            if np.linalg.norm(to_numpy(distance_vector)) <= self.COLLISION_RADIUS:
                return

        # If the actor speed is 0, the collision isn't its fault
        actor_velocity = self.actor.get_velocity()
        actor_speed = np.linalg.norm(to_numpy(actor_velocity))
        if actor_speed < self.EPSILON:
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
        
        # NOTE: we only record the final one
        self.st_detail = {
            "occurred": True,
            "details": {
                "timestamp": GameTime.get_time(),
                "location": {
                    "x": actor_location.x,
                    "y": actor_location.y,
                    "z": actor_location.z
                },
                "other_actor": {
                    "id": event.other_actor.id,
                    "type_id": event.other_actor.type_id
                },
            }
        }