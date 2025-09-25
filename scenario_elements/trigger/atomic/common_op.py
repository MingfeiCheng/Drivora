import py_trees

from .base import AtomicCondition
from tools.timer import GameTime

EPSILON = 0.001

class TriggerTimer(AtomicCondition):

    def __init__(self, actor, name, duration=float("inf")):
        """
        Setup actor
        """
        super(TriggerTimer, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

        self._duration = duration
        self._start_time = 0

    def initialise(self):
        """
        Initialize the start time of this condition
        """
        self._start_time = GameTime.get_time()
        super(TriggerTimer, self).initialise()

    def update(self):
        """
        Check if the _actor stands still (v=0)
        """
        new_status = py_trees.common.Status.RUNNING

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS
            # logger.debug('{}: duration: {} current: {}', self._actor.id, self._duration, GameTime.get_time())

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        return new_status
    
class StandStill(AtomicCondition):

    """
    This class contains a standstill behavior of a scenario

    Important parameters:
    - actor: CARLA actor to execute the behavior
    - name: Name of the condition
    - duration: Duration of the behavior in seconds

    The condition terminates with SUCCESS, when the actor does not move
    """

    def __init__(self, actor, name, duration=float("inf")):
        """
        Setup actor
        """
        super(StandStill, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor

        self._duration = duration
        self._start_time = 0

    def initialise(self):
        """
        Initialize the start time of this condition
        """
        self._start_time = GameTime.get_time()
        super(StandStill, self).initialise()

    def update(self):
        """
        Check if the _actor stands still (v=0)
        """
        new_status = py_trees.common.Status.RUNNING

        velocity_vec = self._actor.get_velocity()
        
        velocity = (velocity_vec.x ** 2 + velocity_vec.y ** 2 + velocity_vec.z ** 2) ** 0.5

        if velocity > EPSILON:
            self._start_time = GameTime.get_time()

        if GameTime.get_time() - self._start_time > self._duration:
            new_status = py_trees.common.Status.SUCCESS

        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status