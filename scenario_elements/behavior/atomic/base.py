from __future__ import print_function

import copy
import operator
import py_trees


class AtomicBehavior(py_trees.behaviour.Behaviour):

    """
    Base class for all atomic behaviors used to setup a scenario

    *All behaviors should use this class as parent*

    Important parameters:
    - name: Name of the atomic behavior
    """

    def __init__(self, name, actor=None):
        """
        Default init. Has to be called via super from derived class
        """
        super(AtomicBehavior, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name
        self._actor = actor

    def setup(self, unused_timeout=15):
        """
        Default setup
        """
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        """
        Initialise setup terminates WaypointFollowers
        Check whether WF for this actor is running and terminate all active WFs
        """
        if self._actor is not None:
            try:
                check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
                terminate_wf = copy.copy(check_attr(py_trees.blackboard.Blackboard()))
                py_trees.blackboard.Blackboard().set(
                    "terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            except AttributeError:
                # It is ok to continue, if the Blackboard variable does not exist
                pass
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
