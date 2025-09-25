from __future__ import print_function

import py_trees

class AtomicCondition(py_trees.behaviour.Behaviour):

    """
    Base class for all atomic conditions used to setup a scenario

    *All behaviors should use this class as parent*

    Important parameters:
    - name: Name of the atomic condition
    """

    def __init__(self, name):
        """
        Default init. Has to be called via super from derived class
        """
        super(AtomicCondition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name

    def setup(self, unused_timeout=15):
        """
        Default setup
        """
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        """
        Initialise setup
        """
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Default terminate. Can be extended in derived class
        """
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
