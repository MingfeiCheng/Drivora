import py_trees
import numpy as np

def to_numpy(vec):
    """
    Convert a carla.Vector3D to a numpy array
    if use Carla version < 0.9.14, uncomment the return line
    """
    return np.array([vec.x, vec.y, vec.z])
    # return vec
    
class Criterion(py_trees.behaviour.Behaviour):

    """
    Base class for all criteria used to evaluate a scenario for success/failure

    Important parameters (PUBLIC):
    - name: Name of the criterion
    - actor: Actor of the criterion
    - optional: Indicates if a criterion is optional (not used for overall analysis)
    - terminate on failure: Whether or not the criteria stops on failure

    - test_status: Used to access the result of the criterion
    - success_value: Result in case of success (e.g. max_speed, zero collisions, ...)
    - acceptable_value: Result that does not mean a failure,  but is not good enough for a success
    - actual_value: Actual result after running the scenario
    - units: units of the 'actual_value'. This is a string and is used by the result writter
    """

    def __init__(self,
                 name,
                 actor,
                 optional=False,
                 terminate_on_failure=False):
        super(Criterion, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

        self.name = name
        self.actor = actor
        self.optional = optional
        self._terminate_on_failure = terminate_on_failure
        self.test_status = "INIT"   # Either "INIT", "RUNNING", "SUCCESS", "ACCEPTABLE" or "FAILURE"

        # Attributes to compare the current state (actual_value), with the expected ones
        self.success_value = 0
        self.acceptable_value = None
        self.actual_value = 0
        self.units = "times"

        self.events = []  # List of events (i.e collision, sidewalk invasion...)
        
        self.st_detail = {
            "occurred": False,
            "details": {}
        }

    def initialise(self):
        """
        Initialise the criterion. Can be extended by the user-derived class
        """
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        """
        Terminate the criterion. Can be extended by the user-derived class
        """
        if self.test_status in ('RUNNING', 'INIT'):
            self.test_status = "SUCCESS"

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        
        