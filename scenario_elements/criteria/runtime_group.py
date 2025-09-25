import py_trees
import traceback

from loguru import logger
from typing import List

from .atomic.base import Criterion
from .runtime_single import RuntimeSingleTest       
        
class RuntimeGroupTest(Criterion):
    """
    A group of detectors, the criterion is successful when all the sub-criteria are successful
    """
    
    def __init__(
        self, 
        name: str = "RuntimeGroupTest",
        detectors: List[RuntimeSingleTest] = None,
        terminate_on_failure: bool = True,
    ):
        super(RuntimeGroupTest, self).__init__(name, None, False, terminate_on_failure)
                
        self.sub_criteria = detectors if detectors is not None else []
        self.sub_criteria_can_stop = [False for _ in range(len(self.sub_criteria))]
        
        self.st_detail = {}
        self.events = []
        self.test_status = "RUNNING"
        self.success_value = 1 # this defines the success value of the criterion, expected to be 1
        self.actual_value = 0 # this is the actual value of the criterion, updated during the update phase
                
    def initialise(self):
        super(RuntimeGroupTest, self).initialise()
        for criterion in self.sub_criteria:
            criterion.initialise()
        
    def update(self):
        
        new_status = py_trees.common.Status.RUNNING
        
        if self.test_status == "FAILURE":
            new_status = py_trees.common.Status.FAILURE
            return new_status
        
        if self.test_status == "SUCCESS":
            new_status = py_trees.common.Status.SUCCESS
            return new_status
        
        all_stop_count = 0
        for i, criterion in enumerate(self.sub_criteria):
            try:
                criterion_status = criterion.update()
            except Exception as e:
                logger.error(f"RuntimeGroupTest {self.name} update error in sub criterion {criterion.name}: {e}")
                self.test_status = "FAILURE"
                self.actual_value = 0
                new_status = py_trees.common.Status.FAILURE
                traceback.print_exc()
                # return new_status
    
        for i, criterion in enumerate(self.sub_criteria):
            self.st_detail[criterion.actor_id] = criterion.st_detail
            if criterion.get_stop():
                all_stop_count += 1
        
        if all_stop_count == len(self.sub_criteria):
            all_stop = True
        else:
            all_stop = False
        
        if all_stop:
            logger.warning(f"RuntimeGroupTest: {self.name} all sub criteria finished.")
            
            self.test_status = "SUCCESS"
            self.actual_value = 1
            new_status = py_trees.common.Status.SUCCESS
            
            for i, criterion in enumerate(self.sub_criteria):            
                if criterion.test_status == "FAILURE":
                    self.test_status = "FAILURE"
                    self.actual_value = 0
                    new_status = py_trees.common.Status.FAILURE   
            
            logger.warning(f"RuntimeGroupTest: {self.name} st_detail: {self.st_detail}")
            
        return new_status
                
        # self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, status, new_status))
        # logger.debug(f"RuntimeGroupTest: {self.name}, new_status: {new_status}, actual_value: {self.actual_value}, success_value: {self.success_value}")
        # return new_status

    def terminate(self, new_status):
        """
        Cleanup sensor
        """
        for criterion in self.sub_criteria:
            criterion.terminate(new_status)
        
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        super(RuntimeGroupTest, self).terminate(new_status)