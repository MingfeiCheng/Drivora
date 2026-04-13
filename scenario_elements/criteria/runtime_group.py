import py_trees
import traceback

from loguru import logger
from typing import List

from .atomic.base import Criterion
from .runtime_single import RuntimeSingleTest


class RuntimeGroupTest(Criterion):
    """
    A group of detectors. Succeeds when all sub-criteria finish;
    fails if any sub-criterion reports FAILURE.
    """

    def __init__(
        self,
        name: str = "RuntimeGroupTest",
        detectors: List[RuntimeSingleTest] = None,
        terminate_on_failure: bool = True,
    ):
        super().__init__(name, None, False, terminate_on_failure)

        self.sub_criteria = detectors if detectors is not None else []

        self.st_detail = {}
        self.events = []
        self.test_status = "RUNNING"
        self.success_value = 1
        self.actual_value = 0

    def initialise(self):
        super().initialise()
        for criterion in self.sub_criteria:
            criterion.initialise()

    def update(self):
        if self.test_status == "FAILURE":
            return py_trees.common.Status.FAILURE
        if self.test_status == "SUCCESS":
            return py_trees.common.Status.SUCCESS

        new_status = py_trees.common.Status.RUNNING

        # Update all sub-criteria and collect results in one pass
        all_stopped = True
        for criterion in self.sub_criteria:
            try:
                criterion.update()
            except Exception as e:
                logger.error(f"RuntimeGroupTest {self.name} update error in {criterion.name}: {e}")
                traceback.print_exc()
                self.test_status = "FAILURE"
                self.actual_value = 0
                return py_trees.common.Status.FAILURE

            self.st_detail[criterion.actor_id] = criterion.st_detail
            if not criterion.get_stop():
                all_stopped = False

        if all_stopped:
            logger.warning(f"RuntimeGroupTest: {self.name} all sub criteria finished.")
            self.test_status = "SUCCESS"
            self.actual_value = 1
            new_status = py_trees.common.Status.SUCCESS

            for criterion in self.sub_criteria:
                if criterion.test_status == "FAILURE":
                    self.test_status = "FAILURE"
                    self.actual_value = 0
                    new_status = py_trees.common.Status.FAILURE

        return new_status

    def terminate(self, new_status):
        for criterion in self.sub_criteria:
            criterion.terminate(new_status)

        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))
        super().terminate(new_status)
