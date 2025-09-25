import py_trees

from typing import Optional

from .sub_scenario import ScenarioTree

from scenario_runner.ctn_operator import CtnSimOperator
from scenario_elements.behavior.atomic import Idle

class BasicScenario(ScenarioTree):
    
    def __init__(
        self, 
        name: str,
        ctn_operator: CtnSimOperator,
        terminate_on_failure: bool = True,
        criteria_enable: bool = True,
        debug_mode: bool = False,
        timeout: Optional[float] = None
    ):
        super(BasicScenario, self).__init__(
            name=name,
            ctn_operator=ctn_operator,
            terminate_on_failure=terminate_on_failure,
            criteria_enable=criteria_enable,
            debug_mode=debug_mode,
            timeout=timeout
        )
        
        self.list_scenarios = [] # to store the sub-scenarios
        self.ego_vehicles = {} # to store the ego vehicles/ carla actors
        self.ego_gps_routes = {} # to store the ego gps routes if any
        self.ego_routes = {} # to store the ego routes if any
        self.ego_destinations = {} # to store the ego destinations if any
        self.ego_trigger_times = {} # to store the ego triggers if any

    def initialize(self):
        self._setup_ego_vehicles()
        self._setup_scenarios()
        
        super(BasicScenario, self).initialize()
    
    def _setup_ego_vehicles(self):
        raise NotImplementedError("_setup_ego_vehicles() must be implemented in the subclass.")
        
    def _setup_scenarios(self):
        raise NotImplementedError("_setup_scenarios() must be implemented in the subclass.")
    
    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        behavior = py_trees.composites.Parallel(
            name="WaypointScenarioBehavior",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        
        scenario_behaviors = []
        # todo: consider add trigger position?
        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario_tree is not None:
                # only execute once
                name = "{} - {}".format(i, scenario.scenario_tree.name)
                oneshot_idiom = self.oneshot_behavior(
                    name=name,
                    variable_name=name,
                    behaviour=scenario.scenario_tree)
                scenario_behaviors.append(oneshot_idiom)

        behavior.add_children(scenario_behaviors)
        behavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        return behavior
    
    @staticmethod
    def oneshot_behavior(
        variable_name, 
        behaviour, 
        name=None
    ):
        """
        This is taken from py_trees.idiom.oneshot.
        """
        if not name:
            name = behaviour.name

        subtree_root = py_trees.composites.Selector(name=name)

        # Initialize the variables
        blackboard = py_trees.blackboard.Blackboard()
        _ = blackboard.set(variable_name, False)

        # Wait until the scenario has ended
        check_flag = py_trees.blackboard.CheckBlackboardVariable(
            name=variable_name + " Done?",
            variable_name=variable_name,
            expected_value=True,
            clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
        )
        set_flag = py_trees.blackboard.SetBlackboardVariable(
            name="Mark Done",
            variable_name=variable_name,
            variable_value=True
        )
        # If it's a sequence, don't double-nest it in a redundant manner
        if isinstance(behaviour, py_trees.composites.Sequence):
            behaviour.add_child(set_flag)
            sequence = behaviour
        else:
            sequence = py_trees.composites.Sequence(name="OneShot")
            sequence.add_children([behaviour, set_flag])

        subtree_root.add_children([check_flag, sequence])
        return subtree_root