#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides ScenarioTree, the base class for sub-scenario behavior trees.
"""

import operator

import py_trees

from typing import Optional

from scenario_runner.ctn_operator import CtnSimOperator

from scenario_elements.behavior.atomic import UpdateAllActorControls
from scenario_elements.criteria.atomic import TimeOut, Criterion

class ScenarioTree:
    """
    Base class for sub-scenario behavior trees.

    Controls the behaviors of sub-scenarios by composing two parallel trees:
    - behavior_tree: trigger_node -> running_node -> end_node
    - criteria: runtime monitor nodes running in parallel

    Subclasses should override the _create_* methods to define actors, behaviors,
    and test criteria.
    """
    def __init__(
        self, 
        name: str, # NOTE: we ignore config here, not unified
        ctn_operator: CtnSimOperator,
        terminate_on_failure: bool = True,
        criteria_enable: bool = True, # NOTE: for each sub-scenario, you should edit your own criteria
        debug_mode: bool = False,
        timeout: Optional[float] = None
    ):
        # assigned parameters
        self.name = name
        self.ctn_operator = ctn_operator
        self.terminate_on_failure = terminate_on_failure
        self.criteria_enable = criteria_enable
        self.debug_mode = debug_mode
        self.timeout = timeout
        
        # some convenient settings
        self.world = self.ctn_operator.get_world()
        self.map = self.ctn_operator.get_map()
        self.is_sync_mode = self.ctn_operator.is_sync_mode
        
        if debug_mode:
            py_trees.logging.level = py_trees.logging.Level.DEBUG

        # internal parameters
        self.other_actors = {}
        self.scenario_tree = None
        self.criteria_tree = None
        self.timeout_node = None
        
    def initialize(self):
        # create the scenario step by step
        # 1. initialize the scenario with env and actors
        self._initialize_actors() # NOTE: you would be better to initialize actors, envs, etc. here
        self._initialize_environment()
        
        # better tick here after initialization
        if self.is_sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # 2. create the scenario tree
        self.scenario_tree = py_trees.composites.Parallel(
            name=self.name,
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        
        # create the behavior tree - sequence
        self.behavior_tree = py_trees.composites.Sequence(
            name=f"{self.name}_behavior_sequence"
        )  # Placeholder for the actual behavior tree
        
        # TODO: do we need trigger node?
        trigger_node = self._setup_scenario_trigger()
        if trigger_node:
            self.behavior_tree.add_child(trigger_node)
        
        behavior_node = self._create_behavior()
        if behavior_node:
            self.behavior_tree.add_child(behavior_node)
        
        # TODO: do we need end node?
        end_node = self._setup_scenario_end()
        if end_node:
            self.behavior_tree.add_child(end_node)
        
        # add the behavior tree to the scenario tree
        self.scenario_tree.add_child(self.behavior_tree)
            
        # Create the criteria tree
        if self.criteria_enable:
            criteria = self._create_test_criteria()

            if isinstance(criteria, py_trees.composites.Composite):
                self.criteria_tree = criteria

            elif isinstance(criteria, list):
                if len(criteria) > 0:
                    for criterion in criteria:
                        criterion.terminate_on_failure = self.terminate_on_failure

                    self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria",
                                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
                    self.criteria_tree.add_children(criteria)
                    self.criteria_tree.setup(timeout=1)
                else:
                    self.criteria_tree = None
            else:
                raise ValueError("WARNING: Scenario {} couldn't be setup, make sure the criteria is either "
                                 "a list or a py_trees.composites.Composite".format(self.name))

            if self.criteria_tree:
                self.scenario_tree.add_child(self.criteria_tree)

        # Create the timeout behavior
        if self.timeout:
            self.timeout_node = self._create_timeout_behavior()
            if self.timeout_node:
                self.scenario_tree.add_child(self.timeout_node)

        # Add other nodes
        self.scenario_tree.add_child(UpdateAllActorControls())
        self.scenario_tree.setup(timeout=1)
        
    def tick(self):
        if self.scenario_tree is None:
            raise RuntimeError("Scenario tree is not initialized, call initialize() first.")
        return self.scenario_tree.tick_once()
        
    def _initialize_actors(self):
        """Initialize actors for this sub-scenario. Override in subclass."""
        pass

    def _initialize_environment(self):
        """Initialize environment (weather, traffic lights, etc.). Override in subclass."""
        pass

    def _setup_scenario_trigger(self):
        """Return a trigger node, or None to skip. Override in subclass."""
        return None

    def _setup_scenario_end(self):
        """Return an end node, or None to skip. Override in subclass."""
        return None

    def _create_behavior(self):
        """Return the behavior sub-tree. Must be implemented in subclass."""
        raise NotImplementedError("_create_behavior() must be implemented in subclass")

    def _create_test_criteria(self) -> Optional[list]:
        """Return a list of criteria nodes or a Composite. Override in subclass."""
        return []

    def _create_timeout_behavior(self):
        """Return a timeout node. Override in subclass for custom timeout logic."""
        return TimeOut(self.timeout, name="TimeOut")

    @staticmethod
    def _extract_nodes_from_tree(tree):
        """Return all leaf nodes from the given behavior tree."""
        leaves = []
        queue = [tree]
        while queue:
            node = queue.pop(0)
            if node.children:
                queue.extend(node.children)
            else:
                leaves.append(node)

        if len(leaves) == 1 and isinstance(leaves[0], py_trees.composites.Parallel):
            return []

        return leaves

    def get_criteria(self):
        """Return all Criterion leaf nodes from the criteria tree."""
        if not self.criteria_tree:
            return []
        return [
            node for node in self._extract_nodes_from_tree(self.criteria_tree)
            if isinstance(node, Criterion)
        ]
    
    def terminate(self):
        
        # Get list of all nodes in the tree
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)

        # Cleanup all instantiated controllers
        actor_dict = {}
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass
        for actor_id in actor_dict:
            actor_dict[actor_id].reset()
        py_trees.blackboard.Blackboard().set("ActorsWithController", {}, overwrite=True)
    
    def remove_all_actors(self):
        """Remove all actors spawned by this sub-scenario."""
        for key, actor in self.other_actors.items():
            if actor is not None:
                self.ctn_operator.remove_actor(actor)
        self.other_actors = {}
