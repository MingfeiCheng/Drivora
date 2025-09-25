#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide BasicScenario, the basic class of all the scenarios.
"""

from __future__ import print_function

import py_trees
import operator

from loguru import logger
from typing import Optional

from scenario_runner.ctn_operator import CtnSimOperator

from scenario_elements.behavior.atomic import UpdateAllActorControls
from scenario_elements.criteria.atomic import TimeOut, Criterion

class ScenarioTree:
    """
    NOTE: we do not need time out here, is controlled by global scenario manager
    
    A class to represent a sub-behavior tree for parallel execution.
    This is a placeholder for future implementation.
    
    This class only controls the behaviors of sub-scenarios.
    
    Normally, we parallelly provide two trees:
    behavior_tree: trigger_node -> running_node -> end_node
    criteria: runtime monitor_node parallelly runs
    
    Tutorial:
    in the create methods, you can create your own actors, agents, and behaviors
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
        self.other_actors = {} # record the other actors in this sub-scenario
         
        # move initialize method to caller
        # calling by demands
        self.scenario_tree = None  # the root of the scenario tree
        
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
            
        # Create the criteria tree (if needed)
        if self.criteria_enable:
            criteria = self._create_test_criteria()

            # All the work is done, thanks!
            if isinstance(criteria, py_trees.composites.Composite):
                self.criteria_tree = criteria

            # Lazy mode, but its okay, we'll create the parallel behavior tree for you.
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
                
        else:
            self.timeout_node = None

        # Add other nodes
        self.scenario_tree.add_child(UpdateAllActorControls())
        self.scenario_tree.setup(timeout=1)
        
    # ======= Running configs =======
    def tick(self):
        """
        Tick the scenario tree
        """
        if self.scenario_tree is not None:
            return self.scenario_tree.tick_once()
        else:
            raise RuntimeError("Scenario tree is not initialized, please call initialize() first.")
        
    # define the initilization method
    def _initialize_actors(self):
        """_summary_
        This method aims to provide the initlization of the sub-scenario tree, including:
        1. mainly create the actors
        """
        pass
    
    def _initialize_environment(self):
        """_summary_
        This method aims to provide the initlization of the sub-scenario tree, including:
        1. set the weather
        2. set the traffic light
        """
        pass
        
    # create behavior tree nodes
    def _setup_scenario_trigger(self):
        # NOTE: you can replace this with your own trigger node logic
        return None
    
    def _setup_scenario_end(self):
        # NOTE: you can replace this with your own end node logic
        return None
    
    def _create_behavior(self):
        """_summary_
        The note likes a sub-treee to define the behavior of the node
        """
        # NOTE: you must replace this with your own running node logic
        raise NotImplementedError("This method should be implemented in subclass")
    
    # create runtime criteria nodes
    def _create_test_criteria(self) -> Optional[list]:
        """
        Create a runtime monitor for the sub-behavior tree.
        This is a placeholder for future implementation.
        """
        return []    
    
    def _create_timeout_behavior(self):
        """
        Default initialization of the timeout behavior.
        Override this method in child class to provide custom initialization.
        """
        return TimeOut(self.timeout, name="TimeOut")  # Timeout node

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in list(node_list):
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def get_criteria(self):
        """
        Return the list of test criteria, including all the leaf nodes.
        Some criteria might have trigger conditions, which have to be filtered out.
        """
        criteria = []
        if not self.criteria_tree:
            return criteria

        criteria_nodes = self._extract_nodes_from_tree(self.criteria_tree)
        for criterion in criteria_nodes:
            if isinstance(criterion, Criterion):
                criteria.append(criterion)

        return criteria
    
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
        """
        Remove all actors
        """
        if not hasattr(self, 'other_actors'):
            return
        
        for i, _ in self.other_actors.items():
            if self.other_actors[i] is not None:
                was_destroyed = self.ctn_operator.remove_actor(self.other_actors[i])
                # if was_destroyed:
                #     logger.debug(f"Actor {self.other_actors[i].id} destroyed successfully.")
                # else:
                #     logger.warning(f"Actor {self.other_actors[i].id} could not be destroyed.")
                self.other_actors[i] = None
        self.other_actors = {}
