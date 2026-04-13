import carla
import py_trees

from loguru import logger
from typing import List, Dict

from scenario_elements.trigger.atomic import TriggerTimer, StandStill
from scenario_elements.behavior.atomic import ActorDestroy, ActorTransformSetter
from scenario_elements.behavior.ai_walker.behavior import AIWalkerBehavior, AIWalkerConfig

from scenario_runner.ctn_operator import CtnSimOperator
from scenario_corpus.base.sub_scenario import ScenarioTree

class AIWalkerScenario(ScenarioTree):
    """NPC AI walker scenario tree."""

    def __init__(
        self,
        name: str,
        configs: List[AIWalkerConfig],
        ctn_operator: CtnSimOperator
    ):
        self.configs = configs
        self.npc_configs: Dict[str, AIWalkerConfig] = {cfg.id: cfg for cfg in self.configs}

        super().__init__(
            name=name,
            ctn_operator=ctn_operator
        )
        
    def _initialize_actors(self):
        """Spawn NPC walkers at their configured waypoints."""
        _carla_map = self.map
        for npc_id, npc_cfg in self.npc_configs.items():
            # 1. request new actor
            # -> spawn point of the actor
            npc_spawn_point = npc_cfg.get_initial_waypoint()
            npc_spawn_waypoint = _carla_map.get_waypoint(
                location=carla.Location(
                    x=npc_spawn_point.x,
                    y=npc_spawn_point.y,
                    z=npc_spawn_point.z
                )
            )
            npc_spawn_transform = npc_spawn_waypoint.transform
            npc_spawn_transform.location.z += 1.0 # no need this, as align at the beginning __init__
            
            created_actor = self.ctn_operator.request_new_actor(
                model=npc_cfg.model, 
                spawn_point=npc_spawn_transform,  # aligned Carla transform
                rolename=npc_cfg.rolename,
                autopilot=False,  
                color=npc_cfg.color, 
                actor_category=npc_cfg.category,  
                attribute_filter=None,
                tick=True
            )

            if created_actor is None:
                logger.warning(f"Failed to create NPC vehicle '{npc_id}' with model '{npc_cfg.model}' at spawn point {npc_spawn_transform.location}.")
                continue
                            
            if npc_id not in self.other_actors:
                self.other_actors[npc_id] = created_actor
            else:
                logger.warning(f"NPC vehicle ID '{npc_id}' already exists in the scenario tree. Skipping creation.")
                continue
    
    def _create_behavior(self):
        
        actor_behavior_pool = py_trees.composites.Parallel(
            name="aiwalker_behavior_pool",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        
        for actor_id, actor_config in self.npc_configs.items():
            if actor_id not in self.other_actors:
                continue

            actor = self.other_actors[actor_id]

            py_trees_name = f"ai_walker_{actor_config.id}"

            actor_tree = py_trees.composites.Sequence(name=py_trees_name)
            actor_start_wp = actor_config.route[0]
            actor_transform = carla.Transform(
                location=carla.Location(
                    x=actor_start_wp.x,
                    y=actor_start_wp.y,
                    z=0.5
                ),
                rotation=carla.Rotation(
                    pitch=actor_start_wp.pitch,
                    yaw=actor_start_wp.yaw,
                    roll=actor_start_wp.roll
                )
            )
            start_transform = ActorTransformSetter(actor, actor_transform)
            actor_tree.add_child(start_transform)
            actor_tree.add_child(
                TriggerTimer(actor, name=f"{py_trees_name}_behavior_trigger_time", duration=actor_config.trigger_time))

            actor_behavior = py_trees.composites.Parallel(
                name=f"{py_trees_name}_behavior",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
            )

            actor_behavior.add_child(AIWalkerBehavior(
                self.ctn_operator,
                actor,
                actor_config
            ))
            actor_behavior.add_child(StandStill(actor, name=f"{py_trees_name}_behavior_standstill", duration=3))

            actor_tree.add_child(actor_behavior)
            actor_tree.add_child(StandStill(actor, name=f"{py_trees_name}_behavior_standstill", duration=5)) # 80 < 90
            actor_tree.add_child(ActorDestroy(actor, self.ctn_operator))

            actor_behavior_pool.add_child(actor_tree)

        return actor_behavior_pool
    
    def remove_all_actors(self):
        """Remove all walker actors and their AI controllers."""
        super().remove_all_actors()

        # Clean up AI walker controllers
        world = self.ctn_operator.get_world()
        if world is not None:
            for controller in world.get_actors().filter('controller.ai.walker'):
                controller.stop()
                controller.destroy()