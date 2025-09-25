from scenario_runner.ctn_operator import CtnSimOperator
from scenario_corpus.base.sub_scenario import ScenarioTree

from scenario_elements.behavior.traffic_light.behavior import TrafficLightBehavior, TrafficLightBehaviorConfig

class TrafficLightScenario(ScenarioTree):
    """
    Traffic light scenario, used to manage traffic light scenarios
    TODO: unfinished
    """

    def __init__(
            self,
            name: str,
            config: TrafficLightBehaviorConfig,
            ctn_operator: CtnSimOperator,
    ):
        super(TrafficLightScenario, self).__init__(name=name, ctn_operator=ctn_operator)

        # assign configs and fix the z-axis of the spawn points
        self.config = config

    def _create_behavior(self):
        return TrafficLightBehavior(self.ctn_operator, self.config.pattern, self.config.yellow_time, self.config.red_time)