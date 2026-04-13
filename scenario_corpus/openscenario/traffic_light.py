from scenario_runner.ctn_operator import CtnSimOperator
from scenario_corpus.base.sub_scenario import ScenarioTree

from scenario_elements.behavior.traffic_light.behavior import TrafficLightBehavior, TrafficLightBehaviorConfig

class TrafficLightScenario(ScenarioTree):
    """Traffic light scenario using rule-based cycling controller."""

    def __init__(
            self,
            name: str,
            config: TrafficLightBehaviorConfig,
            ctn_operator: CtnSimOperator,
    ):
        super().__init__(name=name, ctn_operator=ctn_operator)
        self.config = config

    def _create_behavior(self):
        return TrafficLightBehavior(
            self.ctn_operator,
            pattern=self.config.pattern,
            yellow_time=self.config.yellow_time,
            red_time=self.config.red_time,
            green_time=self.config.green_time,
        )